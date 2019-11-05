""" Environment file wrapper for CARLA """

import gym
from gym.spaces import Box, Discrete, Tuple

from datetime import datetime
import os
import glob
import sys
import traceback
import random
import json
import numpy as np
import math
import copy
import cv2
import collections
import queue
import time

import environment.carla_9_4.scenarios as scenarios
import environment.carla_9_4.server as server
import environment.carla_9_4.planner as planner
import environment.carla_9_4.controller as controller
import environment.carla_9_4.sensors as sensors
from environment.carla_9_4.reward import compute_reward
from environment.carla_9_4.agents.navigation.roaming_agent import RoamingAgent
from environment.carla_9_4.config import DEFAULT_ENV, DISCRETE_ACTIONS, episode_measurements
import scipy.misc
from scipy.misc import imsave
from agents.tf.ae.util import *

try:
    import carla
except Exception as e:
    print("Failed to import Carla")
    raise e

from carla import ColorConverter as cc

CARLA_LOGS = os.path.expanduser("~/CARLA_LOGS/"+str(datetime.now()))
if not os.path.exists(CARLA_LOGS):
    os.makedirs(CARLA_LOGS)

class CarlaEnv(gym.Env):
    def __init__(self, config=DEFAULT_ENV, vis_wrapper=None, vis_wrapper_vae=None, logger=None):
        self.config = DEFAULT_ENV
        self._update_config(config)
        self.CarlaServer = None
        self.episode_measurements = episode_measurements
        self.episode_id = None
        self.vehicle_actor = None
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        
        # Can pass in train/test weather as an array
        self.weather = None
        self.camera_queue = queue.Queue()
        self.target_speed = self.config['target_speed']
        self.args_longitudinal_dict = {
            'K_P': 0.1,
            'K_D': 0.0005,
            'K_I': 0.4,
            'dt': 1/10.0}
        self.actor_list = []

        self.image_data = None
        self.source_transform = None
        self.destination_transform = None
        self.global_planner = None
        self.trace_route = None
        self.episode_num = 0
        self.total_steps = 0
        self.semantic_image = None
        self.unseen = False

        self.logger = logger
        self.vis_wrapper = vis_wrapper
        self.vis_wrapper_vae = vis_wrapper_vae

        self.dist_to_trajectory = None
        if(self.config['grayscale']):
            self.im_channels = 1
        else:
            self.im_channels = 3
        
        self.controller = controller.PIDLongitudinalController(K_P=self.args_longitudinal_dict['K_P'], K_D=self.args_longitudinal_dict['K_D'], K_I=self.args_longitudinal_dict['K_I'], dt=self.args_longitudinal_dict['dt'])

        # Start Carla Server
        serverStarted = False
        serverStartRetries = 0
        while ((not serverStarted) and serverStartRetries < self.config['server_retries']):
            try:
                self.CarlaServer = server.CarlaServer(config=self.config)
                serverStarted = True
            except Exception as e:
                print("Error in starting carla server : {}".format(traceback.format_exc()))
                self.CarlaServer.close()
                error = e
                serverStartRetries += 1
        
        time.sleep(10)

        # Create new client
        self.client =  self._spawn_client()
        print("server_version", self.client.get_server_version())

        # Commenting load_world, assuming default is set as Town01 in CARLA binary config
        # since sometimes, it causes timeout issues in the beginning
        # self._world = self.client.load_world(self.config['city_name'])
        
        self._world = self.client.get_world()

        settings = self._world.get_settings()
        if(self.config['sync_mode']):            
            settings.synchronous_mode = True
        
        if self.config["server_fps"] is not None and self.config["server_fps"] != 0:
            settings.fixed_delta_seconds =  1.0 / float(self.config["server_fps"])
        
        # We want to enable rendering
        settings.no_rendering_mode = False

        self._world.apply_settings(settings)

        self._map = self._world.get_map()
        
        self.blueprint_library = self._world.get_blueprint_library()
        self.spawn_points = self._world.get_map().get_spawn_points()

        if(self.config['train_config'] == 'baselines'):
            self.action_space = Discrete(len(DISCRETE_ACTIONS))
            image_space = Box(0, 255, shape=(self.config["y_res"], self.config["x_res"], self.im_channels * self.config["framestack"]), dtype=np.uint8)
            self.observation_space = image_space

        if(self.config['train_config'] == 'PPO'):
            if self.config["action_type"] == 'merged_gas':
                # Streer, Throttle
                self.action_space = Box(low=np.array([-0.5, -0.5]), high=np.array([0.5, 0.5]), dtype=np.float32)
            elif self.config["action_type"] == 'merged_speed':
                # Steer, Speed
                self.action_space = Box(low=np.array([-0.5, -10.0]), high=np.array([0.5, 10.0]), dtype=np.float32)
            elif self.config["action_type"] == 'merged_speed_tanh':
                # Steer, Speed
                self.action_space = Box(low=np.array([-0.5, -1.0]), high=np.array([0.5, 1.0]), dtype=np.float32)
            elif self.config["action_type"] == 'steer_only':
                # Steer only
                self.action_space = Box(low=np.array([-0.5]), high=np.array([0.5]), dtype=np.float32)
 
            if self.config["input_type"] == 'wp':
                self.observation_space = Box(low=np.array([-4.0]), high=np.array([4.0]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_constant' or self.config["input_type"] == 'wp_noise':
                self.observation_space = Box(low=-4.0, high=4.0, shape=(1, 2), dtype=np.float32)
            elif self.config["input_type"] == 'vae':
                self.observation_space = Box(low=np.finfo(np.float32).min,
                                        high=np.finfo(np.float32).max,
                                        shape=(1, 768), dtype=np.float32)
            elif self.config["input_type"] == 'wp_vae':
                self.observation_space = Box(low=np.finfo(np.float32).min,
                                        high=np.finfo(np.float32).max,
                                        shape=(1, 769), dtype=np.float32)

    def _update_config(self, config):
        for key, val in config.items():
            self.config[key] = val

    def _spawn_client(self, hostname='localhost', port_number=None):
        port_number = self.CarlaServer.server_port
        client = carla.Client(hostname, port_number)
        client.set_timeout(self.config["client_timeout_seconds"])
        return client

    def step(self, action):
        try:
            obs = self._step(action)
            return obs
        except Exception:
            print("Error during step, terminating episode early", traceback.format_exc())
            self.reset()

    def _step(self, action):
        if(self.config['discrete_actions']):
            action = DISCRETE_ACTIONS[int(action)]
            target_speed = float(np.clip(action[0] + 10.0, 0, self.target_speed))
            self.episode_measurements['target_speed'] = target_speed
            current_speed = self.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            throttle = self.controller.pid_control(target_speed, current_speed)
            brake = 0.0
            # throttle = float(np.clip(action[0], 0, 1))
            # brake = float(np.abs(np.clip(action[0], -1, 0)))
            steer = float(np.clip(action[1], -1, 1))
            reverse = False
            hand_brake = False

            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=0
            )
        else:
            control = self.get_control(action)

        #Print actions
        if self.config['verbose']:
            print("steer", control.steer, "throttle", control.throttle, "brake", control.brake,
                  "reverse", control.reverse)

        #Store control for this step
        self.episode_measurements['control_steer'] = control.steer
        self.episode_measurements['control_throttle'] = control.throttle
        self.episode_measurements['control_brake'] = control.brake
        self.episode_measurements['control_reverse'] = control.reverse
        self.episode_measurements['control_hand_brake'] = control.hand_brake
        

        world_frame = None

        #TODO: Increment steps inside of frame_skip?
        for _ in range(self.config["frame_skip"]):
            self.vehicle_actor.apply_control(control)
            world_frame = self._world.tick()
        self.num_steps += 1

        if not self.unseen:
            self.total_steps +=1
        self.episode_measurements['num_steps'] = self.num_steps

        # Read in preprocessed image
        sensor_image = self._read_data(world_frame)

        # Set state variables for reward calculation
        self.episode_measurements['num_collisions'] = self.collision_sensor.num_collisions
        if self.config["enable_lane_invasion_sensor"]:
            self.episode_measurements['num_laneintersections'] = self.lane_invasion_sensor.num_laneintersections
        self.location = self.vehicle_actor.get_location()
        self.episode_measurements['distance_to_goal'] = self.location.distance(self.destination_transform.location)
        if self.episode_measurements['min_distance_to_goal'] >= self.location.distance(self.destination_transform.location):
            self.episode_measurements['min_distance_to_goal'] = self.location.distance(self.destination_transform.location)
        self.episode_measurements['speed'] = self.get_speed_from_velocity(self.vehicle_actor.get_velocity())

        if self.config["algo"] == "AE":
            next_orientation, self.dist_to_trajectory = 0, 0
        else:
            next_orientation, self.dist_to_trajectory = self.global_planner.get_next_orientation_new(self.vehicle_actor.get_transform())
        
        self.episode_measurements['dist_to_trajectory'] = self.dist_to_trajectory
        
        reward = compute_reward(name=self.config['reward_function'],
                             prev_measurement=self.prev_measurement,
                             cur_measurement=self.episode_measurements,
                             config=self.config,
                             verbose=self.config["verbose"])
        self.total_reward += reward
        self.episode_measurements['reward'] = reward
        self.episode_measurements['total_reward'] = self.total_reward

        done = self._compute_done_condition()

        self.episode_measurements['done'] = done
        self.prev_measurement = copy.deepcopy(self.episode_measurements)

        obs = {}
        #TODO: Get branch_idx from planner and set accordingly.
        branch_idx = 1
        
        if self.config["input_type"] == 'vae' or self.config["input_type"] == 'wp_vae':
            semantic_image = sensor_image[:,:,0]
            semantic_image = reduce_classes(semantic_image)
            image_labels = convert_to_one_hot(semantic_image, num_classes=5)
            encoded_image = self.vae_observation(image_labels)
            encoded_image = encoded_image / self.config["vae_encoding_norm_factor"]
            obs['semantic_image'] = semantic_image
        
        if self.config["input_type"] == "ae_train":
            semantic_image = sensor_image[:,:,0]
            obs['semantic_image'] = semantic_image

        obs['image'] = sensor_image
        obs['speed'] = np.expand_dims(
            np.array([self.episode_measurements['speed']]), axis=0)  # * 3.6 / 30
        obs['dist_to_target'] = np.array(
            [self.episode_measurements['distance_to_goal']])

        obs['orientation'] = np.array([next_orientation])
        if self.config["input_type"] == 'wp_constant':
            obs['orientation'] = np.array([0.0, next_orientation])
        elif self.config["input_type"] == 'wp_noise':
            obs['orientation'] = np.array([np.random.normal(0.0, 1.0), next_orientation])

        reward = np.expand_dims(np.array([reward]), axis=0)
        done = np.expand_dims(np.array([done]), axis=0)

        if self.config["train_config"] == "PPO":
            if self.config["videos"]:
                if self.vis_wrapper is not None:
                    if self.config["input_type"] == 'vae' or self.config["input_type"] == 'wp_vae':
                        self.vis_wrapper.save_image(convert_to_rgb(obs['semantic_image'], reduced_classes=True).astype(np.uint8), self.num_steps)
                    else:
                        self.vis_wrapper.save_image(obs['image'], self.num_steps)
                if self.vis_wrapper_vae is not None:
                    self.vis_wrapper_vae.save_image(convert_to_rgb(convert_from_one_hot(self.vae.decode(encoded_image)[0]), reduced_classes=True).astype(np.uint8), self.num_steps)
            if not self.unseen and self.logger is not None:
                self.logger.log_scalar('timesteps/train/orientation', next_orientation, self.total_steps)
                # self.logger.log_scalar('timesteps/train/orientation_old', next_orientation_old, self.total_steps)
                self.logger.log_scalar('timesteps/train/throttle', control.throttle, self.total_steps)
                self.logger.log_scalar('timesteps/train/speed', self.episode_measurements['speed'] * 3.6, self.total_steps)
                self.logger.log_scalar('timesteps/train/steer', control.steer, self.total_steps)
                self.logger.log_scalar('timesteps/train/target_speed', self.episode_measurements['target_speed'], self.total_steps)
                self.logger.log_scalar('timesteps/train/dist_to_trajectory_reward', self.episode_measurements['dist_to_trajectory_reward'], self.total_steps)
                self.logger.log_scalar('timesteps/train/speed_reward', self.episode_measurements['speed_reward'], self.total_steps)
                self.logger.log_scalar('timesteps/train/steer_reward', self.episode_measurements['steer_reward'], self.total_steps)
                self.logger.log_scalar('timesteps/train/step_reward', self.episode_measurements['step_reward'], self.total_steps)
                            
            if done:
                self.episode_num += 1
                if not self.unseen and self.logger is not None:
                    self.logger.log_scalar('episodes/train/dist_to_target', self.episode_measurements['distance_to_goal'], self.episode_num)
                    self.logger.log_scalar('episodes/train/diff_dist_to_target', (self.episode_measurements['distance_to_goal'] - self.episode_measurements['min_distance_to_goal']), self.episode_num)
                    self.logger.log_scalar('episodes/train/reward', self.episode_measurements['total_reward'], self.episode_num)
                    self.logger.log_scalar('timesteps/train/dist_to_target', self.episode_measurements['distance_to_goal'], self.total_steps)
                    self.logger.log_scalar('timesteps/train/diff_dist_to_target', (self.episode_measurements['distance_to_goal'] - self.episode_measurements['min_distance_to_goal']), self.total_steps)
                    self.logger.log_scalar('timesteps/train/reward', self.episode_measurements['total_reward'], self.total_steps)
                if self.config["videos"]:
                    if self.vis_wrapper is not None:
                        self.vis_wrapper.generate_video(self.episode_num)
                        self.vis_wrapper.remove_images()
                    if self.vis_wrapper_vae is not None:
                        self.vis_wrapper_vae.generate_video(self.episode_num)
                        self.vis_wrapper_vae.remove_images()
        if self.config["input_type"] == 'vae':
            return encoded_image, reward, done, self.episode_measurements
        elif self.config["input_type"] == "wp_vae":
            orientation = np.expand_dims(obs['orientation'], axis = 0)
            fused_input = np.hstack([encoded_image, orientation])
            return fused_input, reward, done, self.episode_measurements
        elif self.config["input_type"] == "wp":
            return obs['orientation'], reward, done, self.episode_measurements
        elif self.config["input_type"] == "wp_noise":
            orientation = np.expand_dims(obs['orientation'], axis = 0)
            return orientation, reward, done, self.episode_measurements
        else:
            return obs, reward, done, self.episode_measurements

    def _set_scenario(self, unseen=False, town="Town01", index=0):
        if self.config["scenarios"] == "straight":
            # self.source_transform, self.destination_transform = scenarios.get_fixed_long_straight_path_Town01()
            self.source_transform, self.destination_transform = scenarios.get_straight_path(unseen, town, index)
        elif self.config["scenarios"] == "left_right_curved":
            self.source_transform, self.destination_transform = scenarios.get_left_right_randomly(unseen)
        elif self.config["scenarios"] == "right_curved":
            self.source_transform, self.destination_transform = scenarios.get_right_turn(unseen)
        elif self.config["scenarios"] == "left_curved":
            self.source_transform, self.destination_transform = scenarios.get_left_turn(unseen)
        elif self.config["scenarios"] == "t_junction":
            self.source_transform, self.destination_transform = scenarios.get_t_junction_path(unseen)
        elif self.config["scenarios"] == "curved":
            # self.source_transform, self.destination_transform = scenarios.get_fixed_long_curved_path_Town01()
            self.source_transform, self.destination_transform = scenarios.get_curved_path(unseen, town, index)
        elif self.config["scenarios"] == "navigation" or self.config["scenarios"] == "dynamic_navigation":
            self.source_transform, self.destination_transform = scenarios.get_navigation_path(unseen, town, index)
        else:
            raise ValueError("Scenarios Config not set!")

    def set_vae(self, vae):
        self.vae = vae
    
    def vae_observation(self, observation_image):
        if self.config["train_vae"]:
            self.vae.buffer_append(observation_image)
        ob = self.vae.encode(observation_image)
        return ob

    def get_control(self, action):
        """ Get Control object for Carla from action
        Input:
            - action: tuple containing (steer, throttle, brake) in [-1, 1]
        Output:
            - control: Control object for Carla
        """
        if self.config["action_type"] != "control":
            action = action.flatten()

        if self.config["action_type"] is "sep_gas":
            steer = float(action[0])
            throttle = float(action[1])
            brake = float(action[2])
        elif self.config["action_type"] is "merged_gas":
            steer = float(action[0])
            gas = float(action[1])
            # gas = gas + 0.25
            gas = np.clip(gas, 0.0, 0.7)
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config["action_type"] == "steer_only":
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(20.0)
            current_speed = self.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            throttle = self.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config["action_type"] == "throttle_only":
            steer = float(0.0)
            target_speed = float(np.clip(action[0], 0, self.target_speed))
            current_speed = self.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            throttle = self.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config["action_type"] == "merged_speed":
            # steer = float(action[0])
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(np.clip(action[1] + 10.0, 0, self.target_speed))
            current_speed = self.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            throttle = self.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config["action_type"] == "merged_speed_tanh":
            # steer = float(action[0])
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(np.clip((action[1] + 1) * 10.0, 0, self.target_speed))
            current_speed = self.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            throttle = self.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config["action_type"] == "control":
            return action
        
        self.episode_measurements["target_speed"] = target_speed
            
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0)

        return control

    def reset(self, unseen=False, index=0):
        return self._reset(unseen, index)

    def destroy_all_existing_actors(self):
        # Delete all existing actors
        for _ in range(len(self.actor_list)):
            try:
                actor = self.actor_list.pop()
                actor.destroy()
            except Exception as e:
                print("Error during destroying actor {0}:{1}: {2}".format(actor.type_id, actor.id,traceback.format_exc()))

    def clear_episode_measurements(self):
        for key, val in self.episode_measurements.items():
            self.episode_measurements[key] = 0
    
    def _reset(self, unseen=False, index=0):
        #TODO: Keep track of current location, and distance to goal (i.e. update eps meas params)

        self.clear_episode_measurements()

        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        # self.prev_image = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None
        self.unseen = unseen

        # Destroy
        self.destroy_all_existing_actors()

        self.camera_queue.queue.clear()

        try:
            vehicle_bp = self.blueprint_library.find(self.config['vehicle_type'])
            # vehicle_bp = self.blueprint_library.find(random.choice(self.config['vehicle_types']))
        except Exception as e:
            print("Error during vehicle creation: {}".format(traceback.format_exc()))

        # Set source and destination based on scenario
        # Currently scenarios are defined only for Town01
        if self.config["use_scenarios"] and (self.config["city_name"] == "Town01" or self.config["city_name"] == "Town02"):
            self._set_scenario(unseen=unseen, index=index, town=self.config["city_name"])
        else:
            self.source_transform, self.destination_transform = random.choice(self.spawn_points), random.choice(self.spawn_points)

        self.vehicle_actor = self._world.try_spawn_actor(vehicle_bp, self.source_transform)
        self.actor_list.append(self.vehicle_actor)
        self.location = self.vehicle_actor.get_location()

        if self.config["num_npc"] > 0:
            self.spawn_npc(self.config["num_npc"])    

        #TODO: Generalize this code to attach 'n' different sensors to the vehicle
        #Attach a sensor to the vehicle
        if self.config["semantic"]:
            sensor = self.config['sensors'][1]
        else:
            sensor = self.config['sensors'][0]

        camera = self.blueprint_library.find(sensor)
        camera.set_attribute('image_size_x', self.config['sensor_x_res'])
        camera.set_attribute('image_size_y', self.config['sensor_y_res'])
        camera.set_attribute('sensor_tick', self.config['sensor_tick'])
        camera.set_attribute('fov', '120')

        camera_transform = carla.Transform(carla.Location(x=5.0, z=20.0), carla.Rotation(pitch=270.0))
        self.camera_actor = self._world.spawn_actor(camera, camera_transform, attach_to=self.vehicle_actor)
        self.actor_list.append(self.camera_actor)
        
        self.camera_actor.listen(self.camera_queue.put)
        
        self.collision_sensor = sensors.CollisionSensor(self.vehicle_actor)
        self.actor_list.append(self.collision_sensor.sensor)

        if self.config["enable_lane_invasion_sensor"]:
            self.lane_invasion_sensor = sensors.LaneInvasionSensor(self.vehicle_actor)
            self.actor_list.append(self.lane_invasion_sensor.sensor)
            
        # Set state variables for reward calculation
        self.episode_measurements['num_collisions'] = self.collision_sensor.num_collisions
        if self.config["enable_lane_invasion_sensor"]:
            self.episode_measurements['num_laneintersections'] = self.lane_invasion_sensor.num_laneintersections
        self.location = self.vehicle_actor.get_location()
        self.episode_measurements['distance_to_goal'] = self.location.distance(self.destination_transform.location)
        self.episode_measurements['min_distance_to_goal'] = 1000000.0
        self.episode_measurements['speed'] = self.get_speed_from_velocity(self.vehicle_actor.get_velocity())

        
        time.sleep(1)

        # TODO: fix bug with no sensor_image. empty image for now
        # x_res = int(self.config["sensor_x_res"])
        # y_res = int(self.config["sensor_y_res"])
        #sensor_image = np.zeros(shape=(x_res, y_res, self.im_channels))
        # TODO: Change this to return the full measurement vector (like the step function)

        obs = {}

        # Ticking for 15 frames to handle car initialization in air
        for _ in range(15):
            world_frame = self._world.tick()

        image = self._read_data(world_frame)

        self.global_planner = planner.GlobalPlanner()
        self.trace_route  = self.global_planner._trace_route(self._map,
                                self.source_transform, self.destination_transform)
        self.global_planner.set_global_plan(self.trace_route)

        if self.config["algo"] == "AE":
            next_orientation, self.dist_to_trajectory = 0, 0
        else:
            next_orientation, self.dist_to_trajectory = self.global_planner.get_next_orientation_new(self.vehicle_actor.get_transform())
        
        obs['image'] = image
        if self.config["input_type"] == 'vae' or self.config["input_type"] == 'wp_vae':
            semantic_image = image[:,:,0]
            semantic_image = reduce_classes(semantic_image)
            image_labels = convert_to_one_hot(semantic_image, num_classes=5)
            encoded_image = self.vae_observation(image_labels)
            encoded_image = encoded_image / self.config["vae_encoding_norm_factor"]
            # print("Maximum value in encoded VAE features: {}".format(np.amax(encoded_image)))
            # print("Minimum value in encoded VAE features: {}".format(np.amin(encoded_image)))
            obs['semantic_image'] = semantic_image
        
        if self.config["input_type"] == "ae_train":
            semantic_image = image[:,:,0]
            obs['semantic_image'] = semantic_image
    
        obs['speed'] = np.expand_dims(np.array([self.episode_measurements['speed']]), axis=0) # * 3.6 / 30
        obs['dist_to_target'] = np.array([self.episode_measurements['distance_to_goal']])
        obs['orientation']= np.array([next_orientation])
        if self.config["input_type"] == 'wp_constant':
            obs['orientation'] = np.array([0.0, next_orientation])
        elif self.config["input_type"] == 'wp_noise':
            obs['orientation'] = np.array([np.random.normal(0.0, 1.0), next_orientation])
        self.prev_measurement = copy.deepcopy(self.episode_measurements)

        if self.config["input_type"] == 'vae':
            return encoded_image
        elif self.config["input_type"] == "wp_vae":
            orientation = np.expand_dims(obs['orientation'], axis = 0)
            fused_input = np.hstack([encoded_image, orientation])
            return fused_input
        elif self.config["input_type"] == "wp":
            return obs['orientation']
        elif self.config["input_type"] == "wp_noise":
            orientation = np.expand_dims(obs['orientation'], axis = 0)
            return orientation
        else:
            return obs
        
    def try_spawn_random_vehicle_at(self, blueprints, transform):
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self._world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            self.actor_list.append(vehicle)
            vehicle.set_autopilot()
            print('spawned %r at %s' % (vehicle.type_id, transform.location.x))
            return True
        return False
    
    def spawn_npc(self, number_of_vehicles):
        blueprints = self._world.get_blueprint_library().filter('vehicle.*')
        spawn_points = list(self._world.get_map().get_spawn_points())
        random.shuffle(spawn_points)

        if self.config["verbose"]:
            print('found %d spawn points.' % len(spawn_points))
        
        count = number_of_vehicles
        for spawn_point in spawn_points:
            if self.try_spawn_random_vehicle_at(blueprints, spawn_point):
                count -= 1
            if count <= 0:
                break

        while count > 0:
            if self.try_spawn_random_vehicle_at(blueprints, random.choice(spawn_points)):
                count -= 1

    def get_speed_from_velocity(self, velocity):
        speed = np.sqrt(velocity.x ** 2 + velocity.y **2 + velocity.z **2)
        return speed

    def _read_data(self, world_frame, timeout=10.0):

        cam_image = self._read_camera_data(world_frame, timeout)
        cam_image_p = self._preprocess_image(cam_image)
        return cam_image_p

    def _preprocess_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if(self.config['preprocess_crop_image']):
            array = array[200:500, 300:500] 

        array = cv2.resize(array, (self.config["x_res"], self.config["y_res"]), interpolation=cv2.INTER_NEAREST)

        return array

    def _read_camera_data(self, world_frame, timeout):

        data  = self._retrieve_data(self.camera_queue, timeout, world_frame)
        return data

    def _retrieve_data(self, sensor_queue, timeout, world_frame):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == world_frame:
                return data
            else:
                if self.config["verbose"]:
                    print("difference in frames, world_frame={0}, data_frame={1}".format(world_frame, data.frame))

    def _compute_done_condition(self):

        # Episode termination conditions
        success = self.episode_measurements["distance_to_goal"] < self.config["dist_for_success"]
        offlane = self.episode_measurements["offlane_steps"] > self.config["max_offlane_steps"]
        static = self.episode_measurements["static_steps"] > self.config["max_static_steps"]
        collision = np.absolute(self.episode_measurements["collision_reward"]) > 0
        maxStepsTaken = self.episode_measurements["num_steps"] > self.config['max_steps']
        offlane = False
        static = False

        # Do not want to terminate on reaching goal
        # in case of VAE training
        if self.config["algo"] == "AE":
            success = False

        if success:
            termination_state = 'success'
        elif collision:
            termination_state = 'collision'
        elif offlane:
            termination_state = 'offlane'
        elif static:
            termination_state = 'static'
        elif maxStepsTaken:
            termination_state = 'max_steps'
        else:
            termination_state = 'none'
        
        if self.config["verbose"]:
            print("Termination State: {}".format(termination_state))

        self.episode_measurements['termination_state'] = termination_state

        done = success or collision or offlane or static or maxStepsTaken
        return done

    def printInfo(self):
        print("Vehicle transform:{0}".format(self.vehicle_actor.get_transform()))
        print("Vehicle velocity:{0}".format(self.vehicle_actor.get_velocity()))

    def close(self):
        self.destroy_all_existing_actors()

        if not self.CarlaServer is None:
            self.CarlaServer.close()

    def __del__(self):
        self.close()

if __name__ == "__main__":
    env = CarlaEnv()
    env.reset()
