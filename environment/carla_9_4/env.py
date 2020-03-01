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
from environment.carla_9_4.agents.navigation.agent import Agent
from environment.carla_9_4.config import DEFAULT_ENV, DISCRETE_ACTIONS, episode_measurements
import scipy.misc
from scipy.misc import imsave
from agents.tf.ae.util import *
import matplotlib
import matplotlib.pyplot as plt

try:
    import carla
except Exception as e:
    print("Failed to import Carla")
    raise e

from carla import ColorConverter as cc
from carla.libcarla import Transform
from carla.libcarla import Location
from carla.libcarla import Rotation

from environment.carla_9_4.env_util import check_if_vehicle_in_same_lane

class CarlaEnv(gym.Env):
    def __init__(self, config=DEFAULT_ENV, vis_wrapper=None, vis_wrapper_vae=None, logger=None, log_dir=None):
        self.config = DEFAULT_ENV
        self._update_config(config)
        self.CarlaServer = None
        self.episode_measurements = episode_measurements
        self.episode_id = None
        self.vehicle_actor = None
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.log_dir = log_dir
        
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

        # Queue for stacked frames and measurements
        self.stacked_observation_queue = queue.Queue(maxsize=self.config['frame_stack_size'])

        self.image_data = None
        self.source_transform = None
        self.destination_transform = None
        self.global_planner = None
        self.trace_route = None
        self.episode_num = 0
        self.validation_episode_num = 0
        self.total_steps = 0
        self.semantic_image = None
        self.unseen = False
        self.index = 0

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
        
        time.sleep(60)

        # Create new client
        self.client =  self._spawn_client()
        print("server_version", self.client.get_server_version())

        # Commenting load_world, assuming default is set as Town01 in CARLA binary config
        # since sometimes, it causes timeout issues in the beginning
        self._world = self.client.load_world(self.config['city_name'])

        # time.sleep(20)
        
        self._world = self.client.get_world()

        settings = self._world.get_settings()
        if(self.config['sync_mode']):            
            settings.synchronous_mode = True
        
        if self.config["server_fps"] is not None and self.config["server_fps"] != 0:
            settings.fixed_delta_seconds =  1.0 / float(self.config["server_fps"])
        
        # We want to enable rendering
        settings.no_rendering_mode = False

        self._world.apply_settings(settings)

        time.sleep(20)

        self._map = self._world.get_map()
        
        self.blueprint_library = self._world.get_blueprint_library()
        self.spawn_points = self._world.get_map().get_spawn_points()
        
        if self.config["testing"]:
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in self.config['spawn_points_fixed_idx']]
        else:
            spawn_pt_idx = np.random.permutation(len(self.spawn_points))
            np.save(os.path.join(self.log_dir, "spawn_pt_order"), spawn_pt_idx)
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in spawn_pt_idx]

        if(self.config['train_config'] == 'baselines'):
            self.action_space = Discrete(len(DISCRETE_ACTIONS))
            image_space = Box(0, 255, shape=(self.config["y_res"], self.config["x_res"], self.im_channels * self.config["framestack"]), dtype=np.uint8)
            self.observation_space = image_space

        # TODO: Verify the limits and bounds of observation spaces
        if(self.config['train_config'] == 'PPO'):
            if self.config["action_type"] == 'merged_gas':
                # Streer, Throttle
                self.action_space = Box(low=np.array([-0.5, -0.5]), high=np.array([0.5, 0.5]), dtype=np.float32)
            elif self.config["action_type"] == 'merged_speed':
                # Steer, Speed
                self.action_space = Box(low=np.array([-0.5, -10.0]), high=np.array([0.5, 10.0]), dtype=np.float32)
            elif self.config["action_type"] == 'merged_speed_tanh' or self.config["action_type"] == 'merged_speed_scaled_tanh':
                # Steer, Speed
                self.action_space = Box(low=np.array([-0.5, -1.0]), high=np.array([0.5, 1.0]), dtype=np.float32)
            elif self.config["action_type"] == "merged_speed_pid_test":
                self.action_space = Box(low=np.array([-0.5, -20.0]), high=np.array([0.5, 20.0]), dtype=np.float32)
            elif self.config["action_type"] == 'steer_only':
                # Steer only
                self.action_space = Box(low=np.array([-0.5]), high=np.array([0.5]), dtype=np.float32)
 
            if self.config["input_type"] == 'wp':
                self.observation_space = Box(low=np.array([-4.0]), high=np.array([4.0]), dtype=np.float32)
            elif self.config["input_type"] in ['wp_constant', 'wp_noise', 'wp_obs_dist', 'wp_obs_bool']:
                self.observation_space = Box(low=np.array([[-4.0, -1.0]]), high=np.array([[4.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_obs_bool_noise':
                limit = np.hstack((np.array([[4]]), np.ones((1, 1 + self.config["noise_dim"]))))
                self.observation_space = Box(low=-limit, high=limit, shape=(1, 2 + self.config["noise_dim"]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_obs_bool_speed_steer_goal_light':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, -0.5, 0.0, -1.0]]), high=np.array([[4.0, 1.0, 1.0, 0.5, 1.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_goal_light':
                self.observation_space = Box(low=np.array([[-4.0, -1.0, -1.0, 0.0, -0.5, -1.0, 0.0, -1.0]]), high=np.array([[4.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0]]), dtype=np.float32)
            elif self.config["input_type"] == 'vae':
                self.observation_space = Box(low=np.finfo(np.float32).min,
                                        high=np.finfo(np.float32).max,
                                        shape=(1, 400), dtype=np.float32)
            elif self.config["input_type"] == 'wp_vae':
                self.observation_space = Box(low=np.finfo(np.float32).min,
                                        high=np.finfo(np.float32).max,
                                        shape=(1, 401), dtype=np.float32)
            elif self.config["input_type"] == 'wp_vae_speed_steer_goal':
                self.observation_space = Box(low=np.finfo(np.float32).min,
                                        high=np.finfo(np.float32).max,
                                        shape=(1, 404), dtype=np.float32)
            elif self.config["input_type"] == 'wp_vae_speed_steer_ldist_goal_light':
                self.observation_space = Box(low=np.finfo(np.float32).min,
                                        high=np.finfo(np.float32).max,
                                        shape=(1, 406), dtype=np.float32)
        
        self.vehicle_blueprints = self._world.get_blueprint_library().filter('vehicle.*')
        self.traffic_actors = self._world.get_actors().filter("*traffic_light*")

        if self.config["disable_two_wheeler"]:
            self.vehicle_blueprints = [x for x in self.vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        
        self.episode_measurements["episode_num"] = 0
        self.episode_measurements['obstacle_visible'] = False
        self.episode_measurements['obstacle_dist'] = -1
        self.episode_measurements['obstacle_speed'] = -1
        self.next_waypoints = None
        self.target_speeds_array = []
        self.speeds_array = []
        self.throttles_array = []
        self.steers_array = []
        self.brakes_array = []
        self.obstacle_dist_array = []
        self.step_reward_array = []
        self.collision_reward_array = []
        self.dist_to_trajectory_reward_array = []
        self.speed_reward_array = []
        self.dist_to_target_array = []
        
    def _update_config(self, config):
        for key, val in config.items():
            self.config[key] = val

    def _spawn_client(self, hostname='localhost', port_number=None):
        port_number = self.CarlaServer.server_port
        client = carla.Client(hostname, port_number)
        client.set_timeout(self.config["client_timeout_seconds"])
        return client

    def create_observations(self, obs):
        obs['observation'] = np.array([self.episode_measurements['next_orientation']])

        if self.config["input_type"] == 'wp_constant':
            obs['observation'] = np.array([self.episode_measurements['next_orientation'], 0.0])
        elif self.config["input_type"] == 'wp_noise':
            obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.random.normal(0.0, 1.0, self.config["noise_dim"])))
        elif self.config["input_type"] == 'wp_obs_dist':
            obs_dist = self.episode_measurements['obstacle_dist'] / self.config["obstacle_dist_norm"]
            obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([obs_dist])))
        elif self.config["input_type"] == 'wp_obs_bool':
            obs_bool = self.episode_measurements['obstacle_visible']
            obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([obs_bool])))
        elif self.config["input_type"] == 'wp_obs_bool_noise':
            obs_bool = self.episode_measurements['obstacle_visible']
            obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([obs_bool]), np.random.normal(0.0, 1.0, self.config["noise_dim"])))
        elif self.config["input_type"] == 'wp_obs_bool_speed_steer_goal_light':
            speed = self.episode_measurements['speed'] / 10
            obs_bool = self.episode_measurements['obstacle_visible']
            steer = self.episode_measurements['control_steer']
            distance_to_goal_trajec = self.episode_measurements['distance_to_goal_trajec'] / 500
            light = self.episode_measurements['red_light_dist']
            # normalization
            if light != -1:
                light /= self.config['proximity_threshold']
            obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([obs_bool]), np.array([speed]), np.array([steer]), np.array([distance_to_goal_trajec]), np.array([light])))
        elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_goal_light':
            speed = self.episode_measurements['speed'] / 10
            obstacle_dist = self.episode_measurements['obstacle_dist']
            obstacle_speed = self.episode_measurements['obstacle_speed']
            steer = self.episode_measurements['control_steer']
            ldist = self.episode_measurements['dist_to_trajectory']
            distance_to_goal_trajec = self.episode_measurements['distance_to_goal_trajec'] / 500
            light = self.episode_measurements['red_light_dist']
            # normalization
            if obstacle_dist != -1:
                obstacle_dist = obstacle_dist / self.config['proximity_threshold']

            if obstacle_speed != -1:
                obstacle_speed = obstacle_speed / 10

            if light != -1:
                light /= self.config['proximity_threshold']
            obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([distance_to_goal_trajec]), np.array([light])))
        elif self.config["input_type"] == 'wp_vae_speed_steer_goal':
            speed = self.episode_measurements['speed'] / 10
            steer = self.episode_measurements['control_steer']
            distance_to_goal_trajec = self.episode_measurements['distance_to_goal_trajec'] / 500
            obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([speed]), np.array([steer]), np.array([distance_to_goal_trajec])))
        elif self.config["input_type"] == 'wp_vae_speed_steer_ldist_goal_light':
            speed = self.episode_measurements['speed'] / 10
            steer = self.episode_measurements['control_steer']
            ldist = self.episode_measurements['dist_to_trajectory']
            distance_to_goal_trajec = self.episode_measurements['distance_to_goal_trajec'] / 500
            light = self.episode_measurements['red_light_dist']
            # normalization
            if light != -1:
                light /= self.config['proximity_threshold']
            obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([distance_to_goal_trajec]), np.array([light])))

    def step(self, action):
        try:
            obs = self._step(action)
            return obs
        except Exception:
            print("Error during step, terminating episode early", traceback.format_exc())
            self.reset()

    def _step(self, action):
        # if(self.config['discrete_actions']):
        #     action = DISCRETE_ACTIONS[int(action)]
        #     target_speed = float(np.clip(action[0] + 10.0, 0, self.target_speed))
        #     self.episode_measurements['target_speed'] = target_speed
        #     current_speed = self.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
        #     throttle = self.controller.pid_control(target_speed, current_speed)
        #     brake = 0.0
        #     # throttle = float(np.clip(action[0], 0, 1))
        #     # brake = float(np.abs(np.clip(action[0], -1, 0)))
        #     steer = float(np.clip(action[1], -1, 1))
        #     reverse = False
        #     hand_brake = False

        #     control = carla.VehicleControl(
        #         throttle=throttle,
        #         steer=steer,
        #         brake=brake,
        #         hand_brake=False,
        #         reverse=False,
        #         manual_gear_shift=False,
        #         gear=0
        #     )
        # else:
        #     control = self.get_control(action)

        # #Print actions
        # if self.config['verbose']:
        #     print("steer", control.steer, "throttle", control.throttle, "brake", control.brake,
        #           "reverse", control.reverse)

        # #Store control for this step
        # self.episode_measurements['control_steer'] = control.steer
        # self.episode_measurements['control_throttle'] = control.throttle
        # self.episode_measurements['control_brake'] = control.brake
        # self.episode_measurements['control_reverse'] = control.reverse
        # self.episode_measurements['control_hand_brake'] = control.hand_brake
        

        world_frame = None

        reward = 0

        if not self.config["use_pid_in_frame_skip"]:
            # compute control using PID for each timestep
            control = self.get_control(action)

            #Store control for this step
            self.episode_measurements['control_steer'] = control.steer
            self.episode_measurements['control_throttle'] = control.throttle
            self.episode_measurements['control_brake'] = control.brake
            self.episode_measurements['control_reverse'] = control.reverse
            self.episode_measurements['control_hand_brake'] = control.hand_brake

        for _ in range(self.config["frame_skip"]):

            if self.config["use_pid_in_frame_skip"]:
                # compute control using PID for each timestep
                control = self.get_control(action)

                #Store control for this step
                self.episode_measurements['control_steer'] = control.steer
                self.episode_measurements['control_throttle'] = control.throttle
                self.episode_measurements['control_brake'] = control.brake
                self.episode_measurements['control_reverse'] = control.reverse
                self.episode_measurements['control_hand_brake'] = control.hand_brake

            self.vehicle_actor.apply_control(control)
            world_frame = self._world.tick()
            self.num_steps += 1

            if not self.unseen:
                self.total_steps +=1
        
            self.episode_measurements['num_steps'] = self.num_steps

            # Set state variables for reward calculation
            self.episode_measurements['num_collisions'] = self.collision_sensor.num_collisions
            if self.config["enable_lane_invasion_sensor"]:
                self.episode_measurements['num_laneintersections'] = self.lane_invasion_sensor.num_laneintersections
                self.episode_measurements['out_of_road'] = int(self.lane_invasion_sensor.out_of_road)
            self.location = self.vehicle_actor.get_location()
            self.episode_measurements['distance_to_goal'] = self.location.distance(self.destination_transform.location)
            if self.episode_measurements['min_distance_to_goal'] >= self.location.distance(self.destination_transform.location):
                self.episode_measurements['min_distance_to_goal'] = self.location.distance(self.destination_transform.location)
            self.episode_measurements['speed'] = self.get_speed_from_velocity(self.vehicle_actor.get_velocity())

            if self.config["algo"] == "AE":
                next_orientation, self.dist_to_trajectory = 0, 0
            else:
                next_orientation, self.dist_to_trajectory, distance_to_goal_trajec, self.next_waypoints = self.global_planner.get_next_orientation_new(self.vehicle_actor.get_transform())
            self.episode_measurements['next_orientation'] = next_orientation
            self.episode_measurements['distance_to_goal_trajec'] = distance_to_goal_trajec
            self.episode_measurements['dist_to_trajectory'] = self.dist_to_trajectory

            # Update obstacle distance measurements
            # if self.config["input_type"] in ["wp_obs_bool", "wp_obs_bool_noise"]:
            self._update_env_obs()
            self.obstacle_dist_array.append(self.episode_measurements['obstacle_dist'])
            if self.config["scenarios"] == "straight_dynamic":
                self._update_straight_dynamic_obs()

            reward += compute_reward(name=self.config['reward_function'],
                                prev_measurement=self.prev_measurement,
                                cur_measurement=self.episode_measurements,
                                config=self.config,
                                verbose=self.config["verbose"])
            
            done = self._compute_done_condition()

            self.episode_measurements['done'] = done
            self.prev_measurement = copy.deepcopy(self.episode_measurements)

            self.target_speeds_array.append(self.episode_measurements['target_speed'])
            self.speeds_array.append(self.episode_measurements['speed'] * 3.6)
            self.throttles_array.append(control.throttle)
            self.steers_array.append(control.steer)
            self.brakes_array.append(control.brake)
            self.step_reward_array.append(self.episode_measurements['step_reward'])
            self.collision_reward_array.append(self.episode_measurements['collision_reward'])
            self.dist_to_trajectory_reward_array.append(self.episode_measurements['dist_to_trajectory_reward'])
            self.speed_reward_array.append(self.episode_measurements['speed_reward'])
            self.dist_to_target_array.append(self.episode_measurements['distance_to_goal_trajec'])

            if done:
                break
        
        self.total_reward += reward
        self.episode_measurements['reward'] = reward
        self.episode_measurements['total_reward'] = self.total_reward

        obs = {}
        #TODO: Get branch_idx from planner and set accordingly.
        branch_idx = 1

        
        # Read in preprocessed image
        sensor_image = self._read_data(world_frame)
        encoded_observation = None
        if self.config["input_type"] in ['vae', 'wp_vae', 'wp_vae_speed_steer_goal']:
            semantic_image = sensor_image[:,:,0]
            semantic_image = reduce_classes(semantic_image)
            image_labels = convert_to_one_hot(semantic_image, num_classes=5)
            self._add_to_stacked_queue(self.stacked_observation_queue, image_labels)
            stacked_observation = np.concatenate(list(self.stacked_observation_queue.queue), axis=2)
            # encoded_observation = self.vae_observation(image_labels)
            encoded_observation = self.vae_observation(stacked_observation)
            encoded_observation = encoded_observation / self.config["vae_encoding_norm_factor"]
            obs['semantic_image'] = semantic_image
        
        if self.config["input_type"] == "ae_train":
            semantic_image = sensor_image[:,:,0]
            obs['semantic_image'] = semantic_image

        obs['image'] = sensor_image
        obs['speed'] = np.expand_dims(
            np.array([self.episode_measurements['speed']]), axis=0)  # * 3.6 / 30
        obs['dist_to_target'] = np.array(
            [self.episode_measurements['distance_to_goal']])

        self.create_observations(obs)
        # if self.config["input_type"] == 'wp_constant':
        #     obs['observation'] = np.array([0.0, self.episode_measurements['next_orientation']])
        # elif self.config["input_type"] == 'wp_noise':
        #     obs['observation'] = np.concatenate((np.random.normal(0.0, 1.0, self.config["noise_dim"]), np.array([self.episode_measurements['next_orientation']])))
        # elif self.config["input_type"] == 'wp_obs_dist':
        #     obs_dist = self.episode_measurements['obstacle_dist'] / self.config["obstacle_dist_norm"]
        #     obs['observation'] = np.concatenate((np.array([obs_dist]), np.array([self.episode_measurements['next_orientation']])))
        # elif self.config["input_type"] == 'wp_obs_bool':
        #     obs_bool = self.episode_measurements['obstacle_visible']
        #     obs['observation'] = np.concatenate((np.array([obs_bool]), np.array([self.episode_measurements['next_orientation']])))
        # elif self.config["input_type"] == 'wp_obs_bool_noise':
        #     obs_bool = self.episode_measurements['obstacle_visible']
        #     obs['observation'] = np.concatenate((np.random.normal(0.0, 1.0, self.config["noise_dim"]), np.array([obs_bool]), np.array([self.episode_measurements['next_orientation']])))
        # elif self.config["input_type"] == 'wp_vae_speed_steer_goal':
        #     speed = self.episode_measurements['speed'] / 20
        #     steer = self.episode_measurements['control_steer']
        #     distance_to_goal_trajec = distance_to_goal_trajec / 100
        #     obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([speed]), np.array([steer]), np.array([distance_to_goal_trajec])))
        reward = np.expand_dims(np.array([reward]), axis=0)
        done = np.expand_dims(np.array([done]), axis=0)

        if self.config["train_config"] == "PPO":
            # Save videos now only for validation runs
            if self.config["videos"] and self.unseen:
                if self.vis_wrapper is not None:
                    # if self.config["input_type"] in ['vae', 'wp_vae', 'wp_vae_speed_steer_goal']:
                    #     # self.vis_wrapper.save_semantic_image(obs['semantic_image'], self.num_steps)
                    #     self.vis_wrapper.save_pil_image(convert_to_rgb(obs['semantic_image'], reduced_classes=True).astype(np.uint8), self.num_steps, self.episode_measurements)
                    # else:
                    #     # path = os.path.join(self.log_dir, "ae_images")
                    #     # if not os.path.exists(path):
                    #     #     os.makedirs(path)
                    #     # np.savez_compressed(os.path.join(path, format(self.total_steps, '08d')), img=convert_to_one_hot(reduce_classes(obs['image'][:, :, 0]), num_classes=5))
                    self.vis_wrapper.save_pil_image(convert_to_rgb(reduce_classes(obs['image'][:, :, 0]), reduced_classes=True).astype(np.uint8), self.num_steps, self.episode_measurements)
                if self.vis_wrapper_vae is not None:
                    self.vis_wrapper_vae.save_pil_image(convert_to_rgb(convert_from_one_hot(self.vae.decode(encoded_observation)[0, :, :, -5:]), reduced_classes=True).astype(np.uint8), self.num_steps, self.episode_measurements)
            if not self.unseen and self.logger is not None and self.total_steps % self.config["log_freq"] == 0:
                # self.logger.log_scalar('timesteps/train/orientation', self.episode_measurements['next_orientation'], self.total_steps)
                # self.logger.log_scalar('timesteps/train/orientation_old', next_orientation_old, self.total_steps)
                self.logger.log_scalar('timesteps/train/c_throttle', control.throttle, self.total_steps)
                self.logger.log_scalar('timesteps/train/c_speed', self.episode_measurements['speed'] * 3.6, self.total_steps)
                self.logger.log_scalar('timesteps/train/c_steer', control.steer, self.total_steps)
                self.logger.log_scalar('timesteps/train/c_brake', self.episode_measurements['control_brake'], self.total_steps)
                self.logger.log_scalar('timesteps/train/c_speed_target', self.episode_measurements['target_speed'], self.total_steps)
                self.logger.log_scalar('timesteps/train/reward_dist_to_trajectory', self.episode_measurements['dist_to_trajectory_reward'], self.total_steps)
                self.logger.log_scalar('timesteps/train/reward_speed', self.episode_measurements['speed_reward'], self.total_steps)
                # self.logger.log_scalar('timesteps/train/steer_reward', self.episode_measurements['steer_reward'], self.total_steps)
                self.logger.log_scalar('timesteps/train/reward_step', self.episode_measurements['step_reward'], self.total_steps)
                self.logger.log_scalar('timesteps/train/reward_collision', self.episode_measurements['collision_reward'], self.total_steps)
                self.logger.log_scalar('timesteps/train/obstacle_visible', self.episode_measurements['obstacle_visible'], self.total_steps)

                if self.config["scenarios"] == "straight_dynamic":
                    self._update_straight_dynamic_obs()
                    # car_spawn_point = Transform(Location(x=92.10997772216797, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672))
                    # location = self.vehicle_actor.get_location()
                    # distance_to_car = location.distance(car_spawn_point.location)

                    # self.episode_measurements['obstacle_dist'] = distance_to_car

                    if self.episode_measurements['obstacle_dist'] < 10:
                        speed_near_car = self.episode_measurements['speed'] * 3.6
                        target_speed_near_car = self.episode_measurements['target_speed']
                    else:
                        speed_near_car = -10
                        target_speed_near_car = -10
            
                    self.logger.log_scalar('timesteps/train/near_car_speed', speed_near_car, self.total_steps)
                    self.logger.log_scalar('timesteps/train/near_car_target_speed', target_speed_near_car, self.total_steps)
                    self.logger.log_scalar('timesteps/train/obstacle_dist', self.episode_measurements['obstacle_dist'], self.total_steps)

                    # if distance_to_car < 20:
                    #     self.episode_measurements['obstacle_visible'] = True
                    # else:
                    #     self.episode_measurements['obstacle_visible'] = False
            
            if done:

                # Training runs
                if not self.unseen:
                    self.episode_num += 1
                    
                    # Commenting out plots for all episodes

                    # path = self.log_dir + 'episode_info_plots/'
                    # ep_idx = 'E_' + str(self.episode_num) + '_t_' + str(self.total_steps)
                    # plot_episode_info(path,
                    #     self.target_speeds_array,
                    #     self.speeds_array,
                    #     self.throttles_array,
                    #     self.steers_array,
                    #     self.brakes_array,
                    #     self.dist_to_target_array,
                    #     self.step_reward_array,
                    #     self.collision_reward_array,
                    #     self.dist_to_trajectory_reward_array,
                    #     self.speed_reward_array,
                    #     ep_idx)

                # Validation runs
                else:
                    self.validation_episode_num += 1
                    plotname = 'ValEp_' + str(self.validation_episode_num) + '_TrainEp_' + str(self.episode_num) + '_step_' + str(self.total_steps) + "_ind_" + str(self.index)
                    
                    if self.config["testing"]:
                        path = self.log_dir + 'test_episode_info_plots/'
                    else:
                        path = self.log_dir + 'val_episode_info_plots/'
                    plot_episode_info(path,
                        self.target_speeds_array,
                        self.speeds_array,
                        self.throttles_array,
                        self.steers_array,
                        self.brakes_array,
                        self.obstacle_dist_array,
                        self.step_reward_array,
                        self.collision_reward_array,
                        self.dist_to_trajectory_reward_array,
                        self.speed_reward_array,
                        plotname)
                
                self.episode_measurements["episode_num"] = self.episode_num

                if self.logger is not None:

                    if not self.unseen:
                        self.logger.log_scalar('episodes/train/dist_to_target', self.episode_measurements['distance_to_goal'], self.episode_num)
                        # self.logger.log_scalar('episodes/train/diff_dist_to_target', (self.episode_measurements['distance_to_goal'] - self.episode_measurements['min_distance_to_goal']), self.episode_num)
                        self.logger.log_scalar('episodes/train/reward', self.episode_measurements['total_reward'], self.episode_num)
                        self.logger.log_scalar('timesteps/train/dist_to_target', self.episode_measurements['distance_to_goal'], self.total_steps)
                        # self.logger.log_scalar('timesteps/train/diff_dist_to_target', (self.episode_measurements['distance_to_goal'] - self.episode_measurements['min_distance_to_goal']), self.total_steps)
                        self.logger.log_scalar('timesteps/train/reward', self.episode_measurements['total_reward'], self.total_steps)

                        # New logs
                        self.logger.log_scalar('episodes/train/reward_collision', self.episode_measurements['collision_reward'], self.episode_num)
                        self.logger.log_scalar('episodes/train/out_of_road', self.episode_measurements['out_of_road'], self.episode_num)
                        self.logger.log_scalar('episodes/train/collision_occured', self.episode_measurements['is_collision'], self.episode_num)
                        self.logger.log_scalar('episodes/train/obstacle_dist', self.episode_measurements['obstacle_dist'], self.episode_num)

                    elif self.unseen:

                        self.logger.log_scalar('test/dist_to_target_' + str(self.index), self.episode_measurements['distance_to_goal'], self.total_steps)
                        self.logger.log_scalar('test/reward_' + str(self.index), self.episode_measurements['total_reward'], self.total_steps)

                        self.logger.log_scalar('test/reward_collision_' + str(self.index), self.episode_measurements['collision_reward'], self.total_steps)
                        self.logger.log_scalar('test/out_of_road_' + str(self.index), self.episode_measurements['out_of_road'], self.total_steps)

                # Save videos now only for validation runs
                if self.config["videos"] and self.unseen:
                    if self.vis_wrapper is not None:
                        # self.vis_wrapper.generate_video(self.episode_num)
                        self.vis_wrapper.generate_video(self.validation_episode_num, self.total_steps, self.index)
                        self.vis_wrapper.remove_images()
                    if self.vis_wrapper_vae is not None:
                        # self.vis_wrapper_vae.generate_video(self.episode_num)
                        self.vis_wrapper_vae.generate_video(self.validation_episode_num, self.total_steps, self.index)
                        self.vis_wrapper_vae.remove_images()

        if self.config["input_type"] == 'vae':
            return encoded_observation, reward, done, self.episode_measurements
        elif self.config["input_type"] in ['wp_vae', 'wp_vae_speed_steer_goal', 'wp_vae_speed_steer_ldist_goal_light']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            fused_input = np.hstack([encoded_observation, observation])
            return fused_input, reward, done, self.episode_measurements
        elif self.config["input_type"] == "wp":
            return obs['observation'], reward, done, self.episode_measurements
        elif self.config["input_type"] in ['wp_noise', 'wp_constant', 'wp_obs_dist', 'wp_obs_bool', 'wp_obs_bool_noise',
                                           'wp_obs_bool_speed_steer_goal_light', 'wp_obs_info_speed_steer_ldist_goal_light']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            return observation, reward, done, self.episode_measurements
        else:
            return obs, reward, done, self.episode_measurements
    
    def _add_to_stacked_queue(self, object_queue, object_to_add):

        assert (object_queue is not None and object_to_add is not None)

        if object_queue.full():
            # Pop out earlier stacked frame if queue is full
            object_queue.get()
        object_queue.put(object_to_add)

    def _update_straight_dynamic_obs(self):
        car_spawn_point = Transform(Location(x=92.10997772216797, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672))
        location = self.vehicle_actor.get_location()
        distance_to_car = location.distance(car_spawn_point.location)

        self.episode_measurements['obstacle_dist'] = distance_to_car

        if distance_to_car < 20:
            self.episode_measurements['obstacle_visible'] = True
        else:
            self.episode_measurements['obstacle_visible'] = False

    def is_within_distance_ahead(self, target_transform, current_transform, max_distance):
        """
        Check if a target object is within a certain distance in front of a reference object.
        :param target_transform: location of the target object
        :param current_transform: location of the reference object
        :param orientation: orientation of the reference object
        :param max_distance: maximum allowed distance
        :return: True if target object is within max_distance ahead of the reference object
        """
        target_vector = np.array([target_transform.location.x - current_transform.location.x, target_transform.location.y - current_transform.location.y])
        norm_target = np.linalg.norm(target_vector)

        # If the vector is too short, we can simply stop here
        if norm_target < 0.001:
            return True, norm_target

        if norm_target > max_distance:
            return False, norm_target

        fwd = current_transform.get_forward_vector()
        forward_vector = np.array([fwd.x, fwd.y])
        d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

        return d_angle < 90.0, norm_target

    def _update_env_obs(self):
        ego_vehicle_location = self.vehicle_actor.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        self._update_obs_detector(ego_vehicle_waypoint)
        self._update_traffic_light_states()

        if self.config['verbose']:
            print(self.episode_measurements['dist_to_light'],
                self.episode_measurements['nearest_traffic_actor_id'],
                self.episode_measurements['nearest_traffic_actor_state'],
                self.episode_measurements['initial_dist_to_red_light'],
                self.episode_measurements['red_light_dist'])

    def _update_obs_detector(self, ego_vehicle_waypoint):
        self.episode_measurements['obstacle_visible'] = False

        min_obs_distance = 100000000
        found_obstacle = False
        for target_vehicle in self.actor_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self.vehicle_actor.id or "vehicle" not in target_vehicle.type_id:
                continue

            # # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            d_bool, distance = self.is_within_distance_ahead(target_vehicle.get_transform(),
                                        self.vehicle_actor.get_transform(),
                                        self.config['proximity_threshold'])

            if not d_bool:
                continue
            else:
                if not check_if_vehicle_in_same_lane(self.vehicle_actor, target_vehicle, self.next_waypoints, self._map):
                    continue
                
                found_obstacle = True
                self.episode_measurements['obstacle_visible'] = True
                
                if distance < min_obs_distance:
                    self.episode_measurements['obstacle_dist'] = distance
                    self.episode_measurements['obstacle_speed'] = self.get_speed_from_velocity(target_vehicle.get_velocity())
                    min_obs_distance = distance
        
        if not found_obstacle:
            self.episode_measurements['obstacle_dist'] = -1
            self.episode_measurements['obstacle_speed'] = -1

    def _update_traffic_light_states(self):
        # TODO: Pass correct target waypoint to find_nearest_traffic_light() for US style traffic.
        traffic_actor, dist = self.vehicle_agent.find_nearest_traffic_light(self.traffic_actors)

        if traffic_actor is not None:
            if traffic_actor.state == carla.TrafficLightState.Red:
                self.episode_measurements['red_light_dist'] = dist

                if self.episode_measurements['initial_dist_to_red_light'] == -1 \
                    or (self.episode_measurements['nearest_traffic_actor_id'] != -1 and traffic_actor.id != self.episode_measurements['nearest_traffic_actor_id']):
                    self.episode_measurements['initial_dist_to_red_light'] = dist

            else:
                self.episode_measurements['red_light_dist'] = -1
                self.episode_measurements['initial_dist_to_red_light'] = -1

            self.episode_measurements['nearest_traffic_actor_id'] = traffic_actor.id
            self.episode_measurements['nearest_traffic_actor_state'] = traffic_actor.state
        else:
            self.episode_measurements['red_light_dist'] = -1
            self.episode_measurements['initial_dist_to_red_light'] = -1
            self.episode_measurements['nearest_traffic_actor_id'] = -1
            self.episode_measurements['nearest_traffic_actor_state'] = None

        self.episode_measurements['dist_to_light'] = dist

    def _set_scenario(self, unseen=False, town="Town01", index=0):
        if self.config["scenarios"] == "straight":
            # self.source_transform, self.destination_transform = scenarios.get_fixed_long_straight_path_Town01()
            self.source_transform, self.destination_transform = scenarios.get_straight_path(unseen, town, index)
            self.config["num_episodes"] = 25
        elif self.config["scenarios"] == "long_straight":
            self.source_transform, self.destination_transform = scenarios.get_long_straight_path(unseen, town, index)
            self.config["num_episodes"] = 1
        elif self.config["scenarios"] == "straight_dynamic":
            self.source_transform, self.destination_transform = scenarios.get_straight_dynamic_path(unseen, town, index)
        elif self.config["scenarios"] == "crowded":
            self.source_transform, self.destination_transform = scenarios.get_crowded_path(unseen, town, index)
        elif self.config["scenarios"] == "straight_crowded":
            self.source_transform, self.destination_transform = scenarios.get_straight_crowded_path(unseen, town, index)
        elif self.config["scenarios"] == "town3":
            self.source_transform, self.destination_transform = scenarios.get_curved_town03_path(unseen, town, index)
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
            self.config["num_episodes"] = 25
        elif self.config["scenarios"] == "navigation" or self.config["scenarios"] == "dynamic_navigation":
            self.source_transform, self.destination_transform = scenarios.get_navigation_path(unseen, town, index)
            self.config["num_episodes"] = 25
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
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.config["enable_brake"])
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config["action_type"] == "merged_speed_scaled_tanh":
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = (action[1] * 1.5) + 1
            target_speed = float(np.clip(target_speed * 10, 0, self.target_speed))
            current_speed = self.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.config["enable_brake"])
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config["action_type"] == "merged_speed_pid_test":
            # steer = float(action[0])
            steer = (float(action[0]))
            target_speed = float(action[1])
            current_speed = self.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.config["enable_brake"])
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
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
        self.clear_episode_measurements()

        self.num_steps = 0 # Episode level step count
        self.total_reward = 0 # Episode level total reward
        self.prev_measurement = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None
        self.unseen = unseen
        self.index = index

        # Destroy
        self.destroy_all_existing_actors()

        self.camera_queue.queue.clear()
        self.stacked_observation_queue.queue.clear()

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
        self.vehicle_agent = Agent(self.vehicle_actor, self.config['proximity_threshold'])
        self.actor_list.append(self.vehicle_actor)
        self.location = self.vehicle_actor.get_location()

        if self.config["num_npc"] > 0:
            self.spawn_npc(self.config["num_npc"], unseen)    

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
        # camera.set_attribute('fov', '120')
        camera.set_attribute('fov', '90')

        # camera_transform = carla.Transform(carla.Location(x=5.0, z=20.0), carla.Rotation(pitch=270.0))
        camera_transform = carla.Transform(carla.Location(x=13.0, z=18.0), carla.Rotation(pitch=270.0))
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
            self.episode_measurements['out_of_road'] = int(self.lane_invasion_sensor.out_of_road)
        self.location = self.vehicle_actor.get_location()
        self.episode_measurements['distance_to_goal'] = self.location.distance(self.destination_transform.location)
        self.episode_measurements['min_distance_to_goal'] = 1000000.0
        self.episode_measurements['speed'] = self.get_speed_from_velocity(self.vehicle_actor.get_velocity())

        self.episode_measurements['total_steps'] = self.total_steps
        self.episode_measurements['initial_dist_to_red_light'] = -1
        # time.sleep(1)

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
            next_orientation, self.dist_to_trajectory, distance_to_goal_trajec, self.next_waypoints = self.global_planner.get_next_orientation_new(self.vehicle_actor.get_transform())
        
        self.episode_measurements['next_orientation'] = next_orientation
        self.episode_measurements['distance_to_goal_trajec'] = distance_to_goal_trajec
        self.episode_measurements['dist_to_trajectory'] = self.dist_to_trajectory

        # Update obstacle distance measurements
        # if self.config['input_type'] in ["wp_obs_bool", "wp_obs_bool_noise"]:
        self._update_env_obs()
        if self.config["scenarios"] == "straight_dynamic":
            self._update_straight_dynamic_obs()

        obs['image'] = image
        encoded_observation = None
        if self.config["input_type"] in ['vae', 'wp_vae', 'wp_vae_speed_steer_goal']:
            semantic_image = image[:,:,0]
            semantic_image = reduce_classes(semantic_image)
            image_labels = convert_to_one_hot(semantic_image, num_classes=5)
            
            for _ in range(self.config['frame_stack_size']):
                # Update stacked frames and measurements
                self._add_to_stacked_queue(self.stacked_observation_queue, image_labels)

            stacked_observation = np.concatenate(list(self.stacked_observation_queue.queue), axis=2)
            # encoded_observation = self.vae_observation(image_labels)
            encoded_observation = self.vae_observation(stacked_observation)
            encoded_observation = encoded_observation / self.config["vae_encoding_norm_factor"]
            obs['semantic_image'] = semantic_image
        
        if self.config["input_type"] == "ae_train":
            semantic_image = image[:,:,0]
            obs['semantic_image'] = semantic_image
    
        obs['speed'] = np.expand_dims(np.array([self.episode_measurements['speed']]), axis=0) # * 3.6 / 30
        obs['dist_to_target'] = np.array([self.episode_measurements['distance_to_goal']])

        self.create_observations(obs)
        # if self.config["input_type"] == 'wp_constant':
        #     obs['observation'] = np.array([0.0, self.episode_measurements['next_orientation']])
        # elif self.config["input_type"] == 'wp_noise':
        #     obs['observation'] = np.concatenate((np.random.normal(0.0, 1.0, self.config["noise_dim"]), np.array([self.episode_measurements['next_orientation']])))
        # elif self.config["input_type"] == 'wp_obs_dist':
        #     obs_dist = self.episode_measurements['obstacle_dist'] / self.config["obstacle_dist_norm"]
        #     obs['observation'] = np.concatenate((np.array([obs_dist]), np.array([self.episode_measurements['next_orientation']])))
        # elif self.config["input_type"] == 'wp_obs_bool':
        #     obs_bool = int(self.episode_measurements['obstacle_visible'])
        #     obs['observation'] = np.concatenate((np.array([obs_bool]), np.array([self.episode_measurements['next_orientation']])))
        # elif self.config["input_type"] == 'wp_obs_bool_noise':
        #     obs_bool = int(self.episode_measurements['obstacle_visible'])
        #     obs['observation'] = np.concatenate((np.random.normal(0.0, 1.0, self.config["noise_dim"]), np.array([obs_bool]), np.array([self.episode_measurements['next_orientation']])))
        # elif self.config["input_type"] == 'wp_vae_speed_steer_goal':
        #     speed = 0
        #     steer = 0
        #     distance_to_goal_trajec = 0
        #     obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([speed]), np.array([steer]), np.array([distance_to_goal_trajec])))
        self.prev_measurement = copy.deepcopy(self.episode_measurements)

        self.target_speeds_array = []
        self.speeds_array = []
        self.throttles_array = []
        self.steers_array = []
        self.brakes_array = []
        self.obstacle_dist_array = []
        self.step_reward_array = []
        self.collision_reward_array = []
        self.dist_to_trajectory_reward_array = []
        self.speed_reward_array = []
        self.dist_to_target_array = []

        if self.config["input_type"] == 'vae':
            return encoded_observation
        elif self.config["input_type"] in ['wp_vae', 'wp_vae_speed_steer_goal', 'wp_vae_speed_steer_ldist_goal_light']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            fused_input = np.hstack([encoded_observation, observation])
            return fused_input
        elif self.config["input_type"] == "wp":
            return obs['observation']
        elif self.config["input_type"] in ['wp_noise', 'wp_constant', 'wp_obs_dist', 'wp_obs_bool', 'wp_obs_bool_noise',
                                           'wp_obs_bool_speed_steer_goal_light', 'wp_obs_info_speed_steer_ldist_goal_light']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            return observation
        else:
            return obs
        
    def try_spawn_random_vehicle_at(self, blueprints, transform):
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        
        # TODO: uncomment below to enable autopilot
        if not self.config["scenarios"] == "straight_dynamic":
            blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self._world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            self.actor_list.append(vehicle)
            # TODO: uncomment below to enable autopilot
            if not self.config["scenarios"] == "straight_dynamic":
                vehicle.set_autopilot()

            if self.config["verbose"]:
                print('spawned %r at %s' % (vehicle.type_id, transform.location.x))
            return True
        return False
    
    def spawn_npc(self, number_of_vehicles, unseen):
        
        # TODO: remove hard coded logic
        if self.config["scenarios"] == "straight_dynamic":
            spawn_points = [Transform(Location(x=88.61997985839844, y=249.42999267578125, z=1.32), Rotation(yaw=90.00004577636719)),
            Transform(Location(x=92.10997772216797, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672))]
        elif self.config["scenarios"] == "crowded":
            spawn_points = scenarios.get_crowded_npcs(number_of_vehicles)
            # print('CROWDED SPAWNING: ', spawn_points)
        elif self.config["scenarios"] == "long_straight":
            spawn_points_1 = scenarios.get_long_straight_npcs()
            if unseen:
                if self.config["test_fixed_spawn_points"]:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
                    random.shuffle(spawn_points)
            else:
                if self.config["train_fixed_spawn_points"]:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
        elif self.config["scenarios"] == "straight_crowded":
            spawn_points = scenarios.get_straight_crowded_npcs(number_of_vehicles)
            # print('STRAIGHT CROWDED SPAWNING: ', spawn_points)
        elif self.config["scenarios"] == "town3":
            spawn_points = scenarios.get_curved_town03_npcs(number_of_vehicles)
            # print('TOWN 3 SPAWNING: ', spawn_points)
        else:
            # Testing
            if unseen:
                if self.config["test_fixed_spawn_points"]:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
                    random.shuffle(spawn_points)
            else:
                if self.config["train_fixed_spawn_points"]:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
                    random.shuffle(spawn_points)


        if self.config["verbose"]:
            print('found %d spawn points.' % len(spawn_points))

        if self.config["scenarios"] == "long_straight":
            for spawn_point in spawn_points_1:
                self.try_spawn_random_vehicle_at(self.vehicle_blueprints, spawn_point)

        count = number_of_vehicles
        for spawn_point in spawn_points:
            if self.try_spawn_random_vehicle_at(self.vehicle_blueprints, spawn_point):
                count -= 1
            if count <= 0:
                break

        while count > 0:
            print("in while loop")
            if self.try_spawn_random_vehicle_at(self.vehicle_blueprints, random.choice(spawn_points)):
                count -= 1

    def get_speed_from_velocity(self, velocity):
        speed = np.sqrt(velocity.x ** 2 + velocity.y **2 + velocity.z **2)
        return speed

    def _read_data(self, world_frame, timeout=240.0):

        cam_image = self._read_camera_data(world_frame, timeout)
        cam_image_p = self._preprocess_image(cam_image)
        return cam_image_p

    def _preprocess_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        # if(self.config['preprocess_crop_image']):
        #     array = array[200:500, 300:500]
        # array = cv2.resize(array, (self.config["x_res"], self.config["y_res"]), interpolation=cv2.INTER_NEAREST)

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
        collision = self.episode_measurements["is_collision"]
        runover_light = self.episode_measurements["runover_light"]
        maxStepsTaken = self.episode_measurements["num_steps"] > self.config['max_steps']
        offlane = False

        if not self.config["enable_static"]:
            static = False
        if self.config["disable_collision"]:
            collision = False
        if not self.config["terminate_on_light"]:
            runover_light = False
        if self.config["enable_lane_invasion_collision"]:
            offlane = self.episode_measurements['num_laneintersections'] > 0
        # Do not want to terminate on reaching goal
        # in case of VAE training
        if self.config["algo"] == "AE":
            success = False

        if success:
            termination_state = 'success'
        elif collision:
            termination_state = 'collision'
        elif runover_light:
            termination_state = 'runover_light'
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

        done = success or collision or runover_light or offlane or static or maxStepsTaken
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

def plot_episode_info(path,
                target_speeds_array,
                speeds_array,
                throttles_array,
                steers_array,
                brakes_array,
                obstacle_dist_array,
                step_reward_array,
                collision_reward_array,
                dist_to_trajectory_reward_array,
                speed_reward_array,
                episode_num):
    
    if not os.path.exists(path):
        os.makedirs(path)
    observations = np.arange(len(target_speeds_array))
    
    target_speeds_array = np.array(target_speeds_array)
    speeds_array = np.array(speeds_array)
    throttles_array = np.array(throttles_array)
    steers_array = np.array(steers_array)
    brakes_array = np.array(brakes_array)
    step_reward_array = np.array(step_reward_array)
    collision_reward_array = np.array(collision_reward_array)
    obstacle_dist_array = np.array(obstacle_dist_array)
    dist_to_trajectory_reward_array = np.array(dist_to_trajectory_reward_array)
    speed_reward_array = np.array(speed_reward_array)
    
    fig, axs = plt.subplots(5, 2, figsize=(12, 12))
    fig.suptitle('Episode info plots for episode idx {} '.format(episode_num))

    axs[0, 0].plot(observations, target_speeds_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[0, 0].set_xlabel('Timesteps')
    axs[0, 0].set_ylabel('Target Speed - Stochastic')

    axs[1, 0].plot(observations, speeds_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[1, 0].set_xlabel('Timesteps')
    axs[1, 0].set_ylabel('Actual Speed - Stochastic')

    
    axs[2, 0].plot(observations, throttles_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[2, 0].set_xlabel('Timesteps')
    axs[2, 0].set_ylabel('Throttle')

    axs[3, 0].plot(observations, step_reward_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[3, 0].set_xlabel('Timesteps')
    axs[3, 0].set_ylabel('Step reward')

    axs[4, 0].plot(observations, dist_to_trajectory_reward_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[4, 0].set_xlabel('Timesteps')
    axs[4, 0].set_ylabel('dist_to_trajectory reward')


    axs[0, 1].plot(observations, steers_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[0, 1].set_xlabel('Timesteps')
    axs[0, 1].set_ylabel('Steer - Stochastic')


    axs[1, 1].plot(observations, obstacle_dist_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[1, 1].set_xlabel('Timesteps')
    axs[1, 1].set_ylabel('Obstacle Distance')

    axs[2, 1].plot(observations, brakes_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[2, 1].set_xlabel('Timesteps')
    axs[2, 1].set_ylabel('Break')

    axs[3, 1].plot(observations, collision_reward_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[3, 1].set_xlabel('Timesteps')
    axs[3, 1].set_ylabel('collision_reward')

    axs[4, 1].plot(observations, speed_reward_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[4, 1].set_xlabel('Timesteps')
    axs[4, 1].set_ylabel('speed_reward')

    axs[0,0].grid(True)
    axs[0,1].grid(True)
    axs[1,0].grid(True)
    axs[1,1].grid(True)
    axs[2,0].grid(True)
    axs[2,1].grid(True)
    axs[3,0].grid(True)
    axs[3,1].grid(True)
    axs[4,0].grid(True)
    axs[4,1].grid(True)
    
    plt.grid(True)
    plt.savefig(path + '{}.png'.format(episode_num))
    plt.close()

if __name__ == "__main__":
    env = CarlaEnv()
    env.reset()
