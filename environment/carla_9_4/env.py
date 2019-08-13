""" Environment file wrapper for CARLA """

import gym
from gym import error, utils
from gym.spaces import Box, Discrete, Tuple
from gym.utils import seeding

from datetime import datetime
import os
import glob
import sys
import traceback
import random
import queue
import json
import numpy as np
import math
import copy
import cv2
import collections
import time

import environment.carla_9_4.scenarios as scenarios
import environment.carla_9_4.server as server
import environment.carla_9_4.planner as planner
import environment.carla_9_4.controller as controller
import scipy.misc
from scipy.misc import imsave
from environment.carla_9_4.agents.navigation.roaming_agent import RoamingAgent



# Keeping this for now (since we may need to log images later)
# SENSOR_LOG_DIR = '/../../misc/logs'

RETRIES_ON_ERROR=5

CARLA_9_4_PATH = os.environ.get("CARLA_9_4_PATH")
if CARLA_9_4_PATH == None:
    raise ValueError("Set $CARLA_9_4_PATH to directory that contains CarlaUE4.sh")

try:
    sys.path.append(glob.glob(CARLA_9_4_PATH+'/**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    import carla
except Exception as e:
    print("Failed to import Carla")
    raise e

# from environment.carla_9_4.agents.navigation.agent import *
# from environment.carla_9_4.agents.navigation.local_planner import LocalPlanner
# from environment.carla_9_4.agents.navigation.local_planner import compute_connection, RoadOption
# from environment.carla_9_4.agents.navigation.global_route_planner import GlobalRoutePlanner
# from environment.carla_9_4.agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
# from environment.carla_9_4.agents.tools.misc import vector
import environment.carla_9_4.sensors as sensors
from carla import ColorConverter as cc

# Dict storing basic environ config params
# NOTE: Doing this since it's more convenient to pass in a dict (compared to __init__ args)
# TODO: Split into server specific and client specific
DEFAULT_ENV = {
    "server_path" : CARLA_9_4_PATH,
    "server_binary" : CARLA_9_4_PATH + '/CarlaUE4.sh',
    "server_process" : None,
    # X Rendering Resolution (NOTE: Doesn't change anything! Link to Issue #17)
    "render_res_x" : 800,
    # Y Rendering Resolution (NOTE: Doesn't change anything! Link to Issue #17)
    "render_res_y" : 800,
    # Note data type here is string (since that is what the blueprint attribute set API requires)
    "sensor_x_res" : '800',
    "sensor_y_res" : '800',
    # Input X Res (Default set to Atari)
    "x_res": 84,
    # Input Y Res (Default set to Atari)
    "y_res": 84,
    "server_fps" : 10,
    "server_port" : None,
    "city_name" : "Town01",
    "frame_skip": 1,
    "enable_planner" : True,
    "reward_function" : 'corl',
    "save_images_to_disk" : False,
    "record_sim": False,
    "write_data": True,
    # Print measurements to screen
    "print_obs" : True,
    "client" : None,
    "discrete_actions": True,

    # Number of frames stacked together
    "framestack" : 1,
    "grayscale" : False,
    "num_vehicles" : 1,
    "num_pedestrians" : 0,
    "max_steps" : 400,
    "next_command": None,
    "verbose": True,
    "vehicle_type": 'vehicle.toyota.prius',
    "vehicle_types": ['vehicle.ford.mustang', 'vehicle.audi.a2', 'vehicle.audi.tt', 'vehicle.bmw.isetta', 'vehicle.carlamotors.carlacola', 
                      'vehicle.citroen.c3', 'vehicle.bmw.grandtourer', 'vehicle.mercedes-benz.coupe',
                      'vehicle.toyota.prius', 'vehicle.dodge_charger.police', 'vehicle.nissan.patrol',
                      'vehicle.tesla.model3', 'vehicle.seat.leon', 'vehicle.lincoln.mkz2017',
                      'vehicle.volkswagen.t2', 'vehicle.nissan.micra', 'vehicle.chevrolet.impala', 'vehicle.mini.cooperst',
                      'vehicle.jeep.wrangler_rubicon'],
    "target_speed": 20,
    "sensors": ["sensor.camera.rgb", "sensor.camera.semantic_segmentation"],
    "action_type": "merged_gas",
    "sensor_tick": '1.0',
    "dist_for_success" : 4.0,
    "max_offlane_steps" : 20,
    "max_static_steps" : 500,
    "log_measurements_to_file": False,
    "train_config": 'baselines',
    "sync_mode": True,
    # NOTE: crop does not work with framestack yet. need to add.
    "preprocess_crop_image": False,
    "scenarios" : "straight",
    "semantic" : False
}
# DISCRETE_ACTIONS = {
#     # Coast
#     0: [0.5, -0.5],
#     # Forward
#     1: [0.5, -0.4],
#     # Brake
#     2: [0.5, -0.3],
#     # Left
#     3: [0.5, -0.2],
#     # Right
#     4: [0.5, -0.1],
#     # Forward left
#     5: [0.5, 0.0],
#     # Forward right
#     6: [0.5, 0.1],
#     # Brake left
#     7: [0.5, 0.2],
#     # Brake right
#     8: [0.5, 0.3],

#     9: [0.5, 0.4],
#     10: [0.5, 0.5]
# }

# DISCRETE_ACTIONS = {
#     # Coast
#     0: [0.0, 0.0],
#     # Forward
#     1: [0.5, 0.0],
#     # Forward left
#     2: [0.25, -0.3],
#     3: [0.25, -0.1],
#     # Forward right
#     4: [0.25, 0.1],
#     5: [0.25, 0.3],
#     # Brake
#     6: [-0.5, 0.0],
#     # Brake left
#     7: [-0.25, -0.3],
#     8: [-0.25, -0.1],
#     # Brake right
#     9: [-0.25, 0.1],
#     10: [-0.25, 0.3]
# }

# DISCRETE_ACTIONS = {
#     # Coast
#     0: [10.0, 0.0],
#     # Forward
#     1: [20.0, 0.0],
#     # Forward left
#     2: [15.0, -0.3],
#     3: [15.0, -0.1],
#     # Forward right
#     4: [15.0, 0.1],
#     5: [15.0, 0.3],
#     # Brake
#     6: [0.0, 0.0],
#     # Brake left
#     7: [5.0, -0.3],
#     8: [5.0, -0.1],
#     # Brake right
#     9: [5.0, 0.1],
#     10: [5.0, 0.3]
# }

DISCRETE_ACTIONS = {
    # Coast
    0: [10.0, -0.5],
    # Forward
    1: [10.0, -0.4],
    # Brake
    2: [10.0, -0.3],
    # Left
    3: [10.0, -0.2],
    # Right
    4: [10.0, -0.1],
    # Forward left
    5: [10.0, 0.0],
    # Forward right
    6: [10.0, 0.1],
    # Brake left
    7: [10.0, 0.2],
    # Brake right
    8: [10.0, 0.3],

    9: [10.0, 0.4],
    10: [10.0, 0.5]
}

# DISCRETE_ACTIONS = {
#     # Coast
#     0: [0.0, 0.0],
#     # Forward
#     1: [2.0, 0.0],
#     # Forward left
#     2: [4.0, 0.0],
#     3: [6.0, 0.0],
#     # Forward right
#     4: [8.0, 0.0],
#     5: [10.0, 0.0],
#     # Brake
#     6: [12.0, 0.0],
#     # Brake left
#     7: [14.0, 0.0],
#     8: [16.0, 0.0],
#     # Brake right
#     9: [18.0, 0.0],
#     10: [20.0, 0.0]
# }

episode_measurements = {
    "episode_id": None,
    "num_steps": None,
    "location": None,
    "speed": None,
    "distance_to_goal": None,
    "num_collisions": 0,
    "num_laneintersections": 0,
    "static_steps": 0,
    "offlane_steps": 0
    # intersection_offroad
    # intersection_otherlane
    # next_command
}

CARLA_LOGS = os.path.expanduser("~/CARLA_LOGS/"+str(datetime.now()))
if not os.path.exists(CARLA_LOGS):
    os.makedirs(CARLA_LOGS)

class CarlaEnv(gym.Env):
    def __init__(self, config=DEFAULT_ENV, vis_wrapper=None, logger=None):
        self.config = DEFAULT_ENV
        self._update_config(config)
        self.CarlaServer = None
        self.episode_measurements = episode_measurements
        self.server_port = self.config["server_port"]
        # TODO: Check planner API from 0.9

        self.episode_id = None
        self.client = None
        self.vehicle_actor = None
        self._world = None
        self._map = None
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.prev_image = 0
        # File to log measurements to
        self.measurements_log = None
        # Can pass in train/test weather as an array
        self.weather = None
        # Scenario is a list of weather/poses tuples
        self.scenario = None
        self.start_pos = None
        self.end_pos = None
        #Agent defaults (for planner)
        self._proximity_threshold = 10.0
        self._local_planner = None
        self._hop_resolution = 2.0
        self._current_plan = None
        self._image_queue = collections.deque(maxlen=self.config['framestack'])
        self.server_process = None
        self.CarlaServer = None
        self.target_speed = self.config['target_speed']
        self.args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.02,
            'K_I': 0,
            'dt': 1.0/10.0}
        # self.args_lateral_dict = {
        #     'K_P': 1.95,
        #     'K_D': 0.01,
        #     'K_I': 1.4,
        #     'dt': 1.0/10.0}
        self.args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 1,
            'dt': 1.0/10.0}
        self.actor_list = []
        self.other_vehicle_actor_list = []
        self.other_vehicle_agent_list = []
        self.other_vehicle_control_list = []

        self.image_data = None
        # Set default source and destination points (in _reset function)
        self.source_transform = None
        self.destination_transform = None
        self.global_planner = None
        self.trace_route = None
        self.episode_num = 0
        self.total_steps = 0

        self.logger = logger
        self.vis_wrapper = vis_wrapper

        self.dist_to_trajectory = None
        # Compute number of channels in sensor image
        # We use this later in the preprocess step to reshape the data
        # im_channels refers to number of channels the agent receives after preprocessing
        if(self.config['grayscale']):
            self.im_channels = 1
        else:
            self.im_channels = 3
        
        self.controller = controller.PIDLongitudinalController(K_P=0.1, K_D=0.0005, K_I=1.0, dt=1/10.0) 

        # Start Carla Server
        serverStarted = False
        serverStartRetries = 0
        while ((not serverStarted) and serverStartRetries < RETRIES_ON_ERROR):
            try:
                self.CarlaServer = server.CarlaServer(config=self.config)
                serverStarted = True
            except Exception as e:
                print("Error in starting carla server : {}".format(traceback.format_exc()))
                self.CarlaServer.close()
                error = e
                serverStartRetries += 1

        if(self.config['train_config'] == 'baselines'):
            self.action_space = Discrete(len(DISCRETE_ACTIONS))
            image_space = Box(
            0,
            255,
            shape=(self.config["y_res"], self.config["x_res"],
                    self.im_channels * self.config["framestack"]),
            dtype=np.uint8)
            # observation space is image and vector of measurements
            # vector of measurements is:
            # current speed, distance to goal, damage from collisions,
            # current high-level command by planner, in one-hot encoding.
            # self.observation_space = Tuple(
            # [
            #     image_space,
            #     # Discrete(len(COMMANDS_ENUM)),  # next_command
            #     Box(0, 1024.0, shape=(2, ), dtype=np.float32)
            # ])
            self.observation_space = image_space

        if(self.config['train_config'] == 'PPO'):
            # Streer, Throttle
            # self.action_space = Box(low=np.array([-0.5, -0.5]), high=np.array([0.5, 0.5]), dtype=np.float32)
            
            # Steer, Speed
            self.action_space = Box(low=np.array([-0.5, -10.0]), high=np.array([0.5, 10.0]), dtype=np.float32)
            
            # Steer only
            # self.action_space = Box(low=np.array([-0.5]), high=np.array([0.5]), dtype=np.float32)
            
            # Speed only
            # self.action_space = Box(low=np.array([0.0]), high=np.array([20.0]), dtype=np.float32)
            
            # self.action_space = Box(low=np.array([0.0]), high=np.array([0.7]), dtype=np.float32)
            self.observation_space = Box(low=np.array([-4.0]), high=np.array([4.0]), dtype=np.float32)
            # vae_image_space = Box(low=np.finfo(np.float32).min,
            #                          high=np.finfo(np.float32).max,
            #                          shape=(1, 512), dtype=np.float32)
            # self.observation_space = Box(low=np.finfo(np.float32).min,
            #                          high=np.finfo(np.float32).max,
            #                          shape=(1, 513), dtype=np.float32)

    def _update_config(self, config):
        for key, val in config.items():
            self.config[key] = val

    def _spawn_client(self, hostname='localhost', port_number=None):
        port_number = self.CarlaServer.server_port
        client = carla.Client(hostname, port_number)
        client.set_timeout(40.0)
        return client

    def step(self, action):
        try:
            obs = self._step(action)
            return obs
        except Exception:
            print("Error during step, terminating episode early",
        traceback.format_exc())

    def _step(self, action):
        #TODO: Add other vehicle + traffic light check methods
        for i in range(self.config["num_vehicles"] - 1):
            control = self.other_vehicle_agent_list[i].run_step()
            control.manual_gear_shift = False
            self.other_vehicle_control_list.append(control)
        #NOTE: Only mapping to one action for now (target speed)

        # speed = action
        # self._local_planner.set_speed(speed)
        # control = self._local_planner.run_step()

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

        #TODO: Increment steps inside of frame_skip?

        for _ in range(self.config["frame_skip"]):
            self.vehicle_actor.apply_control(control)
            for i in range(self.config["num_vehicles"] - 1):
                self.other_vehicle_actor_list[i].apply_control(self.other_vehicle_control_list[i])
            self._world.tick()
            timestamp = self._world.wait_for_tick()
        self.num_steps += 1
        self.total_steps +=1
        self.episode_measurements['num_steps'] = self.num_steps

        # Read in preprocessed image
        sensor_image = self._read_data()

        # print('-'*50)
        # print('In step. Read sensor image of type:', type(sensor_image))
        # print('-'*50)

        # Set state variables for reward calculation
        self.episode_measurements['num_collisions'] = self.collision_sensor.num_collisions
        self.episode_measurements['num_laneintersections'] = self.lane_invasion_sensor.num_laneintersections
        self.location = self.vehicle_actor.get_location()
        self.episode_measurements['distance_to_goal'] = self.location.distance(self.destination_transform.location)
        self.episode_measurements['speed'] = self.get_speed_from_velocity(self.vehicle_actor.get_velocity())

        next_orientation, self.dist_to_trajectory = self.global_planner.get_next_orientation_new(self.vehicle_actor.get_transform())
        next_orientation_old, _ = self.global_planner.get_next_orientation(self.vehicle_actor.get_transform())
        reward = self._compute_reward(name=self.config['reward_function'],
                                    prev_measurement=self.prev_measurement,
                                    cur_measurement=self.episode_measurements)
        self.total_reward += reward
        self.episode_measurements['reward'] = reward
        self.episode_measurements['total_reward'] = self.total_reward

        done = self._compute_done_condition()

        self.episode_measurements['done'] = done
        self.prev_measurement = copy.deepcopy(self.episode_measurements)

        if self.config["log_measurements_to_file"] and CARLA_LOGS:
            if not self.measurements_log:
                self.measurements_log = open(os.path.join(CARLA_LOGS,
                "measurements_{}.json".format(self.episode_id)), "a")
            self.measurements_log.write(json.dumps(self.episode_measurements))
            self.measurements_log.write("\n")

            if done:
                self.measurements_log.close()
                self.measurements_file = None
        #Only increment step after writing log (successful)

        #TODO: badcast errors in carla-3.5egg file
        # print("Vehicle transform:{0}".format(self.vehicle_actor.get_transform()))
        # print("Vehicle velocity:{0}".format(self.vehicle_actor.get_velocity()))

        obs = {}
        #TODO: Get branch_idx from planner and set accordingly.
        branch_idx = 1

        obs['image'] = sensor_image
        if self.config["input_type"] == 'vae' or self.config["input_type"] == "wp_vae":
            obs['image'] = self.vae.decode(sensor_image)[0, :, :, :].astype(np.uint8)
        # import scipy.misc
        # import random
        # import string
        # random = ''.join([random.choice(string.ascii_letters + string.digits) for n in xrange(6)])
        # scipy.misc.imsave('images/' + random + '.jpg', sensor_image)
        
        obs['speed'] = np.expand_dims(
            np.array([self.episode_measurements['speed']]), axis=0)  # * 3.6 / 30
        obs['dist_to_target'] = np.array(
            [self.episode_measurements['distance_to_goal']])
        obs['branch_mask'] = np.expand_dims(np.eye(4)[branch_idx], axis=0)

        obs['orientation'] = np.array([next_orientation])

        print("orientation {0}".format(next_orientation))
        print("old orientation {0}".format(next_orientation_old))
        reward = np.expand_dims(np.array([reward]), axis=0)
        done = np.expand_dims(np.array([done]), axis=0)

        if self.config["train_config"] == "PPO":
            self.vis_wrapper.save_image(obs['image'], self.num_steps)
            self.logger.log_scalar('timesteps/train/orientation', next_orientation, self.total_steps)
            self.logger.log_scalar('timesteps/train/orientation_old', next_orientation_old, self.total_steps)
            self.logger.log_scalar('timesteps/train/throttle', control.throttle, self.total_steps)
            self.logger.log_scalar('timesteps/train/speed', self.episode_measurements['speed'], self.total_steps)
            self.logger.log_scalar('timesteps/train/steer', control.steer, self.total_steps)
            self.logger.log_scalar('timesteps/train/target_speed', self.episode_measurements['target_speed'], self.total_steps)
                            
            if done:
                self.episode_num += 1
                self.logger.log_scalar('episodes/train/dist_to_target', self.episode_measurements['distance_to_goal'], self.episode_num)
                self.logger.log_scalar('episodes/train/reward', self.episode_measurements['total_reward'], self.episode_num)
                self.logger.log_scalar('timesteps/train/dist_to_target', self.episode_measurements['distance_to_goal'], self.total_steps)
                self.logger.log_scalar('timesteps/train/reward', self.episode_measurements['total_reward'], self.total_steps)
                self.vis_wrapper.generate_video(self.episode_num)
                self.vis_wrapper.remove_images()
        if self.config["input_type"] == 'vae':
            return sensor_image, reward, done, self.episode_measurements
        elif self.config["input_type"] == "wp_vae":
            orientation = np.expand_dims(obs['orientation'], axis = 0)
            fused_input = np.hstack([sensor_image, orientation])
            return fused_input, reward, done, self.episode_measurements
        else:
            return obs['orientation'], reward, done, self.episode_measurements

    def _set_scenario(self, unseen=False, town="Town01", index=0):
        if self.config["scenarios"] == "straight":
            self.source_transform, self.destination_transform = scenarios.get_straight_path(unseen, town, index)
        elif self.config["scenarios"] == "left_right_curved":
            self.source_transform, self.destination_transform = scenarios.get_left_right_randomly(unseen)
        elif self.config["scenarios"] == "right_curved":
            # self.source_transform, self.destination_transform = scenarios.get_train_right_turn()
            self.source_transform, self.destination_transform = scenarios.get_right_turn(unseen)
        elif self.config["scenarios"] == "left_curved":
            self.source_transform, self.destination_transform = scenarios.get_left_turn(unseen)
        elif self.config["scenarios"] == "curved":
            self.source_transform, self.destination_transform = scenarios.get_curved_path(unseen, town, index)
        elif self.config["scenarios"] == "navigation" or self.config["scenarios"] == "dynamic_navigation":
            self.source_transform, self.destination_transform = scenarios.get_navigation_path(unseen, town, index)
        else:
            raise ValueError("Scenarios Config not set!")

    def set_vae(self, vae):
        self.vae = vae
    
    def vae_observation(self, observation_image):
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
            steer = float(action[0])
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
            steer = float(action[0])
            # steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(np.clip(action[1] + 10.0, 0, self.target_speed))
            current_speed = self.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            throttle = self.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        
        
        self.episode_measurements["target_speed"] = target_speed
            

        # Avoid fake braking (from Codevilla conditional imitation learning code)
        # Needed for imitation learning agent to succeed on benchmarks, should not
        # be used with RL agents
        #if (brake < 0.1) or (brake < acc):
        #    brake = 0.0

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
    
    def populate_spawn_points(self, spawn_points, source_transform):
        print(len(spawn_points))
        points = []
        for point in spawn_points:
            if point.location.x == source_transform.location.x and point.location.y == source_transform.location.y and \
            point.location.z == source_transform.location.z and point.rotation.yaw == source_transform.rotation.yaw:
                continue
            points.append(point)
        print(len(points))
        return points

    def destroy_other_actors(self):
        for _ in range(len(self.other_vehicle_actor_list)):
            try:
                actor = self.other_vehicle_actor_list.pop()
                actor.destroy()
            except Exception as e:
                print("Error during destroying actor {0}:{1}: {2}".format(actor.type_id, actor.id,traceback.format_exc()))
        
        self.other_vehicle_agent_list.clear()
        self.other_vehicle_control_list.clear()

    def _reset(self, unseen=False, index=0):
        #TODO: Keep track of current location, and distance to goal (i.e. update eps meas params)

        self.clear_episode_measurements()

        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.prev_image = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None

        # Destroy
        self.destroy_all_existing_actors()

        # Create new client
        self.client =  self._spawn_client()

        self._world = self.client.get_world()

        if(self.config['sync_mode']):
            settings = self._world.get_settings()
            settings.synchronous_mode = True
            self._world.apply_settings(settings)

        self._map = self._world.get_map()

        blueprint_library = self._world.get_blueprint_library()
        spawn_points = self._world.get_map().get_spawn_points()
        # f = open("spawn_points.txt", "w")
        # f.write("Printing all spawn points")
        # for point in spawn_points:
        #     f.write("Transform(Location(x={0}, y={1}, z={2}), Rotation(yaw={3}))\n".format(point.location.x, point.location.y, point.location.z, point.rotation.yaw))
        # f.close()
        try:
            vehicle_bp = blueprint_library.find(self.config['vehicle_type'])
            # vehicle_bp = blueprint_library.find(random.choice(self.config['vehicle_types']))
        except Exception as e:
            print("Error during vehicle creation: {}".format(traceback.format_exc()))

        # Set source and destination based on scenario
        # Currently scenarios are defined only for Town01
        if self.config["city_name"] == "Town01" or self.config["city_name"] == "Town02":
            self._set_scenario(unseen=unseen, index=index, town=self.config["city_name"])
        else:
            # Set source and destination at random spawn points
            # get_spawn_points() returns a list of carla.libcarla.Transform
            # which has attributes location and rotation
            self.source_transform, self.destination_transform = random.choice(spawn_points), random.choice(spawn_points)

        self.vehicle_actor = self._world.try_spawn_actor(vehicle_bp, self.source_transform)
        self.actor_list.append(self.vehicle_actor)
        self.location = self.vehicle_actor.get_location()
        # print('Spawned vehicle actor at', self.location)

        if self.config["scenarios"] == "dynamic_navigation":
            self.destroy_other_actors()
            # spawn_points = self.populate_spawn_points(spawn_points, self.source_transform)
            # vehicle_positions = random.sample(spawn_points, len(spawn_points))
            for i in range(self.config["num_vehicles"] - 1):
                vehicle_initialized = False
                
                other_vehicle_actor = None
                vehicle_bp = blueprint_library.find(self.config['vehicle_type'])
                # vehicle_bp = blueprint_library.find(random.choice(self.config['vehicle_types']))
                while not other_vehicle_actor:
                    start_pose = random.choice(spawn_points)
                    other_vehicle_actor = self._world.try_spawn_actor(vehicle_bp, start_pose)
                    # if driving_vehicle and self._auto_pilot:
                    #     driving_vehicle.set_autopilot(self._auto_pilot)
                print("Initiliazed vehicle actor successfully!! ")
                other_vehicle_agent = RoamingAgent(other_vehicle_actor)
                self.actor_list.append(other_vehicle_actor)
                self.other_vehicle_actor_list.append(other_vehicle_actor)
                self.other_vehicle_agent_list.append(other_vehicle_agent)
                

        #TODO: Generalize this code to attach 'n' different sensors to the vehicle
        #Attach a sensor to the vehicle
        sensor = self.config['sensors'][1]
        camera = blueprint_library.find(sensor)
        camera.set_attribute('image_size_x', self.config['sensor_x_res'])
        camera.set_attribute('image_size_y', self.config['sensor_y_res'])
        camera.set_attribute('sensor_tick', self.config['sensor_tick'])
        camera.set_attribute('fov', '120')
        camera_transform = carla.Transform(carla.Location(x=5.0, z=30.0), carla.Rotation(pitch=270.0))
        self.camera_actor = self._world.spawn_actor(camera, camera_transform, attach_to=self.vehicle_actor)
        self.actor_list.append(self.camera_actor)
        self.collision_sensor = sensors.CollisionSensor(self.vehicle_actor)
        self.actor_list.append(self.collision_sensor.sensor)

        self.lane_invasion_sensor = sensors.LaneInvasionSensor(self.vehicle_actor)
        self.actor_list.append(self.lane_invasion_sensor.sensor)
        # Register callback to put images in the queue
        # Prefer to write raw data (of type 'memoryview') since we won't use all data
        # written to memory (hence typecasting before would be waste of compute)
        if(self.config['write_data']):
            self.camera_actor.listen(lambda image: self._write_data(image))
        if(self.config['save_images_to_disk']):
            self.camera_actor.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame_number))
        if(self.config['record_sim']):
            log_id = str(episode_measurements['episode_id'])
            self.client.start_recorder(log_id, self.vehicle_actor)

        #Attach planner to vehicle actor
        #TODO: Check how to give steering as input to PID? Target speed is present as input
        #TODO: Clean up destination init (pass in a location)
        # if self.config["enable_planner"]:
        #     self._local_planner = LocalPlanner(self.vehicle_actor, opt_dict={'target_speed' : self.target_speed})
        #     self._set_destination(location=self.destination_transform.location)
        # Get start and end positions (to figure out when to end the episode)
        # print("Start pos {}, End Pos {}".format(
        #     spawn_point.location, self.start_coord,
        #     self.scenario["end_pos_id"], self.end_coord))

        # Set state variables for reward calculation
        self.episode_measurements['num_collisions'] = self.collision_sensor.num_collisions
        self.episode_measurements['num_laneintersections'] = self.lane_invasion_sensor.num_laneintersections
        self.location = self.vehicle_actor.get_location()
        self.episode_measurements['distance_to_goal'] = self.location.distance(self.destination_transform.location)
        self.episode_measurements['speed'] = self.get_speed_from_velocity(self.vehicle_actor.get_velocity())

        print('-'*50)
        print('Waiting for sensor to initialize')
        print('-'*50)
        time.sleep(2)

        #TODO: fix bug with no sensor_image. empty image for now
        # x_res = int(self.config["sensor_x_res"])
        # y_res = int(self.config["sensor_y_res"])
        #sensor_image = np.zeros(shape=(x_res, y_res, self.im_channels))
        #TODO: Change this to return the full measurement vector (like the step function)

        obs = {}
        #TODO: Get branch_idx from planner and set accordingly.
        branch_idx = 1

        print('-'*50)
        print('Initializing environment')
        print('-'*50)

        for _ in range(60):
            self._world.tick()
            timestamp = self._world.wait_for_tick()
        image = self._read_data()

        self.global_planner = planner.GlobalPlanner()
        self.trace_route  = self.global_planner._trace_route(self._map,
                                self.source_transform, self.destination_transform)
        self.global_planner.set_global_plan(self.trace_route)

        next_orientation, self.dist_to_trajectory = self.global_planner.get_next_orientation_new(self.vehicle_actor.get_transform())
        next_orientation_old, _ = self.global_planner.get_next_orientation(self.vehicle_actor.get_transform())

        obs['image'] = image
        if self.config["input_type"] == 'vae' or self.config["input_type"] == "wp_vae":
            obs['image'] = self.vae.decode(image)[0, :, :, :].astype(np.uint8)
        obs['speed'] = np.expand_dims(np.array([self.episode_measurements['speed']]), axis=0) # * 3.6 / 30
        obs['dist_to_target'] = np.array([self.episode_measurements['distance_to_goal']])
        obs['branch_mask'] = np.expand_dims(np.eye(4)[branch_idx], axis=0)
        obs['orientation']= np.array([next_orientation])
        self.prev_measurement = copy.deepcopy(self.episode_measurements)

        if self.config["input_type"] == 'vae':
            return image
        elif self.config["input_type"] == "wp_vae":
            orientation = np.expand_dims(obs['orientation'], axis = 0)
            fused_input = np.hstack([image, orientation])
            return fused_input
        else:
            return obs['orientation']

    def get_speed_from_velocity(self, velocity):

        speed = np.sqrt(velocity.x ** 2 + velocity.y **2 + velocity.z **2)
        return speed

    # def _set_destination(self,location):
    #     """Generate waypoints and feed into local + global planner
    #     Parameters
    #     ----------
    #     location: Final destination waypoint
    #     """
    #     start_waypoint = self._map.get_waypoint(self.vehicle_actor.get_location())
    #     end_waypoint = self._map.get_waypoint(
    #         carla.Location(location.x, location.y, location.z))
    #     solution = []

    #     # Setting up global router
    #     dao = GlobalRoutePlannerDAO(self.vehicle_actor.get_world().get_map())
    #     grp = GlobalRoutePlanner(dao)
    #     grp.setup()

    #     # Obtain route plan
    #     x1 = start_waypoint.transform.location.x
    #     y1 = start_waypoint.transform.location.y
    #     x2 = end_waypoint.transform.location.x
    #     y2 = end_waypoint.transform.location.y
    #     route = grp.plan_route((x1, y1), (x2, y2))

    #     current_waypoint = start_waypoint
    #     route.append(RoadOption.VOID)
    #     for action in route:

    #         #   Generate waypoints to next junction
    #         wp_choice = current_waypoint.next(self._hop_resolution)
    #         while len(wp_choice) == 1:
    #             current_waypoint = wp_choice[0]
    #             solution.append((current_waypoint, RoadOption.LANEFOLLOW))
    #             wp_choice = current_waypoint.next(self._hop_resolution)

    #             #   Stop at destination
    #             if current_waypoint.transform.location.distance(
    #                 end_waypoint.transform.location) < self._hop_resolution: break
    #         if action == RoadOption.VOID: break

    #         #   Select appropriate path at the junction
    #         if len(wp_choice) > 1:

    #             # Current heading vector
    #             current_transform = current_waypoint.transform
    #             current_location = current_transform.location
    #             projected_location = current_location + \
    #                 carla.Location(
    #                     x=math.cos(math.radians(current_transform.rotation.yaw)),
    #                     y=math.sin(math.radians(current_transform.rotation.yaw)))
    #             v_current = vector(current_location, projected_location)
    #             direction = 0
    #             if action == RoadOption.LEFT:
    #                 direction = 1
    #             elif action == RoadOption.RIGHT:
    #                 direction = -1
    #             elif action == RoadOption.STRAIGHT:
    #                 direction = 0
    #             select_criteria = float('inf')

    #             #   Choose correct path
    #             for wp_select in wp_choice:
    #                 v_select = vector(
    #                     current_location, wp_select.transform.location)
    #                 cross = float('inf')
    #                 if direction == 0:
    #                     cross = abs(np.cross(v_current, v_select)[-1])
    #                 else:
    #                     cross = direction*np.cross(v_current, v_select)[-1]
    #                 if cross < select_criteria:
    #                     select_criteria = cross
    #                     current_waypoint = wp_select

    #             #   Generate all waypoints within the junction
    #             #   along selected path
    #             solution.append((current_waypoint, action))
    #             current_waypoint = current_waypoint.next(self._hop_resolution)[0]
    #             while current_waypoint.is_intersection:
    #                 solution.append((current_waypoint, action))
    #                 current_waypoint = current_waypoint.next(self._hop_resolution)[0]

    #     assert solution

    #     self._current_plan = solution
    #     self._local_planner.set_global_plan(self._current_plan)

    def _write_data(self, image):
        
        if self.config["semantic"]:
            image.convert(cc.CityScapesPalette)
        sensor_data = image.raw_data
        
        if(self.config['framestack'] == 1):
            self._save_sensor_data(sensor_data)
        else:
            # NOTE: Typecasting here since can't do a deepcopy with 'memoryview objects'
            # TODO: Find a workaround, since typecasting then discarding is inefficient.
            self._image_queue.append(np.array(sensor_data))

    def _save_sensor_data(self, sensor_data):
        self.image_data = sensor_data

    def _read_sensor_data(self):
        return np.array(self.image_data)

    def _read_data(self):
        #TODO: Read data in from sensor callback and then call preprocess function
        #sensor data is Image object for all sensors (besides LIDAR)

        if(self.config['framestack'] == 1):
            sensor_data = self._read_sensor_data()
            im_processed = self._preprocess_core(sensor_data)
        else:
            data_array = []
            # Use this loop since the callback is continuously writing into the queue
            # hence we only read in 'framestack' number of images
            # Original Atari DQN paper is unclear on order of stacking
            _image_queue_snapshot = copy.deepcopy(self._image_queue)
            for image in _image_queue_snapshot:
                data_array.append(self._preprocess_core(image))
            # data_array = list(copy.deepcopy(self._image_queue))
            #Compute ndims (to compute which axis to stack along)
            ndim = self.config['framestack']
            # Stack all the images along last axis
            im_processed = np.concatenate((data_array[:]), axis=2)
        if self.config["input_type"] == 'vae' or self.config["input_type"] == 'wp_vae':
            im_processed = self.vae_observation(im_processed)
        return im_processed

    def _preprocess_core(self, image):
        # sensor_x_res is a str (reason mentioned near definition). reshape requires int
        x_res =int(self.config["sensor_x_res"])
        y_res =int(self.config["sensor_y_res"])
        # NOTE: BGRA array is returned by RGB sensor
        data = image.reshape(x_res, y_res, 4)
        # Convert from BGRA to RGB image
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGB)
        
        if(self.config['grayscale']):
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        if(self.config['preprocess_crop_image']):
            # Cut from top and bottom
            data = data[115:510, :]

        # preprocess
        data = data[:500, :]
        data = cv2.resize(data, (self.config["x_res"], self.config["y_res"]), interpolation=cv2.INTER_AREA)
        # The cv2 resize converts to self.config["x_res"], self.config["y_res"]. We need the last channel to framestack later.
        if(self.config['grayscale']):
            data = data.reshape(self.config["x_res"], self.config["y_res"], 1)
        # TODO: Need to check better forms of normalization. Add a config flag for normalization
        # image = (image.astype(np.float32) - 128) / 128
        # data = data / 255.0
        return data

    def _compute_reward(self, name, prev_measurement, cur_measurement):
        if name == 'corl':
            reward = self._compute_reward_corl(prev_measurement, cur_measurement)
        elif name == 'cirl':
            reward = self._compute_reward_cirl(prev_measurement, cur_measurement)
        elif name == 'corl2':
            reward = self._compute_reward_corl2(prev_measurement, cur_measurement)
        elif name == 'corlT':
            reward = self._compute_reward_corlT(prev_measurement, cur_measurement)
        elif name == "simple":
            reward = self._compute_reward_simple(prev_measurement, cur_measurement)
        return reward

    def _compute_reward_cirl(self, prev, current):
        # 1) Abnormal steer penalty
        """
        if (control.steer > 0) and (directions == 3):
            # Turn right when should go left
            steer_reward = -15
        elif (control.steer < 0) and (directions == 4):
            # Turn left when should go right
            steer_reward = -15
        elif (abs(control.steer) > 0.2) and (directions in [0, 2, 5]):
            # Turn when should go straight
            # TODO: directions 0, 2 could mean follow lane that is turning
            steer_reward = -20
        else:
            steer_reward = 0
        """
        steer_reward = 0
        self.episode_measurements["steer_reward"] = steer_reward

        # 2) Collision penalty
        no_collisions = (current["num_collisions"] - prev["num_collisions"])
        collision = no_collisions > 0
        collision_reward = -30 if collision else 0
        self.episode_measurements["collision_reward"] = collision_reward

        # 3) Sidewalk and opposite lane overlap penalty
        no_lane_intersections = (current["num_laneintersections"] - prev["num_laneintersections"])
        lane_change = no_lane_intersections > 0
        lane_intersection_reward = -30 if lane_change else 0
        self.episode_measurements["lane_intersection_reward"] = lane_intersection_reward

        # 4) Speed reward (in km/h)
        #TODO: Incorporate directions once planner is ready. Default assumed to go straight.
        # converted to km/h
        speed = current["speed"] * 3.6
        speed_reward = speed if (speed < 30) else (60 - speed)
        # if directions in [0, 2]:
        #     # If following lane or going straight, limit speed to 30km/h
        #     speed_reward = speed if (speed < 30) else (60 - speed)
        # else:
        #     # If approaching intersection, limit speed to 20km/h
        #     speed_reward = speed if (speed < 20) else (40 - speed)
        self.episode_measurements["speed_reward"] = speed_reward

        # Total reward (approximately scaled to [0, 1] range)
        reward = steer_reward + collision_reward + lane_intersection_reward + speed_reward
        reward /= 30

        if np.absolute(lane_intersection_reward) > 0:
            self.episode_measurements["offlane_steps"] += 1
        if current["speed"] == 0:
            self.episode_measurements["static_steps"] += 1

        return reward

    def _compute_reward_corl(self, prev, current):
        cur_dist = current["distance_to_goal"]
        prev_dist = prev["distance_to_goal"]

        if self.config["verbose"]:
            print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

        # Distance travelled toward the goal in m
        distance_reward = 10000 * (prev_dist - cur_dist)
        self.episode_measurements["distance_reward"] = distance_reward

        # Change in speed (km/h)
        speed_reward = 0.05 * (current["speed"] - prev["speed"])
        self.episode_measurements["speed_reward"] = speed_reward

        # Collision damage
        collision_reward = -.00002 * (current["num_collisions"] - prev["num_collisions"])
        self.episode_measurements["collision_reward"] = collision_reward

        # New sidewalk intersection
        lane_intersection_reward = -2 * (current["num_laneintersections"] - prev["num_laneintersections"])
        self.episode_measurements["lane_intersection_reward"] = lane_intersection_reward

        reward = distance_reward + speed_reward + collision_reward + lane_intersection_reward

        # Update state variables
        if np.absolute(lane_intersection_reward) > 0:
            self.episode_measurements["offlane_steps"] += 1
        if current["speed"] == 0:
            self.episode_measurements["static_steps"] += 1
        return reward

    def _compute_reward_corl2(self, prev, current):
        cur_dist = current["distance_to_goal"]
        prev_dist = prev["distance_to_goal"]

        if self.config["verbose"]:
            print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

        goal_distance_reward = 1/(cur_dist)**0.5
        self.episode_measurements["goal_distance_reward"] = goal_distance_reward

        # Distance travelled toward the goal in m
        distance_reward = 0.01 * (prev_dist - cur_dist)
        self.episode_measurements["distance_reward"] = distance_reward

        # Change in speed (km/h)
        speed_reward = 0.05 * (current["speed"] - prev["speed"])
        self.episode_measurements["speed_reward"] = speed_reward

        # Collision damage
        if((current["num_collisions"] - prev["num_collisions"]) > 0):
            collision_reward = -1
        else:
            collision_reward = 0
        self.episode_measurements["collision_reward"] = collision_reward

        # New sidewalk intersection
        if((current["num_laneintersections"] - prev["num_laneintersections"]) > 0):
            lane_intersection_reward = -1
        else:
            lane_intersection_reward = 0
        self.episode_measurements["lane_intersection_reward"] = lane_intersection_reward

        # # Collision damage
        # collision_reward = -.00002 * (current["num_collisions"] - prev["num_collisions"])
        # self.episode_measurements["collision_reward"] = collision_reward

        # # New sidewalk intersection
        # lane_intersection_reward = -2 * (current["num_laneintersections"] - prev["num_laneintersections"])
        # self.episode_measurements["lane_intersection_reward"] = lane_intersection_reward


        reward = goal_distance_reward + speed_reward + distance_reward + collision_reward + lane_intersection_reward

        print("goal_distance_reward, speed_reward, distance_reward, collision_reward, lane_intersection_reward, reward")
        print(goal_distance_reward, speed_reward, distance_reward, collision_reward, lane_intersection_reward, reward)
        # Update state variables
        if np.absolute(lane_intersection_reward) > 0:
            self.episode_measurements["offlane_steps"] += 1
        if current["speed"] == 0:
            self.episode_measurements["static_steps"] += 1
        return reward
    
    def _compute_reward_simple(self, prev, current):
        cur_dist = current["distance_to_goal"]
        prev_dist = prev["distance_to_goal"]

        if self.config["verbose"]:
            print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

        dist_to_trajectory_reward = -1 * self.dist_to_trajectory
        
        speed_reward = current["speed"]
        acceleration_reward = (current["speed"] - prev["speed"])
        
        # Collision damage
        if((current["num_collisions"] - prev["num_collisions"]) > 0):
            collision_reward = -1
        else:
            collision_reward = 0
        self.episode_measurements["collision_reward"] = collision_reward

        # New sidewalk intersection
        if((current["num_laneintersections"] - prev["num_laneintersections"]) > 0):
            lane_intersection_reward = -1
        else:
            lane_intersection_reward = 0
        self.episode_measurements["lane_intersection_reward"] = lane_intersection_reward

        reward = dist_to_trajectory_reward + speed_reward

        print("dist_to_trajectory_reward, speed_reward, acceleration_reward, collision_reward, lane_intersection_reward, reward")
        print(dist_to_trajectory_reward, speed_reward, acceleration_reward, collision_reward, lane_intersection_reward, reward)
        # Update state variables
        if np.absolute(lane_intersection_reward) > 0:
            self.episode_measurements["offlane_steps"] += 1
        if current["speed"] == 0:
            self.episode_measurements["static_steps"] += 1
        return reward
    
    def _compute_reward_corlT(self, prev, current):
       cur_dist = current["distance_to_goal"]
       prev_dist = prev["distance_to_goal"]

       # Distance travelled toward the goal in m
       #distance_reward = np.clip(prev_dist - cur_dist, -10.0, 10.0)
       distance_reward = 1/(cur_dist)**0.5
       self.episode_measurements["distance_reward"] = distance_reward

       # Change in speed (km/h)
       speed_reward = 0.05 * (current["speed"] - prev["speed"])
       self.episode_measurements["speed_reward"] = speed_reward

       # Collision damage
       collision_reward = -.00002 * (current["num_collisions"] - prev["num_collisions"])
       self.episode_measurements["collision_reward"] = collision_reward

       # New sidewalk intersection
       lane_intersection_reward = -2 * (current["num_laneintersections"] - prev["num_laneintersections"])
       self.episode_measurements["lane_intersection_reward"] = lane_intersection_reward

       reward = distance_reward + speed_reward + collision_reward + lane_intersection_reward

       # Update state variables
       if np.absolute(lane_intersection_reward) > 0:
           self.episode_measurements["offlane_steps"] += 1
       if current["speed"] == 0:
           self.episode_measurements["static_steps"] += 1
       return reward

    def _compute_done_condition(self):

        # Episode termination conditions
        success = self.episode_measurements["distance_to_goal"] < self.config["dist_for_success"]
        offlane = self.episode_measurements["offlane_steps"] > self.config["max_offlane_steps"]
        static = self.episode_measurements["static_steps"] > self.config["max_static_steps"]
        collision = np.absolute(self.episode_measurements["collision_reward"]) > 0
        maxStepsTaken = self.episode_measurements["num_steps"] > self.config['max_steps']
        offlane = False
        static = False
        maxStepsTaken = False

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
