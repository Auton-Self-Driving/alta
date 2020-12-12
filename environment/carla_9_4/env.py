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

import ae.util as util

import yaml
import pickle
from scipy.interpolate import interp1d
import torch
from detectron2.config import CfgNode
# from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.defaults import DefaultPredictor
from AdelaiDet.tools.train_net import Trainer


import environment.carla_9_4.scenarios as scenarios
import environment.carla_9_4.server as server
import environment.carla_9_4.planner as planner
import environment.carla_9_4.controller as controller
import environment.carla_9_4.sensors as sensors
from environment.carla_9_4.reward import compute_reward
from environment.carla_9_4.agents.navigation.roaming_agent import RoamingAgent
from environment.carla_9_4.agents.navigation.agent import Agent
from environment.carla_9_4.agents.navigation.basic_agent import BasicAgent
from environment.carla_9_4.config import DEFAULT_ENV, DISCRETE_ACTIONS, episode_measurements
import scipy.misc
from scipy.misc import imsave
from agents.tf.ae.util import *
import matplotlib
import matplotlib.pyplot as plt

# import ipdb
# st = ipdb.set_trace

try:
    import carla
except Exception as e:
    print("Failed to import Carla")
    raise e

from carla import ColorConverter as cc
from carla.libcarla import Transform
from carla.libcarla import Location
from carla.libcarla import Rotation
import psutil

from environment.carla_9_4.env_util import (
    check_if_vehicle_in_same_lane,
    get_world_coords_from_latlong,
    convert_route_from_GPS_world
)


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
        self.rgb_camera_queue = queue.Queue()
        self.front_camera_queue = queue.Queue()
        self.rv_camera_queue = queue.Queue()
        self.target_speed = self.config['target_speed']
        self.args_longitudinal_dict = {
            'K_P': 0.1,
            'K_D': 0.0005,
            'K_I': 0.4,
            'dt': 1/10.0}
        self.actor_list = []

        # Used for testing comparison with autopilot mode.
        self.collision_sensor_list = []
        self.vehicle_agent_list = []
        self.control_list = {}
        self.goal_destination_list = {}
        self.total_successes = 0

        # Queue for stacked frames and measurements
        self.rv_stack_size = 1
        self.stacked_observation_queue = queue.Queue(maxsize=self.config['frame_stack_size'])
        self.rv_stacked_observation_queue = queue.Queue(maxsize=self.rv_stack_size)

        self.image_data = None
        self.source_transform = None
        self.destination_transform = None
        self.scenario_route = None
        self.global_planner = None
        self.trace_route = None
        self.episode_num = 0
        self.validation_episode_num = 0
        self.total_steps = 0
        self.semantic_image = None
        self.unseen = False
        self.index = 0
        self.expert_agent = False

        self.logger = logger
        self.vis_wrapper = vis_wrapper
        self.vis_wrapper_vae = vis_wrapper_vae

        self.dist_to_trajectory = None
        if(self.config['grayscale']):
            self.im_channels = 1
        else:
            self.im_channels = 3

        self.controller = controller.PIDLongitudinalController(K_P=self.args_longitudinal_dict['K_P'], K_D=self.args_longitudinal_dict['K_D'], K_I=self.args_longitudinal_dict['K_I'], dt=self.args_longitudinal_dict['dt'])


        # traffic lights detection model
        # print(os.getcwd())

        with open('../../AdelaiDet_model/config.yaml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            model = Trainer.build_model(CfgNode(cfg))
            ckpt = torch.load('../../AdelaiDet_model/state_dict.pth', map_location=torch.device('cuda'))
            # ckpt = DetectionCheckpointer(model)
            # loaded = ckpt._load_file('../../AdelaiDet_model/model_final.pth')
        with open('../../AdelaiDet_model/interpolator.pkl', 'rb') as f:
            self.dist_interpolator = pickle.load(f)
        self.traffic_light_detector = DefaultPredictor(CfgNode(cfg))
        # self.traffic_light_detector.model.load_state_dict(loaded['model']) # OpenCV BGR format image input expected
        self.traffic_light_detector.model.load_state_dict(ckpt) # OpenCV BGR format image input expected

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
        # time.sleep(120)

        # Create new client
        self.client =  self._spawn_client()
        print("server_version", self.client.get_server_version())

        # Commenting load_world, assuming default is set as Town01 in CARLA binary config
        # since sometimes, it causes timeout issues in the beginning
        self._world = self.client.load_world(self.config['city_name'])

        # time.sleep(600)

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

        if self.config["use_offline_map"]:
            with open(self.config["map_path"], 'r') as f:
                map_content = f.read()
                self._map = carla.Map(self.config["city_name"], map_content)
        else:
            self._map = self._world.get_map()
        self.blueprint_library = self._world.get_blueprint_library()
        self.spawn_points = self._world.get_map().get_spawn_points()

        self.tm = self.client.get_trafficmanager(4050)
        self.tm.set_synchronous_mode(True)

        if self.config["testing"]:
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in self.config['spawn_points_fixed_idx']]
        else:
            spawn_pt_idx = np.random.permutation(len(self.spawn_points))
            np.save(os.path.join(self.log_dir, "spawn_pt_order"), spawn_pt_idx)
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in spawn_pt_idx]

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
            elif self.config["action_type"] == 'discrete':
                # Discrete actions
                self.action_space = Discrete(len(DISCRETE_ACTIONS))
            elif self.config["action_type"] == 'control':
                # Discrete actions
                self.action_space = Discrete(len(DISCRETE_ACTIONS))


            if self.config["input_type"] == 'wp':
                self.observation_space = Box(low=np.array([-4.0]), high=np.array([4.0]), dtype=np.float32)

            elif self.config["input_type"] in ['wp_constant', 'wp_noise', 'wp_obs_dist', 'wp_obs_bool']:
                self.observation_space = Box(low=np.array([[-4.0, -1.0]]), high=np.array([[4.0, 1.0]]), dtype=np.float32)

            elif self.config["input_type"] == 'wp_ldist_goal':
                self.observation_space = Box(low=np.array([[-4.0, -1.0, 0.0]]), high=np.array([[4.0, 1.0, 1.0]]), dtype=np.float32)

            elif self.config["input_type"] == 'wp_obs_bool_noise':
                limit = np.hstack((np.array([[4]]), np.ones((1, 1 + self.config["noise_dim"]))))
                self.observation_space = Box(low=-limit, high=limit, shape=(1, 2 + self.config["noise_dim"]), dtype=np.float32)

            elif self.config["input_type"] == 'wp_speed':
                self.observation_space = Box(low=np.array([[-4.0, 0.0]]), high=np.array([[4.0, 12.0]]), dtype=np.float32)

            elif self.config["input_type"] == 'wp_speed_goal':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0]]), high=np.array([[4.0, 1.0, 1.0]]), dtype=np.float32)

            elif self.config["input_type"] == 'wp_speed_steer_goal':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, -0.5, 0.0]]), high=np.array([[4.0, 1.0, 0.5, 1.0]]), dtype=np.float32)

            elif self.config["input_type"] == 'wp_speed_steer_goal_obs_bool':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, -0.5, 0.0, 0.0]]), high=np.array([[4.0, 1.0, 0.5, 10.0, 1.0]]), dtype=np.float32)

            elif self.config["input_type"] == 'wp_obs_bool_speed_steer_goal_light':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, -0.5, 0.0, 0.0]]), high=np.array([[4.0, 1.0, 1.0, 0.5, 1.0, 1.0]]), dtype=np.float32)

            elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_goal_light':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, 0.0, -0.5, -1.0, 0.0, 0.0]]), high=np.array([[4.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0]]), dtype=np.float32)

            elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_goal':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, 0.0, -0.5, -1.0, 0.0]]), high=np.array([[4.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]]), dtype=np.float32)

            elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_light':
                self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, 0.0, -0.5, -1.0, 0.0]]), high=np.array([[4.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]]), dtype=np.float32)

            elif self.config["input_type"] == 'wp_angles_obs_info_speed_steer_ldist_light':
                self.observation_space = Box(low=np.array([[-4.0, -4.0, -4.0, -4.0, -4.0, -1.0, -1.0, 0.0, -0.5, -1.0, -1.0]]),
                                                high=np.array([[4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]]), dtype=np.float32)

            elif self.config["input_type"] == 'wp_vecs_obs_info_speed_steer_ldist_light':
                self.observation_space = Box(low=np.array([[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, -0.5, -1.0, -1.0]]),
                                        high=np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]]), dtype=np.float32)

            elif self.config["input_type"] == 'wp_angles_vecs_obs_info_speed_steer_ldist_light':
                self.observation_space = Box(low=np.array([[-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, -0.5, -1.0, -1.0]]),
                                        high=np.array([[4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]]), dtype=np.float32)

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
                                        # shape=(1, 406), dtype=np.float32) # Model used for Learning to drive using Waypoints (last layer dim = 16)
                                        shape=(1, 1606), dtype=np.float32) # Model used for Learning to Drive with Dynamic Actors (last layer dim = 64)

            elif self.config["input_type"] == 'wp_vae_obs_info_speed_steer_ldist_goal_light':
                self.observation_space = Box(low=np.finfo(np.float32).min,
                                        high=np.finfo(np.float32).max,
                                        # shape=(1, 408), dtype=np.float32) # Model used for Learning to drive using Waypoints (last layer dim = 16)
                                        shape=(1, 1608), dtype=np.float32) # Model used for Learning to Drive with Dynamic Actors (last layer dim = 64)

            elif self.config["input_type"] == 'wp_cnn_obs_info_speed_steer_ldist_goal_light' or self.config["input_type"] == 'wp_bev_rv_obs_info_speed_steer_ldist_goal_light':
                if not self.config["single_channel_image"]:
                    if self.config["binarized_image"]:
                        dim = 2
                    else:
                        dim = 5
                else:
                    dim = 1
                self.observation_space = Box(low=np.finfo(np.float32).min,
                                        high=np.finfo(np.float32).max,
                                        shape=(1, (int(self.config['sensor_y_res']) * int(self.config['sensor_x_res']) * dim * self.config['frame_stack_size']) + 8), dtype=np.float32)
                                        # shape=(1, 12296), dtype=np.float32)
                                        # shape=(1, 20488), dtype=np.float32)

        self.vehicle_blueprints = self._world.get_blueprint_library().filter('vehicle.*')
        self.traffic_actors = self._world.get_actors().filter("*traffic_light*")

        if self.config["disable_two_wheeler"]:
            self.vehicle_blueprints = [x for x in self.vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        self.episode_measurements["episode_num"] = 0
        self.episode_measurements['obstacle_visible'] = False
        self.episode_measurements['obstacle_dist'] = -1
        self.episode_measurements['obstacle_speed'] = -1
        self.episode_measurements['obstacle_orientation'] = -1
        self.next_waypoints = None
        self.next_wp_vectors = None
        self.next_wp_angles = None

        self.episode_measurements['dist_to_light'] = -1
        self.episode_measurements['nearest_traffic_actor_id'] = -1
        self.episode_measurements['nearest_traffic_actor_state'] = None
        self.episode_measurements['initial_dist_to_red_light'] = -1
        self.episode_measurements['red_light_dist'] = -1
        self.episode_measurements['traffic_light_orientation'] = -1
        self.episode_measurements["runover_light"] = False
        self.vehicle_collisions = 0
        self.static_collisions = 0
        self.total_collisions = 0
        self.total_distance = 0
        self.traffic_light_violations = 0

        self.target_speeds_array = []
        self.speeds_array = []
        self.throttles_array = []
        self.obstacle_speed_array = []
        self.dist_to_trajectory_array = []
        self.steers_array = []
        self.brakes_array = []
        self.wp_orientation_array = []
        self.input_steer_array = []
        self.obstacle_dist_array = []
        self.step_reward_array = []
        self.collision_reward_array = []
        self.dist_to_trajectory_reward_array = []
        self.speed_reward_array = []
        self.dist_to_target_array = []
        self.red_light_dist_array = []

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

        elif self.config["input_type"] == 'wp_ldist_goal':
            ldist = self.episode_measurements['dist_to_trajectory']
            distance_to_goal_trajec = self.episode_measurements['distance_to_goal_trajec'] / 500
            obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([ldist]), np.array([distance_to_goal_trajec])))

        elif self.config["input_type"] == 'wp_obs_bool_noise':
            obs_bool = self.episode_measurements['obstacle_visible']
            obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([obs_bool]), np.random.normal(0.0, 1.0, self.config["noise_dim"])))

        elif self.config["input_type"] == 'wp_speed':
            obs_speed = self.episode_measurements['speed'] / 10
            obs['observation'] = np.concatenate((np.array(self.episode_measurements['next_orientation']), np.array([obs_speed])))

        elif self.config["input_type"] == 'wp_speed_goal':
            obs_speed = self.episode_measurements['speed'] / 10
            distance_to_goal_trajec = self.episode_measurements['distance_to_goal_trajec'] / 100
            obs['observation'] = np.concatenate((np.array(self.episode_measurements['next_orientation']), np.array([obs_speed]), np.array([distance_to_goal_trajec])))

        elif self.config["input_type"] == 'wp_speed_steer_goal':
            obs_speed = self.episode_measurements['speed'] / 10
            distance_to_goal_trajec = self.episode_measurements['distance_to_goal_trajec'] / 100
            steer = self.episode_measurements['control_steer']
            obs['observation'] = np.concatenate((np.array(self.episode_measurements['next_orientation']), np.array([obs_speed]), np.array([steer]), np.array([distance_to_goal_trajec])))

        elif self.config["input_type"] == 'wp_speed_steer_goal_obs_bool':
            obs_speed = self.episode_measurements['speed'] / 10
            distance_to_goal_trajec = self.episode_measurements['distance_to_goal_trajec'] / 100
            steer = self.episode_measurements['control_steer']
            obs_bool = self.episode_measurements['obstacle_visible']
            obs['observation'] = np.concatenate((np.array(self.episode_measurements['next_orientation']), np.array([obs_speed]), np.array([steer]), np.array([distance_to_goal_trajec]), np.array([obs_bool])))

        elif self.config["input_type"] == 'wp_obs_bool_speed_steer_goal_light':

            speed = self.episode_measurements['speed'] / 10
            obs_bool = self.episode_measurements['obstacle_visible']
            steer = self.episode_measurements['control_steer']
            distance_to_goal_trajec = self.episode_measurements['distance_to_goal_trajec'] / 500
            light = self.episode_measurements['red_light_dist']

            # normalization
            if light != -1:
                light /= self.config['traffic_light_proximity_threshold']
            else:
                light = self.config['default_obs_traffic_val']

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
                obstacle_dist = obstacle_dist / self.config['vehicle_proximity_threshold']
            else:
                obstacle_dist = self.config['default_obs_traffic_val']

            if obstacle_speed != -1:
                obstacle_speed = obstacle_speed / 20
            else:
                obstacle_speed = self.config['default_obs_traffic_val']

            if light != -1:
                light /= self.config['traffic_light_proximity_threshold']
            else:
                light = self.config['default_obs_traffic_val']

            obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([distance_to_goal_trajec]), np.array([light])))

        elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_goal':

            speed = self.episode_measurements['speed'] / 10
            obstacle_dist = self.episode_measurements['obstacle_dist']
            obstacle_speed = self.episode_measurements['obstacle_speed']
            steer = self.episode_measurements['control_steer']
            ldist = self.episode_measurements['dist_to_trajectory']
            distance_to_goal_trajec = self.episode_measurements['distance_to_goal_trajec'] / 500

            # normalization

            if obstacle_dist != -1:
                obstacle_dist = obstacle_dist / self.config['vehicle_proximity_threshold']
            else:
                obstacle_dist = self.config['default_obs_traffic_val']

            if obstacle_speed != -1:
                obstacle_speed = obstacle_speed / 20
            else:
                obstacle_speed = self.config['default_obs_traffic_val']

            obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([distance_to_goal_trajec])))

        elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_light':

            speed = self.episode_measurements['speed'] / 10
            obstacle_dist = self.episode_measurements['obstacle_dist']
            obstacle_speed = self.episode_measurements['obstacle_speed']
            steer = self.episode_measurements['control_steer']
            ldist = self.episode_measurements['dist_to_trajectory']
            light = self.episode_measurements['red_light_dist']

            # normalization

            if obstacle_dist != -1:
                obstacle_dist = obstacle_dist / self.config['vehicle_proximity_threshold']
            else:
                obstacle_dist = self.config['default_obs_traffic_val']

            if obstacle_speed != -1:
                obstacle_speed = obstacle_speed / 20
            else:
                obstacle_speed = self.config['default_obs_traffic_val']

            if light != -1:
                light /= self.config['traffic_light_proximity_threshold']
            else:
                light = self.config['default_obs_traffic_val']

            obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([light])))


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
                light /= self.config['traffic_light_proximity_threshold']
            else:
                light = self.config['default_obs_traffic_val']

            obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([distance_to_goal_trajec]), np.array([light])))

        elif self.config["input_type"] in ['wp_vae_obs_info_speed_steer_ldist_goal_light', 'wp_cnn_obs_info_speed_steer_ldist_goal_light', 'wp_bev_rv_obs_info_speed_steer_ldist_goal_light']:
            speed = self.episode_measurements['speed'] / 10
            obstacle_dist = self.episode_measurements['obstacle_dist']
            obstacle_speed = self.episode_measurements['obstacle_speed']
            steer = self.episode_measurements['control_steer']
            ldist = self.episode_measurements['dist_to_trajectory']
            distance_to_goal_trajec = self.episode_measurements['distance_to_goal_trajec'] / 500
            light = self.episode_measurements['red_light_dist']

            # normalization

            if obstacle_dist != -1:
                obstacle_dist = obstacle_dist / self.config['vehicle_proximity_threshold']
            else:
                obstacle_dist = self.config['default_obs_traffic_val']

            if obstacle_speed != -1:
                obstacle_speed = obstacle_speed / 20
            else:
                obstacle_speed = self.config['default_obs_traffic_val']

            if light != -1:
                light /= self.config['traffic_light_proximity_threshold']
            else:
                light = self.config['default_obs_traffic_val']

            obs['observation'] = np.concatenate((np.array([self.episode_measurements['next_orientation']]), np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([distance_to_goal_trajec]), np.array([light])))



        elif self.config["input_type"] == 'wp_angles_obs_info_speed_steer_ldist_light':
            wp_angles_array, wp_vectors_array = self.get_wp_obs_input()
            speed = self.episode_measurements['speed'] / 10
            obstacle_dist = self.episode_measurements['obstacle_dist']
            obstacle_speed = self.episode_measurements['obstacle_speed']
            steer = self.episode_measurements['control_steer']
            ldist = self.dist_to_trajectory
            light = self.episode_measurements['red_light_dist']

            # normalization
            if obstacle_dist != -1:
                obstacle_dist = obstacle_dist / self.config['vehicle_proximity_threshold']
            else:
                obstacle_dist = self.config['default_obs_traffic_val']

            if obstacle_speed != -1:
                obstacle_speed = obstacle_speed / 20
            else:
                obstacle_speed = self.config['default_obs_traffic_val']

            if light != -1:
                light /= self.config['traffic_light_proximity_threshold']
            else:
                light = self.config['default_obs_traffic_val']
            obs['orientation'] = np.concatenate((wp_angles_array, np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([light])))

        elif self.config["input_type"] == 'wp_vecs_obs_info_speed_steer_ldist_light':
            wp_angles_array, wp_vectors_array = self.get_wp_obs_input()

            # normalize vectors by 10, assuming max norm of vector would be 10
            wp_vectors_array = wp_vectors_array / 10
            speed = self.episode_measurements['speed'] / 10
            obstacle_dist = self.episode_measurements['obstacle_dist']
            obstacle_speed = self.episode_measurements['obstacle_speed']
            steer = self.episode_measurements['control_steer']
            ldist = self.dist_to_trajectory
            light = self.episode_measurements['red_light_dist']
            # normalization
            if obstacle_dist != -1:
                obstacle_dist = obstacle_dist / self.config['vehicle_proximity_threshold']
            else:
                obstacle_dist = self.config['default_obs_traffic_val']

            if obstacle_speed != -1:
                obstacle_speed = obstacle_speed / 20
            else:
                obstacle_speed = self.config['default_obs_traffic_val']

            if light != -1:
                light /= self.config['traffic_light_proximity_threshold']
            else:
                light = self.config['default_obs_traffic_val']
            obs['orientation'] = np.concatenate((wp_vectors_array, np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([light])))

        elif self.config["input_type"] == 'wp_angles_vecs_obs_info_speed_steer_ldist_light':
            wp_angles_array, wp_vectors_array = self.get_wp_obs_input()

            # normalize vectors by 10, assuming max norm of vector would be 10
            wp_vectors_array = wp_vectors_array / 10
            speed = self.episode_measurements['speed'] / 10
            obstacle_dist = self.episode_measurements['obstacle_dist']
            obstacle_speed = self.episode_measurements['obstacle_speed']
            steer = self.episode_measurements['control_steer']
            ldist = self.dist_to_trajectory
            light = self.episode_measurements['red_light_dist']
            # normalization
            if obstacle_dist != -1:
                obstacle_dist = obstacle_dist / self.config['vehicle_proximity_threshold']
            else:
                obstacle_dist = self.config['default_obs_traffic_val']

            if obstacle_speed != -1:
                obstacle_speed = obstacle_speed / 10
            else:
                obstacle_speed = self.config['default_obs_traffic_val']

            if light != -1:
                light /= self.config['traffic_light_proximity_threshold']
            else:
                light = self.config['default_obs_traffic_val']
            obs['orientation'] = np.concatenate((np.array([next_orientation]), wp_angles_array, wp_vectors_array, np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([light])))

    def step(self, action):
        try:
            if not self.config['test_comparison']:
                obs = self._step(action)
                return obs
            else:
                self._step_test_comparison(action)
                return None
        except Exception:
            print("Error during step, terminating episode early", traceback.format_exc())
            raise
            # self.reset()

    def get_action_for_test_comparison(self):
        for ind, agent in enumerate(self.vehicle_agent_list):
            control = agent.run_step()
            self.control_list[ind] = control
        self.step(None)

    def _step_test_comparison(self, action):

        world_frame = None

        for _ in range(self.config["frame_skip"]):

            batch = []
            for ind, vehicle in enumerate(self.actor_list):
                batch.append(carla.command.ApplyVehicleControl(vehicle, self.control_list[ind]))

            self.client.apply_batch_sync(batch)

            world_frame = self._world.tick()
            self.num_steps += 1

            self.episode_measurements['num_steps'] = self.num_steps
            self.episode_measurements['num_collisions'] = sum([sensor.num_collisions for sensor in self.collision_sensor_list])

            self.episode_measurements['distance_to_goal'] = [vehicle.get_location().distance(self.goal_destination_list[ind].location) for ind, vehicle in enumerate(self.actor_list)]
            self.episode_measurements['speed'] = [self.get_speed_from_velocity(vehicle.get_velocity()) for vehicle in self.actor_list]
            print("Num Steps: {}, Num Collisons: {}".format(self.episode_measurements['num_steps'], self.episode_measurements['num_collisions']))

            for ind, goal_distance in enumerate(self.episode_measurements['distance_to_goal']):
                if goal_distance < self.config["dist_for_success"]:
                    print("***************************DONE: {}***********************".format(ind + 1))
                    random_waypoint = random.choice(self.spawn_points)
                    self.goal_destination_list[ind] = random_waypoint
                    self.vehicle_agent_list[ind].set_destination((random_waypoint.location.x,
                                                                  random_waypoint.location.y,
                                                                  random_waypoint.location.z))

            self.prev_measurement = copy.deepcopy(self.episode_measurements)

        # obs = {}
        # sensor_image = self._read_data(self.camera_queue, world_frame)

        # obs['image'] = sensor_image
        # self.vis_wrapper.save_image(obs['image'], self.num_steps)

        return None, None, None, self.episode_measurements

    def _step(self, action):

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

                #Print actions
                if self.config['verbose']:
                    print("steer", control.steer, "throttle", control.throttle, "brake", control.brake,
                  "reverse", control.reverse)
                    print("steps", self.num_steps)


                #Store control for this step
                self.episode_measurements['control_steer'] = control.steer
                self.episode_measurements['control_throttle'] = control.throttle
                self.episode_measurements['control_brake'] = control.brake
                self.episode_measurements['control_reverse'] = control.reverse
                self.episode_measurements['control_hand_brake'] = control.hand_brake

            self.obstacle_dist_array.append(self.episode_measurements['obstacle_dist'])
            self.obstacle_speed_array.append(self.episode_measurements['obstacle_speed'])
            self.wp_orientation_array.append(self.episode_measurements['next_orientation'])
            self.input_steer_array.append(self.episode_measurements['control_steer'])
            self.speeds_array.append(self.episode_measurements['speed'] * 3.6)
            self.red_light_dist_array.append(self.episode_measurements['red_light_dist'])
            self.dist_to_trajectory_array.append(self.episode_measurements['dist_to_trajectory'])
            self.dist_to_target_array.append(self.episode_measurements['distance_to_goal_trajec'])

            self.vehicle_actor.apply_control(control)
            world_frame = self._world.tick()
            self.num_steps += 1

            if not self.unseen:
                self.total_steps +=1

            self.episode_measurements['num_steps'] = self.num_steps

            # Set state variables for reward calculation
            self.episode_measurements['num_collisions'] = self.collision_sensor.num_collisions
            self.episode_measurements['collision_actor_id'] = self.collision_sensor.actor_id
            self.episode_measurements['collision_actor_type'] = self.collision_sensor.actor_type
            if self.config["enable_lane_invasion_sensor"]:
                self.episode_measurements['num_laneintersections'] = self.lane_invasion_sensor.num_laneintersections
                self.episode_measurements['out_of_road'] = int(self.lane_invasion_sensor.out_of_road)
            self.location = self.vehicle_actor.get_location()
            self.episode_measurements['distance_to_goal'] = self.location.distance(self.destination_transform.location)
            if self.episode_measurements['min_distance_to_goal'] >= self.location.distance(self.destination_transform.location):
                self.episode_measurements['min_distance_to_goal'] = self.location.distance(self.destination_transform.location)
            self.episode_measurements['speed'] = self.get_speed_from_velocity(self.vehicle_actor.get_velocity())

            # if self.config["algo"] == "AE":
            #     next_orientation, self.dist_to_trajectory = 0, 0
            # else:
            next_orientation, self.dist_to_trajectory, distance_to_goal_trajec, self.next_waypoints, self.next_wp_angles, self.next_wp_vectors = self.global_planner.get_next_orientation_new(self.vehicle_actor.get_transform())
            self.episode_measurements['next_orientation'] = next_orientation
            self.episode_measurements['distance_to_goal_trajec'] = distance_to_goal_trajec
            self.episode_measurements['dist_to_trajectory'] = self.dist_to_trajectory


            # Read in preprocessed image
            # sensor_image = self._read_data(self.camera_queue, world_frame)
            sensor_image = front_image = self._read_data(self.front_camera_queue, world_frame)
            rv_sensor_image = self._read_data(self.rv_camera_queue, world_frame)
            # Update obstacle distance measurements
            rgb_image = self._read_data(self.rgb_camera_queue, world_frame)
            self._update_env_obs(front_rgb_image=rgb_image)
            #self._update_env_obs()

            if self.config["scenarios"] == "straight_dynamic":
                self._update_straight_dynamic_obs()

            reward += compute_reward(name=self.config['reward_function'],
                                prev_measurement=self.prev_measurement,
                                cur_measurement=self.episode_measurements,
                                config=self.config,
                                verbose=self.config["verbose"])

            obs_collision = self.episode_measurements['num_collisions'] - self.prev_measurement['num_collisions'] > 0

            if obs_collision and self.episode_measurements["collision_actor_id"] != self.prev_measurement["collision_actor_id"]:
                self.total_collisions += 1
                if 'vehicle' in self.episode_measurements['collision_actor_type']:
                    self.vehicle_collisions += 1
                else:
                    self.static_collisions += 1
            elif not obs_collision:
                self.episode_measurements["collision_actor_id"] = None

            self.episode_measurements['is_collision'] = obs_collision
            if self.episode_measurements['runover_light']:
                self.traffic_light_violations += 1

            if self.config["verbose"]:
                print("Collisions Total: {}, Vehicle: {}, Static: {}".format(self.total_collisions, self.vehicle_collisions, self.static_collisions))
                print("Traffic Light Violations: {}".format(self.traffic_light_violations))

            done = self._compute_done_condition()

            self.episode_measurements['done'] = done
            self.prev_measurement = copy.deepcopy(self.episode_measurements)

            self.target_speeds_array.append(self.episode_measurements['target_speed'])
            self.throttles_array.append(control.throttle)
            self.steers_array.append(control.steer)
            self.brakes_array.append(control.brake)
            self.episode_measurements['step_reward'] = 0
            self.step_reward_array.append(self.episode_measurements['step_reward'])
            self.collision_reward_array.append(self.episode_measurements['collision_reward'])
            self.episode_measurements['dist_to_trajectory_reward'] = 0
            self.dist_to_trajectory_reward_array.append(self.episode_measurements['dist_to_trajectory_reward'])
            self.speed_reward_array.append(self.episode_measurements['speed_reward'])

            if done:
                break

        self.total_reward += reward
        self.episode_measurements['reward'] = reward
        self.episode_measurements['total_reward'] = self.total_reward

        #TODO: Get branch_idx from planner and set accordingly.
        branch_idx = 1

        # rgb_image = self._read_data(self.rgb_camera_queue, world_frame)
        # front_image = self._read_data(self.front_camera_queue, world_frame)
        visual_observation = None

        if self.config["input_type"] in ['vae', 'wp_vae', 'wp_vae_speed_steer_goal',
                                         'wp_vae_speed_steer_ldist_goal_light', 'wp_vae_obs_info_speed_steer_ldist_goal_light',
                                         'wp_cnn_obs_info_speed_steer_ldist_goal_light', 'wp_bev_rv_obs_info_speed_steer_ldist_goal_light']:

            if self.config["semantic"]:
                semantic_image = sensor_image[:,:,0]
                rv_semantic_image = rv_sensor_image[:,:,0]

                semantic_image = reduce_classes(semantic_image, binarized_image=self.config['binarized_image'])
                rv_semantic_image = reduce_classes(rv_semantic_image, binarized_image=self.config['binarized_image'])

                obs['semantic_image'] = semantic_image
                obs['rv_semantic_image'] = rv_semantic_image

                if not self.config['single_channel_image']:
                    if self.config['binarized_image']:
                        semantic_image = convert_to_one_hot(semantic_image, num_classes=2)
                        rv_semantic_image = convert_to_one_hot(rv_semantic_image, num_classes=2)
                    else:
                        semantic_image = convert_to_one_hot(semantic_image, num_classes=5)
                        rv_semantic_image = convert_to_one_hot(rv_semantic_image, num_classes=5)

                self._add_to_stacked_queue(self.stacked_observation_queue, semantic_image)
                self._add_to_stacked_queue(self.rv_stacked_observation_queue, rv_semantic_image)
            else:
                self._add_to_stacked_queue(self.stacked_observation_queue, sensor_image)
                self._add_to_stacked_queue(self.rv_stacked_observation_queue, rv_sensor_image)


            if not self.config['single_channel_image']:
                stacked_observation = np.concatenate(list(self.stacked_observation_queue.queue), axis=2)
                rv_stacked_observation = np.concatenate(list(self.rv_stacked_observation_queue.queue), axis=2)
            else:
                stacked_observation = np.stack(list(self.stacked_observation_queue.queue), axis=2)
                rv_stacked_observation = np.stack(list(self.rv_stacked_observation_queue.queue), axis=2)

            if 'vae' in self.config["input_type"]:
                visual_observation = self.vae_observation(stacked_observation)
                rv_visual_observation = self.vae_observation(rv_stacked_observation)
                visual_observation = visual_observation / self.config["vae_encoding_norm_factor"]
                rv_visual_observation = rv_visual_observation / self.config["vae_encoding_norm_factor"]
            else:
                visual_observation = stacked_observation
                rv_visual_observation = rv_stacked_observation

        if self.config["input_type"] == "ae_train":
            semantic_image = sensor_image[:,:,0]
            rv_semantic_image = rv_sensor_image[:,:,0]
            obs['semantic_image'] = semantic_image
            obs['rv_semantic_image'] = rv_semantic_image

        obs['speed'] = np.expand_dims(
            np.array([self.episode_measurements['speed']]), axis=0)  # * 3.6 / 30
        obs['dist_to_target'] = np.array(
            [self.episode_measurements['distance_to_goal']])

        # Update observation input in obs dictionary
        self.create_observations(obs)

        reward = np.expand_dims(np.array([reward]), axis=0)
        done = np.expand_dims(np.array([done]), axis=0)

        if self.config["train_config"] == "PPO":
            # Save videos now only for validation runs
            if self.config["videos"] and self.unseen:
                if self.vis_wrapper is not None:
                    # TODO: Check and uncomment when running with VAE
                    # if self.config["input_type"] in ['vae', 'wp_vae', 'wp_vae_speed_steer_goal']:
                    #     # self.vis_wrapper.save_semantic_image(obs['semantic_image'], self.num_steps)
                    #     self.vis_wrapper.save_pil_image(convert_to_rgb(obs['semantic_image'], reduced_classes=True, binarized_image=self.config['binarized_image']).astype(np.uint8), self.num_steps, self.episode_measurements)
                    # else:
                    #     # Saving image logic for Auto-Encoder training
                    #     # path = os.path.join(self.log_dir, "ae_images")
                    #     # if not os.path.exists(path):
                    #     #     os.makedirs(path)
                    #     # np.savez_compressed(os.path.join(path, format(self.total_steps, '08d')), img=convert_to_one_hot(reduce_classes(obs['image'][:, :, 0]), num_classes=5))

                    # Logic for combined videos
                    # temp_image = np.hstack((front_image, rgb_image, convert_to_rgb(reduce_classes(obs['image'][:, :, 0], binarized_image=self.config['binarized_image']), reduced_classes=True, binarized_image=self.config['binarized_image']).astype(np.uint8)))
                    # self.vis_wrapper.save_image(temp_image, self.num_steps)

                    if self.config["semantic"]:
                        self.vis_wrapper.save_pil_image(convert_to_rgb(reduce_classes(obs['rv_image'][:, :, 0], binarized_image=self.config['binarized_image']), reduced_classes=True, binarized_image=self.config['binarized_image']).astype(np.uint8), self.num_steps, self.episode_measurements)
                    else:
                        self.vis_wrapper.save_pil_image(obs['image'], self.num_steps, self.episode_measurements)
                if self.vis_wrapper_vae is not None:

                    # Logic for combined videos
                    # temp_image = np.hstack((front_image, rgb_image, convert_to_rgb(convert_from_one_hot(self.vae.decode(visual_observation)[0, :, :, -5:]), reduced_classes=True, binarized_image=self.config['binarized_image']).astype(np.uint8)))
                    # self.vis_wrapper_vae.save_image(temp_image, self.num_steps)
                    # self.vis_wrapper_vae.save_pil_image(convert_to_rgb(convert_from_one_hot(self.vae.decode(visual_observation)[0, :, :, -5:]), reduced_classes=True, binarized_image=self.config['binarized_image']).astype(np.uint8), self.num_steps, self.episode_measurements)
                    self.vis_wrapper_vae.save_pil_image(convert_to_rgb(convert_from_one_hot(self.vae.decode(visual_observation)[0, :, :, -13:]), reduced_classes=False, binarized_image=self.config['binarized_image']).astype(np.uint8), self.num_steps, self.episode_measurements)
            if not self.unseen and self.logger is not None and self.total_steps % self.config["log_freq"] == 0:
                # self.logger.log_scalar('timesteps/train/orientation', self.episode_measurements['next_orientation'], self.total_steps)
                # self.logger.log_scalar('timesteps/train/orientation_old', next_orientation_old, self.total_steps)
                # self.logger.log_scalar('timesteps/train/c_throttle', control.throttle, self.total_steps)
                # self.logger.log_scalar('timesteps/train/c_speed', self.episode_measurements['speed'] * 3.6, self.total_steps)
                # self.logger.log_scalar('timesteps/train/c_steer', control.steer, self.total_steps)
                # self.logger.log_scalar('timesteps/train/c_brake', self.episode_measurements['control_brake'], self.total_steps)
                # self.logger.log_scalar('timesteps/train/c_speed_target', self.episode_measurements['target_speed'], self.total_steps)
                # self.logger.log_scalar('timesteps/train/reward_dist_to_trajectory', self.episode_measurements['dist_to_trajectory_reward'], self.total_steps)
                # self.logger.log_scalar('timesteps/train/reward_speed', self.episode_measurements['speed_reward'], self.total_steps)
                # self.logger.log_scalar('timesteps/train/steer_reward', self.episode_measurements['steer_reward'], self.total_steps)
                # self.logger.log_scalar('timesteps/train/reward_step', self.episode_measurements['step_reward'], self.total_steps)
                # self.logger.log_scalar('timesteps/train/reward_collision', self.episode_measurements['collision_reward'], self.total_steps)
                # self.logger.log_scalar('timesteps/train/reward_light', self.episode_measurements['light_reward'], self.total_steps)
                # self.logger.log_scalar('timesteps/train/obstacle_visible', self.episode_measurements['obstacle_visible'], self.total_steps)

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
                    if self.episode_num % 100 == 0:
                        path = self.log_dir + 'train_episode_info_plots/'
                        plotname = 'TrainEp_' + str(self.episode_num) + '_step_' + str(self.total_steps)
                        plot_episode_info(path,
                            self.target_speeds_array,
                            self.speeds_array,
                            self.throttles_array,
                            self.steers_array,
                            # self.brakes_array,
                            self.wp_orientation_array,
                            self.obstacle_dist_array,
                            self.step_reward_array,
                            self.collision_reward_array,
                            self.dist_to_trajectory_reward_array,
                            self.red_light_dist_array,
                            plotname)

                # Validation runs
                else:
                    self.validation_episode_num += 1
                    plotname = 'ValEp_' + str(self.validation_episode_num) + '_TrainEp_' + str(self.episode_num) + '_step_' + str(self.total_steps) + "_ind_" + str(self.index)
                    self.episode_measurements['val_ep_idx'] = plotname
                    if self.config["testing"]:
                        path = self.log_dir + 'test_episode_info_plots_{}/'.format(self.config['city_name'])
                    else:
                        path = self.log_dir + 'val_episode_info_plots_{}/'.format(self.config['city_name'])
                    plot_episode_info(path,
                        self.target_speeds_array,
                        self.speeds_array,
                        self.throttles_array,
                        self.steers_array,
                        # self.brakes_array,
                        self.wp_orientation_array,
                        self.obstacle_dist_array,
                        self.step_reward_array,
                        self.collision_reward_array,
                        self.dist_to_trajectory_reward_array,
                        self.red_light_dist_array,
                        plotname)

                    if self.config["testing"]:
                        np.savez_compressed(os.path.join(path, 'test_stats_{}.npz'.format(self.validation_episode_num)),
                                            target_speed=self.target_speeds_array, current_speed=self.speeds_array, steer=self.steers_array,
                                            input_steer=self.input_steer_array, throttle=self.throttles_array, brake=self.brakes_array,
                                            obstacle_dist=self.obstacle_dist_array, obstacle_speed=self.obstacle_speed_array,
                                            wp_orientation=self.wp_orientation_array, red_light_dist=self.red_light_dist_array,
                                            dist_to_trajectory=self.dist_to_trajectory_array, dist_to_goal=self.dist_to_target_array,
                                            step_reward=self.step_reward_array, collision_reward=self.collision_reward_array,
                                            dist_to_trajectory_reward=self.dist_to_trajectory_reward_array, speed_reward=self.speed_reward_array)

                self.episode_measurements["episode_num"] = self.episode_num

                if self.logger is not None:

                    if not self.unseen and self.episode_num % 100 == 0:
                        self.logger.log_scalar('episodes/train/dist_to_target', self.episode_measurements['distance_to_goal'], self.episode_num)
                        # self.logger.log_scalar('episodes/train/diff_dist_to_target', (self.episode_measurements['distance_to_goal'] - self.episode_measurements['min_distance_to_goal']), self.episode_num)
                        self.logger.log_scalar('episodes/train/reward', self.episode_measurements['total_reward'], self.episode_num)
                        self.logger.log_scalar('timesteps/train/dist_to_target', self.episode_measurements['distance_to_goal'], self.total_steps)
                        # self.logger.log_scalar('timesteps/train/diff_dist_to_target', (self.episode_measurements['distance_to_goal'] - self.episode_measurements['min_distance_to_goal']), self.total_steps)
                        self.logger.log_scalar('timesteps/train/reward', self.episode_measurements['total_reward'], self.total_steps)

                        # Termination logs
                        self.logger.log_scalar('episodes/train/term_obstacle', self.episode_measurements['obs_collision'], self.episode_num)
                        if self.config["enable_lane_invasion_sensor"]:
                            self.logger.log_scalar('episodes/train/term_out_of_road', self.episode_measurements['out_of_road'], self.episode_num)
                            self.logger.log_scalar('episodes/train/term_lane_change', self.episode_measurements['lane_change'], self.episode_num)
                        self.logger.log_scalar('episodes/train/term_runover_light', self.episode_measurements['runover_light'], self.episode_num)
                        success = 1 if self.episode_measurements['termination_state'] == 'success' else 0
                        self.logger.log_scalar('episodes/train/term_success', success, self.episode_num)
                        static = 1 if self.episode_measurements['termination_state'] == 'static' else 0
                        self.logger.log_scalar('episodes/train/term_static', static, self.episode_num)
                        max_steps = 1 if self.episode_measurements['termination_state'] == 'max_steps' else 0
                        self.logger.log_scalar('episodes/train/term_max_steps', max_steps, self.episode_num)
                        # self.logger.log_scalar('episodes/train/reward_collision', self.episode_measurements['collision_reward'], self.episode_num)
                        # self.logger.log_scalar('episodes/train/obstacle_dist', self.episode_measurements['obstacle_dist'], self.episode_num)


                    elif self.unseen:

                        self.logger.log_scalar('test/dist_to_target_' + str(self.index), self.episode_measurements['distance_to_goal'], self.total_steps)
                        self.logger.log_scalar('test/reward_' + str(self.index), self.episode_measurements['total_reward'], self.total_steps)

                        # self.logger.log_scalar('test/reward_collision_' + str(self.index), self.episode_measurements['collision_reward'], self.total_steps)
                        # self.logger.log_scalar('test/out_of_road_' + str(self.index), self.episode_measurements['out_of_road'], self.total_steps)

                # Save videos now only for validation runs
                if self.config["videos"] and self.unseen:
                    if self.vis_wrapper is not None:
                        self.vis_wrapper.generate_video(self.validation_episode_num, self.total_steps, self.index)
                        self.vis_wrapper.remove_images()
                    if self.vis_wrapper_vae is not None:
                        self.vis_wrapper_vae.generate_video(self.validation_episode_num, self.total_steps, self.index)
                        self.vis_wrapper_vae.remove_images()

        if self.config["input_type"] == 'vae':
            return visual_observation, reward, done, self.episode_measurements
        elif self.config["input_type"] in ['wp_vae', 'wp_vae_speed_steer_goal', 'wp_vae_speed_steer_ldist_goal_light', 'wp_vae_obs_info_speed_steer_ldist_goal_light']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            fused_input = np.hstack([visual_observation, observation])
            return fused_input, reward, done, self.episode_measurements
        elif self.config["input_type"] in ['wp_cnn_obs_info_speed_steer_ldist_goal_light']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            visual_observation = visual_observation.reshape((1, -1))
            fused_input = np.hstack([visual_observation, observation])
            return fused_input, reward, done, self.episode_measurements
        elif self.config["input_type"] in ['wp_bev_rv_obs_info_speed_steer_ldist_goal_light']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            visual_observation = visual_observation.reshape((1, -1))
            fused_input = np.hstack([visual_observation, observation])
            return fused_input, reward, done, self.episode_measurements, rv_visual_observation
        elif self.config["input_type"] == "wp":
            return obs['observation'], reward, done, self.episode_measurements
        elif self.config["input_type"] in ['wp_noise', 'wp_constant', 'wp_obs_dist', 'wp_obs_bool', 'wp_obs_bool_noise', 'wp_ldist_goal',
                                           'wp_speed', 'wp_speed_goal','wp_speed_steer_goal', 'wp_speed_steer_goal_obs_bool',
                                           'wp_obs_bool_speed_steer_goal_light', 'wp_obs_info_speed_steer_ldist_goal_light',
                                           'wp_obs_info_speed_steer_ldist_goal', 'wp_obs_info_speed_steer_ldist_light',
                                           'wp_angles_obs_info_speed_steer_ldist_light', 'wp_vecs_obs_info_speed_steer_ldist_light',
                                            'wp_angles_vecs_obs_info_speed_steer_ldist_light']:
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
            return True, 0, norm_target

        if norm_target > max_distance:
            return False, -1, norm_target

        fwd = current_transform.get_forward_vector()
        forward_vector = np.array([fwd.x, fwd.y])
        d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

        return d_angle < 90.0, d_angle, norm_target

    def _update_env_obs(self, front_rgb_image=None):
        if not self.config['disable_obstacle_info']:
            self._update_obs_detector()

        if not self.config['disable_traffic_light']:
            if front_rgb_image is None:
                self._update_traffic_light_states()
            else:
                self._update_traffic_light_states_nonprivilege(front_rgb_image)

            if self.config['verbose']:
                print(self.episode_measurements['dist_to_light'],
                    self.episode_measurements['nearest_traffic_actor_id'],
                    self.episode_measurements['nearest_traffic_actor_state'],
                    self.episode_measurements['initial_dist_to_red_light'],
                    self.episode_measurements['red_light_dist'])

    def _update_obs_detector(self):
        self.episode_measurements['obstacle_visible'] = False
        self.episode_measurements['obstacle_orientation'] = -1

        min_obs_distance = 100000000
        found_obstacle = False
        for target_vehicle in self.actor_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self.vehicle_actor.id or "vehicle" not in target_vehicle.type_id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            d_bool, d_angle, distance = self.is_within_distance_ahead(target_vehicle.get_transform(),
                                        self.vehicle_actor.get_transform(),
                                        self.config['vehicle_proximity_threshold'])

            if not d_bool:
                continue
            else:
                if not check_if_vehicle_in_same_lane(self.vehicle_actor, target_vehicle, self.next_waypoints, self._map):
                    continue

                found_obstacle = True
                self.episode_measurements['obstacle_visible'] = True
                self.episode_measurements['obstacle_orientation'] = d_angle

                if distance < min_obs_distance:
                    self.episode_measurements['obstacle_dist'] = distance
                    self.episode_measurements['obstacle_speed'] = self.get_speed_from_velocity(target_vehicle.get_velocity())
                    min_obs_distance = distance

        if not found_obstacle:
            self.episode_measurements['obstacle_dist'] = -1
            self.episode_measurements['obstacle_speed'] = -1

    def _update_traffic_light_states(self):
        # TODO: Pass correct target waypoint to find_nearest_traffic_light() for US style traffic.
        traffic_actor, dist, traffic_light_orientation = self.vehicle_agent.find_nearest_traffic_light(self.traffic_actors)
        if traffic_light_orientation is not None:
            self.episode_measurements['traffic_light_orientation'] = traffic_light_orientation
        else:
            self.episode_measurements['traffic_light_orientation'] = -1

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



    def _update_traffic_light_states_nonprivilege(self, front_rgb_image):
        front_rgb_image = front_rgb_image[:, :, ::-1].copy() # RGB -> GBR
        res = self.traffic_light_detector(front_rgb_image)
        print(res, flush=True)
        traffic_actor, dist, traffic_light_orientation = self.vehicle_agent.find_nearest_traffic_light(self.traffic_actors)

        if len(res['instances']) == 0: # no lights
            pass
        # elif res['instances'].scores[0].item() > .5:
        else:
            area = res['instances'].pred_boxes.area().tolist()
            color = res['instances'].pred_classes.tolist() # 0: Green, 1: Red
            score = res['instances'].scores.tolist()
            num_ins = len(res['instances'])
            print('cls:', color)
            print('score:', score)
            for _cls, _score, _area in zip(color, score, area):
                if _cls == 1:
            # found_red = color.index(1)
            # if color == 1: # and score > .667:
                    dist_pred = self.dist_interpolator(area)
                    print('detector Red, dist: {:.4f}, score: {:.4f}, num_ins: {}'.format(dist_pred, _score, num_ins), flush=True)

        if traffic_light_orientation is not None:
            self.episode_measurements['traffic_light_orientation'] = traffic_light_orientation
        else:
            self.episode_measurements['traffic_light_orientation'] = -1

        if traffic_actor is not None:
            if traffic_actor.state == carla.TrafficLightState.Red:
                print('privilege Red, dist: {:.4f}'.format(dist), flush=True)
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


    def _set_updated_scenario(self, unseen=False, town="Town01", index=0):
        if self.config["scenarios"] == "straight":
            source_idx, destination_idx = scenarios.get_straight_path_updated(unseen, town, index)
            self.config["num_episodes"] = 25
        elif self.config["scenarios"] == "curved":
            source_idx, destination_idx = scenarios.get_curved_path_updated(unseen, town, index)
            self.config["num_episodes"] = 25
        elif self.config["scenarios"] == "navigation" or self.config["scenarios"] == "dynamic_navigation":
            source_idx, destination_idx = scenarios.get_navigation_path_updated(unseen, town, index)
            self.config["num_episodes"] = 25
        else:
            raise ValueError("Scenarios Config not set!")

        self.source_transform = self.spawn_points[source_idx]
        self.destination_transform = self.spawn_points[destination_idx]

    def _set_scenario(self, unseen=False, town="Town01", index=0):
        if self.config["scenarios"] == "straight":
            # self.source_transform, self.destination_transform = scenarios.get_fixed_long_straight_path_Town01()
            self.source_transform, self.destination_transform = scenarios.get_straight_path(unseen, town, index)
            self.config["num_episodes"] = 25
        elif self.config["scenarios"] == "long_straight":
            self.source_transform, self.destination_transform = scenarios.get_long_straight_path(unseen, town)
            self.config["num_episodes"] = 2
        elif self.config["scenarios"] == "long_straight_junction":
            self.source_transform, self.destination_transform = scenarios.get_long_straight_junction_path(unseen, town, index)
            self.config["num_episodes"] = 3
        elif self.config["scenarios"] == "straight_dynamic":
            self.source_transform, self.destination_transform = scenarios.get_straight_dynamic_path(unseen, town)
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
            self.source_transform, self.destination_transform = scenarios.get_t_junction_path(unseen, town, index)
        elif self.config["scenarios"] == "curved":
            # self.source_transform, self.destination_transform = scenarios.get_fixed_long_curved_path_Town01()
            self.source_transform, self.destination_transform = scenarios.get_curved_path(unseen, town, index)
            self.config["num_episodes"] = 25
        elif self.config["scenarios"] == "navigation" or self.config["scenarios"] == "dynamic_navigation":
            self.source_transform, self.destination_transform = scenarios.get_navigation_path(unseen, town, index)
            self.config["num_episodes"] = 25
        elif self.config["scenarios"] == "no_crash_empty" or self.config["scenarios"] == "no_crash_regular" or self.config["scenarios"] == "no_crash_dense":
            source_idx, destination_idx = scenarios.get_no_crash_path(unseen, town, index)
            self.source_transform = self.spawn_points[source_idx]
            self.destination_transform = self.spawn_points[destination_idx]
            self.config["num_episodes"] = 25
        elif self.config["scenarios"] == "challenge_test_scenario":
            route = scenarios.get_test_route()

            self.scenario_route = convert_route_from_GPS_world(route, self._map)
            self.source_transform = self.scenario_route[0]
            self.destination_transform = self.scenario_route[-1]
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
        elif self.config["action_type"] == "discrete":
            # Discrete actions
            # No need to clip actions in case of discrete state-space
            # since it is chosen to be in range.
            discrete_actions = DISCRETE_ACTIONS[int(action)]
            target_speed, steer = float(discrete_actions[0]), float(discrete_actions[1])
            current_speed = self.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.config["enable_brake"])
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config["action_type"] == "control":
            target_speed = -1
            self.episode_measurements["target_speed"] = target_speed
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
        if not self.config['test_comparison']:
            return self._reset(unseen, index)
        else:
            return self._reset_test_comparison(unseen, index)

    def destroy_all_existing_actors(self):
        # Delete all existing actors
        if self.config['test_comparison']:
            self.actor_list = self.actor_list + self.collision_sensor_list + [self.camera_actor]
        for _ in range(len(self.actor_list)):
            try:
                actor = self.actor_list.pop()
                actor.destroy()
            except Exception as e:
                print("Error during destroying actor {0}:{1}: {2}".format(actor.type_id, actor.id,traceback.format_exc()))

    def clear_episode_measurements(self):

        # Below logic is to avoid clearing of following measurements,
        # when env reset is called automatically in DummyVec env.
        # These are used in testing logic, hence their values are required.
        for key, val in self.episode_measurements.items():
            if key in ['termination_state_code', 'termination_state']:
                continue

            self.episode_measurements[key] = 0

    def _reset_test_comparison(self, unseen=False, index=0):
        self.clear_episode_measurements()

        self.num_steps = 0 # Episode level step count
        self.total_reward = 0 # Episode level total reward
        self.prev_measurement = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None
        self.unseen = unseen

        # Destroy
        self.destroy_all_existing_actors()
        self.collision_sensor_list.clear()
        self.vehicle_agent_list.clear()
        self.control_list.clear()
        self.goal_destination_list.clear()

        self.camera_queue.queue.clear()

        try:
            vehicle_bp = self.blueprint_library.find(self.config['vehicle_type'])
            # vehicle_bp = self.blueprint_library.find(random.choice(self.config['vehicle_types']))
        except Exception as e:
            print("Error during vehicle creation: {}".format(traceback.format_exc()))

        if self.config['city_name'] == 'Town01':
            source_spawn_points_inds = np.random.permutation(255)
        elif self.config['city_name'] == 'Town02':
            source_spawn_points_inds = np.random.permutation(101)

        source_spawn_points = [self.spawn_points[ind] for ind in source_spawn_points_inds]

        count = self.config["num_npc"]
        for spawn_point in source_spawn_points:
            if self.try_spawn_random_vehicle_at(self.vehicle_blueprints, spawn_point):
                count -= 1
            if count <= 0:
                break

        # TODO: Generalize this code to attach 'n' different sensors to the vehicle
        # Attach a sensor to the vehicle
        sensor = self.config['sensors'][0]

        camera = self.blueprint_library.find(sensor)
        camera.set_attribute('image_size_x', self.config['sensor_x_res'])
        camera.set_attribute('image_size_y', self.config['sensor_y_res'])
        camera.set_attribute('sensor_tick', self.config['sensor_tick'])
        camera.set_attribute('fov', '90')

        camera_transform = carla.Transform(carla.Location(x=13.0, z=18.0), carla.Rotation(pitch=270.0))
        self.camera_actor = self._world.spawn_actor(camera, camera_transform, attach_to=self.actor_list[0])
        # self.actor_list.append(self.camera_actor)

        self.camera_actor.listen(self.camera_queue.put)

        # Ticking for 15 frames to handle car initialization in air
        for _ in range(15):
            world_frame = self._world.tick()

        for ind, vehicle in enumerate(self.actor_list):
            agent = BasicAgent(vehicle)
            self.vehicle_agent_list.append(agent)
            random_waypoint = random.choice(self.spawn_points)
            self.goal_destination_list[ind] = random_waypoint
            agent.set_destination((random_waypoint.location.x,
                                   random_waypoint.location.y,
                                   random_waypoint.location.z))

        self.episode_measurements['num_collisions'] = sum([sensor.num_collisions for sensor in self.collision_sensor_list])
        self.episode_measurements['distance_to_goal'] = [vehicle.get_location().distance(self.goal_destination_list[ind].location) for ind, vehicle in enumerate(self.actor_list)]
        self.episode_measurements['speed'] = [self.get_speed_from_velocity(vehicle.get_velocity()) for vehicle in self.actor_list]

        # obs = {}
        # image = self._read_data(self.camera_queue, world_frame)
        # obs['image'] = image

        self.prev_measurement = copy.deepcopy(self.episode_measurements)

        return None


    # @profile
    def _reset(self, unseen=False, index=0, expert_agent=False):
        self.clear_episode_measurements()

        self.num_steps = 0 # Episode level step count
        self.total_reward = 0 # Episode level total reward
        self.prev_measurement = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None
        self.unseen = unseen

        if self.config["scenarios"] in ["long_straight", "long_straight_junction"] and not self.unseen:
            # Way to test two scenarios with and without dynamic actors
            # in training run in long_straight scenario
            self.index = (self.index + 1) % self.config["num_episodes"]
        else:
            self.index = index
        self.expert_agent = expert_agent

        # Destroy
        self.destroy_all_existing_actors()

        self.camera_queue.queue.clear()
        self.rgb_camera_queue.queue.clear()
        self.front_camera_queue.queue.clear()
        self.rv_camera_queue.queue.clear()
        self.stacked_observation_queue.queue.clear()

        try:
            vehicle_bp = self.blueprint_library.find(self.config['vehicle_type'])
            # vehicle_bp = self.blueprint_library.find(random.choice(self.config['vehicle_types']))
        except Exception as e:
            print("Error during vehicle creation: {}".format(traceback.format_exc()))

        # Set source and destination based on scenario
        # Currently scenarios are defined only for Town01
        if self.config["use_scenarios"] and (self.config["city_name"] == "Town01" or self.config["city_name"] == "Town02"):
            if self.config["updated_scenarios"]:
                self._set_updated_scenario(unseen=unseen, index=self.index, town=self.config["city_name"])
            else:
                self._set_scenario(unseen=unseen, index=self.index, town=self.config["city_name"])
        else:
            self.source_transform, self.destination_transform = random.choice(self.spawn_points), random.choice(self.spawn_points)


        # Spawning vehicle actor with retry logic as it fails to spawn sometimes
        self.vehicle_actor = None
        NUM_RETRIES = 5
        for _ in range(NUM_RETRIES):
            self.vehicle_actor = self._world.try_spawn_actor(vehicle_bp, self.source_transform)
            if self.vehicle_actor is not None:
                break
            else:
                print("Unable to spawn vehicle actor at {0}, {1}.".format(self.source_transform.location.x, self.source_transform.location.y))
                print("Number of existing actors, {0}".format(len(self.actor_list)))
                self.destroy_all_existing_actors()
                time.sleep(120)

        if self.vehicle_actor is not None:
            self.actor_list.append(self.vehicle_actor)
            self.location = self.vehicle_actor.get_location()
        else:
            raise Exception("Failed in spawning vehicle actor.")

        # Agent uses proximity_threshold to detect traffic lights.
        # Hence we use traffic_light_proximity_threshold while creating an Agent.
        self.vehicle_agent = Agent(self.vehicle_actor, self.config['traffic_light_proximity_threshold'])

        spawn_vehicles = False
        if self.config["scenarios"] == "long_straight":

            if self.config["num_npc"] > 0 and (self.index % self.config["num_episodes"] == 1):
                spawn_vehicles = True

        elif self.config["scenarios"] == "long_straight_junction":

            if self.config["num_npc"] > 0 and (self.index % self.config["num_episodes"] == 0):
                spawn_vehicles = True

        else:
            if self.config["num_npc"] > 0:
                spawn_vehicles = True

        if spawn_vehicles:
            if self.config["sample_npc"]:
                self.spawn_npc(np.random.randint(low=self.config["num_npc_lower_threshold"],
                    high=self.config["num_npc_upper_threshold"]), unseen)
            else:
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
        camera.set_attribute('fov', '120')
        # camera.set_attribute('fov', '90')

        # front_seg_cam = self.blueprint_library.find(sensor)
        # front_seg_cam.set_attribute('image_size_x', self.config['sensor_x_res'])
        # front_seg_cam.set_attribute('image_size_y', self.config['sensor_y_res'])
        # front_seg_cam.set_attribute('sensor_tick', self.config['sensor_tick'])
        # # camera.set_attribute('fov', '120')
        # front_seg_cam.set_attribute('fov', '90')

        # Orientation for top-down (BEV) facing camera
        bev_camera_transform = carla.Transform(carla.Location(x=13.0, z=18.0), carla.Rotation(pitch=270.0))

        self.camera_actor = self._world.spawn_actor(camera, bev_camera_transform, attach_to=self.vehicle_actor)
        self.actor_list.append(self.camera_actor)

        self.camera_actor.listen(self.camera_queue.put)

        rgb_camera = self.blueprint_library.find(self.config['sensors'][0])
        rgb_camera.set_attribute('image_size_x', '512')
        rgb_camera.set_attribute('image_size_y', '512')
        rgb_camera.set_attribute('sensor_tick', self.config['sensor_tick'])
        rgb_camera.set_attribute('fov', '90')
        # Additional front-view/range-view(rv) camera
        rv_camera = self.blueprint_library.find(sensor)
        rv_camera.set_attribute('image_size_x', self.config['sensor_x_res'])
        rv_camera.set_attribute('image_size_y', self.config['sensor_y_res'])
        rv_camera.set_attribute('sensor_tick', self.config['sensor_tick'])
        # camera.set_attribute('fov', '120')
        rv_camera.set_attribute('fov', '90')

        # Orientation for forward-facing camera
        rv_camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0.0))

        self.rv_camera_actor = self._world.spawn_actor(rv_camera, rv_camera_transform, attach_to=self.vehicle_actor)
        self.actor_list.append(self.rv_camera_actor)

        self.rv_camera_actor.listen(self.rv_camera_queue.put)

        # # rgb_camera_transform = carla.Transform(carla.Location(x=5.0, z=20.0), carla.Rotation(pitch=270.0))
        # rgb_camera_transform = carla.Transform(carla.Location(x=13.0, z=18.0), carla.Rotation(pitch=270.0))
        rgb_camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0.0))
        self.rgb_camera_actor = self._world.spawn_actor(rgb_camera, rgb_camera_transform, attach_to=self.vehicle_actor)
        self.actor_list.append(self.rgb_camera_actor)

        self.rgb_camera_actor.listen(self.rgb_camera_queue.put)


        front_camera = self.blueprint_library.find(self.config['sensors'][1])
        front_camera.set_attribute('image_size_x', '512')
        front_camera.set_attribute('image_size_y', '512')
        front_camera.set_attribute('sensor_tick', self.config['sensor_tick'])
        # front_camera.set_attribute('fov', '120')
        front_camera.set_attribute('fov', '90')

        # # front_camera_transform = carla.Transform(carla.Location(x=5.0, z=20.0), carla.Rotation(pitch=270.0))
        # front_camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=8.0))
        self.front_camera_actor = self._world.spawn_actor(front_camera, rgb_camera_transform, attach_to=self.vehicle_actor)
        self.actor_list.append(self.front_camera_actor)

        self.front_camera_actor.listen(self.front_camera_queue.put)

        self.collision_sensor = sensors.CollisionSensor(self.vehicle_actor)
        self.actor_list.append(self.collision_sensor.sensor)

        if self.config["enable_lane_invasion_sensor"]:
            self.lane_invasion_sensor = sensors.LaneInvasionSensor(self.vehicle_actor)
            self.actor_list.append(self.lane_invasion_sensor.sensor)

        # Set state variables for reward calculation
        self.episode_measurements['num_collisions'] = self.collision_sensor.num_collisions
        self.episode_measurements['collision_actor_id'] = self.collision_sensor.actor_id
        self.episode_measurements['collision_actor_type'] = self.collision_sensor.actor_type
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

        # image = self._read_data(self.camera_queue, world_frame)
        image = rgb_image = self._read_data(self.rgb_camera_queue, world_frame)
        front_image = self._read_data(self.front_camera_queue, world_frame)

        # collect data
        # seg_img = util.convert_to_rgb(front_image)


        rv_image = self._read_data(self.rv_camera_queue, world_frame)

        self.global_planner = planner.GlobalPlanner()

        if self.config["use_route_to_plan"]:
            self.trace_route = []
            for idx in range(len(self.scenario_route) - 1):
                source = self.scenario_route[idx]
                destination = self.scenario_route[idx+1]
                trace_route = self.global_planner._trace_route(self._map,
                                source, destination)
                self.trace_route.extend(trace_route)
        else:
            self.trace_route  = self.global_planner._trace_route(self._map,
                                self.source_transform, self.destination_transform)
        self.global_planner.set_global_plan(self.trace_route)

        if self.expert_agent:
            self.vehicle_agent = BasicAgent(self.vehicle_actor, proximity_threshold=self.config['traffic_light_proximity_threshold'])
            self.vehicle_agent.set_destination((self.destination_transform.location.x,
                                            self.destination_transform.location.y,
                                            self.destination_transform.location.z))
        else:
            # Agent uses proximity_threshold to detect traffic lights.
            # Hence we use traffic_light_proximity_threshold while creating an Agent.
            self.vehicle_agent = Agent(self.vehicle_actor, self.config['traffic_light_proximity_threshold'])

        # if self.config["algo"] == "AE":
        #     next_orientation, self.dist_to_trajectory, distance_to_goal_trajec = 0, 0
        # else:
        next_orientation, self.dist_to_trajectory, distance_to_goal_trajec, self.next_waypoints, self.next_wp_angles, self.next_wp_vectors = self.global_planner.get_next_orientation_new(self.vehicle_actor.get_transform())

        self.episode_measurements['next_orientation'] = next_orientation
        # print('self.config["algo"]', self.config["algo"], flush=True)
        # distance_to_goal_trajec = 0
        self.episode_measurements['distance_to_goal_trajec'] = distance_to_goal_trajec
        if self.unseen:
            self.total_distance += distance_to_goal_trajec
        self.episode_measurements['dist_to_trajectory'] = self.dist_to_trajectory

        # Update obstacle distance measurements
        self._update_env_obs(front_rgb_image=rgb_image)
        #self._update_env_obs()

        if self.config["scenarios"] == "straight_dynamic":
            self._update_straight_dynamic_obs()

        obs['image'] = obs['semantic_image'] = front_image
        obs['rgb_image'] = rgb_image
        obs['rv_image'] = rv_image

        visual_observation = None
        if self.config["input_type"] in ['vae', 'wp_vae', 'wp_vae_speed_steer_goal',
                                         'wp_vae_speed_steer_ldist_goal_light', 'wp_vae_obs_info_speed_steer_ldist_goal_light',
                                         'wp_cnn_obs_info_speed_steer_ldist_goal_light', 'wp_bev_rv_obs_info_speed_steer_ldist_goal_light']:

            if self.config["semantic"]:
                semantic_image = image[:,:,0]
                rv_semantic_image = rv_image[:,:,0]

                semantic_image = reduce_classes(semantic_image, binarized_image=self.config['binarized_image'])
                rv_semantic_image = reduce_classes(rv_semantic_image, binarized_image=self.config['binarized_image'])
                obs['semantic_image'] = semantic_image
                obs['rv_semantic_image'] = rv_semantic_image
                if not self.config['single_channel_image']:
                    if self.config['binarized_image']:
                        semantic_image = convert_to_one_hot(semantic_image, num_classes=2)
                        rv_semantic_image = convert_to_one_hot(rv_semantic_image, num_classes=2)
                    else:
                        semantic_image = convert_to_one_hot(semantic_image, num_classes=5)
                        rv_semantic_image = convert_to_one_hot(rv_semantic_image, num_classes=5)

                for _ in range(self.rv_stack_size):
                    self._add_to_stacked_queue(self.rv_stacked_observation_queue, rv_semantic_image)

                for _ in range(self.config['frame_stack_size']):
                    # Update stacked frames and measurements
                    self._add_to_stacked_queue(self.stacked_observation_queue, semantic_image)
            else:
                for _ in range(self.rv_stack_size):
                    self._add_to_stacked_queue(self.rv_stacked_observation_queue, rv_image)

                for _ in range(self.config['frame_stack_size']):
                    # Update stacked frames and measurements
                    self._add_to_stacked_queue(self.stacked_observation_queue, image)

            if not self.config['single_channel_image']:
                stacked_observation = np.concatenate(list(self.stacked_observation_queue.queue), axis=2)
                rv_stacked_observation = np.concatenate(list(self.rv_stacked_observation_queue.queue), axis=2)
            else:
                stacked_observation = np.stack(list(self.stacked_observation_queue.queue), axis=2)
                rv_stacked_observation = np.stack(list(self.rv_stacked_observation_queue.queue), axis=2)

            if 'vae' in self.config["input_type"]:
                visual_observation = self.vae_observation(stacked_observation)
                rv_visual_observation = self.vae_observation(rv_stacked_observation)
                visual_observation = visual_observation / self.config["vae_encoding_norm_factor"]
                rv_visual_observation = rv_visual_observation / self.config["vae_encoding_norm_factor"]
            else:
                visual_observation = stacked_observation
                rv_visual_observation = rv_stacked_observation

        if self.config["input_type"] == "ae_train":
            semantic_image = image[:,:,0]
            rv_semantic_image = rv_image[:,:,0]

            obs['semantic_image'] = semantic_image
            obs['rv_semantic_image'] = rv_semantic_image

        obs['speed'] = np.expand_dims(np.array([self.episode_measurements['speed']]), axis=0) # * 3.6 / 30
        obs['dist_to_target'] = np.array([self.episode_measurements['distance_to_goal']])

        # Update observation input in obs dictionary
        self.create_observations(obs)

        self.prev_measurement = copy.deepcopy(self.episode_measurements)

        self.target_speeds_array = []
        self.speeds_array = []
        self.throttles_array = []
        self.obstacle_speed_array = []
        self.dist_to_trajectory_array = []
        self.steers_array = []
        self.brakes_array = []
        self.wp_orientation_array = []
        self.input_steer_array = []
        self.obstacle_dist_array = []
        self.step_reward_array = []
        self.collision_reward_array = []
        self.dist_to_trajectory_reward_array = []
        self.speed_reward_array = []
        self.dist_to_target_array = []
        self.red_light_dist_array = []

        if self.config["input_type"] == 'vae':
            return visual_observation

        elif self.config["input_type"] in ['wp_vae', 'wp_vae_speed_steer_goal', 'wp_vae_speed_steer_ldist_goal_light', 'wp_vae_obs_info_speed_steer_ldist_goal_light']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            fused_input = np.hstack([visual_observation, observation])
            return fused_input

        elif self.config["input_type"] in ['wp_cnn_obs_info_speed_steer_ldist_goal_light']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            visual_observation = visual_observation.reshape((1, -1))
            fused_input = np.hstack([visual_observation, observation])
            return fused_input

        elif self.config["input_type"] in ['wp_bev_rv_obs_info_speed_steer_ldist_goal_light']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            visual_observation = visual_observation.reshape((1, -1))
            fused_input = np.hstack([visual_observation, observation])
            return fused_input, rv_visual_observation

        elif self.config["input_type"] == "wp":
            return obs['observation']

        elif self.config["input_type"] in ['wp_noise', 'wp_constant', 'wp_obs_dist', 'wp_obs_bool', 'wp_obs_bool_noise', 'wp_ldist_goal',
                                           'wp_speed', 'wp_speed_goal','wp_speed_steer_goal', 'wp_speed_steer_goal_obs_bool',
                                           'wp_obs_bool_speed_steer_goal_light', 'wp_obs_info_speed_steer_ldist_goal_light',
                                           'wp_obs_info_speed_steer_ldist_goal', 'wp_obs_info_speed_steer_ldist_light',
                                           'wp_angles_obs_info_speed_steer_ldist_light', 'wp_vecs_obs_info_speed_steer_ldist_light',
                                           'wp_angles_vecs_obs_info_speed_steer_ldist_light']:

            observation = np.expand_dims(obs['observation'], axis = 0)
            return observation
        else:
            return obs

    def get_wp_obs_input(self):
        '''
        Create wp angles input
        '''
        num_wp = 5
        wp_angles_array = None
        wp_vectors_array = None

        n = len(self.next_wp_angles)
        if n == 0:
            print("No next waypoints found. Giving zero as input.")
            wp_angles_array = np.zeros(num_wp)
            wp_vectors_array = np.zeros(2 * num_wp)

        elif n == num_wp:
            wp_angles_array = np.array(self.next_wp_angles)
            wp_vectors_array = np.array(self.next_wp_vectors)

        elif n < num_wp:
            # Fill using last entry
            last_angle = self.next_wp_angles[-1]
            last_vec = self.next_wp_vectors[-1]

            for _ in range(num_wp-n):
                self.next_wp_angles.append(last_angle)
                self.next_wp_vectors.append(last_vec)
            wp_angles_array = np.array(self.next_wp_angles)
            wp_vectors_array = np.array(self.next_wp_vectors)
        else:
            print("Error: More than {0} waypoints returned from planner.".format(num_wp))
            print("Taking required number of entries.")
            wp_angles_array = np.array(self.next_wp_angles[:num_wp])
            wp_vectors_array = np.array(self.next_wp_vectors[:num_wp])

        wp_vectors_array = wp_vectors_array.reshape(-1)

        return wp_angles_array, wp_vectors_array

    def try_spawn_random_vehicle_at(self, blueprints, transform):
        # blueprint = random.choice(blueprints)

        # To spawn same type of vehicle
        blueprint = blueprints[0]
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # TODO: uncomment below to enable autopilot
        if not self.config["scenarios"] == "straight_dynamic" and not self.config['test_comparison']:
            blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self._world.try_spawn_actor(blueprint, transform)
        tm_port = self.tm.get_port()
        if vehicle is not None:
            self.actor_list.append(vehicle)
            # TODO: uncomment below to enable autopilot
            if not self.config["scenarios"] == "straight_dynamic" and not self.config['test_comparison']:
                vehicle.set_autopilot(True, tm_port)

            if self.config['test_comparison']:
                self.collision_sensor_list.append(sensors.CollisionSensor(vehicle))

            if self.config["verbose"]:
                print('spawned %r at %s' % (vehicle.type_id, transform.location.x))
            return True
        return False

    def spawn_npc(self, number_of_vehicles, unseen):

        # TODO: remove hard coded logic
        if self.config["scenarios"] == "straight_dynamic":
            # vehicle spawn_points corresponding to 84, 40
            spawn_points = [Transform(Location(x=-2.4200193881988525, y=187.97000122070312, z=1.32), Rotation(yaw=89.9996109008789)),
                        Transform(Location(x=1.5599803924560547, y=187.9700164794922, z=1.32), Rotation(yaw=-90.00040435791016))]

            # vehicle spawn_points corresponding to 96, 140
            # spawn_points = [Transform(Location(x=88.61997985839844, y=249.42999267578125, z=1.32), Rotation(yaw=90.00004577636719)),
            # Transform(Location(x=92.10997772216797, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672))]
        elif self.config["scenarios"] == "crowded":
            spawn_points = scenarios.get_crowded_npcs(number_of_vehicles)
            print('CROWDED SPAWNING: ', spawn_points)
        elif self.config["scenarios"] in ["long_straight", "long_straight_junction"]:
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
                    random.shuffle(spawn_points)

        elif self.config["scenarios"] == "straight_crowded":
            spawn_points = scenarios.get_straight_crowded_npcs(number_of_vehicles)
            print('STRAIGHT CROWDED SPAWNING: ', spawn_points)
        elif self.config["scenarios"] == "town3":
            spawn_points = scenarios.get_curved_town03_npcs(number_of_vehicles)
            print('TOWN 3 SPAWNING: ', spawn_points)

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

        if self.config["scenarios"] in ["long_straight", "long_straight_junction"]:
            for spawn_point in spawn_points_1:
                self.try_spawn_random_vehicle_at(self.vehicle_blueprints, spawn_point)

        count = number_of_vehicles
        for spawn_point in spawn_points:
            if self.try_spawn_random_vehicle_at(self.vehicle_blueprints, spawn_point):
                count -= 1
            if count <= 0:
                break

    def get_speed_from_velocity(self, velocity):
        speed = np.sqrt(velocity.x ** 2 + velocity.y **2 + velocity.z **2)
        return speed

    def _read_data(self, camera_queue, world_frame, timeout=240.0):

        cam_image = self._read_camera_data(camera_queue, world_frame, timeout)
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
        # array = cv2.resize(array, (64, 64), interpolation=cv2.INTER_AREA)

        return array

    def _read_camera_data(self, camera_queue, world_frame, timeout):

        data  = self._retrieve_data(camera_queue, timeout, world_frame)
        return data

    # @profile
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
        offlane = self.episode_measurements["offlane_steps"] > self.config["max_offlane_steps"] # Unused
        static = self.episode_measurements["static_steps"] > self.config["max_static_steps"]
        collision = self.episode_measurements["is_collision"] = False
        runover_light = self.episode_measurements["runover_light"]
        maxStepsTaken = self.episode_measurements["num_steps"] > self.config['max_steps']
        offlane = False

        # Conditions to check there is obstacle or red light ahead for last 2 timesteps
        obstacle_ahead = self.episode_measurements['obstacle_dist'] != -1 and self.prev_measurement['obstacle_dist'] != -1
        red_light = self.episode_measurements['red_light_dist'] != -1 and self.prev_measurement['red_light_dist'] != -1

        if not self.config["enable_static"]:
            static = False
        if self.config["disable_collision"]:
            collision = False
        if self.config["disable_traffic_light"] or not self.config["terminate_on_light"]:
            runover_light = False
        if self.config["enable_lane_invasion_sensor"] and self.config["enable_lane_invasion_collision"]:
            offlane = self.episode_measurements['num_laneintersections'] > 0

        # Do not want to terminate on reaching goal
        # in case of VAE training
        if self.config["algo"] == "AE":
            success = False

        if success:
            termination_state = 'success'
            termination_state_code = 0
        elif collision:
            if 'obs_collision' in self.episode_measurements and self.episode_measurements['obs_collision']:
                termination_state = 'obs_collision'
                termination_state_code = 1
            elif self.config["enable_lane_invasion_sensor"] and self.episode_measurements["out_of_road"]:
                termination_state = 'out_of_road'
                termination_state_code = 2
            elif self.config["enable_lane_invasion_sensor"] and self.episode_measurements['lane_change']:
                termination_state = 'lane_invasion'
                termination_state_code = 3
            else:
                termination_state = 'unexpected_collision'
                termination_state_code = 4
        elif runover_light:
            termination_state = 'runover_light'
            termination_state_code = 5
        elif offlane:
            termination_state = 'offlane'
            termination_state_code = 6
        elif static:
            termination_state = 'static'
            termination_state_code = 7
        elif maxStepsTaken:
            if obstacle_ahead:
                termination_state = 'max_steps_obstacle'
                termination_state_code = 8
            elif red_light:
                termination_state = 'max_steps_light'
                termination_state_code = 9
            else:
                termination_state = 'max_steps'
                termination_state_code = 10
        else:
            termination_state = 'none'
            termination_state_code = 11

        if self.config["verbose"]:
            print("Termination State: {}".format(termination_state))

        self.episode_measurements['termination_state'] = termination_state
        self.episode_measurements['termination_state_code'] = termination_state_code

        done = success or collision or runover_light or offlane or static or maxStepsTaken
        return done

    def printInfo(self):
        print("Vehicle transform:{0}".format(self.vehicle_actor.get_transform()))
        print("Vehicle velocity:{0}".format(self.vehicle_actor.get_velocity()))

    def close(self):

        try:
            self.destroy_all_existing_actors()

            if not self.CarlaServer is None:
                self.CarlaServer.close()

        except Exception as e:
                print("********** Exception in closing env **********")
                print(e)
                print(traceback.format_exc())

    def __del__(self):
        self.close()

# @profile
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
                red_light_dist_array,
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
    red_light_dist_array = np.array(red_light_dist_array)

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
    # axs[2, 1].set_ylabel('Break')
    axs[2, 1].set_ylabel('Orientation')

    axs[3, 1].plot(observations, collision_reward_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[3, 1].set_xlabel('Timesteps')
    axs[3, 1].set_ylabel('collision_reward')

    axs[4, 1].plot(observations, red_light_dist_array, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[4, 1].set_xlabel('Timesteps')
    axs[4, 1].set_ylabel('Dist to red light')

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

# Used for plotting debugging related plots
# def plot_episode_info(path,
#                 target_speeds_array,
#                 speeds_array,
#                 throttles_array,
#                 steers_array,
#                 brakes_array,
#                 dist_to_target_array,
#                 step_reward_array,
#                 collision_reward_array,
#                 dist_to_trajectory_reward_array,
#                 speed_reward_array,
#                 episode_num):

#     if not os.path.exists(path):
#         os.makedirs(path)


#     target_speeds_array = np.array(target_speeds_array)
#     speeds_array = np.array(speeds_array)
#     throttles_array = np.array(throttles_array)
#     steers_array = np.array(steers_array)
#     brakes_array = np.array(brakes_array)
#     step_reward_array = np.array(step_reward_array)
#     collision_reward_array = np.array(collision_reward_array)
#     dist_to_target_array = np.array(dist_to_target_array)
#     dist_to_trajectory_reward_array = np.array(dist_to_trajectory_reward_array)
#     speed_reward_array = np.array(speed_reward_array)

#     target_speeds_array = target_speeds_array[-50:]
#     speeds_array = speeds_array[-50:]

#     observations = np.arange(len(target_speeds_array))
#     fig, axs = plt.subplots(1, 1, figsize=(12, 12))
#     fig.suptitle('Episode info plots for episode idx {} '.format(episode_num))

#     axs.plot(observations, target_speeds_array, linestyle='-', linewidth=2, markersize=8, label='target')
#     axs.set_xlabel('Timesteps')
#     axs.set_ylabel('Target and Actual Speed')

#     axs.plot(observations, speeds_array, linestyle='-', linewidth=2, markersize=8, label='actual')

#     axs.grid(True)

#     plt.grid(True)
#     plt.savefig(path + 'episode_info_{}.png'.format(episode_num))
#     plt.close()

if __name__ == "__main__":
    env = CarlaEnv()
    env.reset()
