""" Environment file wrapper for CARLA """

import gym
from gym.spaces import Box, Discrete, Tuple

from datetime import datetime
import os
import traceback
import random
import numpy as np
import math
import copy
import queue
import time
import matplotlib.pyplot as plt

import environment.carla_9_4.scenarios as scenarios
import environment.carla_9_4.server as server
import environment.carla_9_4.planner as planner
import environment.carla_9_4.controller as controller
import environment.carla_9_4.sensors as sensors
from environment.carla_9_4.reward import compute_reward
from environment.carla_9_4.dashcam import Visualizer


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
    def __init__(self, config, logger=None):
        self.config = config
        # self._update_config(config)
        self.CarlaServer = None
        self.episode_measurements = self.config['episode_measurements']
        self.episode_id = None
        self.vehicle_actor = None
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.log_dir = os.path.expanduser(self.config['log_dir'])
        ################################################################################
        # if 'num_agents' in self.config and self.config['algo'] == 'A2C':
        if self.config['verbose']: print('##### USE MULTI-AGENT #####', flush=True)
        self.ego_vehicle_list = [None] * self.config['num_agents']
        self.ego_agent_list = [None] * self.config['num_agents']
        # else:
        #     self.config['num_agents'] = 1
        self.curr_num_agents = 0
        self.world_frame = 0
        ################################################################################
        # Can pass in train/test weather as an array
        self.weather = None
        self.target_speed = self.config['target_speed']
        self.vehicle_collisions = 0
        self.static_collisions = 0
        self.total_collisions = 0
        self.total_distance = 0
        self.traffic_light_violations = 0
        self.args_longitudinal_dict = {
            'K_P': 0.1,
            'K_D': 0.0005,
            'K_I': 0.4,
            'dt': 1/10.0}
        self.actor_list = []

        # Queue for stacked frames and measurements
        self.rv_stack_size = 1

        self.logger = logger
        self.controller = controller.PIDLongitudinalController(
            K_P=self.args_longitudinal_dict['K_P'],
            K_D=self.args_longitudinal_dict['K_D'],
            K_I=self.args_longitudinal_dict['K_I'],
            dt=self.args_longitudinal_dict['dt'])

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

        # Create new client
        self.client =  self._spawn_client()
        print("server_version", self.client.get_server_version())

        # Commenting load_world, assuming default is set as Town01 in CARLA binary config
        # since sometimes, it causes timeout issues in the beginning
        # print(self.client.get_available_maps())
        self._world = self.client.load_world('/Game/Carla/Maps/' + self.config['city_name'])

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

        self._map = self._world.get_map()
        self.blueprint_library = self._world.get_blueprint_library()
        self.spawn_points = self._world.get_map().get_spawn_points()

        self.tm = self.client.get_trafficmanager(random.randint(10000, 60000))
        self.tm.set_synchronous_mode(True)

        if self.config["testing"]:
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in self.config['spawn_points_fixed_idx']]
        else:
            spawn_pt_idx = np.random.permutation(len(self.spawn_points))
            np.save(os.path.join(self.log_dir, "spawn_pt_order"), spawn_pt_idx)
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in spawn_pt_idx]

        # TODO: Verify the limits and bounds of observation spaces
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
            self.action_space = Discrete(len(self.config['discrete_actions']))
        elif self.config["action_type"] == 'control':
            # Discrete actions
            self.action_space = Discrete(len(self.config['discrete_actions']))

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
                                    shape=(1, 1608), dtype=np.float32) # Model used for Learning to Drive with Dynamic Actors (last layer dim = 64)steer_ldist_goal_light':
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

    def _update_config(self, config):
        for key, val in config.items():
            self.config[key] = val

    def _spawn_client(self, hostname='localhost', port_number=None):
        port_number = self.CarlaServer.server_port
        client = carla.Client(hostname, port_number)
        client.set_timeout(self.config["client_timeout_seconds"])
        return client


    def create_observations(self, agent, obs):
        obs['observation'] = np.array([agent.episode_measurements['next_orientation']])

        if self.config["input_type"] == 'wp_constant':
            obs['observation'] = np.array([agent.episode_measurements['next_orientation'], 0.0])

        elif self.config["input_type"] == 'wp_noise':
            obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.random.normal(0.0, 1.0, self.config["noise_dim"])))

        elif self.config["input_type"] == 'wp_obs_dist':
            obs_dist = agent.episode_measurements['obstacle_dist'] / self.config["obstacle_dist_norm"]
            obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obs_dist])))

        elif self.config["input_type"] == 'wp_obs_bool':
            obs_bool = agent.episode_measurements['obstacle_visible']
            obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obs_bool])))

        elif self.config["input_type"] == 'wp_ldist_goal':
            ldist = agent.episode_measurements['dist_to_trajectory']
            distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 500
            obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([ldist]), np.array([distance_to_goal_trajec])))

        elif self.config["input_type"] == 'wp_obs_bool_noise':
            obs_bool = agent.episode_measurements['obstacle_visible']
            obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obs_bool]), np.random.normal(0.0, 1.0, self.config["noise_dim"])))

        elif self.config["input_type"] == 'wp_speed':
            obs_speed = agent.episode_measurements['speed'] / 10
            obs['observation'] = np.concatenate((np.array(agent.episode_measurements['next_orientation']), np.array([obs_speed])))

        elif self.config["input_type"] == 'wp_speed_goal':
            obs_speed = agent.episode_measurements['speed'] / 10
            distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 100
            obs['observation'] = np.concatenate((np.array(agent.episode_measurements['next_orientation']), np.array([obs_speed]), np.array([distance_to_goal_trajec])))

        elif self.config["input_type"] == 'wp_speed_steer_goal':
            obs_speed = agent.episode_measurements['speed'] / 10
            distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 100
            steer = agent.episode_measurements['control_steer']
            obs['observation'] = np.concatenate((np.array(agent.episode_measurements['next_orientation']), np.array([obs_speed]), np.array([steer]), np.array([distance_to_goal_trajec])))

        elif self.config["input_type"] == 'wp_speed_steer_goal_obs_bool':
            obs_speed = agent.episode_measurements['speed'] / 10
            distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 100
            steer = agent.episode_measurements['control_steer']
            obs_bool = agent.episode_measurements['obstacle_visible']
            obs['observation'] = np.concatenate((np.array(agent.episode_measurements['next_orientation']), np.array([obs_speed]), np.array([steer]), np.array([distance_to_goal_trajec]), np.array([obs_bool])))

        elif self.config["input_type"] == 'wp_obs_bool_speed_steer_goal_light':

            speed = agent.episode_measurements['speed'] / 10
            obs_bool = agent.episode_measurements['obstacle_visible']
            steer = agent.episode_measurements['control_steer']
            distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 500
            light = agent.episode_measurements['red_light_dist']

            # normalization
            if light != -1:
                light /= self.config['traffic_light_proximity_threshold']
            else:
                light = self.config['default_obs_traffic_val']

            obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obs_bool]), np.array([speed]), np.array([steer]), np.array([distance_to_goal_trajec]), np.array([light])))

        elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_goal_light':

            speed = agent.episode_measurements['speed'] / 10
            obstacle_dist = agent.episode_measurements['obstacle_dist']
            obstacle_speed = agent.episode_measurements['obstacle_speed']
            steer = agent.episode_measurements['control_steer']
            ldist = agent.episode_measurements['dist_to_trajectory']
            distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 500
            light = agent.episode_measurements['red_light_dist']

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

            obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([distance_to_goal_trajec]), np.array([light])))

        elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_goal':

            speed = agent.episode_measurements['speed'] / 10
            obstacle_dist = agent.episode_measurements['obstacle_dist']
            obstacle_speed = agent.episode_measurements['obstacle_speed']
            steer = agent.episode_measurements['control_steer']
            ldist = agent.episode_measurements['dist_to_trajectory']
            distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 500

            # normalization

            if obstacle_dist != -1:
                obstacle_dist = obstacle_dist / self.config['vehicle_proximity_threshold']
            else:
                obstacle_dist = self.config['default_obs_traffic_val']

            if obstacle_speed != -1:
                obstacle_speed = obstacle_speed / 20
            else:
                obstacle_speed = self.config['default_obs_traffic_val']

            obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([distance_to_goal_trajec])))

        elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_light':

            speed = agent.episode_measurements['speed'] / 10
            obstacle_dist = agent.episode_measurements['obstacle_dist']
            obstacle_speed = agent.episode_measurements['obstacle_speed']
            steer = agent.episode_measurements['control_steer']
            ldist = agent.episode_measurements['dist_to_trajectory']
            light = agent.episode_measurements['red_light_dist']

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

            obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([light])))


        elif self.config["input_type"] == 'wp_vae_speed_steer_goal':
            speed = agent.episode_measurements['speed'] / 10
            steer = agent.episode_measurements['control_steer']
            distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 500
            obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([speed]), np.array([steer]), np.array([distance_to_goal_trajec])))

        elif self.config["input_type"] == 'wp_vae_speed_steer_ldist_goal_light':
            speed = agent.episode_measurements['speed'] / 10
            steer = agent.episode_measurements['control_steer']
            ldist = agent.episode_measurements['dist_to_trajectory']
            distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 500
            light = agent.episode_measurements['red_light_dist']

            # normalization
            if light != -1:
                light /= self.config['traffic_light_proximity_threshold']
            else:
                light = self.config['default_obs_traffic_val']

            obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([distance_to_goal_trajec]), np.array([light])))

        elif self.config["input_type"] in ['wp_vae_obs_info_speed_steer_ldist_goal_light', 'wp_cnn_obs_info_speed_steer_ldist_goal_light', 'wp_bev_rv_obs_info_speed_steer_ldist_goal_light']:
            speed = agent.episode_measurements['speed'] / 10
            obstacle_dist = agent.episode_measurements['obstacle_dist']
            obstacle_speed = agent.episode_measurements['obstacle_speed']
            steer = agent.episode_measurements['control_steer']
            ldist = agent.episode_measurements['dist_to_trajectory']
            distance_to_goal_trajec = agent.episode_measurements['distance_to_goal_trajec'] / 500
            light = agent.episode_measurements['red_light_dist']

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

            obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([distance_to_goal_trajec]), np.array([light])))



        elif self.config["input_type"] == 'wp_angles_obs_info_speed_steer_ldist_light':
            wp_angles_array, wp_vectors_array = self.get_wp_obs_input(agent)
            speed = agent.episode_measurements['speed'] / 10
            obstacle_dist = agent.episode_measurements['obstacle_dist']
            obstacle_speed = agent.episode_measurements['obstacle_speed']
            steer = agent.episode_measurements['control_steer']
            ldist = agent.dist_to_trajectory
            light = agent.episode_measurements['red_light_dist']

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
            obs['observation'] = np.concatenate((wp_angles_array, np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([light])))

        elif self.config["input_type"] == 'wp_vecs_obs_info_speed_steer_ldist_light':
            wp_angles_array, wp_vectors_array = self.get_wp_obs_input(agent)

            # normalize vectors by 10, assuming max norm of vector would be 10
            wp_vectors_array = wp_vectors_array / 10
            speed = agent.episode_measurements['speed'] / 10
            obstacle_dist = agent.episode_measurements['obstacle_dist']
            obstacle_speed = agent.episode_measurements['obstacle_speed']
            steer = agent.episode_measurements['control_steer']
            ldist = agent.dist_to_trajectory
            light = agent.episode_measurements['red_light_dist']
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
            obs['observation'] = np.concatenate((wp_vectors_array, np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([light])))

        elif self.config["input_type"] == 'wp_angles_vecs_obs_info_speed_steer_ldist_light':
            wp_angles_array, wp_vectors_array = self.get_wp_obs_input(agent)

            # normalize vectors by 10, assuming max norm of vector would be 10
            wp_vectors_array = wp_vectors_array / 10
            speed = agent.episode_measurements['speed'] / 10
            obstacle_dist = agent.episode_measurements['obstacle_dist']
            obstacle_speed = agent.episode_measurements['obstacle_speed']
            steer = agent.episode_measurements['control_steer']
            ldist = agent.dist_to_trajectory
            light = agent.episode_measurements['red_light_dist']
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

            obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), wp_angles_array, wp_vectors_array, np.array([obstacle_dist]), np.array([obstacle_speed]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([light])))

    def step(self, action=None):
        try:
            # if self.config['test_comparison']:
            #     self._step_test_comparison(action)
            #     return None
            # elif self.config['algo'] == 'A2C':
                # new_obs, reward, done, ep_info = self._step(action[0])
                # return [new_obs], [reward], [done], [ep_info]
            self.list_step() # action here will be an action list
            # else:
            #     obs = self._step(action)
            # return obs
        except Exception:
            print("Error during step, terminating episode early", traceback.format_exc())
            raise

    def get_action_for_test_comparison(self):
        pass

    def _update_control(self, agent):
        control = self.get_control(agent, agent.action)
        #Store control for this step
        agent.episode_measurements['control_steer'] = control.steer
        agent.episode_measurements['control_throttle'] = control.throttle
        agent.episode_measurements['control_brake'] = control.brake
        agent.episode_measurements['control_reverse'] = control.reverse
        agent.episode_measurements['control_hand_brake'] = control.hand_brake
        return control

    def list_step(self):
        # action_list here should be a list of action
        self.world_frame = None

        for rk, agent in enumerate(self.ego_agent_list):
            if agent.action is None: continue
            agent.curr_reward = 0
            if not self.config["use_pid_in_frame_skip"]:
                control = self._update_control(agent)

        for _ in range(self.config["frame_skip"]):
            for rk, agent in enumerate(self.ego_agent_list):
                if agent.done or agent.action is None: continue
                if self.config["use_pid_in_frame_skip"]:
                    control = self._update_control(agent)
                    if self.config['verbose']:
                        print("steer", control.steer, "throttle", control.throttle, "brake", control.brake,
                    "reverse", control.reverse)
                        print("steps", agent.curr_ep_num_steps)

                agent.obstacle_dist_array.append(agent.episode_measurements['obstacle_dist'])
                agent.obstacle_speed_array.append(agent.episode_measurements['obstacle_speed'])
                agent.wp_orientation_array.append(agent.episode_measurements['next_orientation'])
                agent.input_steer_array.append(agent.episode_measurements['control_steer'])
                agent.speeds_array.append(agent.episode_measurements['speed'] * 3.6)
                agent.red_light_dist_array.append(agent.episode_measurements['red_light_dist'])
                agent.dist_to_trajectory_array.append(agent.episode_measurements['dist_to_trajectory'])
                agent.dist_to_target_array.append(agent.episode_measurements['distance_to_goal_trajec'])
                agent.vehicle_actor.apply_control(control)
                agent.curr_ep_num_steps += 1
                # if not agent.unseen:
                #     agent.total_num_steps +=1
            ########################################################################################
            self.world_frame = self._world.tick()
            ########################################################################################
            for idx, agent in enumerate(self.ego_agent_list):
                if agent.done or agent.action is None: continue
                agent.episode_measurements['num_steps'] = agent.curr_ep_num_steps
                # Set state variables for reward calculation
                agent.episode_measurements['num_collisions'] = agent.collision_sensor.num_collisions
                agent.episode_measurements['collision_actor_id'] = agent.collision_sensor.actor_id
                agent.episode_measurements['collision_actor_type'] = agent.collision_sensor.actor_type
                if self.config["enable_lane_invasion_sensor"]:
                    agent.episode_measurements['num_laneintersections'] = agent.lane_invasion_sensor.num_laneintersections
                    agent.episode_measurements['out_of_road'] = agent.lane_invasion_sensor.out_of_road
                agent.location = agent.vehicle_actor.get_location()
                agent.episode_measurements['distance_to_goal'] = agent.location.distance(agent.destination_transform.location)
                if agent.episode_measurements['min_distance_to_goal'] >= agent.location.distance(agent.destination_transform.location):
                    agent.episode_measurements['min_distance_to_goal'] = agent.location.distance(agent.destination_transform.location)
                agent.episode_measurements['speed'] = self.get_speed_from_velocity(agent.vehicle_actor.get_velocity())

                next_orientation, agent.dist_to_trajectory, distance_to_goal_trajec, \
                    agent.next_waypoints, agent.next_wp_angles, agent.next_wp_vectors = \
                    agent.global_planner.get_next_orientation_new(agent.vehicle_actor.get_transform())

                agent.episode_measurements['next_orientation'] = next_orientation
                agent.episode_measurements['distance_to_goal_trajec'] = distance_to_goal_trajec
                agent.episode_measurements['dist_to_trajectory'] = agent.dist_to_trajectory

                self._get_ego_input(agent)

                agent.step_reward = compute_reward(name=self.config['reward_function'],
                                    prev_measurement=agent.prev_measurement,
                                    cur_measurement=agent.episode_measurements,
                                    config=self.config,
                                    verbose=self.config["verbose"])
                agent.curr_reward += agent.step_reward

                obs_collision = agent.episode_measurements['num_collisions'] - agent.prev_measurement['num_collisions'] > 0
                # print('obs_collision', obs_collision, agent.episode_measurements['num_collisions'], agent.prev_measurement['num_collisions'])

                if obs_collision and agent.episode_measurements["collision_actor_id"] != agent.prev_measurement["collision_actor_id"]:
                    # print("agent.episode_measurements['is_collision'] =", obs_collision, agent.episode_measurements["collision_actor_id"])
                    self.total_collisions += 1
                    if 'vehicle' in agent.episode_measurements['collision_actor_type']:
                        self.vehicle_collisions += 1
                    else:
                        self.static_collisions += 1
                elif not obs_collision:
                    agent.episode_measurements["collision_actor_id"] = None

                agent.episode_measurements['is_collision'] = obs_collision

                if agent.episode_measurements['runover_light']:
                    self.traffic_light_violations += 1

                if self.config["verbose"]:
                    print("Collisions Total: {}, Vehicle: {}, Static: {}".format(self.total_collisions, self.vehicle_collisions, self.static_collisions))
                    print("Traffic Light Violations: {}".format(self.traffic_light_violations))

                done = self._compute_done_condition(agent)
                # print('[agent {}] 677'.format(agent.rank), agent.episode_measurements['initial_dist_to_red_light'])
                agent.episode_measurements['done'] = done
                agent.done = bool(done)
                agent.prev_measurement = copy.deepcopy(agent.episode_measurements)

                agent.target_speeds_array.append(agent.episode_measurements['target_speed'])
                agent.throttles_array.append(control.throttle)
                agent.steers_array.append(control.steer)
                agent.brakes_array.append(control.brake)
                agent.step_reward_array.append(agent.episode_measurements['step_reward'])
                agent.collision_reward_array.append(agent.episode_measurements['collision_reward'])
                agent.dist_to_trajectory_reward_array.append(agent.episode_measurements['dist_to_trajectory_reward'])
                agent.speed_reward_array.append(agent.episode_measurements['speed_reward'])

                agent.episode_reward += agent.curr_reward
                agent.episode_measurements['reward'] = agent.curr_reward
                agent.episode_measurements['total_reward'] = agent.episode_reward

        for rk, agent in enumerate(self.ego_agent_list):
            if agent.action is None:
                self._get_ego_input(agent)
                agent.prev_measurement = copy.deepcopy(agent.episode_measurements)

    def _step_test_comparison(self, action):
        pass

    def _step(self, action):
        pass

    def _add_to_stacked_queue(self, object_queue, object_to_add):

        assert (object_queue is not None and object_to_add is not None)

        if object_queue.full():
            # Pop out earlier stacked frame if queue is full
            object_queue.get()
        object_queue.put(object_to_add)

    def _update_straight_dynamic_obs(self, agent):
        car_spawn_point = Transform(Location(x=92.10997772216797, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672))
        location = agent.vehicle_actor.get_location()
        distance_to_car = location.distance(car_spawn_point.location)

        agent.episode_measurements['obstacle_dist'] = distance_to_car

        if distance_to_car < 20:
            agent.episode_measurements['obstacle_visible'] = True
        else:
            agent.episode_measurements['obstacle_visible'] = False

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

    def _update_env_obs(self, agent):
        if not self.config['disable_obstacle_info']:
            if self.config['enable_obstacle_sensor']:
                self._update_obs_detector_via_sensor(agent)
            else:
                self._update_obs_detector_via_privilege(agent)

        if not self.config['disable_traffic_light']:
            self._update_traffic_light_states(agent)

            if self.config['verbose']:
                print('[agent {}] light info:'.format(agent.rank),
                    agent.episode_measurements['dist_to_light'],
                    agent.episode_measurements['nearest_traffic_actor_id'],
                    agent.episode_measurements['nearest_traffic_actor_state'],
                    agent.episode_measurements['initial_dist_to_red_light'],
                    agent.episode_measurements['red_light_dist'])


    def _update_obs_detector_via_privilege(self, agent):
        agent.episode_measurements['obstacle_visible'] = False
        agent.episode_measurements['obstacle_orientation'] = -1

        min_obs_distance = 100000000
        found_obstacle = False
        for target_vehicle in self.actor_list + self.ego_vehicle_list:
            # do not account for the ego vehicle
            try:
                if target_vehicle is None or hasattr(target_vehicle, 'done') and target_vehicle.done: continue
                if target_vehicle.id == agent.id or 'vehicle' not in target_vehicle.type_id:
                    continue

                # if the object is not in our lane it's not an obstacle
                target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
                d_bool, d_angle, distance = self.is_within_distance_ahead(target_vehicle.get_transform(),
                                            agent.vehicle_actor.get_transform(),
                                            self.config['vehicle_proximity_threshold'])

                if not d_bool:
                    continue
                else:
                    if not check_if_vehicle_in_same_lane(agent.vehicle_actor, target_vehicle, agent.next_waypoints, self._map):
                        continue

                    found_obstacle = True
                    agent.episode_measurements['obstacle_visible'] = True
                    agent.episode_measurements['obstacle_orientation'] = d_angle

                    if distance < min_obs_distance:
                        agent.episode_measurements['obstacle_dist'] = distance
                        agent.episode_measurements['obstacle_speed'] = self.get_speed_from_velocity(target_vehicle.get_velocity())
                        min_obs_distance = distance
            except Exception as e:
                print('>>> skip this vehicle {} due to [{}]'.format(target_vehicle, e))
                if target_vehicle is None or hasattr(target_vehicle, 'done'):
                    print(target_vehicle.done, target_vehicle.termination_state, target_vehicle.rank, target_vehicle.num_total_steps)
                self.spawn_npc_vehicles()
                time.sleep(4)
                return

        if not found_obstacle:
            agent.episode_measurements['obstacle_dist'] = -1
            agent.episode_measurements['obstacle_speed'] = -1
        # else:
        #     print('obstacle actor {}, dist: {}'.format(target_vehicle, distance))

    def _update_obs_detector_via_sensor(self, agent):
        agent.episode_measurements['obstacle_visible'] = False
        agent.episode_measurements['obstacle_orientation'] = -1

        found_obstacle = False
        same_lane = True
        if agent.obstacle_sensor.distance != -1:
            if 'vehicle' in obstacle_actor.type_id:
                same_lane = check_if_vehicle_in_same_lane(agent.vehicle_actor, obstacle_actor, agent.next_waypoints, self._map)
            found_obstacle = True
            obstacle_actor = agent.obstacle_sensor.obstacle_actor
            agent.episode_measurements['obstacle_visible'] = True
            agent.episode_measurements['obstacle_dist'] = agent.obstacle_sensor.distance
            # if 'vehicle' in obstacle_actor.type_id:
            if hasattr(obstacle_actor, 'get_velocity'):
                agent.episode_measurements['obstacle_speed'] = self.get_speed_from_velocity(obstacle_actor.get_velocity())
            else:
                agent.episode_measurements['obstacle_speed'] = -1
            print('obstacle actor {}, dist: {}, same_lane: {}'.format(obstacle_actor, agent.obstacle_sensor.distance, same_lane))
        if not found_obstacle or not same_lane:
            agent.episode_measurements['obstacle_visible'] = False
            agent.episode_measurements['obstacle_dist'] = -1
            agent.episode_measurements['obstacle_speed'] = -1


    def _update_traffic_light_states(self, agent):
        # TODO: Pass correct target waypoint to find_nearest_traffic_light() for US style traffic.
        traffic_actor, dist, traffic_light_orientation = agent.find_nearest_traffic_light(self.traffic_actors)
        if traffic_light_orientation is not None:
            agent.episode_measurements['traffic_light_orientation'] = traffic_light_orientation
        else:
            agent.episode_measurements['traffic_light_orientation'] = -1

        if traffic_actor is not None:
            if traffic_actor.state == carla.TrafficLightState.Red:
                agent.episode_measurements['red_light_dist'] = dist
                # print('[agent {} init {}] traffic light info'.format(
                #         agent.rank, agent.episode_measurements['initial_dist_to_red_light']), traffic_actor.id, traffic_actor.state, dist)
                if agent.episode_measurements['initial_dist_to_red_light'] == -1 or \
                    (agent.episode_measurements['nearest_traffic_actor_id'] != -1 and traffic_actor.id != agent.episode_measurements['nearest_traffic_actor_id']):
                    agent.episode_measurements['initial_dist_to_red_light'] = dist
                    # print('[agent {} init {}] traffic light info'.format(
                    #     agent.rank, agent.episode_measurements['initial_dist_to_red_light']), traffic_actor.id, traffic_actor.state, dist)
            else:
                agent.episode_measurements['red_light_dist'] = -1
                agent.episode_measurements['initial_dist_to_red_light'] = -1

            agent.episode_measurements['nearest_traffic_actor_id'] = traffic_actor.id
            agent.episode_measurements['nearest_traffic_actor_state'] = traffic_actor.state
            # print('[agent {} init {}] traffic light info'.format(
            #     agent.rank, agent.episode_measurements['initial_dist_to_red_light']), traffic_actor.id, traffic_actor.state, dist)
        else:
            agent.episode_measurements['red_light_dist'] = -1
            agent.episode_measurements['initial_dist_to_red_light'] = -1
            agent.episode_measurements['nearest_traffic_actor_id'] = -1
            agent.episode_measurements['nearest_traffic_actor_state'] = None
            # print('[agent {} init {}] traffic light info'.format(
            #     agent.rank, agent.episode_measurements['initial_dist_to_red_light']), -1, -1, -1)

        agent.episode_measurements['dist_to_light'] = dist
        # print('[agent {} init {}] traffic light info'.format(
        #     agent.rank, agent.episode_measurements['initial_dist_to_red_light']), -1, -1, -1)


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


    def get_control(self, agent, action):
        """ Get Control object for Carla from action
        Input:
            - action: tuple containing (steer, throttle, brake) in [-1, 1]
        Output:
            - control: Control object for Carla
        """

        if self.config["action_type"] != "control":
            action = action.flatten()

        if self.config["action_type"] == "sep_gas":
            steer = float(action[0])
            throttle = float(action[1])
            brake = float(action[2])
        elif self.config["action_type"] == "merged_gas":
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
            current_speed = self.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
            throttle = self.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config["action_type"] == "throttle_only":
            steer = float(0.0)
            target_speed = float(np.clip(action[0], 0, self.target_speed))
            current_speed = self.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
            throttle = self.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config["action_type"] == "merged_speed":
            # steer = float(action[0])
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(np.clip(action[1] + 10.0, 0, self.target_speed))
            current_speed = self.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
            throttle = self.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config["action_type"] == "merged_speed_tanh":
            # steer = float(action[0])
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(np.clip((action[1] + 1) * 10.0, 0, self.target_speed))
            current_speed = self.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
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
            current_speed = self.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
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
            current_speed = self.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
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
            discrete_actions = self.config['discrete_actions'][int(action)]
            target_speed, steer = float(discrete_actions[0]), float(discrete_actions[1])
            current_speed = self.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.config["enable_brake"])
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config["action_type"] == "control":
            target_speed = -1
            agent.episode_measurements["target_speed"] = target_speed
            return action

        agent.episode_measurements["target_speed"] = target_speed

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0)

        return control


    def reset(self, use_idx=False, idx_list=None, rank_list=None, reset_npc=False):
        # if self.config['test_comparison']:
        #     return self._reset_test_comparison(unseen, index)
        # elif self.config['algo'] == 'A2C':
        return self.list_reset(use_idx=use_idx, idx_list=idx_list, rank_list=rank_list, reset_npc=reset_npc)
        # else:
        #     return self._reset(unseen, index)

    def destroy_all_existing_npc_actors(self):
        # Delete all existing actors
        for _ in range(len(self.actor_list)):
            try:
                actor = self.actor_list.pop()
                actor.destroy()
            except Exception as e:
                print("Error during destroying actor {0}:{1}: {2}".format(actor.type_id, actor.id, traceback.format_exc()))

    def destroy_an_existing_ego_agent(self, agent):
        if agent is None: return
        for _ in range(len(agent.actor_list)):
            try:
                actor = agent.actor_list.pop()
                actor.destroy()
            except Exception as e:
                print("Error during destroying sensor actor {0}:{1}: {2}".format(actor.type_id, actor.id, traceback.format_exc()))
        try:
            actor = agent.vehicle_actor
            actor.destroy()
        except Exception as e:
            print("Error during destroying ego vehicle actor {0}:{1}: {2}".format(actor.type_id, actor.id, traceback.format_exc()))
        try:
            del agent
        except Exception as e:
            print("Error during destroying ego agent {0}:{1}: {2}".format(agent, e, traceback.format_exc()))

    def destroy_all_existing_ego_agents(self):
        for agent in self.ego_agent_list:
            self.destroy_an_existing_ego_agent(agent)

    def clear_episode_measurements(self):
        # Below logic is to avoid clearing of following measurements,
        # when env reset is called automatically in DummyVec env.
        # These are used in testing logic, hence their values are required.
        for key, val in self.episode_measurements.items():
            if key in ['termination_state_code', 'termination_state']:
                continue

            self.episode_measurements[key] = 0

    def save_rgb_image(self, agent, rgb_image, save_folder='../../../../alta-logs/a3c/'):
        _folder = '{}/{}/{}'.format(os.path.expanduser(save_folder), agent.rank, agent.episode_id)
        if not os.path.exists(_folder):
            print('creating [{}] to save image'.format(_folder))
            os.makedirs(_folder)
        _filename = '{}/{:08d}.jpg'.format(_folder, agent.num_total_steps)
        plt.imsave(_filename, rgb_image)


    def reset_vehicle_agent(self, agent_list):
        # bind new agent
        for agent in agent_list:
            self.ego_agent_list[agent.rank] = agent

            # set attributes
            agent.image_data = None
            agent.source_transform = agent.vehicle_actor.source_transform
            agent.destination_transform = agent.vehicle_actor.destination_transform
            agent.scenario_route = None
            agent.global_planner = None
            agent.trace_route = None
            agent.episode_num = 0
            agent.validation_episode_num = 0
            agent.semantic_image = None
            agent.index = 0
            agent.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
            agent.curr_ep_num_steps = 0

            agent.episode_measurements = copy.deepcopy(self.config['episode_measurements'])
            agent.previous_measurements = None

            agent.dist_to_trajectory = None
            agent.episode_measurements["episode_num"] = 0
            agent.episode_measurements['obstacle_visible'] = False
            agent.episode_measurements['obstacle_dist'] = -1
            agent.episode_measurements['obstacle_speed'] = -1
            agent.episode_measurements['obstacle_orientation'] = -1
            agent.next_waypoints = None
            agent.next_wp_vectors = None
            agent.next_wp_angles = None

            agent.episode_measurements['dist_to_light'] = -1
            agent.episode_measurements['nearest_traffic_actor_id'] = -1
            agent.episode_measurements['nearest_traffic_actor_state'] = None
            agent.episode_measurements['initial_dist_to_red_light'] = -1
            agent.episode_measurements['red_light_dist'] = -1
            agent.episode_measurements['traffic_light_orientation'] = -1
            agent.episode_measurements["runover_light"] = False

            agent.rv_camera_queue = queue.Queue()

            agent.actor_list = []
            agent.target_speeds_array = []
            agent.speeds_array = []
            agent.throttles_array = []
            agent.obstacle_speed_array = []
            agent.dist_to_trajectory_array = []
            agent.steers_array = []
            agent.brakes_array = []
            agent.wp_orientation_array = []
            agent.input_steer_array = []
            agent.obstacle_dist_array = []
            agent.step_reward_array = []
            agent.collision_reward_array = []
            agent.dist_to_trajectory_reward_array = []
            agent.speed_reward_array = []
            agent.dist_to_target_array = []
            agent.red_light_dist_array = []

            if self.config["semantic"]:
                sensor = self.config['sensors'][1]
            else:
                sensor = self.config['sensors'][0]

            rv_camera = self.blueprint_library.find(sensor)
            rv_camera.set_attribute('image_size_x', self.config['sensor_x_res'])
            rv_camera.set_attribute('image_size_y', self.config['sensor_y_res'])
            rv_camera.set_attribute('sensor_tick', self.config['sensor_tick'])
            # camera.set_attribute('fov', '120')
            rv_camera.set_attribute('fov', '90')

            # Orientation for forward-facing camera
            rv_camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0.0))

            agent.rv_camera_actor = self._world.spawn_actor(rv_camera, rv_camera_transform, attach_to=agent.vehicle_actor)
            agent.actor_list.append(agent.rv_camera_actor)

            agent.rv_camera_actor.listen(agent.rv_camera_queue.put)

            agent.collision_sensor = sensors.CollisionSensor(agent.vehicle_actor)
            agent.actor_list.append(agent.collision_sensor.sensor)

            if self.config["enable_lane_invasion_sensor"]:
                agent.lane_invasion_sensor = sensors.LaneInvasionSensor(agent.vehicle_actor)
                agent.actor_list.append(agent.lane_invasion_sensor.sensor)

            if self.config["enable_obstacle_sensor"]:
                agent.obstacle_sensor = sensors.ObstacleSensor(agent.vehicle_actor, 
                    distance=self.config['vehicle_proximity_threshold'])

                agent.actor_list.append(agent.obstacle_sensor.sensor)

            # Set state variables for reward calculation
            agent.episode_measurements['num_collisions'] = agent.collision_sensor.num_collisions
            agent.episode_measurements['collision_actor_id'] = agent.collision_sensor.actor_id
            agent.episode_measurements['collision_actor_type'] = agent.collision_sensor.actor_type
            if self.config["enable_lane_invasion_sensor"]:
                agent.episode_measurements['num_laneintersections'] = agent.lane_invasion_sensor.num_laneintersections
                agent.episode_measurements['out_of_road'] = int(agent.lane_invasion_sensor.out_of_road)
            agent.location = agent.vehicle_actor.get_location()
            agent.episode_measurements['distance_to_goal'] = agent.location.distance(agent.destination_transform.location)
            agent.episode_measurements['min_distance_to_goal'] = 1000000.0
            agent.episode_measurements['speed'] = self.get_speed_from_velocity(agent.vehicle_actor.get_velocity())

            agent.episode_measurements['total_steps'] = agent.num_total_steps

        # Ticking for 15 frames to handle car initialization in air
        time.sleep(.04)
        # for _ in range(3):
        #     # print(self.world_frame)
        #     self.world_frame = self._world.tick()


    def _get_ego_input(self, agent):
        rv_image = self._read_data(agent.rv_camera_queue, self.world_frame)

        agent.global_planner = planner.GlobalPlanner()

        if self.config["use_route_to_plan"]:
            agent.trace_route = []
            for idx in range(len(self.scenario_route) - 1):
                source = self.scenario_route[idx]
                destination = self.scenario_route[idx + 1]
                trace_route = agent.global_planner._trace_route(self._map,
                                source, destination)
                agent.trace_route.extend(trace_route)
        else:
            agent.trace_route  = agent.global_planner._trace_route(self._map,
                                agent.source_transform, agent.destination_transform)
        agent.global_planner.set_global_plan(agent.trace_route)

        next_orientation, agent.dist_to_trajectory, distance_to_goal_trajec, \
            agent.next_waypoints, agent.next_wp_angles, agent.next_wp_vectors = \
            agent.global_planner.get_next_orientation_new(agent.vehicle_actor.get_transform())

        agent.episode_measurements['next_orientation'] = next_orientation
        agent.episode_measurements['distance_to_goal_trajec'] = distance_to_goal_trajec
        # if agent.unseen:
        #     self.total_distance += distance_to_goal_trajec
        agent.episode_measurements['dist_to_trajectory'] = agent.dist_to_trajectory

        # Update obstacle distance measurements
        obs = {}
        # self._update_env_obs(front_rgb_image=rgb_image)
        self._update_env_obs(agent)

        # if static (stuck by obstacle)
        if agent.episode_measurements['speed'] < self.config['zero_speed_threshold'] and \
            agent.episode_measurements['obstacle_dist'] == -1 and \
            agent.episode_measurements['red_light_dist'] != -1:
            agent.episode_measurements['static_steps'] += 1
        else:
            agent.episode_measurements['static_steps'] = 0

        if self.config["scenarios"] == "straight_dynamic":
            self._update_straight_dynamic_obs(agent)

        obs['rv_image'] = rv_image
        # self.save_rgb_image(agent, rv_image)

        obs['speed'] = np.expand_dims(np.array([agent.episode_measurements['speed']]), axis=0) # * 3.6 / 30
        obs['dist_to_target'] = np.array([agent.episode_measurements['distance_to_goal']])

        # Update observation input in obs dictionary
        self.create_observations(agent, obs)
        # agent.prev_measurement = copy.deepcopy(agent.episode_measurements)

        visual_observation = None
        if self.config["input_type"] == 'vae':
            agent.observation = visual_observation
        elif self.config["input_type"] in ['wp_vae', 'wp_vae_speed_steer_goal', 'wp_vae_speed_steer_ldist_goal_light', 'wp_vae_obs_info_speed_steer_ldist_goal_light']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            fused_input = np.hstack([visual_observation, observation])
            agent.observation = fused_input
        elif self.config["input_type"] in ['wp_cnn_obs_info_speed_steer_ldist_goal_light']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            visual_observation = visual_observation.reshape((1, -1))
            fused_input = np.hstack([visual_observation, observation])
            agent.observation = fused_input
        elif self.config["input_type"] in ['wp_bev_rv_obs_info_speed_steer_ldist_goal_light']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            visual_observation = visual_observation.reshape((1, -1))
            fused_input = np.hstack([visual_observation, observation])
            agent.observation = fused_input, visual_observation
        elif self.config["input_type"] == "wp":
            agent.observation = obs['observation']
        elif self.config["input_type"] in ['wp_noise', 'wp_constant', 'wp_obs_dist', 'wp_obs_bool', 'wp_obs_bool_noise', 'wp_ldist_goal',
                                        'wp_speed', 'wp_speed_goal','wp_speed_steer_goal', 'wp_speed_steer_goal_obs_bool',
                                        'wp_obs_bool_speed_steer_goal_light', 'wp_obs_info_speed_steer_ldist_goal_light',
                                        'wp_obs_info_speed_steer_ldist_goal', 'wp_obs_info_speed_steer_ldist_light',
                                        'wp_angles_obs_info_speed_steer_ldist_light', 'wp_vecs_obs_info_speed_steer_ldist_light',
                                        'wp_angles_vecs_obs_info_speed_steer_ldist_light']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            agent.observation = observation
        else:
            agent.observation = obs

        return agent.observation

    def list_reset(self, use_idx=False, idx_list=None, rank_list=None, reset_npc=False):
        if not idx_list: idx_list = [0] * self.config['num_agents']
        try:
            vehicle_bp = self.blueprint_library.find(self.config['vehicle_type'])
            # vehicle_bp = self.blueprint_library.find(random.choice(self.config['vehicle_types']))
        except Exception as e:
            print("Error during vehicle creation: {}".format(traceback.format_exc()))

        for rk in rank_list:
            prev_agent = self.ego_agent_list[rk]
            self.ego_agent_list[rk] = None
            if prev_agent is not None: self.curr_num_agents -= 1
            try:
                self.destroy_an_existing_ego_agent(prev_agent)
            except:
                print('>>> Error when deleting prev_agent [rank {}]'.format(rk))

            # Spawning vehicle actor with retry logic as it fails to spawn sometimes
            self.vehicle_actor = None
            NUM_RETRIES = 100
            for idx in range(1, NUM_RETRIES + 1):
                # Set source and destination based on scenario
                # Currently scenarios are defined only for Town01
                if reset_npc:
                    self.destroy_all_existing_npc_actors()

                if self.config["use_scenarios"] and (self.config["city_name"] == "Town01" or self.config["city_name"] == "Town02"):
                    if self.config["updated_scenarios"]:
                        self._set_updated_scenario(unseen=use_idx, index=idx_list[rk], town=self.config["city_name"])
                    else:
                        self._set_scenario(unseen=use_idx, index=idx_list[rk], town=self.config["city_name"])
                else:
                    self.source_transform, self.destination_transform = random.choice(self.spawn_points), random.choice(self.spawn_points)

                self.vehicle_actor = self._world.try_spawn_actor(vehicle_bp, self.source_transform)

                if reset_npc:
                    self.spawn_npc_vehicles()

                if self.vehicle_actor is not None:
                    break
                else:
                    print("[rank {}] Unable to spawn ego vehicle [trial {}] at ({:.2f}, {:.2f}).".format(
                        rk, idx, self.source_transform.location.x, self.source_transform.location.y))
                    # print("Number of existing actors, {}".format(len(self.actor_list)))
                    # print("Number of existing ego agents, {}".format(self.curr_num_agents))
                    time.sleep(.04)

            if self.vehicle_actor is not None:
                # print(self.vehicle_actor)
                self.ego_vehicle_list[rk] = self.vehicle_actor
                self.vehicle_actor.source_transform = self.source_transform
                self.vehicle_actor.destination_transform = self.destination_transform
                if self.config['verbose']:
                    print('########## rank {} ##########'.format(rk))
                    print('SRC TRANSFORM =', self.vehicle_actor.source_transform)
                    print('DST TRANSFORM =', self.vehicle_actor.destination_transform)
                self.curr_num_agents += 1
            else:
                raise Exception("Failed in spawning vehicle actor.")

    def spawn_npc_vehicles(self):
        self.destroy_all_existing_npc_actors()
        if self.config["sample_npc"]:
            self.spawn_npc(np.random.randint(low=self.config["num_npc_lower_threshold"],
                high=self.config["num_npc_upper_threshold"]))
        else:
            self.spawn_npc(self.config["num_npc"])


    def _reset_test_comparison(self, unseen=False, index=0):
        pass
    
    # @profile
    def _reset(self, unseen=False, index=0):
        pass

    def get_wp_obs_input(self, agent):
        '''
        Create wp angles input
        '''
        num_wp = 5
        wp_angles_array = None
        wp_vectors_array = None

        n = len(agent.next_wp_angles)
        if n == 0:
            print("No next waypoints found. Giving zero as input.")
            wp_angles_array = np.zeros(num_wp)
            wp_vectors_array = np.zeros(2 * num_wp)

        elif n == num_wp:
            wp_angles_array = np.array(agent.next_wp_angles)
            wp_vectors_array = np.array(agent.next_wp_vectors)

        elif n < num_wp:
            # Fill using last entry
            last_angle = agent.next_wp_angles[-1]
            last_vec = agent.next_wp_vectors[-1]

            for _ in range(num_wp-n):
                agent.next_wp_angles.append(last_angle)
                agent.next_wp_vectors.append(last_vec)
            wp_angles_array = np.array(agent.next_wp_angles)
            wp_vectors_array = np.array(agent.next_wp_vectors)
        else:
            print("Error: More than {0} waypoints returned from planner.".format(num_wp))
            print("Taking required number of entries.")
            wp_angles_array = np.array(agent.next_wp_angles[:num_wp])
            wp_vectors_array = np.array(agent.next_wp_vectors[:num_wp])

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

            if self.config["verbose"]:
                print('spawned %r at %s' % (vehicle.type_id, transform.location.x))
            return True
        return False

    def spawn_npc(self, number_of_vehicles):
        # Testing
        if self.config["test_fixed_spawn_points"]:
            spawn_points = self.spawn_points_fixed_order
        else:
            spawn_points = self.spawn_points
            random.shuffle(spawn_points)

        if self.config["verbose"]:
            print('found %d spawn points.' % len(spawn_points))

        count = number_of_vehicles
        for spawn_point in spawn_points:
            if self.config["verbose"]:
                print('spawn_point:', spawn_point)
            if self.try_spawn_random_vehicle_at(self.vehicle_blueprints, spawn_point):
                count -= 1
            if count <= 0:
                break

    def get_speed_from_velocity(self, velocity):
        speed = np.sqrt(velocity.x ** 2 + velocity.y **2 + velocity.z **2)
        return speed

    def _read_data(self, camera_queue, world_frame, timeout=240.0):

        cam_image = self._read_camera_data(camera_queue, self.world_frame, timeout)
        cam_image_p = self._preprocess_image(cam_image)
        return cam_image_p

    def _preprocess_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    def _read_camera_data(self, camera_queue, world_frame, timeout):

        data  = self._retrieve_data(camera_queue, timeout, world_frame)
        return data

    # @profile
    def _retrieve_data(self, sensor_queue, timeout, world_frame):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.world_frame:
                return data
            else:
                if self.config["verbose"]:
                    print("difference in frames, self.world_frame={0}, data_frame={1}".format(self.world_frame, data.frame))

    def _compute_done_condition(self, agent):

        # Episode termination conditions
        success = agent.episode_measurements["distance_to_goal"] < self.config["dist_for_success"]
        static = agent.episode_measurements["static_steps"] > self.config["max_static_steps"]
        # static = agent.episode_measurements['obstacle_dist'] == -1 and self.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) < 1e-2
        collision = agent.episode_measurements["is_collision"]
        runover_light = agent.episode_measurements["runover_light"]
        maxStepsTaken = agent.episode_measurements["num_steps"] > self.config['max_steps']
        offlane = False

        # Conditions to check there is obstacle or red light ahead for last 2 timesteps
        obstacle_ahead = agent.episode_measurements['obstacle_dist'] != -1 and agent.prev_measurement['obstacle_dist'] != -1
        red_light = agent.episode_measurements['red_light_dist'] != -1 and agent.prev_measurement['red_light_dist'] != -1

        if not self.config["enable_static"]:
            static = False
        if self.config["disable_collision"]:
            collision = False
        if self.config["disable_traffic_light"] or not self.config["terminate_on_light"]:
            runover_light = False
        if self.config["enable_lane_invasion_sensor"] and self.config["enable_lane_invasion_collision"]:
            offlane = agent.episode_measurements['num_laneintersections'] > 0

        # Do not want to terminate on reaching goal
        # in case of VAE training
        if success:
            termination_state = 'success'
            termination_state_code = 0
        elif collision:
            if 'obs_collision' in agent.episode_measurements and agent.episode_measurements['obs_collision']:
                termination_state = 'obs_collision'
                termination_state_code = 1
            else:
                termination_state = 'unexpected_collision'
                termination_state_code = 4
        elif self.config["enable_lane_invasion_sensor"] and agent.episode_measurements["out_of_road"]:
            termination_state = 'out_of_road'
            termination_state_code = 2
        elif self.config["enable_lane_invasion_sensor"] and agent.episode_measurements['lane_change']:
            termination_state = 'lane_invasion'
            termination_state_code = 3
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

        agent.termination_state = termination_state

        agent.episode_measurements['termination_state'] = termination_state
        agent.episode_measurements['termination_state_code'] = termination_state_code

        done = success or collision or runover_light or offlane or static or maxStepsTaken
        return done

    def printInfo(self, agent):
        print("Vehicle transform:{0}".format(agent.vehicle_actor.get_transform()))
        print("Vehicle velocity:{0}".format(agent.vehicle_actor.get_velocity()))

    def close(self):

        try:
            self.destroy_all_existing_npc_actors()
            self.destroy_all_existing_ego_agents()

            if not self.CarlaServer is None:
                self.CarlaServer.close()

        except Exception as e:
                print("********** Exception in closing env **********")
                print(e)
                print(traceback.format_exc())

    def __del__(self):
        self.close()



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    env = CarlaEnv()
    env.reset()
