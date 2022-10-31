""" Environment file wrapper for CARLA """

import gym
from gym.spaces import Box, Discrete, Tuple

from datetime import datetime
import os, sys

sys.path.append('/home/zhehuang/Documents/CARLA/alta')

import traceback
import random
import numpy as np
import math
import copy
import queue
import time
import matplotlib.pyplot as plt
from collections import deque

import environment.carla_9_4.scenarios as scenarios
import environment.carla_9_4.server as server
import environment.carla_9_4.planner as planner
import environment.carla_9_4.controller as controller
import environment.carla_9_4.sensors as sensors
from environment.carla_9_4.reward import compute_reward
from environment.carla_9_4.dashcam import Visualizer

# Leaerboard Import
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../leaderboard'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../scenario_runner'))
from leaderboard.utils.route_manipulation import interpolate_trajectory
from leaderboard.utils.route_parser import RouteParser, TRIGGER_THRESHOLD, TRIGGER_ANGLE_THRESHOLD
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.scenarios.route_scenario import (
    scenario_sampling, build_scenario_instances, convert_transform_to_location, Trigger
)
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from environment.carla_9_4.agents.navigation.local_planner import RoadOption
from environment.carla_9_4.agents.navigation.agent import Agent

# transfuser autopilot
from transfuser_autopilot import AutoPilot


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
from carla.libcarla import LaneType, LaneChange
import psutil

from environment.carla_9_4.env_util import (
    check_if_vehicle_in_same_lane,
    get_world_coords_from_latlong,
    convert_route_from_GPS_world,
    get_vehicle_bb_wp,
)



def draw_arrow_waypoints(world, waypoints, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.

    :param world: carla.world object
    :param waypoints: list or iterable container with the waypoints to draw
    :param z: height in meters
    :return:
    """
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z=z)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1000.0)

"""
HELPERS
"""

TOKEN_TYPE = {
    'EGO': 1,
    'VEHICLE': 2,
    'WAYPOINT': 3
}

class DummyScenarioConfig(object):
    def __init__(self, index, trajectory):
        self.index = index
        self.trajectory = trajectory


def flatten_obs(obs_dict):
    obs_array = np.zeros((1, 100, 8))

    # Three types of tokens -- ego, vehicle, and waypoint

    # 1. EGO TOKEN
    ego_tokens = []
    for i in range(4):
        ego_tokens.append(np.array([
            TOKEN_TYPE['EGO'],
            obs_dict['ego_features']['bounding_box'][i*2][0] / 25,
            obs_dict['ego_features']['bounding_box'][i*2][1] / 25,
            np.radians(obs_dict['ego_features']['theta']) / np.pi,
            obs_dict['ego_features']['speed'] / 10.,
            obs_dict['light'] / 10.,
            obs_dict['dist_to_trajectory'],
            obs_dict['next_orientation']
        ]))

    # 2. VEHICLE TOKENS
    vehicle_tokens = []
    for vehicle_idx in obs_dict['vehicle_features']:
        for i in range(4):
            vehicle_tokens.append(np.array([
                TOKEN_TYPE['VEHICLE'],
                obs_dict['vehicle_features'][vehicle_idx]['bounding_box'][i*2][0] / 25,
                obs_dict['vehicle_features'][vehicle_idx]['bounding_box'][i*2][1] / 25,
                np.radians(obs_dict['vehicle_features'][vehicle_idx]['theta']) / np.pi,
                obs_dict['vehicle_features'][vehicle_idx]['speed'] / 10.
            ]))

    # 3. WAYPOINT TOKENS
    waypoint_tokens = []
    for waypoint_idx, waypoint in enumerate(obs_dict['next_waypoints']):
        waypoint_token = np.array([
            TOKEN_TYPE['WAYPOINT'],
            waypoint[0] / 25,
            waypoint[1] / 25,
            waypoint_idx / len(obs_dict['next_waypoints'])
        ])
        waypoint_tokens.append(waypoint_token)

    # Fill in obs array with tokens
    tokens = ego_tokens + waypoint_tokens + vehicle_tokens
    if len(tokens) > obs_array.shape[1]:
        # too many tokens
        tokens = tokens[:obs_array.shape[1]]
        print('Got {} tokens, expecting {} tokens'.format(len(tokens), obs_array.shape[1]))

    for token_idx, token in enumerate(tokens):
        obs_array[0,token_idx,:len(token)] = token

    return obs_array.flatten()

def transform_to_pov(src, ref, theta):
    """
    Transforms src to ref frame
    src and ref are tuples (x, y)
    """
    sx, sy = src
    rx, ry = ref

    x = sx - rx
    y = sy - ry

    theta = normalize_angle(theta)
    theta = -theta # because we want to transform to 0 rotation offset
    theta = np.radians(theta) # because np expects radians

    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xy = np.array([[x],[y]])
    xy = rot_matrix.dot(xy).flatten()

    return xy[0], xy[1]


def normalize_angle(theta):
    theta = (theta + 360) % 360
    theta = theta if theta <= 180 else theta-360
    return theta

class DummyAgent(Agent):
    def __init__(self, vehicle, rank=0, **kwargs):
        super().__init__(vehicle, **kwargs)
        self.rank = rank
        self.done = False
        self.action = None
        self.id = vehicle.id
        self.type_id = vehicle.type_id
        self.vehicle_actor = vehicle
        self.num_total_steps = 0
        self.episode_reward = 0
        self.curr_reward = 0
        self.step_reward = 0
        self.autopilot = False
        self.observation = None
        self.termination_state = None


class CarlaEnv(gym.Env):
    def __init__(self, config, logger=None, env_rank=0):
        self.config = config
        self.CarlaServer = None
        self.env_rank = env_rank
        self.episode_measurements = self.config['episode_measurements']
        self.episode_id = None
        self.vehicle_actor = None
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.log_dir = os.path.expanduser(self.config['log_dir'])
        ################################################################################
        # if 'num_agents' in self.config and self.config['algo'] == 'A2C':
        if 'challenge' in self.config["scenarios"]:
            self.statistics_manager = StatisticsManager()
            self.config['num_agents'] == 1
            assert self.config['num_agents'] == 1, 'Multi agent in one env under challenge scenarios not supported'
        if self.config['verbose']: print('##### USE MULTI-AGENT #####', flush=True)
        self.ego_vehicle_list = [None] * self.config['num_agents']
        self.ego_agent_list = [None] * self.config['num_agents']
        # else:
        #     self.config['num_agents'] = 1
        self.curr_num_agents = 0
        self.world_frame = 0
        self.last_npc_reset_frame = 0
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
        self.args_lateral_dict={
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4,
            'dt': 1.0/10.0}
        self.actor_list = []

        # Queue for stacked frames and measurements
        self.rv_stack_size = 1

        self.logger = logger

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
        self.avail_map = {name[-6:]: name for name in self.client.get_available_maps()}
        print('self.avail_map:', self.avail_map)
        self._set_world_and_map(self.config['initial_town'])
        self.world_annotations = RouteParser.parse_annotations_file(
            '../../../leaderboard/data/all_towns_traffic_scenarios_public.json')

        time.sleep(20)

        # self._map = self._world.get_map()
        # self.blueprint_library = self._world.get_blueprint_library()
        # self.spawn_points = self._world.get_map().get_spawn_points()

        self.tm_port = random.randint(10000, 60000)
        self.tm = self.client.get_trafficmanager(self.tm_port)
        self.tm.set_synchronous_mode(True)

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_traffic_manager_port(self.tm_port)

        if self.config["testing"]:
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in self.config['spawn_points_fixed_idx']]
        else:
            spawn_pt_idx = np.random.permutation(len(self.spawn_points))
            np.save(os.path.join(self.log_dir, "spawn_pt_order"), spawn_pt_idx)
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in spawn_pt_idx]

        self._setup_observation_and_action_space()

        # self.vehicle_blueprints = self._world.get_blueprint_library().filter('vehicle.*')
        # self.traffic_actors = self._world.get_actors().filter("*traffic_light*")

        if self.config["disable_two_wheeler"]:
            self.vehicle_blueprints = [x for x in self.vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]

    def _setup_observation_and_action_space(self):

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
            self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, 0.0, -1 * self.config['steering_scale'], -1.0, 0.0, 0.0]]),
                high=np.array([[4.0, 1.0, 1.0, 1.0, self.config['steering_scale'], 1.0, 1.0, 1.0]]), dtype=np.float32)

        elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_goal':
            self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, 0.0, -0.5, -1.0, 0.0]]), high=np.array([[4.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]]), dtype=np.float32)

        elif self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_light': # currently using
            self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, 0.0, -self.config['steering_scale'], -1.0, 0.0]]), high=np.array([[4.0, 1.0, 1.0, 1.0, self.config['steering_scale'], 1.0, 1.0]]), dtype=np.float32)

        elif self.config["input_type"] == 'wp_obs_info_side_obs_info_speed_steer_ldist_light':
            self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, -1., 0., -1., 0., 0.0, -self.config['steering_scale'], -1.0, 0.0]]),
            high=np.array([[4.0, 1.0, 1.0, 1., 1., 1., 1., 1.0, self.config['steering_scale'], 1.0, 1.0]]), dtype=np.float32)

        elif self.config["input_type"] == 'wp_obs_more_info_speed_steer_ldist_light': # 5 obs sensors
            self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -self.config['steering_scale'], -1.0, 0.0]]),
             high=np.array([[4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, self.config['steering_scale'], 1.0, 1.0]]), dtype=np.float32)

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

        elif self.config["input_type"] == 'transformer':
            self.observation_space = Box(low=np.finfo(np.float32).min,
                                    high=np.finfo(np.float32).max,
                                    shape=(800,), dtype=np.float32)

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

        elif self.config["input_type"] == 'wp_360_obstacle_speed_steer':
            self.observation_space = Box(low=np.array([[-4.0, 0.0, -0.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]),
                            high=np.array([[4.0, 1.0, 0.5, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]]),
                            dtype=np.float32)

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

    def _update_config(self, config):
        for key, val in config.items():
            self.config[key] = val

    def _spawn_client(self, hostname='localhost', port_number=None):
        port_number = self.CarlaServer.server_port
        client = carla.Client(hostname, port_number)
        client.set_timeout(self.config["client_timeout_seconds"])
        return client

    def _set_world_and_map(self, town_name):
        # Get the world
        self.curr_town = town_name
        # print('[433]', self.curr_town)
        self._world = self.client.load_world(self.curr_town)
        # print('[434]', self.curr_town)
        settings = self._world.get_settings()
        if(self.config['sync_mode']):
            settings.synchronous_mode = True
        if self.config["server_fps"] is not None and self.config["server_fps"] != 0:
            settings.fixed_delta_seconds =  1.0 / float(self.config["server_fps"])
        # Enable rendering
        settings.no_rendering_mode = False
        self._world.apply_settings(settings)
        # Sleep to allow for settings to update
        time.sleep(5)
        # Retrieve map
        self._map = self._world.get_map()
        # Get blueprints
        self.blueprint_library = self._world.get_blueprint_library()
        self.vehicle_blueprints = self._world.get_blueprint_library().filter('vehicle.*')
        self.spawn_points = self._world.get_map().get_spawn_points()
        # Get traffic lights
        self.traffic_actors = self._world.get_actors().filter("*traffic_light*")
        self.scenario_index = 0
        CarlaDataProvider.set_world(self._world)

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
                obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
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
                obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
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
                obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
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

        elif self.config["input_type"] == 'wp_obs_info_side_obs_info_speed_steer_ldist_light':
            speed = agent.episode_measurements['speed'] / 10
            obstacle_dist = agent.episode_measurements['obstacle_dist']
            obstacle_speed = agent.episode_measurements['obstacle_speed']
            obstacle_dist_left = agent.episode_measurements['obstacle_dist_left']
            obstacle_speed_left = agent.episode_measurements['obstacle_speed_left']
            obstacle_dist_right = agent.episode_measurements['obstacle_dist_right']
            obstacle_speed_right = agent.episode_measurements['obstacle_speed_right']
            steer = agent.episode_measurements['control_steer']
            ldist = agent.episode_measurements['dist_to_trajectory']
            light = agent.episode_measurements['red_light_dist']

            # normalization

            if obstacle_dist <= self.config['front_obs_proximity_threshold']:
                obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
            else:
                obstacle_dist = self.config['default_obs_traffic_val']

            if obstacle_dist_left <= self.config['front_obs_proximity_threshold']:
                obstacle_dist_left = obstacle_dist_left / self.config['front_obs_proximity_threshold']
            else:
                obstacle_dist_left = self.config['default_obs_traffic_val']

            if obstacle_dist_right <= self.config['front_obs_proximity_threshold']:
                obstacle_dist_right = obstacle_dist_right / self.config['front_obs_proximity_threshold']
            else:
                obstacle_dist_right = self.config['default_obs_traffic_val']

            if obstacle_speed != -1:
                obstacle_speed = obstacle_speed / 20
            else:
                obstacle_speed = self.config['default_obs_traffic_val']

            if obstacle_speed_left != -1:
                obstacle_speed_left = obstacle_speed_left / 20
            else:
                obstacle_speed_left = self.config['default_obs_traffic_val']

            if obstacle_speed_right != -1:
                obstacle_speed_right = obstacle_speed_right / 20
            else:
                obstacle_speed_right = self.config['default_obs_traffic_val']

            if light != -1:
                light /= self.config['traffic_light_proximity_threshold']
            else:
                light = self.config['default_obs_traffic_val']

            obs['observation'] = np.concatenate((np.array([agent.episode_measurements['next_orientation']]), np.array([obstacle_dist]), np.array([obstacle_speed]),
            np.array([obstacle_dist_left]), np.array([obstacle_speed_left]), np.array([obstacle_dist_right]), np.array([obstacle_speed_right]), np.array([speed]), np.array([steer]), np.array([ldist]), np.array([light])))

        elif self.config["input_type"] == 'wp_obs_more_info_speed_steer_ldist_light':
            feat_list = [agent.episode_measurements['next_orientation']]
            for suffix, sensor in agent.obstacle_sensor.items():
                obstacle_dist = agent.episode_measurements['obstacle_dist_{}'.format(suffix)]
                obstacle_speed = agent.episode_measurements['obstacle_speed_{}'.format(suffix)]
                # normalization
                if obstacle_dist <= sensor.max_distance:
                    obstacle_dist = obstacle_dist / sensor.max_distance
                else:
                    obstacle_dist = self.config['default_obs_traffic_val']

                if obstacle_speed != -1:
                    obstacle_speed = obstacle_speed / 20
                else:
                    obstacle_speed = self.config['default_obs_traffic_val']
                feat_list.extend([obstacle_dist, obstacle_speed])

            speed = agent.episode_measurements['speed'] / 10
            steer = agent.episode_measurements['control_steer']
            ldist = agent.episode_measurements['dist_to_trajectory']
            light = agent.episode_measurements['red_light_dist']

            if light != -1:
                light /= self.config['traffic_light_proximity_threshold']
            else:
                light = self.config['default_obs_traffic_val']

            feat_list.extend([speed, steer, ldist, light])

            obs['observation'] = np.array(feat_list)

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
                obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
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
                obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
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
                obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
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
                obstacle_dist = obstacle_dist / self.config['front_obs_proximity_threshold']
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

        elif self.config['input_type'] == 'transformer':
            sym_dict = self.fetch_symbolic_dict(agent)
            obs['observation'] = flatten_obs(sym_dict) # (1, 100, 8)

        elif self.config['input_type'] == 'wp_360_obstacle_speed_steer':
            speed = agent.episode_measurements['speed'] / 10
            steer = agent.episode_measurements['control_steer']
            ldist = agent.episode_measurements['dist_to_trajectory']
            light = agent.episode_measurements['red_light_dist']

            if light != -1:
                light /= self.config['traffic_light_proximity_threshold']
            else:
                light = self.config['default_obs_traffic_val']

            front_obs_vec = np.array([1.5, 1.5])
            front_obs_vel = np.array([1.5, 1.5])
            front_min_dist = 10000

            front_right_obs_vec = np.array([1.5, 1.5])
            front_right_obs_vel = np.array([1.5, 1.5])
            front_right_min_dist = 10000

            front_left_obs_vec = np.array([1.5, 1.5])
            front_left_obs_vel = np.array([1.5, 1.5])
            front_left_min_dist = 10000

            back_right_obs_vec = np.array([1.5, 1.5])
            back_right_obs_vel = np.array([1.5, 1.5])
            back_right_min_dist = 10000

            back_left_obs_vec = np.array([1.5, 1.5])
            back_left_obs_vel = np.array([1.5, 1.5])
            back_left_min_dist = 10000


            for id, obstacle_data in agent.episode_measurements['obstacle_sensor']['state'].items():
                # Compute dot product of obstacle vector with car vector
                normalized_obstacle_vector = obstacle_data['position'] / np.linalg.norm(obstacle_data['position'])
                # Dot product is simply the first element of the normalized vector
                dot_product = normalized_obstacle_vector[0]

                # Obstacle is in front of vehicle
                if dot_product > 0.995 and obstacle_data['distance'] < front_min_dist:
                    front_min_dist = obstacle_data['distance']
                    front_obs_vec = obstacle_data['position'] / self.config['vehicle_proximity_threshold']
                    front_obs_vel = obstacle_data['velocity'] / 20

                # Obstacle is in front right
                elif dot_product > 0 and obstacle_data['position'][1] > 0 and obstacle_data['distance'] < front_right_min_dist:
                    front_right_min_dist = obstacle_data['distance']
                    front_right_obs_vec = obstacle_data['position'] / self.config['vehicle_proximity_threshold']
                    front_right_obs_vel = obstacle_data['velocity'] / 20

                # Obstacle is in front left
                elif dot_product > 0 and obstacle_data['position'][1] < 0 and obstacle_data['distance'] < front_left_min_dist:
                    front_left_min_dist = obstacle_data['distance']
                    front_left_obs_vec = obstacle_data['position']  / self.config['vehicle_proximity_threshold']
                    front_left_obs_vel = obstacle_data['velocity'] / 20

                # Obstacle is in back right
                elif dot_product <= 0 and obstacle_data['position'][1] > 0 and obstacle_data['distance'] < back_right_min_dist:
                    back_right_min_dist = obstacle_data['distance']
                    back_right_obs_vec = obstacle_data['position']  / self.config['vehicle_proximity_threshold']
                    back_right_obs_vel = obstacle_data['velocity'] / 20

                # Obstacle is in back left
                elif dot_product <= 0 and obstacle_data['position'][1] < 0 and obstacle_data['distance'] < back_left_min_dist:
                    back_left_min_dist = obstacle_data['distance']
                    back_left_obs_vec = obstacle_data['position']  / self.config['vehicle_proximity_threshold']
                    back_left_obs_vel = obstacle_data['velocity'] / 20

            if(light != self.config['default_obs_traffic_val']):
                unnorm_obs_dist = front_obs_vec[0] * self.config['vehicle_proximity_threshold']
                unnorm_light = light * 20

                # If the light is further do nothing
                if(front_obs_vec[0] != self.config['default_obs_traffic_val'] and unnorm_light > unnorm_obs_dist):
                    pass
                else:
                    front_obs_vec = np.array([light, 0]) / 20.0
                    front_obs_vel = np.array([0,0])

            obs['observation'] = np.concatenate(
                (
                    np.array([agent.episode_measurements['next_orientation']]),
                    np.array([speed]),
                    np.array([steer]),
                    np.array([ldist]),
                    np.array([front_obs_vec[0]]),
                    np.array([front_obs_vec[1]]),
                    np.array([front_obs_vel[0]]),
                    np.array([front_obs_vel[1]]),
                    np.array([front_right_obs_vec[0]]),
                    np.array([front_right_obs_vec[1]]),
                    np.array([front_right_obs_vel[0]]),
                    np.array([front_right_obs_vel[1]]),
                    np.array([front_left_obs_vec[0]]),
                    np.array([front_left_obs_vec[1]]),
                    np.array([front_left_obs_vel[0]]),
                    np.array([front_left_obs_vel[1]]),
                    np.array([back_right_obs_vec[0]]),
                    np.array([back_right_obs_vec[1]]),
                    np.array([back_right_obs_vel[0]]),
                    np.array([back_right_obs_vel[1]]),
                    np.array([back_left_obs_vec[0]]),
                    np.array([back_left_obs_vec[1]]),
                    np.array([back_left_obs_vel[0]]),
                    np.array([back_left_obs_vel[1]]),
                )
            )
            #print(obs['observation'])
            #exit()


    def step(self, action=None):
        # action is for stablebaseline
        try:
            # if self.config['test_comparison']:
            #     self._step_test_comparison(action)
            #     return None
            # elif self.config['algo'] == 'A2C':
                # new_obs, reward, done, ep_info = self._step(action[0])
                # return [new_obs], [reward], [done], [ep_info]
            self.list_step(action=action) # action here will be an action list

            if self.config['algo'] == 'stable_baseline_sac':
                agt = self.ego_agent_list[0]
                _rwd = agt.step_reward if hasattr(agt, 'step_reward') else 0
                return agt.observation, _rwd, agt.done, agt.episode_measurements

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

    def list_step(self, action=None):
        # action_list here should be a list of action
        self.world_frame = None

        for rk, agent in enumerate(self.ego_agent_list):
            if action is not None: agent.action = action # for stablebaseline
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
                        print('[step {}][agent {}][steer {:.2f}][throttle {:.2f}][break {:.2f}][reverse {}][speed {:.2f}]'.format(
                            agent.curr_ep_num_steps, agent.rank, control.steer, control.throttle, control.brake, control.reverse,
                            self.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6))
                    elif self.config['test_verbose']:
                        print('[step {}][agent {}][steer {:.2f}][speed {:.2f}]'.format(
                            agent.curr_ep_num_steps, agent.rank, control.steer,
                            self.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6))
                        # print("steer", control.steer, "throttle", control.throttle, "brake", control.brake,
                        #     "reverse", control.reverse)
                        # print("steps", agent.curr_ep_num_steps)

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
            CarlaDataProvider.on_carla_tick()
            self.world_frame = self._world.tick()
            ########################################################################################
            for idx, agent in enumerate(self.ego_agent_list):
                if agent.done or agent.action is None: continue
                if 'challenge' in self.config['scenarios']:
                    agent.running.scenario.scenario_tree.tick_once()
                agent.episode_measurements['num_steps'] = agent.curr_ep_num_steps
                # Set state variables for reward calculation
                agent.episode_measurements['num_collisions'] = agent.collision_sensor.num_collisions
                agent.episode_measurements['collision_actor_id'] = agent.collision_sensor.actor_id
                agent.episode_measurements['collision_actor_type'] = agent.collision_sensor.actor_type
                if self.config['enable_lane_invasion_termination']:
                    if self.config["enable_lane_invasion_sensor"]:
                        # if 'challenge' in self.config['scenarios']:
                        #     agent.episode_measurements['num_laneintersections'] = agent.lane_invasion_sensor.num_laneintersections and \
                        #         RoadOption.CHANGELANELEFT not in set(agent.next_road_opts) and \
                        #         RoadOption.CHANGELANERIGHT not in set(agent.next_road_opts)
                        #     agent.episode_measurements['out_of_road'] = int(agent.lane_invasion_sensor.out_of_road) and \
                        #         RoadOption.CHANGELANELEFT not in set(agent.next_road_opts) and \
                        #         RoadOption.CHANGELANERIGHT not in set(agent.next_road_opts)
                        # else:
                        #     agent.episode_measurements['num_laneintersections'] = agent.lane_invasion_sensor.num_laneintersections
                        #     agent.episode_measurements['out_of_road'] = agent.lane_invasion_sensor.out_of_road
                        agent.episode_measurements['num_laneintersections'] = agent.lane_invasion_sensor.num_laneintersections
                        agent.episode_measurements['unlawful_lane_change'] = agent.episode_measurements['num_laneintersections'] > \
                            agent.prev_measurement['num_laneintersections']
                        agent.episode_measurements['out_of_road'] = agent.lane_invasion_sensor.out_of_road
                        # skip lane changing
                        # print('[945]', [opt.value for opt in agent.next_road_opt_queue])
                        next_opts = set(agent.next_road_opt_queue)
                        for opt in next_opts:
                            # NOTE: not sure the reason but here should use .name to compare
                            if opt.name != RoadOption.CHANGELANELEFT.name and \
                                opt.name != RoadOption.CHANGELANERIGHT.name:
                                continue
                            # if not continued
                            # print('[953]', opt, opt.name, RoadOption.LANEFOLLOW.name, opt == RoadOption.LANEFOLLOW)
                            # print('[952] permitted offlane')
                            agent.episode_measurements['unlawful_lane_change'] = False
                        # print('[937]', agent.next_road_opts)
                        # if RoadOption.CHANGELANELEFT in agent.next_road_opts or \
                            #  RoadOption.CHANGELANERIGHT in agent.next_road_opts:
                            #  print('>>>>>>> [939]', agent.next_road_opts)
                        # if agent.episode_measurements['num_laneintersections'] > 0:
                        #     agent.episode_measurements['offlane_steps'] += 1
                        # else:
                        #     agent.episode_measurements['offlane_steps'] = 0
                    else:
                        self._update_lane_invasion_info_via_privilege(agent)

                agent.location = agent.vehicle_actor.get_location()
                agent.episode_measurements['distance_to_goal'] = agent.location.distance(agent.destination_transform.location)
                if agent.episode_measurements['min_distance_to_goal'] >= agent.location.distance(agent.destination_transform.location):
                    agent.episode_measurements['min_distance_to_goal'] = agent.location.distance(agent.destination_transform.location)
                agent.episode_measurements['speed'] = self.get_speed_from_velocity(agent.vehicle_actor.get_velocity())

                # next_orientation, agent.dist_to_trajectory, distance_to_goal_trajec, \
                #     agent.next_waypoints, agent.next_wp_angles, agent.next_wp_vectors = \
                #     agent.global_planner.get_next_orientation_new(agent.vehicle_actor.get_transform())

                # agent.episode_measurements['next_orientation'] = next_orientation
                # agent.episode_measurements['distance_to_goal_trajec'] = distance_to_goal_trajec
                # agent.episode_measurements['dist_to_trajectory'] = agent.dist_to_trajectory

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

                if (agent.episode_measurements['is_collision'] or agent.episode_measurements['runover_light']) and self.config["verbose"]:
                    print("Collisions Total: {}, Vehicle: {}, Static: {}".format(self.total_collisions, self.vehicle_collisions, self.static_collisions))
                    print("Traffic Light Violations: {}".format(self.traffic_light_violations))

                done = self._compute_done_condition(agent)
                # print('[agent {}] 677'.format(agent.rank), agent.episode_measurements['initial_dist_to_red_light'])
                agent.episode_measurements['done'] = done
                agent.done = bool(done)
                if agent.done and 'challenge' in self.config['scenarios']:
                    _record = agent.stats.compute_route_statistics(agent.scenario_config)
                    print(_record.infractions, _record.scores)
                    if len(agent.stats._registry_route_records) == self.config['num_episodes']:
                        _glb_record = self.statistics_manager.compute_global_statistics(self.config['num_episodes'])
                        print('global statistics:\n{}\n{}'.format(_glb_record.infractions, _glb_record.scores))
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
            if self.config['input_type'] == 'wp_obs_info_side_obs_info_speed_steer_ldist_light':
                self._update_obs_detector_via_privilege(agent)
            elif self.config['input_type'] == 'wp_360_obstacle_speed_steer':
                self._update_obs_detector_via_all_sensor(agent)
            elif self.config['enable_obstacle_sensor']:
                self._update_obs_detector_via_sensor(agent)
            else:
                self._update_obs_detector_via_privilege(agent)

        if not self.config['disable_traffic_light']:
            self._update_traffic_light_states(agent)

            # if self.config['verbose']:
            #     print('[agent {}] light info:'.format(agent.rank),
            #         agent.episode_measurements['dist_to_light'],
            #         agent.episode_measurements['nearest_traffic_actor_id'],
            #         agent.episode_measurements['nearest_traffic_actor_state'],
            #         agent.episode_measurements['initial_dist_to_red_light'],
            #         agent.episode_measurements['red_light_dist'])


    def _update_obs_detector_via_privilege(self, agent):
        agent.episode_measurements['obstacle_visible'] = False
        agent.episode_measurements['obstacle_orientation'] = -1

        agent.episode_measurements['obstacle_dist_left'] = -1
        agent.episode_measurements['obstacle_dist_right'] = -1
        agent.episode_measurements['obstacle_speed_left'] = -1
        agent.episode_measurements['obstacle_speed_right'] = -1

        min_obs_distance = 100000000
        found_obstacle = False
        for target_vehicle in self._world.get_actors():
            # do not account for the ego vehicle
            try:
                if target_vehicle is None or hasattr(target_vehicle, 'done') and target_vehicle.done: continue
                if target_vehicle.id == agent.id or 'vehicle' not in target_vehicle.type_id:
                    # skip self and non-vehicular
                    continue

                # if the object is not in our lane it's not an obstacle
                target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
                # check front obstacle
                d_bool, d_angle, distance = self.is_within_distance_ahead(target_vehicle.get_transform(),
                                            agent.vehicle_actor.get_transform(),
                                            self.config['front_obs_proximity_threshold'])


                side_bool, side_dist, side_orient = self._is_in_neighboring_lane(
                    target_vehicle.get_transform(),
                    agent.vehicle_actor.get_transform(),
                    self.config['front_obs_proximity_threshold'],
                )

                if side_orient == -1: # left
                    if agent.episode_measurements['obstacle_dist_left'] == -1 or \
                        side_dist < agent.episode_measurements['obstacle_dist_left']:
                        agent.episode_measurements['obstacle_dist_left'] = side_dist
                        agent.episode_measurements['obstacle_speed_left'] = \
                            self.get_speed_from_velocity(target_vehicle.get_velocity())
                elif side_orient == 1:
                    if agent.episode_measurements['obstacle_dist_right'] == -1 or \
                        side_dist < agent.episode_measurements['obstacle_dist_right']:
                        agent.episode_measurements['obstacle_dist_right'] = side_dist
                        agent.episode_measurements['obstacle_speed_right'] = \
                            self.get_speed_from_velocity(target_vehicle.get_velocity())

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

        for suffix in agent.obstacle_sensor:
            agent.episode_measurements['obstacle_dist_{}'.format(suffix)] = -1
            agent.episode_measurements['obstacle_speed_{}'.format(suffix)] = -1

        # front obstacle detection
        found_obstacle = False
        same_lane = True
        if agent.obstacle_sensor['front'].frame == self.world_frame:
            if self.config['verbose']: print('FRAME:', self.world_frame, agent.obstacle_sensor['front'].frame)
            obstacle_actor = agent.obstacle_sensor['front'].obstacle_actor
            if 'vehicle' in obstacle_actor.type_id:
                same_lane = check_if_vehicle_in_same_lane(agent.vehicle_actor, obstacle_actor, agent.next_waypoints, self._map)
            found_obstacle = True
            agent.episode_measurements['obstacle_visible'] = True
            agent.episode_measurements['obstacle_dist'] = agent.episode_measurements['obstacle_dist_front'] = agent.obstacle_sensor['front'].distance
            # if 'vehicle' in obstacle_actor.type_id:
            if hasattr(obstacle_actor, 'get_velocity'):
                if self.config['obs_cosine_velocity']:
                    cos = self.cosine_between_obs(obstacle_actor.get_velocity(), agent.vehicle_actor.get_velocity())
                    if cos < 0: cos = 0.
                else:
                    cos = 1.
                agent.episode_measurements['obstacle_speed'] = agent.episode_measurements['obstacle_speed_front'] = self.get_speed_from_velocity(obstacle_actor.get_velocity()) * cos
                # cos = self.cosine_between_velocities(obstacle_actor.get_velocity(), agent.vehicle_actor.get_velocity())
                # print('API test COS', cos)
            else:
                agent.episode_measurements['obstacle_speed'] = agent.episode_measurements['obstacle_speed_front'] = -1
            found_obstacle = found_obstacle and (not self.config['check_obs_same_lane'] or same_lane)
            # for weak_verbose
            if agent.episode_measurements['obstacle_init_id'] != obstacle_actor.id: # initial
                agent.episode_measurements['obstacle_init_id'] = obstacle_actor.id
                agent.episode_measurements['obstacle_init_dist'] = agent.obstacle_sensor['front'].distance
                if self.config['weak_verbose'] and not self.config['verbose'] and not self.config['test_verbose']:
                    print('[step {}][obstacle actor id {}][{}][agent {}][agt speed {:.2f}][obs speed {:.2f}][init dist {:.2f}][curr dist {:.2f}][same_lane {}][found {}]'.format(
                        agent.curr_ep_num_steps, obstacle_actor.id, obstacle_actor.type_id, agent.rank, agent.episode_measurements['speed'] * 3.6, agent.episode_measurements['obstacle_speed'] * 3.6,
                        agent.episode_measurements['obstacle_init_dist'], agent.obstacle_sensor['front'].distance, same_lane, found_obstacle))
            if self.config['verbose'] or self.config['test_verbose']:
                print('[step {}][obstacle actor id {}][{}][agent {}][agt speed {:.2f}][obs speed {:.2f}][init dist {:.2f}][curr dist {:.2f}][same_lane {}][found {}]'.format(
                    agent.curr_ep_num_steps, obstacle_actor.id, obstacle_actor.type_id, agent.rank, agent.episode_measurements['speed'] * 3.6, agent.episode_measurements['obstacle_speed'] * 3.6,
                    agent.episode_measurements['obstacle_init_dist'], agent.obstacle_sensor['front'].distance, same_lane, found_obstacle))
            # if only detect vehicular obstacle
            if self.config['obs_sensor_vehicle_only'] and 'vehicle' not in obstacle_actor.type_id:
                found_obstacle = False

        if not found_obstacle:
            agent.episode_measurements['obstacle_visible'] = False
            agent.episode_measurements['obstacle_dist'] = -1
            agent.episode_measurements['obstacle_speed'] = -1
            agent.episode_measurements['obstacle_dist_front'] = -1
            agent.episode_measurements['obstacle_speed_front'] = -1
            agent.episode_measurements['obstacle_init_dist'] = -1
            agent.episode_measurements['obstacle_init_id'] = -1

        # for other obstacles
        for suffix in agent.obstacle_sensor:
            found_obstacle = False
            same_lane = True
            if suffix == 'front': continue
            if agent.obstacle_sensor[suffix].frame == self.world_frame:
                if self.config['verbose']: print('FRAME:', self.world_frame, agent.obstacle_sensor[suffix].frame)
                obstacle_actor = agent.obstacle_sensor[suffix].obstacle_actor
                if 'vehicle' in obstacle_actor.type_id:
                    same_lane = check_if_vehicle_in_same_lane(agent.vehicle_actor, obstacle_actor, agent.next_waypoints, self._map)
                found_obstacle = True
                agent.episode_measurements['obstacle_dist_{}'.format(suffix)] = agent.obstacle_sensor[suffix].distance
                # if 'vehicle' in obstacle_actor.type_id:
                # if hasattr(obstacle_actor, 'get_velocity') and 'vehicle' not in obstacle_actor.type_id:
                if hasattr(obstacle_actor, 'get_velocity'):
                    agent.episode_measurements['obstacle_speed_{}'.format(suffix)] = self.get_speed_from_velocity(obstacle_actor.get_velocity())
                else:
                    agent.episode_measurements['obstacle_speed_{}'.format(suffix)] = -1
                found_obstacle = found_obstacle and (not self.config['check_obs_same_lane'] or not same_lane)
            if not found_obstacle:
                agent.episode_measurements['obstacle_speed_{}'.format(suffix)] = -1
                agent.episode_measurements['obstacle_speed_{}'.format(suffix)] = -1


    def _update_obs_detector_via_all_sensor(self, agent):
        sensor_readings = {}
        obstacle_set = set()
        for suffix in agent.obstacle_sensor:
            #for idx, k in enumerate(self.sensor_names):
            if(not 'obstacle_sensor' in sensor_readings):

                sensor_readings['obstacle_sensor'] = {
                    "state" : {}
                }


            other_actor, distance = agent.obstacle_sensor[suffix]._read_data()
            if other_actor is not None:
                # Get the current number of actors to see if we added a new unique actor
                num_actors = len(obstacle_set)

                # Add actor id to the set of ids
                obstacle_set.add(other_actor.id)

                # If new actor, get transform and velocity
                if(len(obstacle_set) > num_actors):
                    # Get the pose, and velocity of the new actor relative to our vehicle

                    # Get the obstacle sensor readings
                    ego_actor = agent.vehicle_actor
                    ego_velocity = ego_actor.get_velocity()
                    ego_velocity = np.array([ego_velocity.x, ego_velocity.y, ego_velocity.z])
                    ego_inverse_matrix = np.array(ego_actor.get_transform().get_inverse_matrix())

                    # Get transform of other object
                    other_transform = other_actor.get_transform()
                    other_velocity = other_actor.get_velocity()
                    other_velocity = np.array([other_velocity.x, other_velocity.y, other_velocity.z])

                    # Get the relative transform of the new actor
                    relative_transform = ego_inverse_matrix @ np.array(other_transform.get_matrix())

                    # Extract relative position
                    x = relative_transform[0, 3]
                    y = relative_transform[1, 3]


                    # Compute relative velocity
                    relative_velocity = ego_inverse_matrix[0:3,0:3] @ (other_velocity - ego_velocity)
                    vel_x = relative_velocity[0]
                    vel_y = relative_velocity[1]

                    sensor_readings['obstacle_sensor']["state"][other_actor.id] = {
                        "position" : np.array([x,y]),
                        "velocity" : np.array([vel_x, vel_y]),
                        "distance" : distance
                    }

        agent.episode_measurements['obstacle_sensor'] = sensor_readings['obstacle_sensor']


    def _update_traffic_light_states(self, agent):
        # TODO: Pass correct target waypoint to find_nearest_traffic_light() for US style traffic.
        traffic_actor, dist, traffic_light_orientation = agent.find_nearest_traffic_light(self.traffic_actors)
        found_redlight = False
        if traffic_light_orientation is not None:
            agent.episode_measurements['traffic_light_orientation'] = traffic_light_orientation
        else:
            agent.episode_measurements['traffic_light_orientation'] = -1

        # if agent.vehicle_actor.is_at_traffic_light():
        #     lt_actor = agent.vehicle_actor.get_traffic_light()
        #     agt_loc = agent.vehicle_actor.get_location()
        #     lt_loc = lt_actor.get_location()
        #     print('API test', lt_actor, agent.vehicle_actor.get_traffic_light_state(), lt_loc.distance(agt_loc))
        if traffic_actor is not None:
            if traffic_actor.state != carla.TrafficLightState.Green:
                agent.episode_measurements['red_light_dist'] = dist
                found_redlight = True
                # print('[agent {} init {}] traffic light info'.format(
                #         agent.rank, agent.episode_measurements['initial_dist_to_red_light']), traffic_actor.id, traffic_actor.state, dist)
                if agent.episode_measurements['initial_dist_to_red_light'] == -1 or \
                    (agent.episode_measurements['nearest_traffic_actor_id'] != -1 and traffic_actor.id != agent.episode_measurements['nearest_traffic_actor_id']):
                    if dist < self.config['min_dist_from_red_light']:
                        agent.episode_measurements['red_light_dist'] = -1
                        agent.episode_measurements['initial_dist_to_red_light'] = -1
                        found_redlight = False
                    else:
                        agent.episode_measurements['initial_dist_to_red_light'] = dist
                        found_redlight = True
                    if self.config['weak_verbose'] and not self.config['verbose'] and not self.config['test_verbose']:
                        print('[step {}][traffic light id {}][agent {}][speed {:.2f}][init dist {:.2f}][curr dist {:.2f}][state {}][found {}]'.format(agent.curr_ep_num_steps,
                            traffic_actor.id, agent.rank, agent.episode_measurements['speed'] * 3.6, agent.episode_measurements['initial_dist_to_red_light'],
                            dist, traffic_actor.state, found_redlight))
                if self.config['verbose'] or self.config['test_verbose']:
                    print('[step {}][traffic light id {}][agent {}][speed {:.2f}][init dist {:.2f}][curr dist {:.2f}][state {}][found {}]'.format(agent.curr_ep_num_steps,
                        traffic_actor.id, agent.rank, agent.episode_measurements['speed'] * 3.6, agent.episode_measurements['initial_dist_to_red_light'],
                        dist, traffic_actor.state, found_redlight))
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
        _upd_town = self.curr_town
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
        elif self.config["scenarios"] == "challenge_train_scenario":
            self.source_transform, self.destination_transform, self.wps_list, _upd_town = scenarios.get_leaderboard_route(
                unseen, curr_town=self.curr_town, index=index, max_idx=self.config["min_num_eps_before_switch_town"],
                # avail_map_list=self.avail_map.keys(), mode='train')
                avail_map_list=['Town01', 'Town03'], mode='train')
        elif self.config["scenarios"] == "challenge_test_scenario":
            ######### Not the actual number of eposiodes.
            if self.curr_town == 'Town05':
                self.config['num_episodes'] = 10
            elif self.curr_town == 'Town02':
                self.config['num_episodes'] = 6
            else:
                self.config['num_episodes'] = None
            self.source_transform, self.destination_transform, self.wps_list, _upd_town = scenarios.get_leaderboard_route(
                unseen, curr_town=self.curr_town, index=index, max_idx=1,
                # avail_map_list=self.avail_map.keys(), mode='test')
                # avail_map_list=['Town02', 'Town05'], mode='test')
                avail_map_list=[self.curr_town], mode='test')
                # avail_map_list=['Town02'], mode='test')
        elif self.config["scenarios"] == "leaderboard_navigation":
            self.source_transform, self.destination_transform, self.wps_list, _upd_town = scenarios.get_leaderboard_route(
                unseen, curr_town=self.curr_town, index=index, max_idx=self.config["min_num_eps_before_switch_town"],
                # avail_map_list=['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07'], mode='train')
                # avail_map_list=['Town01', 'Town03'], mode='train')
                # avail_map_list=[self.curr_town], mode='train')
                avail_map_list=self.config['avail_town_list'], mode='train')
        else:
            raise ValueError("Scenarios Config not set!")

        if _upd_town != self.curr_town: # switch to a new town
            print('[1060] update town from {} to {}'.format(self.curr_town, _upd_town), self.scenario_index)
        #     if self.config['num_agents'] != 1:
        #         # self.reset_env()
        #         self.reset(rank_list=list(range(self.config['num_agents'])), reset_npc=True)
        #     self._set_world_and_map(_upd_town)
        return _upd_town

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
            throttle = agent.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config["action_type"] == "throttle_only":
            steer = float(0.0)
            target_speed = float(np.clip(action[0], 0, self.target_speed))
            current_speed = self.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
            throttle = agent.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config["action_type"] == "merged_speed":
            # steer = float(action[0])
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(np.clip(action[1] + 10.0, 0, self.target_speed))
            current_speed = self.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
            gas = agent.controller.pid_control(target_speed, current_speed, enable_brake=self.config["enable_brake"])
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config["action_type"] == "merged_speed_tanh":
            # steer = float(action[0])
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(np.clip((action[1] + 1) * 10.0, 0, self.target_speed))
            current_speed = self.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
            gas = agent.controller.pid_control(target_speed, current_speed, enable_brake=self.config["enable_brake"])
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config["action_type"] == "merged_speed_scaled_tanh":
            # steer = np.clip(float(action[0]), -1.0, 1.0)
            # steer = np.clip(float(action[0]), -.25, .25)
            # steer = np.clip(float(action[0]), -0.5, 0.5)
            steer = np.clip(float(action[0]) * self.config['steering_scale'], -1, 1)
            # target_speed = (action[1] * 1.5) + 1.5
            target_speed = (action[1] * 1.5) + 1
            target_speed = float(np.clip(target_speed * self.target_speed / 2, 0, self.target_speed))

            ##################################
            # if use autopilot
            if hasattr(agent, 'autopilot') and agent.autopilot:
                if agent.episode_measurements['red_light_dist'] != -1 or \
                    agent.episode_measurements['obstacle_dist'] != -1:
                    target_speed = 0
                else:
                    target_speed = self.target_speed
                steer = agent.steer_controller.pid_control(
                    agent.next_waypoints[0], agent.vehicle_actor.get_transform())
                steer = np.clip(steer, -1., 1.)
                #!!! modify agent.action
                agent.action = np.array([steer, (2 * target_speed / self.target_speed - 1) / 1.5])

            if hasattr(agent, 'transfuser_autopilot') and agent.transfuser_autopilot:
                # steer, target_speed = agent.transfuser_agent.run_step()
                steer, target_speed, control = agent.transfuser_agent.run_step()
                # print('[1513]', steer, target_speed)
                #!!! modify agent.action
                target_speed *= 3.6
                agent.action = np.array([steer, (2 * target_speed / self.target_speed - 1) / 1.5])
                agent.episode_measurements["target_speed"] = target_speed
                return control

            current_speed = self.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
            gas = agent.controller.pid_control(target_speed, current_speed, enable_brake=self.config["enable_brake"])
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
            gas = agent.controller.pid_control(target_speed, current_speed, enable_brake=self.config["enable_brake"])
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
            gas = agent.controller.pid_control(target_speed, current_speed, enable_brake=self.config["enable_brake"])
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
        if not rank_list: rank_list = [0]
        # if self.config['test_comparison']:
        #     return self._reset_test_comparison(unseen, index)
        # elif self.config['algo'] == 'A2C':
        self.list_reset(use_idx=use_idx, idx_list=idx_list, rank_list=rank_list, reset_npc=reset_npc)
        if self.config['algo'] == 'stable_baseline_sac':
            self.reset_vehicle_agent([DummyAgent(self.ego_vehicle_list[0])])
            obs, _, _, _ = self.step()
            return obs
        # else:
        #     return self._reset(unseen, index)

    def reset_env(self,):
        # CarlaDataProvider.cleanup()
        # self.statistics_manager.scenario = None
        self.destroy_all_existing_npc_actors()
        self.destroy_all_existing_ego_agents()
        self.ego_vehicle_list = [None] * self.config['num_agents']
        self.ego_agent_list = [None] * self.config['num_agents']
        self.curr_num_agents = 0
        self.world_frame = 0
        self.last_npc_reset_frame = 0

    def destroy_all_existing_npc_actors(self):
        # Delete all existing actors
        for _ in range(len(self.actor_list)):
            try:
                # actor = self.actor_list.pop()
                # actor.destroy()
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.actor_list.clear()
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
        if hasattr(agent, 'transfuser_agent'):
            try:
                agent.transfuser_agent.cleanup()
            except Exception as e:
                print("Error during destroying transfuser actor")
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

    def reset_vehicle_agent(self, agent_list, transfuser=False):
        # bind new agent
        for agent in agent_list:
            self.ego_agent_list[agent.rank] = agent

            # set attributes
            agent._proximity_threshold = self.config['traffic_light_proximity_threshold']
            agent._traffic_light_proximity_threshold = self.config['traffic_light_proximity_threshold']
            agent._front_obs_proximity_threshold = self.config['front_obs_proximity_threshold']

            agent.image_data = None
            agent.source_transform = agent.vehicle_actor.source_transform
            agent.destination_transform = agent.vehicle_actor.destination_transform
            # agent.scenario_route = None
            agent.global_planner = agent.vehicle_actor.global_planner
            if 'challenge' in self.config['scenarios']:
                agent.running = agent.vehicle_actor.running
                agent.stats = agent.vehicle_actor.stats
                agent.scenario_config = agent.vehicle_actor.scenario_config
            # agent.trace_route = None
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
            agent.next_waypoint_queue = deque(maxlen=12)
            agent.next_road_opts = None
            agent.next_road_opt_queue = deque(maxlen=12)

            agent.next_wp_vectors = None
            agent.next_wp_angles = None

            agent.episode_measurements['dist_to_light'] = -1
            agent.episode_measurements['nearest_traffic_actor_id'] = -1
            agent.episode_measurements['nearest_traffic_actor_state'] = None
            agent.episode_measurements['initial_dist_to_red_light'] = -1
            agent.episode_measurements['red_light_dist'] = -1
            agent.episode_measurements['traffic_light_orientation'] = -1
            agent.episode_measurements["runover_light"] = False
            # agent.episode_measurements['offlane_steps'] = 0
            agent.episode_measurements['obstacle_init_dist'] = -1
            agent.episode_measurements['obstacle_init_id'] = -1

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

            # reset controller
            agent.controller = controller.PIDLongitudinalController(
                K_P=self.args_longitudinal_dict['K_P'],
                K_D=self.args_longitudinal_dict['K_D'],
                K_I=self.args_longitudinal_dict['K_I'],
                dt=self.args_longitudinal_dict['dt'])
            agent.steer_controller = controller.PIDLateralController(
                K_P=self.args_lateral_dict['K_P'],
                K_D=self.args_lateral_dict['K_D'],
                K_I=self.args_lateral_dict['K_I'],
                dt=self.args_lateral_dict['dt'])


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
            # rv_camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0.0))
            rv_camera_transform = carla.Transform(carla.Location(x=13.0, z=18.0), carla.Rotation(pitch=270.0))

            agent.rv_camera_actor = self._world.spawn_actor(rv_camera, rv_camera_transform, attach_to=agent.vehicle_actor)
            agent.actor_list.append(agent.rv_camera_actor)

            agent.rv_camera_actor.listen(agent.rv_camera_queue.put)

            agent.collision_sensor = sensors.CollisionSensor(agent.vehicle_actor)
            agent.actor_list.append(agent.collision_sensor.sensor)


            if self.config["enable_lane_invasion_sensor"]:
                agent.lane_invasion_sensor = sensors.LaneInvasionSensor(agent.vehicle_actor)
                agent.actor_list.append(agent.lane_invasion_sensor.sensor)

            if self.config["enable_obstacle_sensor"]:
                # agent.obstacle_sensor = sensors.ObstacleSensor(agent.vehicle_actor,
                #     distance=self.config['front_obs_proximity_threshold'],
                #     hit_radius=self.config['front_obs_sensor_hit_radius'],)

                # agent.actor_list.append(agent.obstacle_sensor.sensor)
                if self.config['input_type'] == 'wp_obs_more_info_speed_steer_ldist_light':
                    obs_sensors = {
                        'front': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['front_obs_proximity_threshold'],
                            hit_radius=self.config['front_obs_sensor_hit_radius'],),
                        'front_right': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['side_obs_proximity_threshold'],
                            hit_radius=self.config['side_obs_sensor_hit_radius'],
                            transform=carla.Transform(rotation=carla.Rotation(yaw=45.))),
                        'back_right': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['side_obs_proximity_threshold'],
                            hit_radius=self.config['side_obs_sensor_hit_radius'],
                            transform=carla.Transform(rotation=carla.Rotation(yaw=135.))),
                        'back_left': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['side_obs_proximity_threshold'],
                            hit_radius=self.config['side_obs_sensor_hit_radius'],
                            transform=carla.Transform(rotation=carla.Rotation(yaw=225.))),
                        'front_left': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['side_obs_proximity_threshold'],
                            hit_radius=self.config['side_obs_sensor_hit_radius'],
                            transform=carla.Transform(rotation=carla.Rotation(yaw=315.))),
                    }

                elif self.config['input_type'] == 'wp_360_obstacle_speed_steer':
                    obs_sensors = {
                        'obstacle_sensor_0': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=0.0,y=0.0,z=0.3),rotation=carla.Rotation(yaw=0))),
                        'obstacle_sensor_1': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=12))),
                        'obstacle_sensor_2': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=24))),
                        'obstacle_sensor_3': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=36))),
                        'obstacle_sensor_4': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=48))),
                        'obstacle_sensor_5': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=60))),
                        'obstacle_sensor_6': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=72))),
                        'obstacle_sensor_7': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=84))),
                        'obstacle_sensor_8': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=96))),
                        'obstacle_sensor_9': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=108))),
                        'obstacle_sensor_10': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=120))),
                        'obstacle_sensor_11': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=132))),
                        'obstacle_sensor_12': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=144))),
                        'obstacle_sensor_13': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=156))),
                        'obstacle_sensor_14': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=168))),
                        'obstacle_sensor_15': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=0.0,z=0.3),rotation=carla.Rotation(yaw=180))),
                        'obstacle_sensor_16': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=192))),
                        'obstacle_sensor_17': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=204))),
                        'obstacle_sensor_18': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=216))),
                        'obstacle_sensor_19': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=228))),
                        'obstacle_sensor_20': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=240))),
                        'obstacle_sensor_21': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=252))),
                        'obstacle_sensor_22': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=264))),
                        'obstacle_sensor_23': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=276))),
                        'obstacle_sensor_24': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=288))),
                        'obstacle_sensor_25': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=300))),
                        'obstacle_sensor_26': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=312))),
                        'obstacle_sensor_27': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=324))),
                        'obstacle_sensor_28': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=336))),
                        'obstacle_sensor_29': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=self.config['all_obs_hit_radius'],
                            transform=carla.Transform(location=carla.Location(x=1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=348))),
                        'obstacle_sensor_30': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=0.1,
                            transform=carla.Transform(location=carla.Location(x=1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=5))),
                        'obstacle_sensor_31': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=0.1,
                            transform=carla.Transform(location=carla.Location(x=1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=-5))),
                        'obstacle_sensor_32': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=0.1,
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=1.0,z=0.3),rotation=carla.Rotation(yaw=175))),
                        'obstacle_sensor_33': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['all_obs_proximity_threshold'],
                            hit_radius=0.1,
                            transform=carla.Transform(location=carla.Location(x=-1.5,y=-1.0,z=0.3),rotation=carla.Rotation(yaw=-175))),
                    }

                else:
                    obs_sensors = {
                        'front': sensors.ObstacleSensor(agent.vehicle_actor,
                            distance=self.config['front_obs_proximity_threshold'],
                            hit_radius=self.config['front_obs_sensor_hit_radius'],),
                    }

                agent.obstacle_sensor = {}
                for orient, sensor in obs_sensors.items():
                    agent.obstacle_sensor[orient] = sensor
                    agent.actor_list.append(sensor.sensor)

            # Set state variables for reward calculation
            # agent.episode_measurements['num_collisions'] = agent.collision_sensor.num_collisions
            # agent.episode_measurements['collision_actor_id'] = agent.collision_sensor.actor_id
            # agent.episode_measurements['collision_actor_type'] = agent.collision_sensor.actor_type
            agent.episode_measurements['num_collisions'] = 0
            agent.episode_measurements['collision_actor_id'] = -1
            agent.episode_measurements['collision_actor_type'] = None

            agent.episode_measurements['num_laneintersections'] = 0
            agent.episode_measurements['out_of_road'] = False
            agent.episode_measurements['unlawful_lane_change'] = False

            agent.location = agent.vehicle_actor.get_location()
            agent.episode_measurements['distance_to_goal'] = agent.location.distance(agent.destination_transform.location)
            agent.episode_measurements['min_distance_to_goal'] = 1000000.0
            agent.episode_measurements['speed'] = self.get_speed_from_velocity(agent.vehicle_actor.get_velocity())

            agent.episode_measurements['total_steps'] = agent.num_total_steps

            # add transfuser
            if transfuser:
                agent.transfuser_agent = AutoPilot()
                agent.transfuser_agent.set_global_plan(self.gps_route, self.gps_route)
                agent.transfuser_agent._init(agent.vehicle_actor)

        # Ticking for 15 frames to handle car initialization in air
        # time.sleep(.04)
        # for _ in range(3):
        #     # print(self.world_frame)
        #     self.world_frame = self._world.tick()

    def _is_static(self, agent):
        if type(agent.obstacle_sensor) == dict:
            for suffix in agent.obstacle_sensor:
                obstacle_key = 'obstacle_dist_{}'.format(suffix)
                if obstacle_key in agent.episode_measurements and \
                    agent.episode_measurements[obstacle_key] != -1:
                    return False
        if agent.episode_measurements['speed'] >= self.config['zero_speed_threshold']:
            return False
        if agent.episode_measurements['obstacle_dist'] != -1:
            return False
        if agent.episode_measurements['red_light_dist'] != -1:
            return False
        if  'obstacle_dist_left' in agent.episode_measurements and \
            agent.episode_measurements['obstacle_dist_left'] != -1:
            return False
        if  'obstacle_dist_right' in agent.episode_measurements and \
            agent.episode_measurements['obstacle_dist_right'] != -1:
            return False
        return True

    def _update_lane_invasion_info_via_privilege(self, agent):
        if agent.done: return
        # agent_bb_wp = get_vehicle_bb_wp(
        #     self._world.get_map(), agent.vehicle_actor,
        #     lane_type=(LaneType.Any | LaneType.NONE))
        agent_wp = self._map.get_waypoint(
            agent.vehicle_actor.get_location(),
            lane_type=(LaneType.Any | LaneType.NONE))

        # # definitely offroading
        # if agent_wp.lane_type == LaneType.NONE:
        #     agent.episode_measurements['out_of_road'] = True
        #     return
        if agent_wp.lane_type not in {
            LaneType.Driving,
            LaneType.Parking,
            LaneType.Biking,
            LaneType.Shoulder,
            LaneType.Entry,
            LaneType.Exit,
            LaneType.OffRamp,
            LaneType.OnRamp,
        }:
            agent.episode_measurements['out_of_road'] = True
            print('[1822] out of road', agent_wp.lane_type)
            return

        # skip intersections
        next_opts = set(agent.next_road_opts)
        for opt in next_opts:
            # NOTE: not sure the reason but here should use .name to compare
            if opt.name == RoadOption.LANEFOLLOW.name:
                continue
            # if not continued
            # print(opt, opt.name, RoadOption.LANEFOLLOW.name, opt == RoadOption.LANEFOLLOW)
            # print('[1824] permitted offlane')
            return

        # if not intersection, should stick on driving lane
        if agent_wp.lane_type != LaneType.Driving:
            agent.episode_measurements['out_of_road'] = True
            print('[1852] out of road', agent_wp.lane_type)
        else:
            num_same_road_wp = 0
            for next_wp in [agent.next_waypoints[0], agent.next_waypoints[-1]]:
                # print('[1834]', agent_wp.road_id, next_wp.road_id,
                #     agent_wp.lane_id, next_wp.lane_id)
                if agent_wp.road_id == next_wp.road_id:
                    num_same_road_wp += 1
                    if agent_wp.lane_id == next_wp.lane_id:
                        return
            # for bb_wp in agent_bb_wp:
            #     for next_wp in agent.next_waypoints:
            #         # print('[1834]', agent_wp.road_id, next_wp.road_id,
            #         #     agent_wp.lane_id, next_wp.lane_id)
            #         if bb_wp.road_id == next_wp.road_id:
            #             num_same_road_wp += 1
            #             if bb_wp.lane_id == next_wp.lane_id:
            #                 return
            # at intersection or something
            if num_same_road_wp == 0: return
        # if not returned, lane invasion happened
        agent.episode_measurements['num_laneintersections'] += 1
        if agent_wp.lane_change == LaneChange.NONE:
            agent.episode_measurements['unlawful_lane_change'] = True

    def _get_ego_input(self, agent):
        rv_image = self._read_data(agent.rv_camera_queue, self.world_frame)

        next_orientation, agent.dist_to_trajectory, distance_to_goal_trajec, \
            agent.next_waypoints, agent.next_wp_angles, agent.next_wp_vectors, agent.next_road_opts = \
            agent.global_planner.get_next_orientation_new(agent.vehicle_actor.get_transform(), append_road_opt=True)

        wp_opt = [(wp, opt) for wp, opt in zip(agent.next_waypoints, agent.next_road_opts)]
        new_wp_starting_idx = 0
        for idx, wp in enumerate(agent.next_waypoint_queue):
            if wp is agent.next_waypoints[0]:
                new_wp_starting_idx = len(agent.next_waypoint_queue) - idx
                break
        agent.next_waypoint_queue.extend(agent.next_waypoints[new_wp_starting_idx:])
        agent.next_road_opt_queue.extend(agent.next_road_opts[new_wp_starting_idx:])

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
        if self._is_static(agent):
            agent.episode_measurements['static_steps'] += 1
        else:
            agent.episode_measurements['static_steps'] = 0

        if self.config["scenarios"] == "straight_dynamic":
            self._update_straight_dynamic_obs(agent)

        obs['rv_image'] = agent.rv_image = rv_image
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
                                        'wp_angles_vecs_obs_info_speed_steer_ldist_light',
                                        'wp_obs_info_side_obs_info_speed_steer_ldist_light',
                                        'wp_obs_more_info_speed_steer_ldist_light','wp_360_obstacle_speed_steer']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            agent.observation = observation
        elif self.config['input_type'] == 'transformer':
            agent.observation = obs['observation']
        else:
            agent.observation = obs

        if self.config['verbose']: print('[agent {}] observation: {}'.format(agent.rank, agent.observation))

        return agent.observation

    def fetch_actor_features(self, actor):
        transform = actor.get_transform()
        velocity = actor.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

        bounding_box_loc = actor.bounding_box.get_world_vertices(transform)
        bounding_box = [(loc.x, loc.y) for loc in bounding_box_loc]

        return {
            'x': transform.location.x,
            'y': transform.location.y,
            'theta': transform.rotation.yaw,
            'speed': speed,
            'bounding_box': bounding_box
        }

    def normalize_actor_features(self, actor_features, ref, theta):
        """
        Normalize actor feature dictionary to reference point
        ref is a tuple (x, y, theta)
        """
        for i, (x,y) in enumerate(actor_features['bounding_box']):
            x,y = transform_to_pov((x,y), ref, theta)
            actor_features['bounding_box'][i] = (x,y)

        x,y = transform_to_pov((actor_features['x'], actor_features['y']), ref, theta)
        actor_features['x'], actor_features['y'] = x,y
        actor_features['theta'] = normalize_angle(actor_features['theta'] - theta)

    def fetch_symbolic_dict(self, ego_agent):
        # get ego kinematics
        ego_actor = ego_agent.vehicle_actor
        ego_features = self.fetch_actor_features(ego_actor)

        ref = ego_features['x'], ego_features['y']
        theta = ego_features['theta']

        self.normalize_actor_features(ego_features, ref, theta)

        # get other entities
        other_actors = self._world.get_actors().filter('*vehicle*')
        vehicle_features = {actor.id: self.fetch_actor_features(actor) for actor in other_actors
            if actor.get_transform().location.distance(ego_actor.get_transform().location) < 20
            and actor.id != ego_actor.id
        }

        for vehicle_id in vehicle_features:
            features = vehicle_features[vehicle_id]
            self.normalize_actor_features(features, ref, theta)

        # normalize waypoints
        # print(len(ego_agent.next_waypoints))
        waypoints = [
            (
                wp.transform.location.x,
                wp.transform.location.y,
                wp.transform.location.z,
            )
            for wp in ego_agent.next_waypoints
        ]
        for i, (x,y,_) in enumerate(waypoints):
            x,y = transform_to_pov((x,y), ref, theta)
            waypoints[i] = (x,y)

        features = {
            'ego_features': ego_features,
            'vehicle_features': vehicle_features,

            'light': ego_agent.episode_measurements['red_light_dist'],

            'next_waypoints': waypoints,
            'next_orientation': ego_agent.episode_measurements['next_orientation'],
            'dist_to_trajectory': ego_agent.episode_measurements['dist_to_trajectory'],

            'obstacle_dist': ego_agent.episode_measurements['obstacle_dist'],
            'obstacle_speed': ego_agent.episode_measurements['obstacle_speed'],

            'x': ref[0],
            'y': ref[1],
            'theta': theta
        }
        return features

    def list_reset(self, use_idx=False, idx_list=None, rank_list=None, reset_npc=False):
        # if not idx_list: idx_list = [0] * self.config['num_agents']
        try:
            vehicle_bp = self.blueprint_library.find(self.config['vehicle_type'])
            # vehicle_bp = self.blueprint_library.find(random.choice(self.config['vehicle_types']))
        except Exception as e:
            print("Error during vehicle creation: {}".format(traceback.format_exc()))

        # print('[1544]', self.config['npc_reset_freq'], self.world_frame, self.last_npc_reset_frame)

        if reset_npc or (self.config['npc_reset_freq'] and self.world_frame - self.last_npc_reset_frame > self.config['npc_reset_freq']):
            self.destroy_all_existing_npc_actors()

        for rk in rank_list:
            prev_agent = self.ego_agent_list[rk]
            self.ego_agent_list[rk] = None
            if prev_agent is not None: self.curr_num_agents -= 1
            try:
                self.destroy_an_existing_ego_agent(prev_agent)
            except:
                print('>>> [rank {}] Error when deleting prev_agent [agent {}]'.format(self.env_rank, rk))

            # Spawning vehicle actor with retry logic as it fails to spawn sometimes
            self.vehicle_actor = None
            NUM_RETRIES = 100
            for idx in range(1, NUM_RETRIES + 1):
                # Set source and destination based on scenario
                # Currently scenarios are defined only for Town01

                # if self.config["use_scenarios"] and (self.config['initial_town'] == "Town01" or self.config['initial_town'] == "Town02"):
                if self.config["use_scenarios"]:
                    if self.config["updated_scenarios"]:
                        self._set_updated_scenario(unseen=use_idx, index=self.scenario_index, town=self.config['initial_town'])
                    else:
                        _upd_town = self._set_scenario(unseen=use_idx, index=self.scenario_index, town=self.config['initial_town'])
                else:
                    self.source_transform, self.destination_transform = random.choice(self.spawn_points), random.choice(self.spawn_points)

                # self.vehicle_actor = self._world.try_spawn_actor(vehicle_bp, self.source_transform)
                try:
                    self.vehicle_actor = CarlaDataProvider.request_new_actor(self.config['vehicle_type'], self.source_transform, 'hero')
                except:
                    self.vehicle_actor = None

                if self.vehicle_actor is not None:
                    if _upd_town != self.curr_town: # switch to a new town
                        # print('[1511] update town from {} to {}'.format(self.curr_town, _upd_town), self.scenario_index)
                        # if self.config['num_agents'] != 1:
                        #     for rk in range(self.config['num_agents']):
                        #         if self.ego_agent_list[rk] is not None:
                        #             print(self.ego_agent_list[rk], self.ego_agent_list[rk].rank)
                        #             self.ego_agent_list[rk].done = True
                            # self.reset_env()
                        # self.reset(rank_list=list(range(self.config['num_agents'])), reset_npc=True)
                        # print('[2000]', self.curr_town)
                        self.reset_env()
                        # for actor in self._world.get_actors():
                        #     try:
                        #         actor.destroy()
                        #     except:
                        #         pass
                        # print('[2002]', self.curr_town)
                        self._set_world_and_map(_upd_town)
                        # print('[2004]', self.curr_town)
                    break
                else:
                    print("[rank {}][agt {}] Unable to spawn ego vehicle [trial {}] at ({:.2f}, {:.2f}).".format(
                        self.env_rank, rk, idx, self.source_transform.location.x, self.source_transform.location.y))
                    # print("Number of existing actors, {}".format(len(self.actor_list)))
                    # print("Number of existing ego agents, {}".format(self.curr_num_agents))
                    # time.sleep(.04)

            if self.vehicle_actor is not None:
                # print(self.vehicle_actor)
                self.ego_vehicle_list[rk] = self.vehicle_actor
                self.vehicle_actor.source_transform = self.source_transform
                self.vehicle_actor.destination_transform = self.destination_transform
                if self.config['verbose']:
                    print('########## agent {} ##########'.format(rk))
                    print('SRC TRANSFORM =', self.vehicle_actor.source_transform)
                    print('DST TRANSFORM =', self.vehicle_actor.destination_transform)
                self.curr_num_agents += 1
                # print('[2021]')

                self.vehicle_actor.global_planner = planner.GlobalPlanner()

                if 'challenge' in self.config["scenarios"]:
                    # print(213, len(self.wps_list), self.wps_list)
                    self.gps_route, self.route, self._global_plan_world_coord = interpolate_trajectory(self._world, self.wps_list)

                    # Print route in debug mode
                    # self._draw_waypoints(self._world, self.route, vertical_shift=1.0, persistency=500)
                    # print('self.route', self.route)
                    CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))
                    # print(222, len(self._global_plan_world_coord), self._global_plan_world_coord[0])
                    self.dense_waypoints = self._global_plan_world_coord

                    potential_scenarios_definitions, _ = RouteParser.scan_route_for_scenarios(
                        self.curr_town, self.route, self.world_annotations)
                    # Sample the scenarios to be used for this route instance.
                    self.sampled_scenarios_definitions = scenario_sampling(potential_scenarios_definitions)
                    # print(236, self.sampled_scenarios_definitions)
                    self.scenarios = build_scenario_instances(self._world, self.vehicle_actor, self.sampled_scenarios_definitions, debug_mode=1)
                    # print(244, self.scenarios)
                    if self.config['use_scenarios']:
                        self.vehicle_actor.running = Trigger(self._world, self.vehicle_actor, self.route, self.scenarios, debug_mode=1)
                    else:
                        self.vehicle_actor.running = Trigger(self._world, self.vehicle_actor, self.route, [], debug_mode=1)

                    # set statistics
                    self.vehicle_actor.stats = self.statistics_manager
                    # print('[2038]', self.scenario_index)
                    _record_idx = len(self.statistics_manager._registry_route_records)
                    self.vehicle_actor.stats.set_route('route_{}'.format(_record_idx), _record_idx)
                    # self.vehicle_actor.stats.set_route('curr_route', 0)
                    self.vehicle_actor.stats.set_scenario(self.vehicle_actor.running.scenario)
                    self.vehicle_actor.scenario_config = DummyScenarioConfig(_record_idx, self.wps_list)

                elif self.config["scenarios"] == 'leaderboard_navigation':
                    self.gps_route, self.route, self._global_plan_world_coord = interpolate_trajectory(self._world, self.wps_list)
                    self.dense_waypoints = self._global_plan_world_coord

                else:
                    self.dense_waypoints  = self.vehicle_actor.global_planner.trace_route(self._map,
                                            self.source_transform, self.destination_transform)
                    # print(self.dense_waypoints)
                # print('[2061]')
                # wp_list = []
                # for wp, _ in self.dense_waypoints:
                #     wp_list.append(wp)
                self._draw_waypoints(self._world, self.dense_waypoints)

                self.scenario_index += 1
                self.vehicle_actor.global_planner.set_global_plan(self.dense_waypoints)

            else:
                raise Exception("Failed in spawning vehicle actor.")

        if reset_npc:
            self.spawn_npc_vehicles()
            self.last_npc_reset_frame = self.world_frame
        elif self.config['npc_reset_freq'] and self.world_frame - self.last_npc_reset_frame > self.config['npc_reset_freq']:
            self.spawn_npc_vehicles()
            self.last_npc_reset_frame = self.world_frame


    def spawn_npc_vehicles(self, num_npc=None):
        self.destroy_all_existing_npc_actors()
        if num_npc is not None: # override all other situations
            self.spawn_npc(num_npc)
        elif self.config["sample_npc"]:
            self.spawn_npc(np.random.randint(low=self.config["num_npc_lower_threshold"],
                high=self.config["num_npc_upper_threshold"]))
        else:
            self.spawn_npc(self.config["num_npc"])


    def _draw_waypoints(self, world, waypoints, vertical_shift=.5, persistency=10000):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].transform.location + carla.Location(z=vertical_shift)

            size = 0.2
            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(128, 128, 128)
            else:  # LANEFOLLOW
                color = carla.Color(0, 255, 0) # Green
                size = 0.1

            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)

        world.debug.draw_point(waypoints[0][0].transform.location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].transform.location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(255, 0, 0), life_time=persistency)


    def _reset_test_comparison(self, unseen=False, index=0):
        pass

    def _is_in_neighboring_lane(self, target_transform, current_transform, max_distance=10.):
        """
        Check if a target object is within a certain distance in the neighboring lane of the ego vehicle.
        :param target_transform: location of the target object
        :param current_transform: location of the reference object
        :param orientation: orientation of the reference object
        :param max_distance: maximum allowed distance
        :return: True if target object is within max_distance ahead of the reference object
        """
        target_vector = np.array([target_transform.location.x - current_transform.location.x, target_transform.location.y - current_transform.location.y])
        norm_target = np.linalg.norm(target_vector)

        # Get the forward vector of the ego vehicle
        fwd = current_transform.get_forward_vector()
        forward_vector = np.array([fwd.x, fwd.y])

        # Get distance to ego vehicle in the heading of the ego vehciel
        d_long = np.dot(forward_vector, target_vector) / np.linalg.norm(forward_vector)

        # If the vector is too short, we can simply stop here
        #TODO decide what to do here
        if norm_target < 0.001:
            return True, 0, 1

        # Vehicle is too far away, we can stop here
        if d_long > max_distance:
            return False, max_distance, 0

        # Next, we need to check if the target object is within the neighboring lane
        # Get carla waypoints for ego vehicle and target vehicle
        ego_waypoint = self._map.get_waypoint(current_transform.location)
        target_waypoint = self._map.get_waypoint(target_transform.location)

        # Get the road_id and lane_id of both of these waypoints
        ego_road_id = ego_waypoint.road_id
        ego_lane_id = ego_waypoint.lane_id
        target_road_id = target_waypoint.road_id
        target_lane_id = target_waypoint.lane_id

        # Check if ego vehicle and target vehicle are on the same road
        if(ego_road_id == target_road_id):
        #     print(f"Ego: {ego_lane_id} Other: {target_lane_id}")

            # Check if ego vehicle and target vehicle are in neighboring lanes
            # If they are return true
            if(ego_lane_id == -1 and target_lane_id == 1) or (ego_lane_id == 1 and target_lane_id == -1):
                return True, d_long, -1
            elif(np.abs(ego_lane_id - target_lane_id) == 1):
                return True, d_long, int(np.sign(target_lane_id - ego_lane_id))
        # -1: left, 1: right for the orientation indicator (the last elem)
        return False, max_distance, 0

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
            if count <= 0:
                break
            if self.config["verbose"]:
                print('spawn_point:', spawn_point)
            if self.try_spawn_random_vehicle_at(self.vehicle_blueprints, spawn_point):
                count -= 1

    def get_speed_from_velocity(self, velocity):
        speed = np.sqrt(velocity.x ** 2 + velocity.y **2 + velocity.z **2)
        return speed

    def cosine_between_velocities(self, v1, v2):
        return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z) / \
            (self.get_speed_from_velocity(v1) * self.get_speed_from_velocity(v2) + 1e-6)

    def cosine_between_obs(self, agt_v, obs_v):
        if self.get_speed_from_velocity(agt_v) < self.config['zero_speed_threshold'] or \
            self.get_speed_from_velocity(obs_v) < self.config['zero_speed_threshold']:
            return 0.
        else:
            return self.cosine_between_velocities(agt_v, obs_v)

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
        # offlane = self.episode_measurements["offlane_steps"] > self.config["max_offlane_steps"]
        # static = agent.episode_measurements['obstacle_dist'] == -1 and \
        #     self.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) < 1e-2
        collision = agent.episode_measurements["is_collision"]
        runover_light = agent.episode_measurements["runover_light"]
        maxStepsTaken = agent.episode_measurements["num_steps"] > self.config['max_steps']
        offlane = False

        # Conditions to check there is obstacle or red light ahead for last 2 timesteps
        obstacle_ahead = agent.episode_measurements['obstacle_dist'] != -1 and agent.prev_measurement['obstacle_dist'] != -1
        red_light = agent.episode_measurements['red_light_dist'] != -1 and agent.prev_measurement['red_light_dist'] != -1

        if not self.config["enable_static_termination"]:
            static = False
        if self.config["disable_collision"]:
            collision = False
        if self.config["disable_traffic_light"] or not self.config["terminate_on_light"]:
            runover_light = False
        if self.config['enable_lane_invasion_termination']:
            # offlane = self.episode_measurements["offlane_steps"] > self.config["max_offlane_steps"]
            offlane = agent.episode_measurements['unlawful_lane_change'] or \
                agent.episode_measurements['out_of_road']


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
        elif agent.episode_measurements['out_of_road'] and self.config['enable_lane_invasion_termination']:
            termination_state = 'out_of_road'
            termination_state_code = 2
        elif agent.episode_measurements['unlawful_lane_change'] and self.config['enable_lane_invasion_termination']:
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
        # print(success, collision, runover_light, offlane, static, maxStepsTaken)
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
    env.reset_env()
