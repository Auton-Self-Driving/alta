""" Environment file wrapper for CARLA """

import gym
from gym.spaces import Box, Discrete, Tuple

from datetime import datetime
import os, sys

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

import carla_environment.environment.env_util as env_util
from carla_environment.environment.observations_manager import ObservationsManager

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
        ################################################################################
        if 'challenge' in self.config["scenarios"]:
            self.statistics_manager = StatisticsManager()
            self.config['num_agents'] == 1
            assert self.config['num_agents'] == 1, 'Multi agent in one env under challenge scenarios not supported'
        if self.config['verbose']: print('##### USE MULTI-AGENT #####', flush=True)
        self.ego_vehicle_list = [None] * self.config['num_agents']
        self.ego_agent_list = [None] * self.config['num_agents']

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
        self._set_world_and_map(self.config['city_name'])
        self.world_annotations = RouteParser.parse_annotations_file(
            '../../../leaderboard/data/all_towns_traffic_scenarios_public.json')

        time.sleep(20)

        tmport_retries = 0
        while True:
            try:
                self.tm_port = random.randint(10000, 60000)
                self.tm = self.client.get_trafficmanager(self.tm_port)
                break
            except Exception as e:
                tmport_retries += 1
                if tmport_retries > 5:
                    self.close()
                    exit()
        self.tm.set_synchronous_mode(True)

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_traffic_manager_port(self.tm_port)

        if self.config["testing"]:
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in self.config['spawn_points_fixed_idx']]
        else:
            spawn_pt_idx = np.random.permutation(len(self.spawn_points))
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in spawn_pt_idx]

        self.obs_manager = ObservationsManager(self.config, self._world)
        self.obs_manager.setup_observation_and_action_space()

        self.traj_manager = env_util.get_trajectory_manager(self.config["action_type"], self.config)

        if self.config["disable_two_wheeler"]:
            self.vehicle_blueprints = [x for x in self.vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]

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

    def step(self, action=None):
        try:
            # Action is set in dppo_agent learn method. 
            # 'action' arg is for stablebaseline
            # For stable baselines look at old file for additional code. 
            self.list_step(action=action) # action here will be an action list
        except Exception:
            print("Error during step, terminating episode early", traceback.format_exc())
            raise

    def list_step(self, action=None): # Environment steps take place here
        # action_list here should be a list of action
        self.world_frame = None

        for rk, agent in enumerate(self.ego_agent_list):

            if action is not None: agent.action = action # for stablebaseline
            if agent.action is None: continue

            agent.curr_reward = 0
            if agent.frame_skip_itr == 0:
                agent.last_acted_location = agent.vehicle_actor.get_transform().location # required in bezier action space
                agent.last_acted_rotation = agent.vehicle_actor.get_transform().rotation # required in bezier action space

            if not self.config["use_pid_in_frame_skip"]:# Doesn't seem to be used
                control = self._update_control(agent)

        for _ in range(self.config["frame_skip"]):

            # Get control of agent (steer, throttle, break, reverse and hand brake)
            for rk, agent in enumerate(self.ego_agent_list):

                if agent.done or agent.action is None: continue

                if self.config["use_pid_in_frame_skip"]:

                    control = self._update_control(agent)

                    # frame_skip_itr required for time dependendent control
                    # used in parameterized trajectory action space
                    agent.frame_skip_itr = (agent.frame_skip_itr + 1) % self.config['sticky_temporal_action_frames']

                    if self.config['verbose']:
                        print('[step {}][agent {}][steer {:.2f}][throttle {:.2f}][break {:.2f}][reverse {}][speed {:.2f}]'.format(
                            agent.curr_ep_num_steps, agent.rank, control.steer, control.throttle, control.brake, control.reverse,
                            env_util.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6))
                     
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
        
                # ### NOTE
                # if agent.episode_measurements['num_steps'] is None or agent.episode_measurements['num_steps'] % 8 == 0:
                #     print(agent.episode_measurements['num_steps'],
                #     self.curr_town,
                #     self.config["scenarios"],
                #     f"Sp:{agent.episode_measurements['speed']:.2f}", 
                #     f"D2G:{agent.episode_measurements['distance_to_goal_trajec']:.2f}",
                #     f"[S:({self.source_transform.location.x:.2f},{self.source_transform.location.y:.2f}),D:({self.destination_transform.location.x:.2f},{self.destination_transform.location.y:.2f})]",
                #     f"[Spawn:({self.spawn_points[0].location.x:.2f},{self.spawn_points[0].location.y:.2f})]",
                #     f"Agent:({agent.vehicle_actor.get_transform().location.x:.2f},{agent.vehicle_actor.get_transform().location.y:.2f})",
                #     f"NPC:({self.actor_list[0].get_transform().location.x:.2f},{self.actor_list[0].get_transform().location.y:.2f})",
                #     f"cos:{env_util.cosine_between_velocities(self.actor_list[0].get_transform().get_forward_vector(),agent.vehicle_actor.get_transform().get_forward_vector()):.2f}",
                #     [agent.episode_measurements['obstacle_speed'],agent.episode_measurements['obstacle_speed_front_left'],agent.episode_measurements['obstacle_speed_front_right'],agent.episode_measurements['obstacle_speed_back_left'],agent.episode_measurements['obstacle_speed_back_right']],
                #     [agent.episode_measurements['obstacle_dist'],agent.episode_measurements['obstacle_dist_front_left'],agent.episode_measurements['obstacle_dist_front_right'],agent.episode_measurements['obstacle_dist_back_left'],agent.episode_measurements['obstacle_dist_back_right']] ) 
                

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
                    if not self.config["disable_lane_invasion_sensor"]:
        
                        agent.episode_measurements['num_laneintersections'] = agent.lane_invasion_sensor.num_laneintersections
                        agent.episode_measurements['unlawful_lane_change'] = agent.episode_measurements['num_laneintersections'] > \
                            agent.prev_measurement['num_laneintersections']
                        agent.episode_measurements['out_of_road'] = agent.lane_invasion_sensor.out_of_road
                        
        
                        next_opts = set(agent.next_road_opt_queue)
                        for opt in next_opts:
                            # NOTE: not sure the reason but here should use .name to compare
                            if opt.name != RoadOption.CHANGELANELEFT.name and \
                                opt.name != RoadOption.CHANGELANERIGHT.name:
                                continue
                            
                            agent.episode_measurements['unlawful_lane_change'] = False
                        
                    else:
                        self._update_lane_invasion_info_via_privilege(agent)

                elif not self.config["disable_lane_invasion_sensor"]:

                    agent.episode_measurements['num_laneintersections'] = agent.lane_invasion_sensor.num_laneintersections
                    agent.episode_measurements['unlawful_lane_change'] = agent.episode_measurements['num_laneintersections'] > \
                        agent.prev_measurement['num_laneintersections']
                    agent.episode_measurements['out_of_road'] = agent.lane_invasion_sensor.out_of_road
    
                    next_opts = set(agent.next_road_opt_queue)
                    for opt in next_opts:
                        # NOTE: not sure the reason but here should use .name to compare
                        if opt.name != RoadOption.CHANGELANELEFT.name and \
                            opt.name != RoadOption.CHANGELANERIGHT.name:
                            continue
                        
                        agent.episode_measurements['unlawful_lane_change'] = False

                agent.location = agent.vehicle_actor.get_location()
                agent.episode_measurements['distance_to_goal'] = agent.location.distance(agent.destination_transform.location)
                if agent.episode_measurements['min_distance_to_goal'] >= agent.location.distance(agent.destination_transform.location):
                    agent.episode_measurements['min_distance_to_goal'] = agent.location.distance(agent.destination_transform.location)
                agent.episode_measurements['speed'] = env_util.get_speed_from_velocity(agent.vehicle_actor.get_velocity())

                self._get_ego_input(agent) # Agent observations and ep measurements constructed here

                agent.step_reward = compute_reward(name=self.config['reward_function'],
                                    prev_measurement=agent.prev_measurement,
                                    cur_measurement=agent.episode_measurements,
                                    config=self.config,
                                    verbose=self.config["verbose"])
                agent.curr_reward += agent.step_reward

                obs_collision = (agent.episode_measurements['num_collisions'] - agent.prev_measurement['num_collisions']) > 0

                if obs_collision and agent.episode_measurements["collision_actor_id"] != agent.prev_measurement["collision_actor_id"]:
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
                for k in agent.episode_reward_breakup:
                    agent.episode_reward_breakup[k] += agent.episode_measurements['reward_breakup'][k]
                agent.episode_measurements['reward'] = agent.curr_reward
                agent.episode_measurements['total_reward'] = agent.episode_reward

                for k in agent.camera_images_array:
                    agent.camera_images_array[k].append(agent.episode_measurements['camera_images'][k])

        for rk, agent in enumerate(self.ego_agent_list):
            if agent.action is None:
                self._get_ego_input(agent)
                agent.prev_measurement = copy.deepcopy(agent.episode_measurements)

    def _update_control(self, agent):
        control = self.get_control(agent, agent.action)
        #Store control for this step
        agent.episode_measurements['control_steer'] = control.steer
        agent.episode_measurements['control_throttle'] = control.throttle
        agent.episode_measurements['control_brake'] = control.brake
        agent.episode_measurements['control_reverse'] = control.reverse
        agent.episode_measurements['control_hand_brake'] = control.hand_brake
        return control

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
            current_speed = env_util.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
            throttle = agent.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config["action_type"] == "throttle_only":
            steer = float(0.0)
            target_speed = float(np.clip(action[0], 0, self.target_speed))
            current_speed = env_util.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
            throttle = agent.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config["action_type"] == "merged_speed":
            # steer = float(action[0])
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(np.clip(action[1] + 10.0, 0, self.target_speed))
            current_speed = env_util.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
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
            current_speed = env_util.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
            gas = agent.controller.pid_control(target_speed, current_speed, enable_brake=self.config["enable_brake"])
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config["action_type"] == "merged_speed_scaled_tanh":
            
            steer = np.clip(float(action[0]) * self.config['steering_scale'], -1, 1)

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

            current_speed = env_util.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
            gas = agent.controller.pid_control(target_speed, current_speed, enable_brake=self.config["enable_brake"])

            # print('[carla_env.get_control()] Throttle : {} Cur Spd : {} Trgt Speed : {} Action : {}'.format(gas,current_speed,target_speed,action[1]))

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
            current_speed = env_util.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
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
            current_speed = env_util.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6
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
        elif self.config["action_type"] == "cubic_bezier_3dof":

            cubic_bezier_pt = lambda t,p0,p1,p2,p3 : p0 * (1-t)**3 + \
                                    3 * p1 * t*(1-t)**2 + \
                                    3 * p2 * (t**2)*(1-t) + \
                                    p3 * t**3

            # TODO improve ctrl pt computation code. Make it depend on self.action_space bounds

            time_on_curve =  (agent.frame_skip_itr + float(self.config['frame_skip'] * 0.1)) / float(self.config['traj_frame_horizon'])

            # First control point location fixed
            # This enforces smooth transition into a turn (physically achievable motion)
            x1, y1 = 2, 0 

            # Second control point is free
            # Bounded in box space where X in [0,4] and Y in [-6,6]
            # Note raw action are in [-1,1] due to tanh. Due to squashing they are scaled and clipped
            x2, y2 = np.clip(action[0]*1.5 + 1,0,4), np.clip(action[1]*1.2*6,-6,6)

            # Destination point. At [8,y] where y varies between [-4,4] 
            x3, y3 = 8,  np.clip(action[2]*1.2*4,-4,4) # dest pt fixed at 8meters away


            # Computing bezier waypoint coordinate in world system
            sign = 1 if (agent.last_acted_rotation.yaw < 90 or agent.last_acted_rotation.yaw > 270) else -1
            x0, y0 = agent.last_acted_location.x          , agent.last_acted_location.y 
            x1, y1 = agent.last_acted_location.x + sign*x1, agent.last_acted_location.y + sign*y1
            x2, y2 = agent.last_acted_location.x + sign*x2, agent.last_acted_location.y + sign*y2
            x3, y3 = agent.last_acted_location.x + sign*x3, agent.last_acted_location.y + sign*y3

            target_x = cubic_bezier_pt(time_on_curve,x0,x1,x2,x3)
            target_y = cubic_bezier_pt(time_on_curve,y0,y1,y2,y3)

            target_waypoint = DummyWaypoint(Transform(Location(target_x,target_y,0),Rotation(0,0,0))) # LateralPID only observes x,y


            # Target Speed at destination
            # TODO: target speed is ramped up along bezier curve
            target_speed = action[3]*1.5 + 1
            target_speed = float(np.clip(target_speed * self.target_speed / 2, 0, self.target_speed))
            current_speed = env_util.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6

            gas = agent.controller.pid_control(target_speed, current_speed, enable_brake=self.config["enable_brake"])
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0

            steer = agent.steer_controller.pid_control(
                    target_waypoint, agent.vehicle_actor.get_transform())
            steer = np.clip(steer, -1., 1.)
        elif self.config["action_type"] == "cubic_bezier_5dof":

            # TODO improve ctrl pt computation code. Make it depend on self.action_space bounds

            time_delay = 0.35

            # This is to rectify the origin of the  traj. 
            # Without this the origin would lie at the rear end of the car
            car_length_offset = 3.5

            time_on_curve = time_delay  +  agent.frame_skip_itr / float(self.config['traj_frame_horizon'])

            if agent.frame_skip_itr == 0:
                # Setting up vehicle centric coordinate space
                self.traj_manager.set_coordinate_system(
                    agent.last_acted_location,
                    agent.last_acted_rotation,
                    car_length_offset)
                # Parameterizing trajectory
                self.traj_manager.populate_trajectory(action)
            else:
                # Update trajectory manager with current vehicle orientation 
                # to suitably rotate car coordinate system
                self.traj_manager.set_current_basis(agent.vehicle_actor.get_transform().rotation)

                # Compute pseudo action that produces remaining traj but
                # with origin at current position on original traj
                agent.action, start_pt = self.traj_manager.get_subsegment_parameters(time_on_curve)
                # agent.action, start_pt = self.traj_manager.get_subsegment_parameters(agent)

                ### FOR DEBUGGING ###
                debug_traj_manager = env_util.get_trajectory_manager(self.config["action_type"], self.config)
                debug_traj_manager.current_basis = self.traj_manager.current_basis
                debug_traj_manager.reference_basis = self.traj_manager.current_basis
                debug_traj_manager.yaw_drift = 0
                debug_traj_manager.origin = self.traj_manager.origin 
                cur_pos_ref = start_pt
                debug_traj_manager.origin += cur_pos_ref[0]*self.traj_manager.reference_basis["heading"] 
                debug_traj_manager.origin += cur_pos_ref[1]*self.traj_manager.reference_basis["right"] 
                debug_traj_manager.populate_trajectory(agent.action)
                env_util.plot_trajectory(time_delay, self._world, debug_traj_manager, col_scheme="non_std")

            agent.trajectory_yaw_drift = self.traj_manager.yaw_drift

            ### FOR DEBUGGING ###
            # if agent.frame_skip_itr == 0:
            env_util.plot_trajectory(time_on_curve, self._world, self.traj_manager)

            target_speed, target_waypoint = self.traj_manager.get_target_speed_waypoint(time_on_curve)
            # target_speed, target_waypoint = self.traj_manager.get_target_speed_waypoint(agent)
            current_speed = env_util.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6

            # self._world.debug.draw_point(
            #     carla.Location(x=target_waypoint.transform.location.x,y=target_waypoint.transform.location.y,z=5),
            #     size=4, color=(255,0,0), life_time=0.3)

            gas = agent.controller.pid_control(target_speed, current_speed, enable_brake=self.config["enable_brake"])
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0

            steer = agent.steer_controller.pid_control(
                    target_waypoint, agent.vehicle_actor.get_transform())
            steer = np.clip(steer, -1., 1.)
        elif self.config["action_type"] == "speed_wp":

            time_delay = 0.1

            # This is to rectify the origin of the  traj. 
            # Without this the origin would lie at the rear end of the car
            car_length_offset = 3.5

            time_on_curve = time_delay +  agent.frame_skip_itr / float(self.config['traj_frame_horizon'])

            if agent.frame_skip_itr == 0:
                # Setting up vehicle centric coordinate space
                self.traj_manager.set_coordinate_system(
                    agent.last_acted_location,
                    agent.last_acted_rotation,
                    car_length_offset)
                # Parameterizing trajectory
                self.traj_manager.populate_trajectory(action)
            else:
                # Update trajectory manager with current vehicle orientation 
                # to suitably rotate car coordinate system
                self.traj_manager.set_current_basis(agent.vehicle_actor.get_transform().rotation)

                # Compute pseudo action that produces remaining traj but
                # with origin at current position on original traj
                agent.action = self.traj_manager.get_subsegment_parameters(agent)
                # agent.action = self.traj_manager.get_subsegment_parameters(time_on_curve)

                ### FOR DEBUGGING ###
                if agent.frame_skip_itr > 0:
                    debug_traj_manager = env_util.get_trajectory_manager(self.config["action_type"], self.config)
                    debug_traj_manager.set_coordinate_system(
                        agent.vehicle_actor.get_transform().location,
                        agent.vehicle_actor.get_transform().rotation,
                        car_length_offset)
                    debug_traj_manager.yaw_drift = 0
                    debug_traj_manager.origin = carla.Vector3D(x=self.traj_manager.origin.x, y=self.traj_manager.origin.y, z=self.traj_manager.origin.z)
                    cur_pos_ref = self.traj_manager.get_points_on_trajectory()[int(time_on_curve*self.traj_manager.points_on_traj)]
                    debug_traj_manager.origin += cur_pos_ref[0]*self.traj_manager.reference_basis["heading"] 
                    debug_traj_manager.origin += cur_pos_ref[1]*self.traj_manager.reference_basis["right"]
                    debug_traj_manager.populate_trajectory(agent.action)
                    env_util.plot_trajectory(time_delay, self._world, debug_traj_manager, col_scheme="non_std")

            agent.trajectory_yaw_drift = self.traj_manager.yaw_drift

            ### FOR DEBUGGING ###
            if agent.frame_skip_itr == 0:
                env_util.plot_trajectory(time_on_curve, self._world, self.traj_manager)

            # target_speed, target_waypoint = self.traj_manager.get_target_speed_waypoint(agent)
            target_speed, target_waypoint = self.traj_manager.get_target_speed_waypoint(time_on_curve)
            current_speed = env_util.get_speed_from_velocity(agent.vehicle_actor.get_velocity()) * 3.6

            gas = agent.controller.pid_control(target_speed, current_speed, enable_brake=self.config["enable_brake"])
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0

            steer = agent.steer_controller.pid_control(
                    target_waypoint, agent.vehicle_actor.get_transform())
            steer = np.clip(steer, -1., 1.)                
        
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

    def _add_to_stacked_queue(self, object_queue, object_to_add):

        assert (object_queue is not None and object_to_add is not None)

        if object_queue.full():
            # Pop out earlier stacked frame if queue is full
            object_queue.get()
        object_queue.put(object_to_add)

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

    def _update_obs_detector_via_privilege(self, agent): # Updates agent measurements based on nearby vehicular obstacles using prev. info
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
                d_bool, d_angle, distance = env_util.is_within_distance_ahead(target_vehicle.get_transform(),
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
                            env_util.get_speed_from_velocity(target_vehicle.get_velocity())
                elif side_orient == 1:
                    if agent.episode_measurements['obstacle_dist_right'] == -1 or \
                        side_dist < agent.episode_measurements['obstacle_dist_right']:
                        agent.episode_measurements['obstacle_dist_right'] = side_dist
                        agent.episode_measurements['obstacle_speed_right'] = \
                            env_util.get_speed_from_velocity(target_vehicle.get_velocity())

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
                        agent.episode_measurements['obstacle_speed'] = env_util.get_speed_from_velocity(target_vehicle.get_velocity())
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

    def _update_obs_detector_via_sensor(self, agent): # Default sensor observation update method
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

            if hasattr(obstacle_actor, 'get_velocity'):
                if self.config['obs_cosine_velocity']:
                    cos = env_util.cosine_between_obs(obstacle_actor.get_velocity(), agent.vehicle_actor.get_velocity(), self.config['zero_speed_threshold'])
                    if cos < 0: cos = 0.
                else:
                    cos = 1.
                agent.episode_measurements['obstacle_speed'] = agent.episode_measurements['obstacle_speed_front'] = env_util.get_speed_from_velocity(obstacle_actor.get_velocity()) * cos
                
            else:
                agent.episode_measurements['obstacle_speed'] = agent.episode_measurements['obstacle_speed_front'] = -1
            found_obstacle = found_obstacle and (not self.config['check_obs_same_lane'] or same_lane)

            if agent.episode_measurements['obstacle_init_id'] != obstacle_actor.id: # initial
                agent.episode_measurements['obstacle_init_id'] = obstacle_actor.id
                agent.episode_measurements['obstacle_init_dist'] = agent.obstacle_sensor['front'].distance
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

                obstacle_actor = agent.obstacle_sensor[suffix].obstacle_actor
                if 'vehicle' in obstacle_actor.type_id:
                    same_lane = check_if_vehicle_in_same_lane(agent.vehicle_actor, obstacle_actor, agent.next_waypoints, self._map)
                found_obstacle = True
                agent.episode_measurements['obstacle_dist_{}'.format(suffix)] = agent.obstacle_sensor[suffix].distance
               
               
                if hasattr(obstacle_actor, 'get_velocity'):
                    agent.episode_measurements['obstacle_speed_{}'.format(suffix)] = env_util.get_speed_from_velocity(obstacle_actor.get_velocity())
                else:
                    agent.episode_measurements['obstacle_speed_{}'.format(suffix)] = -1
                found_obstacle = found_obstacle and (not self.config['check_obs_same_lane'] or not same_lane)
            if not found_obstacle:
                agent.episode_measurements['obstacle_speed_{}'.format(suffix)] = -1
                agent.episode_measurements['obstacle_speed_{}'.format(suffix)] = -1

    def _update_obs_detector_via_all_sensor(self, agent): # Used in 360 degree obs space to populate sensor readings
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
                    # relative_velocity = ego_inverse_matrix[0:3,0:3] @ (other_velocity - ego_velocity)
                    relative_velocity = ego_inverse_matrix[0:3,0:3] @ (other_velocity) ### NOTE # Not relative just transformed velocity
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

        if traffic_actor is not None:
            if traffic_actor.state != carla.TrafficLightState.Green:
                agent.episode_measurements['red_light_dist'] = dist
                found_redlight = True
                if agent.episode_measurements['initial_dist_to_red_light'] == -1 or \
                    (agent.episode_measurements['nearest_traffic_actor_id'] != -1 and traffic_actor.id != agent.episode_measurements['nearest_traffic_actor_id']):
                    if dist < self.config['min_dist_from_red_light']:
                        agent.episode_measurements['red_light_dist'] = -1
                        agent.episode_measurements['initial_dist_to_red_light'] = -1
                        found_redlight = False
                    else:
                        agent.episode_measurements['initial_dist_to_red_light'] = dist
                        found_redlight = True
            else:
                agent.episode_measurements['red_light_dist'] = -1
                agent.episode_measurements['initial_dist_to_red_light'] = -1

            agent.episode_measurements['nearest_traffic_actor_id'] = traffic_actor.id
            agent.episode_measurements['nearest_traffic_actor_state'] = traffic_actor.state
            
        else:
            agent.episode_measurements['red_light_dist'] = -1
            agent.episode_measurements['initial_dist_to_red_light'] = -1
            agent.episode_measurements['nearest_traffic_actor_id'] = -1
            agent.episode_measurements['nearest_traffic_actor_state'] = None

        agent.episode_measurements['dist_to_light'] = dist

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

    def _set_scenario(self, unseen=False, town="Town01", index=0): # Scenarios are set here
        """ Returns Start and end waypoints for an episode based on the chosen senario. 
        self.source_transform, self.destination_transform are indices that index a dict
        mapping integer indices to wapoint coordinates in maps. 

        Returns the current town of the environment
        """

        _upd_town = self.curr_town
        if self.config["scenarios"] == "straight": 
            self.source_transform, self.destination_transform = scenarios.get_straight_path(unseen, town, index)
            self.config["num_episodes"] = 25
        elif self.config["scenarios"] == "straight_overtake": 
            self.source_transform, self.destination_transform = scenarios.get_short_straight_path(unseen, town, index)
            self.config["num_episodes"] = 25
            frac = random.random() * 0.60 + 0.2
            spwn_loc = frac*self.source_transform.location + (1-frac)*self.destination_transform.location
            spwn_rot = self.source_transform.rotation
            self.spawn_points = [Transform(spwn_loc,spwn_rot)]
        elif self.config["scenarios"] == "straight_random_overtake": 
            self.source_transform, self.destination_transform = scenarios.get_straight_path(unseen, town, index)
            self.config["num_episodes"] = 25
            if random.random() > 0.5: # Add obstacle vehicle with 0.5 probability
                frac = random.random() * 0.5 + 0.25
                spwn_loc = frac*self.source_transform.location + (1-frac)*self.destination_transform.location
                spwn_rot = self.source_transform.rotation
                self.spawn_points = [Transform(spwn_loc,spwn_rot)]
                self.stationary_obstacle_vehicle = True
            else:
                self.stationary_obstacle_vehicle = False
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
                avail_map_list=[self.curr_town], mode='test')
        elif self.config["scenarios"] == "leaderboard_navigation":
            self.source_transform, self.destination_transform, self.wps_list, _upd_town = scenarios.get_leaderboard_route(
                unseen, curr_town=self.curr_town, index=index, max_idx=self.config["min_num_eps_before_switch_town"],
                avail_map_list=self.config['avail_town_list'], mode='train')
        else:
            raise ValueError("Scenarios Config not set!")

        if _upd_town != self.curr_town: # switch to a new town
            print('[1060] update town from {} to {}'.format(self.curr_town, _upd_town), self.scenario_index)
        return _upd_town

    def reset(self, use_idx=False, idx_list=None, rank_list=None, reset_npc=False):
        # If using stable baselines SAC refer to old version for additional code 
        if not rank_list: rank_list = [0]
        self.list_reset(use_idx=use_idx, idx_list=idx_list, rank_list=rank_list, reset_npc=reset_npc)
   
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

    def _setup_camera(self, sensor, bev = True):

            camera = self.blueprint_library.find(sensor)
            camera.set_attribute('image_size_x', self.config['sensor_x_res'])
            camera.set_attribute('image_size_y', self.config['sensor_y_res'])
            camera.set_attribute('sensor_tick', self.config['sensor_tick'])
            camera.set_attribute('fov', '90')

            if bev: # BEV Camera
                camera_transform = carla.Transform(carla.Location(x=13.0, z=18.0), carla.Rotation(pitch=270.0))
            else: # Front Facing Camera
                camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0.0))

            return camera, camera_transform

    def reset_vehicle_agent(self, agent_list, transfuser=False): # Resets vehicle agents and their sensors.

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

            # Reward breakup logger
            agent.episode_reward_breakup = {"progress":0,"motion":0,"steer":0,"d2t":0,"light":0,"lane_invasion":0,"obs_collision":0,"static":0}

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


            # Cameras Setup
            if self.config["semantic"]:
                sensor = self.config['sensors'][1]
            else:
                sensor = self.config['sensors'][0]

            bev_camera, bev_camera_transform = self._setup_camera(sensor, bev = True)
            front_camera, front_camera_transform = self._setup_camera(sensor, bev = False)

            agent.camera_actors = {
                "bev":self._world.spawn_actor(bev_camera, bev_camera_transform, attach_to=agent.vehicle_actor),
                "front":self._world.spawn_actor(front_camera, front_camera_transform, attach_to=agent.vehicle_actor),
            }

            agent.camera_images_array = {k : [] for k in agent.camera_actors}

            agent.camera_queues = [] 
            for k in agent.camera_actors.keys():
                agent.camera_queues.append(queue.Queue())
                agent.actor_list.append(agent.camera_actors[k])
                agent.camera_actors[k].listen(agent.camera_queues[-1].put)                

            agent.collision_sensor = sensors.CollisionSensor(agent.vehicle_actor)
            agent.actor_list.append(agent.collision_sensor.sensor)

            if not self.config["disable_lane_invasion_sensor"]:
                agent.lane_invasion_sensor = sensors.LaneInvasionSensor(agent.vehicle_actor)
                agent.actor_list.append(agent.lane_invasion_sensor.sensor)

            if self.config["enable_obstacle_sensor"]:
                
                if self.config['input_type'] in ['wp_obs_more_info_speed_steer_ldist_light', \
                                    'wp_2avg_obs_more_info_speed_steer_ldist_light', \
                                    'wp_list_obs_more_info_steer_ldist_light', \
                                    'wp_list_obs_more_info_speed_steer_ldist_light', \
                                    'wp_obs_more_info_steer_ldist_light']:
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
            agent.episode_measurements['speed'] = env_util.get_speed_from_velocity(agent.vehicle_actor.get_velocity())

            agent.episode_measurements['total_steps'] = agent.num_total_steps

            ####### Required in bezier_curve action space ######
            # Tracks frames since last policy inference 
            agent.frame_skip_itr = 0
            # Agent location and rotation where last action was taken by policy
            agent.last_acted_location = agent.vehicle_actor.source_transform.location
            agent.last_acted_rotation = agent.vehicle_actor.source_transform.rotation
            # Difference between on trajectory yaw and last action location yaw
            agent.trajectory_yaw_drift = 0

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

    def _update_lane_invasion_info_via_privilege(self, agent):
        if agent.done: return
        
        
        agent_wp = self._map.get_waypoint(
            agent.vehicle_actor.get_location(),
            lane_type=(LaneType.Any | LaneType.NONE))

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
            return

        # if not intersection, should stick on driving lane
        if agent_wp.lane_type != LaneType.Driving:
            agent.episode_measurements['out_of_road'] = True
            print('[1852] out of road', agent_wp.lane_type)
        else:
            num_same_road_wp = 0
            for next_wp in [agent.next_waypoints[0], agent.next_waypoints[-1]]:
                if agent_wp.road_id == next_wp.road_id:
                    num_same_road_wp += 1
                    if agent_wp.lane_id == next_wp.lane_id:
                        return
            # at intersection or something
            if num_same_road_wp == 0: return
        # if not returned, lane invasion happened
        agent.episode_measurements['num_laneintersections'] += 1
        if agent_wp.lane_change == LaneChange.NONE:
            agent.episode_measurements['unlawful_lane_change'] = True

    def _get_ego_input(self, agent): # Obs updated here

        next_orientation, agent.dist_to_trajectory, distance_to_goal_trajec, \
            agent.next_waypoints, agent.next_wp_angles, agent.next_wp_vectors, agent.next_road_opts = \
            agent.global_planner.get_next_orientation_new(agent.vehicle_actor.get_transform(), append_road_opt=True, 
                                                            num_next_waypoints=self.config['num_waypoints'])

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
        agent.episode_measurements['dist_to_trajectory'] = agent.dist_to_trajectory

        # Update obstacle distance measurements
        obs = {}

        self._update_env_obs(agent)

        # if static (stuck by obstacle)
        if env_util._is_static(agent, self.config['zero_speed_threshold']):
            agent.episode_measurements['static_steps'] += 1
        else:
            agent.episode_measurements['static_steps'] = 0


        # Gather camera measurements
        agent.camera_images = {}
        cameras = list(agent.camera_actors.keys())
        for idx in range(len(cameras)):
            image = self._read_data(agent.camera_queues[idx], self.world_frame)
            agent.camera_images[cameras[idx]] = image
        agent.episode_measurements['camera_images'] = agent.camera_images


        obs['speed'] = np.expand_dims(np.array([agent.episode_measurements['speed']]), axis=0) # * 3.6 / 30
        obs['dist_to_target'] = np.array([agent.episode_measurements['distance_to_goal']])

        # Update observation input in obs dictionary
        self.obs_manager.create_observations(agent, obs)
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
                                        'wp_obs_more_info_steer_ldist_light',
                                        'wp_obs_more_info_speed_steer_ldist_light',
                                        'wp_2avg_obs_more_info_speed_steer_ldist_light',
                                        'wp_list_obs_more_info_steer_ldist_light',
                                        'wp_list_obs_more_info_speed_steer_ldist_light',
                                        'wp_360_obstacle_speed_steer']:
            observation = np.expand_dims(obs['observation'], axis = 0)
            agent.observation = observation
        elif self.config['input_type'] == 'transformer':
            agent.observation = obs['observation']
        else:
            agent.observation = obs

        if self.config['verbose']: print('[agent {}] observation: {}'.format(agent.rank, agent.observation))  

        return agent.observation

    def list_reset(self, use_idx=False, idx_list=None, rank_list=None, reset_npc=False): # Resets environment and actors

        # Retrieve Blueprint for Vehicle
        try:
            vehicle_bp = self.blueprint_library.find(self.config['vehicle_type'])
            # vehicle_bp = self.blueprint_library.find(random.choice(self.config['vehicle_types']))
        except Exception as e:
            print("Error during vehicle creation: {}".format(traceback.format_exc()))

        # If episode length surpasses 'npc_reset_freq', reset all NPC vehicles
        # CHECK: Possible source of bug if deletion not happening right
        if reset_npc or (self.config['npc_reset_freq'] and self.world_frame - self.last_npc_reset_frame > self.config['npc_reset_freq']):
            self.destroy_all_existing_npc_actors()

        # Iterate over indices of ego agents in curr env and destroy them
        # Then 
        for rk in rank_list: 

            prev_agent = self.ego_agent_list[rk]
            self.ego_agent_list[rk] = None

            if prev_agent is not None: self.curr_num_agents -= 1
            try:
                self.destroy_an_existing_ego_agent(prev_agent)
            except:
                print('>>> [rank {}] Error when deleting prev_agent [agent {}]'.format(self.env_rank, rk))

            # Spawning vehicle actor with retry logic as it fails to spawn sometimes
            
            NUM_RETRIES = 500 
            BAD_SRC_DEST_RETRIES = 5

            for src_dest_itr in range(BAD_SRC_DEST_RETRIES):
                self.vehicle_actor = None

                for idx in range(1, NUM_RETRIES + 1):
                    # Determine start and end points for episode
                    # use_scenarios taps into pre-existing paths and retrieves their start and end.
                    #   Currently scenarios are defined only for Town01
                    # When this is disabled, the start and end are randomly chosen from a set of spawn points
                    if self.config["use_scenarios"]:
                        if self.config["updated_scenarios"]:
                            self._set_updated_scenario(unseen=use_idx, index=self.scenario_index, town=self.config['city_name'])
                        else:
                            _upd_town = self._set_scenario(unseen=use_idx, index=self.scenario_index, town=self.config['city_name'])
                    else:
                        self.source_transform, self.destination_transform = random.choice(self.spawn_points), random.choice(self.spawn_points)

                    try:
                        self.vehicle_actor = CarlaDataProvider.request_new_actor(self.config['vehicle_type'], self.source_transform, 'hero')
                    except:
                        self.vehicle_actor = None
                                        
                    if self.vehicle_actor is not None:
                        if _upd_town != self.curr_town: # switch to a new town
                            self.reset_env()
                            self._set_world_and_map(_upd_town)
                        break
                
                if self.vehicle_actor is not None:

                    self.ego_vehicle_list[rk] = self.vehicle_actor
                    self.vehicle_actor.source_transform = self.source_transform
                    self.vehicle_actor.destination_transform = self.destination_transform
        
                    self.curr_num_agents += 1

                    # Generates list of waypoints connecting source and destination for current scenario
                    # The sequence of generated way points are stored in self.dense_waypoints
                    self.vehicle_actor.global_planner = planner.GlobalPlanner()

                    if 'challenge' in self.config["scenarios"]:
                        self.gps_route, self.route, self._global_plan_world_coord = interpolate_trajectory(self._world, self.wps_list)

                        # Print route in debug mode
                        # self._draw_waypoints(self._world, self.route, vertical_shift=1.0, persistency=500)
                        # print('self.route', self.route)
                        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))

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
                        self.vehicle_actor.stats.set_scenario(self.vehicle_actor.running.scenario)
                        self.vehicle_actor.scenario_config = DummyScenarioConfig(_record_idx, self.wps_list)

                    elif self.config["scenarios"] == 'leaderboard_navigation':
                        self.gps_route, self.route, self._global_plan_world_coord = interpolate_trajectory(self._world, self.wps_list)
                        self.dense_waypoints = self._global_plan_world_coord

                    else:
                        self.dense_waypoints  = self.vehicle_actor.global_planner.trace_route(self._map,
                                                self.source_transform, self.destination_transform)
        
                    self._draw_waypoints(self._world, self.dense_waypoints)

                    self.scenario_index += 1

                    mod_plan = self.vehicle_actor.global_planner.set_global_plan(self.dense_waypoints)

                    if len(self.dense_waypoints) < self.config['num_waypoints']:
                        print("[carla_env.list_reset] Agent {} Num waypoints {}  Plan {} Trial {}".format(rk,len(self.dense_waypoints),str(mod_plan),src_dest_itr+1))
                    else:
                        break
                    

                else:
                    # DEBUG
                    print("$$$$$$$$$$$$$$$$$$ Err in carla_env.list_reset")
                    
                    w = CarlaDataProvider._world
                    closeness = []
                    for ac in w.get_actors():
                        l = ac.get_location()
                        sr = self.source_transform.location
                        d = ((l.x-sr.x)**2+(l.y-sr.y)**2)**0.5  
                        closeness.append((ac.type_id,d,l.z,sr.z))
                    closeness.sort(key = lambda x : x[1])
                    closeness = closeness[-2:]

                    with open('debug.txt', 'a') as f:
                        f.write('^^^^^^^^^^^^^^^^^^^^^^^^^^\n')
                        f.write('Spawn Point = '+str(self.source_transform) + '\n')
                        f.write('Closeness  = '+str(closeness) + '\n')
                        f.write('Retries  = '+str(idx) + '\n')

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

    def _reset(self, unseen=False, index=0):
        pass

    def try_spawn_random_vehicle_at(self, blueprints, transform):
        # blueprint = random.choice(blueprints)

        # To spawn same type of vehicle
        blueprint = blueprints[0]
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # TODO: uncomment below to enable autopilot
        if self.config["scenarios"] not in ["straight_dynamic","straight_overtake"]:
            blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self._world.try_spawn_actor(blueprint, transform)
        tm_port = self.tm.get_port()

        # print("[carla_env/try_spawn_random_vehicle_at]",dir(vehicle))

        if vehicle is not None:

            if self.config["scenarios"] == "straight_overtake":
                vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            elif self.config["scenarios"] == "straight_random_overtake":
                if self.stationary_obstacle_vehicle:
                    vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            # TODO: uncomment below to enable autopilot
            elif not self.config["scenarios"] == "straight_dynamic":
                vehicle.set_autopilot(True, tm_port)

            self.actor_list.append(vehicle)

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

    def _retrieve_data(self, sensor_queue, timeout, world_frame):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.world_frame:
                return data
            else:
                if self.config["verbose"]:
                    print("difference in frames, self.world_frame={0}, data_frame={1}".format(self.world_frame, data.frame))

    def _compute_done_condition(self, agent): # Episode termination
        # Episode termination conditions
        success = agent.episode_measurements["distance_to_goal"] < self.config["dist_for_success"]
        static = agent.episode_measurements["static_steps"] > self.config["max_static_steps"]
        collision = agent.episode_measurements["is_collision"]
        runover_light = agent.episode_measurements["runover_light"]
        maxStepsTaken = agent.episode_measurements["num_steps"] > self.config['max_steps']
        offlane = False
        offroad = False

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
            offlane = agent.episode_measurements['unlawful_lane_change']
        if self.config['enable_off_road_termination']:
            offroad = agent.episode_measurements['out_of_road']


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
        elif offroad and self.config['enable_off_road_termination']:
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

        done = success or collision or runover_light or offlane or offroad or static or maxStepsTaken
        return done

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
