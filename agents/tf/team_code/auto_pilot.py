import os, sys

print(os.getcwd())
sys.path.append(os.getcwd())

import time
import datetime
import pathlib
import json
import random

import numpy as np
import cv2
import carla
from PIL import Image
from collections import deque, defaultdict

# from team_code.map_agent import MapAgent
# from team_code.pid_controller import PIDController

from map_agent import MapAgent


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative

# TEST_CONFIG = {
#     'PPO': False, # else SAC, currently only support those two
#     'checkpoint': './ckptPPOx1_input_700000_Feb051014PM24.pth',
#     'num_agents': 1,
#     'num_npc': 70,
#     # 'num_npc': 120,
#     'sample_npc': False,
#     'scenarios' : 'no_crash_dense',
#     # 'scenarios' : 'challenge_test_scenario',
#     'use_scenarios': True,
#     'city_name' : 'Town02',
#     # 'city_name' : 'Town01',
#     # 'city_name' : 'Town05',
#     'num_episodes' : 25,
#     'target_speed': 20,
#     'steering_scale': 0.5,
#     'testing' : False, # spawn point pending bugs in env line#142
#     'enable_static_termination' : True,
#     'enable_obstacle_sensor': True,
#     'obs_cosine_velocity': True,
#     'check_obs_same_lane': True,
#     'obs_sensor_vehicle_only': True,
#     'front_obs_sensor_hit_radius': .5,
#     'side_obs_sensor_hit_radius': .7854,
#     'disable_traffic_light': False,
#     'terminate_on_light' : False,
#     'enable_lane_invasion_termination' : False,
#     'front_obs_proximity_threshold' : 15,
#     'side_obs_proximity_threshold' : 5,
#     'traffic_light_proximity_threshold' : 15,
#     'min_dist_from_red_light' : 0,
#     'npc_reset_freq': None,
#     'verbose': False,
#     'weak_verbose': False,
#     'test_verbose': True,
#     # 'test_verbose': False,
#     'videos': False,
#     # 'sensor_x_res' : '400',
#     # 'sensor_y_res' : '800',
#     # 'videos': True,
#     'save_buffer': False,
# }

# ENV_CONFIG = {
#     'algo': 'Multi-Agent',
#     'num_envs': 1,
#     'num_agents': 4,
#     'max_num_steps': 16000000,
#     'device': 'cuda:0',
#     'log_dir': '../../../../alta-logs/',
#     'server_path' : CARLA_9_4_PATH,
#     'server_binary' : CARLA_9_4_PATH + '/CarlaUE4.sh',
#     'server_process' : None,
#     # X Rendering Resolution
#     'render_res_x' : 800,
#     # Y Rendering Resolution
#     'render_res_y' : 800,
#     'sensor_x_res' : '1',
#     'sensor_y_res' : '1',
#     # Input X Res (Default set to Atari)
#     'x_res': 84,
#     # Input Y Res (Default set to Atari)
#     'y_res': 84,
#     'server_fps' : 10,
#     'server_port' : None,
#     'server_retries' : 5,
#     'city_name' : 'Town01',
#     'frame_skip': 1,
#     'enable_planner' : True,
#     # 'reward_function': 'obs',
#     'reward_function' : 'simple2',
#     # 'reward_function' : 'simple2_modified',
#     # 'reward_function' : 'simple3',
#     # Print measurements to screen
#     'client' : None,
#     ### 'discrete_actions': True,
#     # Number of frames stacked together
#     'framestack' : 1,
#     ### 'grayscale' : False,
#     'num_pedestrians' : 0,
#     'max_steps' : 50000,
#     'next_command': None,
#     'verbose': False,
#     'weak_verbose': False,
#     'test_verbose': False,
#     'vehicle_type': 'vehicle.toyota.prius',
#     'disable_two_wheeler' : True,
#     'vehicle_types': ['vehicle.ford.mustang', 'vehicle.audi.a2', 'vehicle.audi.tt', 'vehicle.bmw.isetta', 'vehicle.carlamotors.carlacola',
#                       'vehicle.citroen.c3', 'vehicle.bmw.grandtourer', 'vehicle.mercedes-benz.coupe',
#                       'vehicle.toyota.prius', 'vehicle.dodge_charger.police', 'vehicle.nissan.patrol',
#                       'vehicle.tesla.model3', 'vehicle.seat.leon', 'vehicle.lincoln.mkz2017',
#                       'vehicle.volkswagen.t2', 'vehicle.nissan.micra', 'vehicle.chevrolet.impala', 'vehicle.mini.cooperst',
#                       'vehicle.jeep.wrangler_rubicon'],
#     'target_speed': 20,
#     'steering_scale': 0.5,
#     'sensors': ['sensor.camera.rgb', 'sensor.camera.semantic_segmentation'],
#     # 'action_type': 'discrete',
#     'action_type': 'merged_speed_scaled_tanh',
#     # 'action_type': 'merged_speed',
#     'sensor_tick': '0.0',
#     'dist_for_success' : 10.0,
#     'max_offlane_steps' : 0,
#     'max_static_steps' : 200,
#     'log_measurements_to_file': False,
#     'sync_mode': True,
#     # NOTE: crop does not work with framestack yet. need to add.
#     'preprocess_crop_image': False,
#     # 'scenarios' : 'navigation',
#     # 'scenarios' : 'challenge_train_scenario',
#     'scenarios' : 'leaderboard_navigation',
#     'min_num_eps_before_switch_town': 100,
#     'semantic' : False,
#     'client_timeout_seconds' : 6000,
#     # 'carla_gpu': '0',
#     'render_server': False,
#     'steer_penalty_coeff': 0,
#     'vae_encoding_norm_factor' : 10,
#     # 'input_type': 'wp_angles_vecs_obs_info_speed_steer_ldist_light',
#     # 'input_type': 'wp_obs_info_speed_steer_ldist_goal_light',
#     # 'input_type': 'wp_obs_info_speed_steer_ldist_light',
#     # 'input_type': 'wp_obs_info_side_obs_info_speed_steer_ldist_light',
#     'input_type': 'wp_obs_more_info_speed_steer_ldist_light',
#     # 'input_type': 'transformer',
#     'use_scenarios': True,
#     'num_npc' : 0,
#     'sample_npc': True,
#     'num_npc_lower_threshold' : 20,
#     # 'num_npc_upper_threshold' : 200,
#     'num_npc_upper_threshold' : 380,
#     'npc_reset_freq': 10000,
#     'binarized_image': False,
#     'single_channel_image': False,
#     'noise_dim' : 1,
#     'const_collision_penalty': 250,
#     'collision_penalty_speed_coeff': 250,
#     'const_light_penalty': 250,
#     'light_penalty_speed_coeff': 250,
#     'static_penalty': 0,
#     'terminate_on_light' : False,
#     'enable_brake': True,
#     # 'log_freq': 1,
#     'zero_speed_threshold': 0.05,
#     'obstacle_dist_norm' : 60,
#     'spawn_points_fixed_idx' : [
#         54, 234, 108,  12, 175,  71, 116,  99, 196,  63, 205,  46,  96,
#        246, 128, 106, 143,  39,  72, 176, 140, 138,  91,  88, 241,  29,
#         28, 238, 119, 221, 163,  81,  47, 255, 235,  64, 216, 151, 145,
#         77,  35,  56,  68,  49, 154, 149, 201,  27, 212, 195, 230, 157,
#          3,   5,  20, 193,   6,  90,  18,  13, 139,  44, 122, 220, 125,
#        115,  43,   4, 213,  30,  62, 242, 219, 171,  41, 203,  57, 248,
#        204, 226, 245, 135, 164, 153,  14, 188,   7, 123, 117, 222, 183,
#        152, 150, 185, 224,  19, 104, 111,  82,  79,   0,  33,  38, 146,
#         10, 173, 239,  32, 228, 209, 243, 200, 215, 236,  34,  84,  51,
#         73,  53, 170, 217, 237, 102, 156,  45, 253,  37, 210, 118,  86,
#         74,  61, 165, 179, 202, 101,  36, 132, 168, 137, 126, 178,  24,
#          1, 247, 107,  93, 148,  50,  98,  87, 133, 162,   2, 214, 124,
#        112, 211,  75, 121, 191, 113, 141,  26, 231, 174,  76, 207, 109,
#        244, 129, 103,  52,  42,  55, 180,  89, 181,  69,  48,  21,  16,
#        198,  66,  70, 130, 114,  15, 134,  40, 227, 223,  67,  78, 159,
#        252, 147,  17, 166,  11, 131, 161, 105, 167,  95, 172, 233, 251,
#        194,  60,  80, 182,  97,  59, 197,  25, 186, 136, 160, 120, 158,
#        189, 192, 190, 187, 142, 232,   9, 127, 206, 169,  23, 208,  94,
#        218,  83, 155,  65, 254, 249,  92, 240,  85, 100,  58,  22,   8,
#        225,  31, 229, 250, 110, 177, 199, 184, 144],
#     'test_fixed_spawn_points': False,
#     'train_fixed_spawn_points': False,
#     'testing': False,
#     'disable_collision': False,
#     'enable_static_termination': True,
#     'enable_obstacle_sensor': True,
#     'obs_sensor_vehicle_only': False,
#     'obs_cosine_velocity': True,
#     'check_obs_same_lane': True,
#     'front_obs_sensor_hit_radius': .5,
#     'side_obs_sensor_hit_radius': .7854, # pi / 4
#     'use_pid_in_frame_skip' : True,
#     'enable_lane_invasion_sensor' : True,
#     'enable_lane_invasion_termination' : True,
#     'enable_lane_invasion_collision' : True,
#     # 'enable_lane_invasion_termination' : False,
#     # 'enable_lane_invasion_collision' : False,
#     'front_obs_proximity_threshold' : 10,
#     'side_obs_proximity_threshold' : 5,
#     'traffic_light_proximity_threshold' : 10,
#     'min_dist_from_red_light' : 0,
#     'clip_reward' : False,
#     'default_obs_traffic_val': 1,
#     'reward_normalize_factor': 1,
#     'success_reward': 0,
#     'constant_positive_reward': 0,
#     'frame_stack_size' : 1,
#     'num_episodes' : 1,
#     'disable_traffic_light': False,
#     'disable_obstacle_info' : False,
#     'test_comparison': False,
#     'test_with_automatic_control': False,
#     'updated_scenarios': False,
#     'use_route_to_plan' : False,
#     'discrete_actions': DISCRETE_ACTIONS,
#     'episode_measurements': EPISODE_MEASUREMENTS,
# }

# ENV_CONFIG.update(TEST_CONFIG)

WEATHERS = {
        'ClearNoon': carla.WeatherParameters.ClearNoon,
        'ClearSunset': carla.WeatherParameters.ClearSunset,

        'CloudyNoon': carla.WeatherParameters.CloudyNoon,
        'CloudySunset': carla.WeatherParameters.CloudySunset,

        'WetNoon': carla.WeatherParameters.WetNoon,
        'WetSunset': carla.WeatherParameters.WetSunset,

        'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
        'MidRainSunset': carla.WeatherParameters.MidRainSunset,

        'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
        'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,

        'HardRainNoon': carla.WeatherParameters.HardRainNoon,
        'HardRainSunset': carla.WeatherParameters.HardRainSunset,

        'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
        'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
}
WEATHERS_IDS = list(WEATHERS)


def get_entry_point():
    return 'AutoPilot'


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1) # how many seconds until collision

    return collides, p1 + x[0] * v1


def check_episode_has_noise(lat_noise_percent, long_noise_percent):
    lat_noise = False
    long_noise = False
    if random.randint(0, 101) < lat_noise_percent:
        lat_noise = True

    if random.randint(0, 101) < long_noise_percent:
        long_noise = True

    return lat_noise, long_noise


class AutoPilot(MapAgent):

    # for stop signs
    PROXIMITY_THRESHOLD = 30.0  # meters
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 1.0  # meters

    def setup(self, path_to_conf_file):
        print(f"CONF FILE: {path_to_conf_file}")
        super().setup(path_to_conf_file)

    def _init(self):
        super()._init()

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        # for stop signs
        self._target_stop_sign = None # the stop sign affecting the ego vehicle
        self._stop_completed = False # if the ego vehicle has completed the stop sign
        self._affected_by_stop = False # if the ego vehicle is influenced by a stop sign

        # self.config = ENV_CONFIG
        self.config = defaultdict(int)
        self.target_speed = self.config['target_speed']
        # self._agent.rank = 0
        # self._agent.episode_measurements = {}
        # self._agent.episode_measurements['static_steps'] = 0
        # self._agent.episode_measurements['control_steer'] = 0
        # self._agent.episode_measurements['dist_to_light'] = -1
        # self._agent.episode_measurements['nearest_traffic_actor_id'] = -1
        # self._agent.episode_measurements['nearest_traffic_actor_state'] = None
        # self._agent.episode_measurements['initial_dist_to_red_light'] = -1
        # self._agent.episode_measurements['red_light_dist'] = -1
        # self._agent.episode_measurements['num_step_stopped'] = 0
        # self._agent.episode_measurements['traffic_light_orientation'] = -1
        # self._agent.episode_measurements['runover_light'] = False
        # # agent.episode_measurements['offlane_steps'] = 0
        # self._agent.episode_measurements['obstacle_init_dist'] = -1
        # self._agent.episode_measurements['obstacle_init_id'] = -1
        # self._agent.episode_measurements['obstacle_visible'] = False
        # self._agent.episode_measurements['obstacle_dist'] = -1
        # self._agent.episode_measurements['obstacle_speed'] = -1
        # self._agent.episode_measurements['obstacle_orientation'] = -1
        # self._agent._proximity_threshold = self.config['traffic_light_proximity_threshold']
        # self._agent._traffic_light_proximity_threshold = self.config['traffic_light_proximity_threshold']
        # self._agent._front_obs_proximity_threshold = self.config['front_obs_proximity_threshold']
        # self._agent.vehicle_actor = hero_actor
        # # add sensor
        # self._agent.actor_list = []
        # obs_sensors = {
        #     'front': sensors.ObstacleSensor(self._agent.vehicle_actor,
        #         distance=self.config['front_obs_proximity_threshold'],
        #         hit_radius=self.config['front_obs_sensor_hit_radius'],),
        #     'front_right': sensors.ObstacleSensor(self._agent.vehicle_actor,
        #         distance=self.config['side_obs_proximity_threshold'],
        #         hit_radius=self.config['side_obs_sensor_hit_radius'],
        #         transform=carla.Transform(rotation=carla.Rotation(yaw=45.))),
        #     'back_right': sensors.ObstacleSensor(self._agent.vehicle_actor,
        #         distance=self.config['side_obs_proximity_threshold'],
        #         hit_radius=self.config['side_obs_sensor_hit_radius'],
        #         transform=carla.Transform(rotation=carla.Rotation(yaw=135.))),
        #     'back_left': sensors.ObstacleSensor(self._agent.vehicle_actor,
        #         distance=self.config['side_obs_proximity_threshold'],
        #         hit_radius=self.config['side_obs_sensor_hit_radius'],
        #         transform=carla.Transform(rotation=carla.Rotation(yaw=225.))),
        #     'front_left': sensors.ObstacleSensor(self._agent.vehicle_actor,
        #         distance=self.config['side_obs_proximity_threshold'],
        #         hit_radius=self.config['side_obs_sensor_hit_radius'],
        #         transform=carla.Transform(rotation=carla.Rotation(yaw=315.))),
        # }
        # self._agent.obstacle_sensor = {}
        # for orient, sensor in obs_sensors.items():
        #     self._agent.obstacle_sensor[orient] = sensor
        #     self._agent.actor_list.append(sensor.sensor)

    def _update_obs_detector_via_sensor(self, agent):
        agent.episode_measurements['obstacle_visible'] = False
        agent.episode_measurements['obstacle_orientation'] = -1

        for suffix in agent.obstacle_sensor:
            agent.episode_measurements['obstacle_dist_{}'.format(suffix)] = -1
            agent.episode_measurements['obstacle_speed_{}'.format(suffix)] = -1

        # front obstacle detection
        found_obstacle = False
        same_lane = True
        world_frame = CarlaDataProvider._world.get_snapshot().frame
        if agent.obstacle_sensor['front'].frame == world_frame:
            if self.config['verbose']: print('FRAME:', world_frame, agent.obstacle_sensor['front'].frame)
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
                        self.steps, obstacle_actor.id, obstacle_actor.type_id, agent.rank, agent.episode_measurements['speed'] * 3.6, agent.episode_measurements['obstacle_speed'] * 3.6,
                        agent.episode_measurements['obstacle_init_dist'], agent.obstacle_sensor['front'].distance, same_lane, found_obstacle))
            if self.config['verbose'] or self.config['test_verbose']:
                print('[step {}][obstacle actor id {}][{}][agent {}][agt speed {:.2f}][obs speed {:.2f}][init dist {:.2f}][curr dist {:.2f}][same_lane {}][found {}]'.format(
                    self.steps, obstacle_actor.id, obstacle_actor.type_id, agent.rank, agent.episode_measurements['speed'] * 3.6, agent.episode_measurements['obstacle_speed'] * 3.6,
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
            if agent.obstacle_sensor[suffix].frame == world_frame:
                if self.config['verbose']: print('FRAME:', world_frame, agent.obstacle_sensor[suffix].frame)
                obstacle_actor = agent.obstacle_sensor[suffix].obstacle_actor
                if 'vehicle' in obstacle_actor.type_id:
                    same_lane = check_if_vehicle_in_same_lane(agent.vehicle_actor, obstacle_actor, agent.next_waypoints, self._map)
                found_obstacle = True
                agent.episode_measurements['obstacle_dist_{}'.format(suffix)] = agent.obstacle_sensor[suffix].distance
                # if 'vehicle' in obstacle_actor.type_id:
                if hasattr(obstacle_actor, 'get_velocity') and 'vehicle' not in obstacle_actor.type_id:
                    agent.episode_measurements['obstacle_speed_{}'.format(suffix)] = self.get_speed_from_velocity(obstacle_actor.get_velocity())
                else:
                    agent.episode_measurements['obstacle_speed_{}'.format(suffix)] = -1
                found_obstacle = found_obstacle and (not self.config['check_obs_same_lane'] or not same_lane)
            if not found_obstacle:
                agent.episode_measurements['obstacle_speed_{}'.format(suffix)] = -1
                agent.episode_measurements['obstacle_speed_{}'.format(suffix)] = -1

    def _update_traffic_light_states(self, agent):
        # TODO: Pass correct target waypoint to find_nearest_traffic_light() for US style traffic.
        traffic_actor, dist, traffic_light_orientation = agent.find_nearest_traffic_light(self.traffic_actors)
        stop_sign = self._scan_for_stop_sign()
        found_redlight = False
        if traffic_light_orientation is not None:
            agent.episode_measurements['traffic_light_orientation'] = traffic_light_orientation
        else:
            agent.episode_measurements['traffic_light_orientation'] = -1

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
                        print('[step {}][traffic light id {}][agent {}][speed {:.2f}][init dist {:.2f}][curr dist {:.2f}][state {}][found {}]'.format(self.steps,
                            traffic_actor.id, agent.rank, agent.episode_measurements['speed'] * 3.6, agent.episode_measurements['initial_dist_to_red_light'],
                            dist, traffic_actor.state, found_redlight))
                if self.config['verbose'] or self.config['test_verbose']:
                    print('[step {}][traffic light id {}][agent {}][speed {:.2f}][init dist {:.2f}][curr dist {:.2f}][state {}][found {}]'.format(self.steps,
                        traffic_actor.id, agent.rank, agent.episode_measurements['speed'] * 3.6, agent.episode_measurements['initial_dist_to_red_light'],
                        dist, traffic_actor.state, found_redlight))
            else:
                agent.episode_measurements['red_light_dist'] = -1
                agent.episode_measurements['initial_dist_to_red_light'] = -1

            agent.episode_measurements['nearest_traffic_actor_id'] = traffic_actor.id
            agent.episode_measurements['nearest_traffic_actor_state'] = traffic_actor.state
            # print('[agent {} init {}] traffic light info'.format(
            #     agent.rank, agent.episode_measurements['initial_dist_to_red_light']), traffic_actor.id, traffic_actor.state, dist)

        elif stop_sign is not None:
            # if has stopped for this actor, skip this stop sign actor
            if agent.episode_measurements['nearest_traffic_actor_id'] == stop_sign.id and \
                agent.episode_measurements['num_step_stopped'] > 2 * self.target_speed:
                agent.episode_measurements['red_light_dist'] = -1
            else:
                # pretend there is a light ahead for a short period of time
                stop_dist = stop_sign.get_transform().location.distance(self._agent.vehicle_actor.get_location())
                agent.episode_measurements['red_light_dist'] = stop_dist
                agent.episode_measurements['nearest_traffic_actor_id'] = stop_sign.id
                agent.episode_measurements['num_step_stopped'] += 1
                print('[step {}][stop sign id {}][agent {}][speed {:.2f}][curr dist {:.2f}]'.format(self.steps,
                    stop_sign.id, agent.rank, agent.episode_measurements['speed'] * 3.6, stop_dist))
        else:
            agent.episode_measurements['red_light_dist'] = -1
            agent.episode_measurements['initial_dist_to_red_light'] = -1
            agent.episode_measurements['nearest_traffic_actor_id'] = -1
            agent.episode_measurements['nearest_traffic_actor_state'] = None
            agent.episode_measurements['num_step_stopped'] = 0
            # print('[agent {} init {}] traffic light info'.format(
            #     agent.rank, agent.episode_measurements['initial_dist_to_red_light']), -1, -1, -1)

        agent.episode_measurements['dist_to_light'] = dist

    def _configure_planner(self, map_string):
        self.global_planner = GlobalPlanner()
        self.trace_route = []

        # cheating
        plan = []
        for transform, option in self._global_plan_world_coord:
            wp = self._map.get_waypoint(transform.location)
            self.trace_route.append((wp, option))

        # print(self.trace_route)
        wp_list = []
        for wp, _ in self.trace_route:
            wp_list.append(wp)
        draw_waypoints(CarlaDataProvider._world, wp_list)

        self.global_planner.set_global_plan(self.trace_route)

    def _get_vehicle_transform(self, gnss_reading, imu_reading):
        # Convert to x,y,z
        #world_coords = get_world_coords_from_latlong(gnss_reading.latitude, gnss_reading.longitude, gnss_reading.altitude)
        world_coords = get_world_coords_from_latlong(gnss_reading[0], gnss_reading[1], gnss_reading[2], self._map)

        x,y,z = world_coords[0][0], world_coords[1][0], world_coords[2][0]

        # Construct transform
        return carla.Transform(carla.Location(x = x, y = y, z = z), carla.Rotation(yaw = imu_reading[-1]))

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

    def compute_wp_stats(self, vehicle_transform):
        # "Return type: list containing [mean_angle, ldist, distance_to_goal_trajec]"
        mean_angle, ldist, distance_to_goal_trajec, next_waypoints, _, _ = \
            self.global_planner.get_next_orientation_new(vehicle_transform, num_next_waypoints=5)
        return mean_angle, ldist, distance_to_goal_trajec, next_waypoints

    def cosine_between_velocities(self, v1, v2):
        return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z) / \
            (self.get_speed_from_velocity(v1) * self.get_speed_from_velocity(v2) + 1e-6)

    def cosine_between_obs(self, agt_v, obs_v):
        if self.get_speed_from_velocity(agt_v) < self.config['zero_speed_threshold'] or \
            self.get_speed_from_velocity(obs_v) < self.config['zero_speed_threshold']:
            return 0.
        else:
            return self.cosine_between_velocities(agt_v, obs_v)

    def _update_env_obs(self, agent, input_data):
        # Configure planner when we first receive MAP info
        if not self._route_assigned:
            # print('[479], self._global_plan', len(self._global_plan), self._global_plan)
            # print('[480], self._global_plan_world_coord', len(self._global_plan_world_coord), self._global_plan_world_coord)
            self._configure_planner(input_data['OpenDRIVE'][1]['opendrive'])
            self._route_assigned = True

        # self._agent.episode_measurements["mean_angle"], self._agent.episode_measurements['ldist'], \
        #     self._agent.episode_measurements['distance_to_goal_trajec'] = self.compute_wp_stats(vehicle_transform)
        self._agent.episode_measurements['next_orientation'], self._agent.episode_measurements['dist_to_trajectory'], \
            self._agent.episode_measurements['distance_to_goal_trajec'], self._agent.next_waypoints = self.compute_wp_stats(
                self._agent.vehicle_actor.get_transform())
        self._agent.episode_measurements['distance_to_goal_trajec'] = \
            self._agent.episode_measurements['distance_to_goal_trajec'] / 500 # to match env.py preproc

        # self._agent.episode_measurements['steer'] = self.previous_steer

        # self._agent.episode_measurements['speed'] = input_data['SPEED'][1]['speed']
        self._agent.episode_measurements['speed'] = self.get_speed_from_velocity(self._agent.vehicle_actor.get_velocity())
        # print(input_data['SPEED'][1]['speed'])
        if not self.config['disable_obstacle_info']:
            if self.config['input_type'] == 'wp_obs_info_side_obs_info_speed_steer_ldist_light':
                self._update_obs_detector_via_privilege(agent)
            elif self.config['enable_obstacle_sensor']:
                self._update_obs_detector_via_sensor(agent)
            else:
                self._update_obs_detector_via_privilege(agent)

        if not self.config['disable_traffic_light']:
            self._update_traffic_light_states(agent)

        if self._is_static(agent):
            self._agent.episode_measurements['static_steps'] += 1
        else:
            self._agent.episode_measurements['static_steps'] = 0

    def get_speed_from_velocity(self, velocity):
        speed = np.sqrt(velocity.x ** 2 + velocity.y **2 + velocity.z **2)
        return speed

    def create_observations(self, agent):
        obs = {}

        obs['observation'] = np.array([agent.episode_measurements['next_orientation']])

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
        return obs['observation']

    def _get_angle_to(self, pos, theta, target):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle

        return angle

    def _get_control(self, target, far_target, tick_data):
        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        # Steering.
        angle_unnorm = self._get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        target_speed = 4.0 if should_slow else 7.0
        brake = self._should_brake()
        target_speed = target_speed if not brake else 0.0

        self.should_slow = int(should_slow)
        self.should_brake = int(brake)
        self.angle = angle
        self.angle_unnorm = angle_unnorm
        self.angle_far_unnorm = angle_far_unnorm

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        if brake:
            steer *= 0.5
            throttle = 0.0

        return steer, throttle, brake, target_speed

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        # change weather for visual diversity
        if self.step % 10 == 0:
            index = random.choice(range(len(WEATHERS)))
            self.weather_id = WEATHERS_IDS[index]
            weather = WEATHERS[WEATHERS_IDS[index]]
            print (self.weather_id, weather)
            self._world.set_weather(weather)

        data = self.tick(input_data)
        gps = self._get_position(data)

        near_node, near_command = self._waypoint_planner.run_step(gps)
        far_node, far_command = self._command_planner.run_step(gps)

        steer, throttle, brake, target_speed = self._get_control(near_node, far_node, data)

        control = carla.VehicleControl()
        control.steer = steer + 1e-2 * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake)
        # if self.step % 10 == 0 and self.save_path is not None:
        #     self.save(near_node, far_node, near_command, steer, throttle, brake, target_speed, data)

        # self._update_env_obs(self._agent, input_data)
        # obs = self.create_observations(self._agent)
        # action = np.array([steer, target_speed])

        return control

    def _should_brake(self):
        actors = self._world.get_actors()

        vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        light = self._is_light_red(actors.filter('*traffic_light*'))
        walker = self._is_walker_hazard(actors.filter('*walker*'))
        stop_sign = self._is_stop_sign_hazard(actors.filter('*stop*'))

        self.is_vehicle_present = 1 if vehicle is not None else 0
        self.is_red_light_present = 1 if light is not None else 0
        self.is_pedestrian_present = 1 if walker is not None else 0
        self.is_stop_sign_present = 1 if stop_sign is not None else 0

        return any(x is not None for x in [vehicle, light, walker, stop_sign])

    def _point_inside_boundingbox(self, point, bb_center, bb_extent):
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def _is_actor_affected_by_stop(self, actor, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
            return affected

        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._world.get_map().get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                waypoint = waypoint.next(self.WAYPOINT_STEP)[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self._point_inside_boundingbox(actor_location, transformed_tv, stop.trigger_volume.extent):
                affected = True

        return affected

    def _is_stop_sign_hazard(self, stop_sign_list):
        if self._affected_by_stop:
            if not self._stop_completed:
                current_speed = self._get_forward_speed()
                if current_speed < self.SPEED_THRESHOLD:
                    self._stop_completed = True
                    return None
                else:
                    return self._target_stop_sign
            else:
                # reset if the ego vehicle is outside the influence of the current stop sign
                if not self._is_actor_affected_by_stop(self._vehicle, self._target_stop_sign):
                    self._affected_by_stop = False
                    self._stop_completed = False
                    self._target_stop_sign = None
                return None

        ve_tra = self._vehicle.get_transform()
        ve_dir = ve_tra.get_forward_vector()

        wp = self._world.get_map().get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in stop_sign_list:
                if self._is_actor_affected_by_stop(self._vehicle, stop_sign):
                    # this stop sign is affecting the vehicle
                    self._affected_by_stop = True
                    self._target_stop_sign = stop_sign
                    return self._target_stop_sign

        return None

    def _is_light_red(self, lights_list):
        if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
            affecting = self._vehicle.get_traffic_light()

            for light in self._traffic_lights:
                if light.id == affecting.id:
                    return affecting

        return None

    def _is_walker_hazard(self, walkers_list):
        z = self._vehicle.get_location().z
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(walker.get_velocity()))

            if s2 < 0.05:
                v2_hat *= s2

            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat

            collides, collision_point = get_collision(p1, v1, p2, v2)

            if collides:
                return walker

        return None

    def _is_vehicle_hazard(self, vehicle_list):
        z = self._vehicle.get_location().z

        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._vehicle.get_location())
        s1 = max(10, 3.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity()))) # increases the threshold distance
        v1_hat = o1
        v1 = s1 * v1_hat

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
            angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)

            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue

            return target_vehicle

        return None

