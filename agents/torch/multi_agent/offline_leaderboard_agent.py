#!/usr/bin/env python

import sys, os
import carla
import torch
import pyproj
import traceback
import time
import math
import copy

from config import ENV_CONFIG, TEST_CONFIG, OFFLINE_CONFIG

sys.path.append(os.path.abspath(OFFLINE_CONFIG['offline_repo_location']))
sys.path.append(os.path.abspath(OFFLINE_CONFIG['offline_repo_location'] + '/..'))

from trajectory.policies.dvae_bt_policy import DVAEBTPolicy
from trajectory.policies.bt_policy import BTPolicy
from trajectory.policies.dt_policy import DTPolicy
from trajectory.policies.tt_policy import TTPolicy
from trajectory.policies.rlkit_policy import RLKitPolicy
import trajectory.utils as utils

import numpy as np
import environment.carla_9_4.controller as controller
import environment.carla_9_4.sensors as sensors

from environment.carla_9_4.agents.navigation.basic_agent import BasicAgent
from environment.carla_9_4.planner import GlobalPlanner
from environment.carla_9_4.reward import compute_reward
from network import PPOActorCritic_Continuous
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from leaderboard.utils.route_manipulation import interpolate_trajectory

from environment.carla_9_4.dashcam import Visualizer

from environment.carla_9_4.env_util import (
    check_if_vehicle_in_same_lane,
    get_world_coords_from_latlong,
    convert_route_from_GPS_world
)

# transfuser autopilot
from transfuser_autopilot import PIDController

os.environ["OMP_NUM_THREADS"] = '1'
print('--------------------[PID {}]--------------------'.format(os.getpid()))


policy_set = {'dvae_dt', 'bt', 'dt', 'tt', 'iql'}

def infer_policy_class(ckpt_folder):
    folder_list = ckpt_folder.split('/')
    for folder in folder_list:
        if folder in policy_set:
            return folder
    return 'iql'

policy_type = infer_policy_class(OFFLINE_CONFIG['offline_policy_location'])

if policy_type != 'iql':
    class Parser(utils.Parser):
        dataset: str = 'lowerdimobs-random-ttc'
        config: str = 'transformer_rl.config.offline'

    if policy_type == 'tt':
        args = Parser().parse_args('tt_plan')
    else:
        args = Parser().parse_args('plan')

    utils.set_device(ENV_CONFIG['device'])

    dataset = utils.load_from_config(
        OFFLINE_CONFIG['offline_policy_location'],
        'data_config.pkl')

    discount = dataset.discount
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    gpt, gpt_epoch = utils.load_model(
            OFFLINE_CONFIG['offline_policy_location'],
            epoch='latest', device=ENV_CONFIG['device'])
    print('[59 load model]', gpt, gpt_epoch)

    gpt.eval()
else:
    policy = RLKitPolicy(OFFLINE_CONFIG['offline_policy_location'],
        ENV_CONFIG['device'])


if policy_type == 'dvae_dt':

    policy = DVAEBTPolicy(
        gpt,
        args.horizon,
        observation_dim,
        action_dim,
        discount,
        bs=1,
        max_history=args.max_context_transitions,
        device=ENV_CONFIG['device'],
    )
elif policy_type == 'bt':
    policy = BTPolicy(
        gpt,
        observation_dim,
        action_dim,
        discount,
        bs=1,
        max_history=args.max_context_transitions,
        device=ENV_CONFIG['device'],
    )
elif policy_type == 'dt':
    max_return = dataset.get_max_return()
    policy = DTPolicy(
        gpt,
        max_return * 1.0,
        observation_dim,
        action_dim,
        discount,
        bs=1,
        max_history=args.max_context_transitions,
        device=ENV_CONFIG['device'],
    )
elif policy_type == 'tt':
    discretizer = dataset.discretizer
    value_fn = lambda x: discretizer.value_fn(x, 'mean')
    policy = TTPolicy(
        gpt,
        discretizer,
        args.horizon,
        args.beam_width,
        args.n_expand,
        value_fn,
        observation_dim,
        action_dim,
        discount,
        verbose=args.verbose,
        k_obs=args.k_obs,
        k_act=args.k_act,
        cdf_obs=args.cdf_obs,
        cdf_act=args.cdf_act,
        prefix_context=args.prefix_context,
        max_history=args.max_context_transitions,
        device=ENV_CONFIG['device'],
    )

def get_entry_point():
    return 'PPOAgent'

def draw_waypoints(world, waypoints, z=0.5):
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

ENV_CONFIG.update(TEST_CONFIG)

if ENV_CONFIG['input_type'] == 'wp_obs_info_speed_steer_ldist_goal_light':
    N_S, N_A = 8, 2
elif ENV_CONFIG['input_type'] == 'wp_obs_info_speed_steer_ldist_light':
    N_S, N_A = 7, 2
elif ENV_CONFIG['input_type'] == 'wp_obs_info_side_obs_info_speed_steer_ldist_light':
    N_S, N_A = 11, 2
elif ENV_CONFIG['input_type'] == 'wp_obs_more_info_speed_steer_ldist_light':
    N_S, N_A = 15, 2
else:
    N_S, N_A = 7, 2

episode = -1
savetime = lambda: time.strftime('%b%d%I%M%p%S')
vid_log_dir = '{}/{}_{}'.format('./video_logs',
    'LDB_test', savetime())
sub_folder = None
videos = False


class PPOAgent(AutonomousAgent):
    _agent = None
    _route_assigned = False

    def __init__(self, *args, **kwargs):
        global episode, savetime, videos, sub_folder, vid_log_dir
        # self.glb_policy = PPOActorCritic_Continuous(N_S, N_A).to(ENV_CONFIG['device'])
        # _ckpt = torch.load('../../torch/multi_agent/' + TEST_CONFIG['checkpoint'], map_location='cpu')
        # self.glb_policy.load_state_dict(_ckpt['glb_policy'])
        self.config = ENV_CONFIG
        self.target_speed = self.config['target_speed']
        if videos:
            self.viz = Visualizer(images_path=vid_log_dir, video_path=vid_log_dir)
        # have to put init at last since it will call self.setup first
        super().__init__(*args, **kwargs)


    def destroy(self):
        if self._agent is not None:
            for _ in range(len(self._agent.actor_list)):
                try:
                    actor = self._agent.actor_list.pop()
                    print(actor)
                    actor.destroy()
                except Exception as e:
                    print("Error during destroying sensor actor {0}:{1}: {2}".format(
                        actor.type_id, actor.id, traceback.format_exc()))
        time.sleep(1)


    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        global episode, savetime, videos, sub_folder, vid_log_dir
        self.track = Track.MAP
        self._route_assigned = False
        self._agent = None
        # self.previous_steer = 0
        self.steps = 0
        if videos and episode >= 0:
            self.viz.generate_video(sub_folder)
            self.viz.remove_images(sub_folder)
        episode += 1
        sub_folder ='ep{}'.format(episode)

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

    @staticmethod
    def point_inside_boundingbox(point, bb_center, bb_extent):
        """
        X
        :param point:
        :param bb_center:
        :param bb_extent:
        :return:
        """

        # pylint: disable=invalid-name
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

    def _is_actor_affected_by_stop(self, actor, stop, multi_step=5):
        """
        Check if the given actor is affected by the stop
        """
        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        dist = stop_location.distance(current_location)
        if dist > self.config['front_obs_proximity_threshold']:
            return False

        affected = False
        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._map.get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                next_wps = waypoint.next(2.0)
                if not next_wps:
                    break
                waypoint = next_wps[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self.point_inside_boundingbox(actor_location, transformed_tv, stop.trigger_volume.extent):
                affected = True

        return affected

    def _scan_for_stop_sign(self):
        target_stop_sign = None

        ve_tra = CarlaDataProvider.get_transform(self._agent.vehicle_actor)
        ve_dir = ve_tra.get_forward_vector()

        wp = self._map.get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in self._list_stop_signs:
                if self._is_actor_affected_by_stop(self._agent.vehicle_actor, stop_sign):
                    # this stop sign is affecting the vehicle
                    target_stop_sign = stop_sign
                    break

        return target_stop_sign

        fwd = current_transform.get_forward_vector()
        forward_vector = np.array([fwd.x, fwd.y])
        d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

        return d_angle < 90.0, d_angle, norm_target

    def _update_obs_detector_via_privilege(self, agent):
        agent.episode_measurements['obstacle_visible'] = False
        agent.episode_measurements['obstacle_orientation'] = -1

        agent.episode_measurements['obstacle_dist_left'] = -1
        agent.episode_measurements['obstacle_dist_right'] = -1
        agent.episode_measurements['obstacle_speed_left'] = -1
        agent.episode_measurements['obstacle_speed_right'] = -1

        min_obs_distance = 100000000
        found_obstacle = False
        for target_vehicle in CarlaDataProvider.get_world().get_actors():
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
        # print('[agent {} init {}] traffic light info'.format(
        #     agent.rank, agent.episode_measurements['initial_dist_to_red_light']), -1, -1, -1)

    def _preprocess_image(self, image):
        #array = np.reshape(array, (image.shape[0], image.shape[1], 4))
        image = image[:, :, :3]     # BGR
        image = image[:, :, ::-1]   # RGB
        return image

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

    def sensors(self):
        sensors = [
            # {'type': 'sensor.camera.rgb', 'x': 2.0, 'y': 0.0, 'z': 1.4, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            # 'width': 128, 'height': 128, 'fov': 100, 'id': 'Center'},
            # {'type': 'sensor.camera.rgb', 'x': 2.0, 'y': 0.0, 'z': 1.4, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            # 'width': 800, 'height': 400, 'fov': 100, 'id': 'Center_high_res'},
            {'type': 'sensor.camera.rgb', 'x': 13., 'y': 0., 'z': 18., 'roll': 0.0, 'pitch': 270., 'yaw': 0.0,
            'width': 400, 'height': 800, 'fov': 90, 'id': 'BEV'},
            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
            {'type': 'sensor.other.imu', 'x': 2.0, 'y': 0.0, 'z': 1.4, 'roll': 0.0, 'pitch': 0.0,
             'yaw': -90.0, 'id': 'IMU'},
            {'type': 'sensor.opendrive_map', 'reading_frequency': 1, 'id': 'OpenDRIVE'},
           {'type': 'sensor.speedometer',  'reading_frequency': 20, 'id': 'SPEED'},
           ]
        return sensors

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
        input_data['IMU'][1][-1] = (input_data['IMU'][1][-1]*(180/np.pi))
        if input_data['IMU'][1][-1]>180:
            input_data['IMU'][1][-1] = input_data['IMU'][1][-1]-360
        # Configure planner when we first receive MAP info
        if not self._route_assigned:
            # print('[479], self._global_plan', len(self._global_plan), self._global_plan)
            # print('[480], self._global_plan_world_coord', len(self._global_plan_world_coord), self._global_plan_world_coord)
            self._configure_planner(input_data['OpenDRIVE'][1]['opendrive'])
            self._route_assigned = True

        # vehicle_transform = self._get_vehicle_transform(input_data["GPS"][1], input_data['IMU'][1])

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

        if self.config["input_type"] == 'wp_obs_info_speed_steer_ldist_goal_light':

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

        return obs['observation']

    def get_control(self, input_data, action):
        """ Get Control object for Carla from action
        Input:
            - action: tuple containing (steer, throttle, brake) in [-1, 1]
        Output:
            - control: Control object for Carla
        """
        steer = self._turn_controller.step(float(action[0]))
        steer = np.clip(steer, -1., 1.)
        steer = round(steer, 3)
        target_speed = (action[1] * 1.5) + 1
        # print('action[1]', action[1], 'target_speed', target_speed)
        target_speed = float(np.clip(target_speed * self.target_speed / 2, 0, self.target_speed))

        # if static, push a little bit
        # if self._agent.episode_measurements['static_steps'] > self.config['max_static_steps']:
        if self._agent.episode_measurements['static_steps'] > 10:
            target_speed = self.target_speed
            print('[878] static_episode_problem. Need push !!!')
            self.static_episode_problem = True

        # TODO: Need to replace this once we get to know how to extract agent's current velocity from IMU/speedometer sensors
        # current_speed = self.get_speed_from_velocity(input_data['SPEED'][1]['speed']) * 3.6
        # current_speed = input_data['SPEED'][1]['speed'] *  3.6
        # current_speed = input_data['SPEED'][1]['speed']
        # current_speed = self._agent.episode_measurements['speed']
        current_speed = self._agent.episode_measurements['speed'] * 3.6

        delta = np.clip(target_speed - current_speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        brake = False
        if target_speed < current_speed: # brake
            brake = True
            steer *= 0.5
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        return control

    def run_step(self, input_data, timestamp):
        global episode, savetime, videos, sub_folder, vid_log_dir
        self.steps += 1
        # print('[876]')
        if not self._agent:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break
            if hero_actor:
                # init add-ons
                policy.reset()
                self.traffic_actors = CarlaDataProvider.get_world().get_actors().filter("*traffic_light*")
                self._map = CarlaDataProvider.get_map()
                self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
                self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)
                self.static_episode_problem = False
                self._agent = BasicAgent(hero_actor, proximity_threshold=15.)
                self._agent.rank = 0
                self._agent.episode_measurements = {}
                self._agent.episode_measurements['static_steps'] = 0
                self._agent.episode_measurements['control_steer'] = 0
                self._agent.episode_measurements['dist_to_light'] = -1
                self._agent.episode_measurements['nearest_traffic_actor_id'] = -1
                self._agent.episode_measurements['nearest_traffic_actor_state'] = None
                self._agent.episode_measurements['initial_dist_to_red_light'] = -1
                self._agent.episode_measurements['red_light_dist'] = -1
                self._agent.episode_measurements['num_step_stopped'] = 0
                self._agent.episode_measurements['traffic_light_orientation'] = -1
                self._agent.episode_measurements['runover_light'] = False
                # agent.episode_measurements['offlane_steps'] = 0
                self._agent.episode_measurements['obstacle_init_dist'] = -1
                self._agent.episode_measurements['obstacle_init_id'] = -1
                self._agent.episode_measurements['obstacle_visible'] = False
                self._agent.episode_measurements['obstacle_dist'] = -1
                self._agent.episode_measurements['obstacle_speed'] = -1
                self._agent.episode_measurements['obstacle_orientation'] = -1
                self._agent.episode_measurements['num_collisions'] = 0
                self._agent.episode_measurements['collision_actor_id'] = -1
                self._agent.episode_measurements['collision_actor_type'] = None
                self._agent.episode_measurements['distance_to_goal'] = 10000
                self._agent.episode_measurements['num_laneintersections'] = 0
                self._agent.episode_measurements['out_of_road'] = False
                self._agent.episode_measurements['unlawful_lane_change'] = False
                self._agent._proximity_threshold = self.config['traffic_light_proximity_threshold']
                self._agent._traffic_light_proximity_threshold = self.config['traffic_light_proximity_threshold']
                self._agent._front_obs_proximity_threshold = self.config['front_obs_proximity_threshold']
                self._agent.vehicle_actor = hero_actor

                # add sensor
                self._agent.actor_list = []
                self._agent.collision_sensor = sensors.CollisionSensor(self._agent.vehicle_actor)
                self._agent.actor_list.append(self._agent.collision_sensor.sensor)

                # if self.config["enable_lane_invasion_sensor"]:
                #     self._agent.lane_invasion_sensor = sensors.LaneInvasionSensor(self._agent.vehicle_actor)
                #     self._agent.actor_list.append(self._agent.lane_invasion_sensor.sensor)

                if self.config['input_type'] == 'wp_obs_more_info_speed_steer_ldist_light':
                    obs_sensors = {
                        'front': sensors.ObstacleSensor(self._agent.vehicle_actor,
                            distance=self.config['front_obs_proximity_threshold'],
                            hit_radius=self.config['front_obs_sensor_hit_radius'],),
                        'front_right': sensors.ObstacleSensor(self._agent.vehicle_actor,
                            distance=self.config['side_obs_proximity_threshold'],
                            hit_radius=self.config['side_obs_sensor_hit_radius'],
                            transform=carla.Transform(rotation=carla.Rotation(yaw=45.))),
                        'back_right': sensors.ObstacleSensor(self._agent.vehicle_actor,
                            distance=self.config['side_obs_proximity_threshold'],
                            hit_radius=self.config['side_obs_sensor_hit_radius'],
                            transform=carla.Transform(rotation=carla.Rotation(yaw=135.))),
                        'back_left': sensors.ObstacleSensor(self._agent.vehicle_actor,
                            distance=self.config['side_obs_proximity_threshold'],
                            hit_radius=self.config['side_obs_sensor_hit_radius'],
                            transform=carla.Transform(rotation=carla.Rotation(yaw=225.))),
                        'front_left': sensors.ObstacleSensor(self._agent.vehicle_actor,
                            distance=self.config['side_obs_proximity_threshold'],
                            hit_radius=self.config['side_obs_sensor_hit_radius'],
                            transform=carla.Transform(rotation=carla.Rotation(yaw=315.))),
                    }
                else:
                    obs_sensors = {
                        'front': sensors.ObstacleSensor(self._agent.vehicle_actor,
                            distance=self.config['front_obs_proximity_threshold'],
                            hit_radius=self.config['front_obs_sensor_hit_radius'],),
                    }
                self._agent.obstacle_sensor = {}
                for orient, sensor in obs_sensors.items():
                    self._agent.obstacle_sensor[orient] = sensor
                    self._agent.actor_list.append(sensor.sensor)

                # add stop ligt list
                self._list_stop_signs = []
                for _actor in CarlaDataProvider.get_world().get_actors():
                    if 'traffic.stop' in _actor.type_id:
                        self._list_stop_signs.append(_actor)
                print('[1002] self._list_stop_signs', self._list_stop_signs)

        # preprocess_inputs = self.preprocess_inputs(input_data)
        if videos:
            high_res_rgb = self._preprocess_image(input_data['BEV'][1])
            self.viz.save_image(high_res_rgb, sub_folder=sub_folder)
        # print([preprocess_inputs['mean_angle'], preprocess_inputs['ldist'], preprocess_inputs['distance_to_goal_trajec'],
            # preprocess_inputs['steer'], preprocess_inputs['speed']])
        # if(self.ctr%200==0):
        #     st()
        # action = self.get_action(preprocess_inputs)
        self._update_env_obs(self._agent, input_data)
        self._agent.prev_measurement = copy.deepcopy(self._agent.episode_measurements)

        obs = self.create_observations(self._agent)
        # print(obs)

        # state_tensor = torch.from_numpy(obs).to(torch.float).to(ENV_CONFIG['device'])

        if policy_type == 'dvae_dt':
            action, sequence, candidates, world_index, policy_index = policy(
                obs, max_horizon=None, return_plans=True)
        elif policy_type in {'bt', 'dt', 'iql'}:
            action = policy(obs)
        elif policy_type == 'tt':
            action, sequence, candidates = policy(
                obs, max_horizon=None, return_plans=True)
        # print(self.steps, obs, action)
        # action, _ = self.glb_policy.act(state_tensor, deterministic=True)
        self._agent.episode_measurements['num_collisions'] = self._agent.collision_sensor.num_collisions
        self._agent.episode_measurements['collision_actor_id'] = self._agent.collision_sensor.actor_id
        self._agent.episode_measurements['collision_actor_type'] = self._agent.collision_sensor.actor_type

        step_reward = compute_reward(name=self.config['reward_function'],
            prev_measurement=self._agent.prev_measurement,
            cur_measurement=self._agent.episode_measurements,
            config=self.config,
            verbose=self.config["verbose"])

        if policy_type == 'dt' or policy_type == 'tt':
            policy.update_context(obs, action, np.array(step_reward))
        # elif policy_type == 'tt':
            # print(obs.shape, action.shape, action[:, None].shape)
        #     policy.update_context(obs, action, step_reward)

        control = self.get_control(input_data, action)

        self._agent.episode_measurements['control_steer'] = control.steer
        self._agent.episode_measurements['control_throttle'] = control.throttle
        self._agent.episode_measurements['control_brake'] = control.brake
        self._agent.episode_measurements['control_reverse'] = control.reverse
        self._agent.episode_measurements['control_hand_brake'] = control.hand_brake

        self._agent.prev_measurement = copy.deepcopy(self._agent.episode_measurements)

        return control

