#!/usr/bin/env python

import os
import carla
import torch
import pyproj

from collections import deque
import numpy as np

from environment.carla_9_4.agents.navigation.basic_agent import BasicAgent
from environment.carla_9_4.planner import GlobalPlanner
from network import PPOActorCritic_Continuous
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from config import ENV_CONFIG, TEST_CONFIG

os.environ["OMP_NUM_THREADS"] = '1'
print('--------------------[PID {}]--------------------'.format(os.getpid()))

def get_entry_point():
    return 'PPOAgent'

ENV_CONFIG.update(TEST_CONFIG)
N_S, N_A = 11, 2

def _latlon_to_ecef(lat,lon,alt):
    # Projections
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    # Transform from lat/lon to ecef
    x,y,z= pyproj.transform(p1=lla,
        p2 = ecef,
        x = lon,
        y = lat,
        z = alt,
        radians=False)

    return x, y, z

def get_world_coords_from_latlong(latitude, longitude, altitude, world_map):
    origin_latlong = world_map.transform_to_geolocation(carla.Location())

    # Origin in ECEF coordinates
    O_ecef = _latlon_to_ecef(origin_latlong.latitude, origin_latlong.longitude, origin_latlong.altitude)

    # Convert GNSS data to ECEF coordinates
    P_ecef = _latlon_to_ecef(latitude, longitude, altitude)

    # Calculate difference between current location and origin
    #FIXME The /2 constant is a hacky fix to get this working - this shouldn't be here
    delta = np.expand_dims(np.array(P_ecef) - np.array(O_ecef), axis = 1)

    # Create the rotation matrix to convert from ECEF to ENU Coords
    ecef_to_enu_rot = np.array(
        [[-np.sin(longitude), np.cos(longitude), 0],
         [-np.sin(latitude) * np.cos(longitude), -np.sin(latitude) * np.sin(longitude), np.cos(latitude)],
         [np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)]]
    )
    enu = ecef_to_enu_rot @ delta

    # Create rotation matrix to convert from right hand ENU frame to left-hand CARLA frame
    enu_to_carla_rot = np.array(
        [[1, 0, 0],
         [0,-1, 0],
         [0, 0, 1]]
    )
    return enu_to_carla_rot @ enu


def convert_route_from_GPS_world(route, world_map):

    # Example route input
    # route =[({'z': 0.0, 'lat': 48.99822669411668, 'lon': 8.002271601998707}, RoadOption.LEFT),
    #     ({'z': 0.0, 'lat': 48.99822669411668, 'lon': 8.002709765148996}, RoadOption.RIGHT),
    #     ({'z': 0.0, 'lat': 48.99822679980298, 'lon': 8.002735250105061}, RoadOption.STRAIGHT)
    #     ]

    mapped_route = []
    for idx, pt in enumerate(route):
        print(pt)
        altitude = pt[0]['z']
        latitude = pt[0]['lat']
        longitude = pt[0]['lon']
        world_coord = get_world_coords_from_latlong(latitude, longitude, altitude, world_map)
        x, y, z = world_coord[0][0], world_coord[1][0], world_coord[2][0]
        mapped_route.append(carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation()))
    return mapped_route


class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=30)

    def pid_control(self, target_speed, current_speed, enable_brake=False):
        """
        Estimate the throttle of the vehicle based on the PID equations

        :param target_speed:  target speed in Km/h
        :param current_speed: current speed of the vehicle in Km/h
        :return: throttle control in the range [0, 1]
        """
        _e = (target_speed - current_speed)
        self._e_buffer.append(_e)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0
        
        if enable_brake:
            throttle_min_clip = -1.0
        else:
            throttle_min_clip = 0.0
        
        return np.clip((self._K_P * _e) + (self._K_D * _de / self._dt) + (self._K_I * _ie * self._dt), throttle_min_clip, 1.0)


class PPOAgent(AutonomousAgent):
    _agent = None
    _route_assigned = False
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.glb_policy = PPOActorCritic_Continuous(8, 2).to(ENV_CONFIG['device'])
        _ckpt = torch.load('../../torch/multi_agent/' + TEST_CONFIG['checkpoint'], map_location='cpu')
        self.glb_policy.load_state_dict(_ckpt['glb_policy'])
        self.target_speed = 20
        self.args_longitudinal_dict = {
            'K_P': 0.1,
            'K_D': 0.0005,
            'K_I': 0.4,
            'dt': 1/10.0}        
        self.controller = PIDLongitudinalController(
            K_P=self.args_longitudinal_dict['K_P'], 
            K_D=self.args_longitudinal_dict['K_D'], 
            K_I=self.args_longitudinal_dict['K_I'], 
            dt=self.args_longitudinal_dict['dt'],)
        self.ctr = 0

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.MAP
        self._route_assigned = False
        self._agent = None
        self.previous_steer = 0


    def _configure_planner(self, map_string):
        self._map = carla.Map("map", map_string)

        # Instantiate the global planner
        self.scenario_route = convert_route_from_GPS_world(self._global_plan, self._map)

        self.global_planner = GlobalPlanner()
        self.trace_route = []
        for idx in range(len(self.scenario_route) - 1):
            source = self.scenario_route[idx]
            destination = self.scenario_route[idx+1]
            trace_route = self.global_planner.trace_route(self._map,
                            source, destination)
            self.trace_route.extend(trace_route)

        self.global_planner.set_global_plan(self.trace_route)

    def _get_vehicle_transform(self, gnss_reading, imu_reading):
        # Convert to x,y,z
        #world_coords = get_world_coords_from_latlong(gnss_reading.latitude, gnss_reading.longitude, gnss_reading.altitude)
        world_coords = get_world_coords_from_latlong(gnss_reading[0], gnss_reading[1], gnss_reading[2], self._map)

        x,y,z = world_coords[0][0], world_coords[1][0], world_coords[2][0]

        # Construct transform
        return carla.Transform(carla.Location(x = x, y = y, z = z), carla.Rotation(yaw = imu_reading[-1]))

    def sensors(self):
        sensors = [{'type': 'sensor.camera.rgb', 'x': 2.0, 'y': 0.0, 'z': 1.4, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 128, 'height': 128, 'fov': 100, 'id': 'Center'},
            {'type': 'sensor.camera.rgb', 'x': 2.0, 'y': 0.0, 'z': 1.4, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 512, 'height': 512, 'fov': 100, 'id': 'Center_high_res'},
            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
            {'type': 'sensor.other.imu', 'x': 2.0, 'y': 0.0, 'z': 1.4, 'roll': 0.0, 'pitch': 0.0,
             'yaw': -90.0, 'id': 'IMU'}, 
            {'type': 'sensor.opendrive_map', 'reading_frequency': 1, 'id': 'OpenDRIVE'},
           {'type': 'sensor.speedometer',  'reading_frequency': 20, 'id': 'SPEED'},
           ]
        return sensors

    def compute_wp_stats(self, vehicle_transform):
        # "Return type: list containing [mean_angle, ldist, distance_to_goal_trajec]"
        mean_angle, ldist, distance_to_goal_trajec, _, _, _ = self.global_planner.get_next_orientation_new(vehicle_transform)
        return mean_angle, ldist, distance_to_goal_trajec

    def cosine_between_velocities(self, v1, v2):
        return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z) / \
            (self.get_speed_from_velocity(v1) * self.get_speed_from_velocity(v2) + 1e-6)

    def preprocess_inputs(self, input_data):
        input_data['IMU'][1][-1] = (input_data['IMU'][1][-1]*(180/np.pi))
        if input_data['IMU'][1][-1]>180:
            input_data['IMU'][1][-1] = input_data['IMU'][1][-1]-360

        # Configure planner when we first receive MAP info
        if not self._route_assigned:
            self._configure_planner(input_data['OpenDRIVE'][1]['opendrive'])
            self._route_assigned = True   

        processed_input = {}

        actor_list = CarlaDataProvider.get_world().get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # obstacles
        vehicle_state, vehicle = self._agent._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            _dist = vehicle.get_location().distance(self._agent._vehicle.get_location())
            processed_input['obstacle_dist'] = _dist / 15
            print('obstacle', _dist)
            # if hasattr(vehicle, 'get_velocity'):
            cos = self.cosine_between_velocities(vehicle.get_velocity(), self._agent._vehicle.get_velocity())
            if cos < 0: cos = 0.
            processed_input['obstacle_speed'] = self.get_speed_from_velocity(vehicle.get_velocity()) * cos / 20
            # cos = self.cosine_between_velocities(obstacle_actor.get_velocity(), agent.vehicle_actor.get_velocity())
            # print('API test COS', cos)
        else:
            processed_input['obstacle_dist'] = 1
            processed_input['obstacle_speed'] = 1
        # traffic light
        light_state, traffic_light = self._agent._is_light_red(lights_list)
        if light_state:
            light_dist = traffic_light.get_location().distance(self._agent._vehicle.get_location())
            print('red light', light_dist)
            processed_input['light_dist'] = light_dist / 15
        else:
            processed_input['light_dist'] = 1


        vehicle_transform = self._get_vehicle_transform(input_data["GPS"][1], input_data['IMU'][1])
        # print("*"*50)
        # print(input_data['IMU'][1][-1])
        # print(vehicle_transform.location.x, vehicle_transform.location.y, vehicle_transform.location.z, vehicle_transform.rotation.yaw)
        # print("*"*50)
        processed_input["mean_angle"], processed_input['ldist'], processed_input['distance_to_goal_trajec'] = self.compute_wp_stats(vehicle_transform)
        processed_input['distance_to_goal_trajec'] = processed_input['distance_to_goal_trajec'] / 500 # to match env.py preproc
        
        processed_input['steer'] = self.previous_steer
        processed_input['speed'] = input_data['SPEED'][1]['speed']

        return processed_input

    def get_action(self, inputs):
        mean_angle = inputs['mean_angle']
        ldist = inputs['ldist']
        distance_to_goal_trajec = inputs['distance_to_goal_trajec']        

        steer = inputs['steer']
        speed = inputs['speed'] / 10
        # print(speed)
        light = inputs['light_dist']

        obstacle_dist = inputs['obstacle_dist']
        obstacle_speed = inputs['obstacle_speed']

        state = np.concatenate((np.array([mean_angle]), \
            np.array([obstacle_dist]), \
            np.array([obstacle_speed]), \
            np.array([speed]), \
            np.array([steer]), \
            np.array([ldist]), \
            np.array([distance_to_goal_trajec]), \
            np.array([light])))

        state_tensor = torch.from_numpy(state).to(torch.float).to(ENV_CONFIG['device'])
        action, _ = self.glb_policy.act(state_tensor, deterministic=True)

        self.previous_steer = action[0]

        return action

    def get_speed_from_velocity(self, velocity):
        speed = np.sqrt(velocity.x ** 2 + velocity.y **2 + velocity.z **2)
        return speed

    def get_control(self, input_data, action):
        """ Get Control object for Carla from action
        Input:
            - action: tuple containing (steer, throttle, brake) in [-1, 1]
        Output:
            - control: Control object for Carla
        """
        steer = np.clip(float(action[0]), -1.0, 1.0)
        target_speed = (action[1] * 1.5) + 1
        # print('action[1]', action[1], 'target_speed', target_speed)
        target_speed = float(np.clip(target_speed * self.target_speed / 2, 0, self.target_speed))

        # TODO: Need to replace this once we get to know how to extract agent's current velocity from IMU/speedometer sensors
        #current_speed = self.get_speed_from_velocity(input_data['SPEED'][1]['speed']) * 3.6
        # current_speed = input_data['SPEED'][1]['speed'] *  3.6
        current_speed = input_data['SPEED'][1]['speed'] * 3.6

        gas = self.controller.pid_control(target_speed, current_speed, enable_brake=True)
        if gas < 0:
            throttle = 0.0
            brake = abs(gas)
        else:
            throttle = gas
            brake = 0.0
        '''print("#"*50)
        print(current_speed, target_speed)
        print(throttle)
        print(brake)
        print(steer)
        print("#"*50)'''

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0)

        return control

    def run_step(self, input_data, timestamp):
        self.ctr += 1
        if not self._agent:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break
            if hero_actor:
                # self._agent = BasicAgent(hero_actor)
                self._agent = BasicAgent(hero_actor, proximity_threshold=15.)


        preprocess_inputs = self.preprocess_inputs(input_data)
        # print([preprocess_inputs['mean_angle'], preprocess_inputs['ldist'], preprocess_inputs['distance_to_goal_trajec'], 
            # preprocess_inputs['steer'], preprocess_inputs['speed']])
        # if(self.ctr%200==0):
        #     st()
        action = self.get_action(preprocess_inputs)
        # print(action[0])
        control = self.get_control(input_data, action)
        return control

