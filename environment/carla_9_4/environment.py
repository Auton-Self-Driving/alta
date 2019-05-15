import glob
import os
import sys

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

import logging
mport traceback
import random
import numpy as np
from gym import Env
from gym.spaces import Box
import cv2
from carla import ColorConverter as cc
import environment.carla_9_4.server as server
import environment.carla_9_4.scenarios as scenarios
import weakref
import math
import collections
import time
from copy import deepcopy as dc
from carla.libcarla import Transform
from carla.libcarla import Location
from carla.libcarla import Rotation

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

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

RETRIES_ON_ERROR = 5

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
    "x_res": 224,
    # Input Y Res (Default set to Atari)
    "y_res": 224,
    "server_fps" : 10,
    "server_port" : None,
    "city_name" : "Town01",
    "frame_skip": 1,
    "enable_planner" : False,
    "reward_function" : 'new',
    "save_images_to_disk" : False,
    "record_sim": False,
    "write_data": True,
    # Print measurements to screen
    "print_obs" : True,
    "client" : None,
    "discrete_actions": True,

    # Number of frames stacked together
    "framestack" : 4,
    "grayscale" : True,
    "num_vehicles" : 0,
    "num_pedestrians" : 0,
    "max_steps" : 400,
    "next_command": None,
    "verbose": True,
    "vehicle_type": 'vehicle.toyota.prius',
    "target_speed": 20,
    "sensors": ["sensor.camera.rgb", "sensor.camera.semantic_segmentation"],
    "action_type": "merged_gas",
    "sensor_tick": '1.0',
    "dist_for_success" : 4.0,
    "max_offlane_steps" : 20,
    "max_static_steps" : 100,
    "log_measurements_to_file": False,
    "train_config": 'baselines',
    "sync_mode": True,
    # NOTE: crop does not work with framestack yet. need to add.
    "preprocess_crop_image": False
}

# def draw_image(surface, image):
#     array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
#     array = np.reshape(array, (image.height, image.width, 4))
#     array = array[:, :, :3]
#     array = array[:, :, ::-1]
#     image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
#     surface.blit(image_surface, (0, 0))


def get_cv_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    # array = array[:, :, ::-1]
    return array


# def get_font():
#     fonts = [x for x in pygame.font.get_fonts()]
#     default_font = 'ubuntumono'
#     font = default_font if default_font in fonts else fonts[0]
#     font = pygame.font.match_font(font)
#     return pygame.font.Font(font, 14)


# def should_quit():
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             return True
#         elif event.type == pygame.KEYUP:
#             if event.key == pygame.K_ESCAPE:
#                 return True
#     return False


def get_speed_from_velocity(velocity):
    speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    return speed


class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.num_collisions = 0
        self._history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.num_collisions += 1
        # print('Collision with %r, id = %d' % (actor_type, event.other_actor.id))
        # impulse = event.normal_impulse
        # intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        # self._history.append((event.frame_number, intensity))
        # if len(self._history) > 4000:
        #    self._history.pop(0)

    def destroy(self):
        self.sensor.destroy()


class LaneInvasionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.num_laneintersections = 0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_detector')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        # TODO : Handle case of lane invasion for dashed vs solid lane markings
        self.num_laneintersections += 1
        # text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        # self._hud.notification('Crossed line %s' % ' and '.join(text))

    def destroy(self):
        self.sensor.destroy()


class CarlaEnv(Env):
    def __init__(self, config=DEFAULT_ENV, port=None, z_size=512):
        
        self.config = DEFAULT_ENV
        self._update_config(config)

        if port is not None:
            self.config["server_port"] = port

        self.z_size = z_size
        self.action_space = Box(low=np.array([-0.5]), high=np.array([0.5]), dtype=np.float32)
        self.observation_space = Box(low=np.finfo(np.float32).min,
                                     high=np.finfo(np.float32).max,
                                     shape=(1, self.z_size), dtype=np.float32)

        self.actor_list = []
        self.image_queue = None
        self.frame = None
        self.image_queue = queue.Queue()
        
        
        self.server_port = self.config["server_port"]
        self.city_name = self.config["city_name"]
        self.episode_measurements = episode_measurements
        self.prev_measurement = None

        self.clock = pygame.time.Clock()
        self.total_reward = 0

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
        
        while True:
            try:
                client = carla.Client('localhost', self.server_port)
                client.set_timeout(60.0)
                self.world = client.get_world()
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                self.world.apply_settings(settings)
                break
            except:
                print("could not connect. Trying again")
        self.source_transform, self.destination_transform = get_fixed_short_straight_path_Town01()
        self._get_actors()
        
    def _clamp_action(self, action, min_val=-0.5, max_val=0.5):
        return np.clip(action, a_min = min_val, a_max = max_val) 
    
    def _get_control(self, action):
        """ Get Control object for Carla from action
        Input:
            - action: tuple containing (steer, throttle, brake) in [-1, 1]
        Output:
            - control: Control object for Carla
        """
        action = action.flatten()
        self.action = self._clamp_action(action)
        
        if self.config["action_type"] == "sep_gas":
            steer = float(action[0])
            throttle = float(action[1])
            brake = float(action[2])
        elif self.config["action_type"] == "merged_gas":
            steer = float(action[0])
            gas = float(action[1])
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config["action_type"] == "steer_only":
            steer = float(action[0])
            throttle = float(0.50)
            brake = float(0.0)

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0)

        return control

    def step(self, action):
        # self.action = action
        # if action[0] < -0.5:
        #     action[0] = -0.5
        # elif action[0] > 0.5:
        #     action[0] = 0.5

        # control = carla.VehicleControl()
        # control.throttle = 0.5
        # control.steer = float(action[0])
        # print(control.steer)
        # control.brake = 0
        
        control = self._get_control(action)

        self.episode_measurements['control_steer'] = control.steer
        self.episode_measurements['control_throttle'] = control.throttle
        self.episode_measurements['control_brake'] = control.brake
        self.episode_measurements['control_reverse'] = control.reverse
        self.episode_measurements['control_hand_brake'] = control.hand_brake
        
        #Print actions
        if self.config['verbose']:
            print("steer", control.steer, "throttle", control.throttle, "brake", control.brake,
                  "reverse", control.reverse)

        self.episode_measurements['num_collisions'] = self.actor_list[2].num_collisions
        self.episode_measurements['num_laneintersections'] = self.actor_list[3].num_laneintersections
        self.location = self.actor_list[0].get_location()
        self.episode_measurements['distance_to_goal'] = self.location.distance(self.destination_transform.location)
        self.episode_measurements['speed'] = get_speed_from_velocity(velocity=self.actor_list[0].get_velocity())

        self.actor_list[0].apply_control(
            control)  # actor_list = [vehicle, camera, collision_sensor, lane_invasion_sensor]

        obs = {}
        obs['dist_to_target'] = np.array([self.episode_measurements['distance_to_goal']])
        observation = self._get_observation()
        obs['image'] = observation
        obs['orientation'] = np.expand_dims(
	            self._get_orientation_measurements(), axis=0)

        reward = self._compute_reward(self.config['reward_function'], prev_measurement, cur_measurement):
            
        self.total_reward += reward
        self.episode_measurements['reward'] = reward
        self.episode_measurements['total_reward'] = self.total_reward

        done = self._is_game_over(collision=self.actor_list[2].num_collisions,
                                  distance=self.location.distance(self.destination_transform.location))

        self.episode_measurements['done'] = done
        self.prev_measurement = dc(self.episode_measurements)
        info = {}
        if done:
            self.reset()
        reward = np.expand_dims(np.array([reward]), axis=0)
        done = np.expand_dims(np.array([done]), axis=0)
        
        return obs, reward, done, self.episode_measurements

    # def _get_orientation_measurements(self):
    #     vehicle_transform = self.vehicle_actor.get_transform()
    #     waypoint = self._map.get_waypoint(self.vehicle_actor.get_location())
    #     waypoint = waypoint.next(2)[0]

    #     v_begin = vehicle_transform.location
    #     v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
    #                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

    #     v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
    #     w_vec = np.array([waypoint.transform.location.x -
    #                         v_begin.x, waypoint.transform.location.y -
    #                         v_begin.y, 0.0])
    #     _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
    #                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

    #     _cross = np.cross(v_vec, w_vec)
    #     if _cross[2] < 0:
    #         _dot *= -1.0

    #     return np.array([_dot])
    
    def _get_orientation_measurements(self):
       vehicle_transform = self.vehicle_actor.get_transform()
       dest = self.destination_transform
       v_begin = vehicle_transform.location
       v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                        y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

       v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
       w_vec = np.array([dest.location.x -
                         v_begin.x, dest.location.y -
                         v_begin.y, 0.0])
       _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

       _cross = np.cross(v_vec, w_vec)
       if _cross[2] < 0:
           _dot *= -1.0

       return np.array([_dot])
   
    def _update_config(self, config):
        for key, val in config.items():
            self.config[key] = val
    
    def _clear_episode_measurements(self):
        for key, val in self.episode_measurements.items():
            self.episode_measurements[key] = 0

    def _get_observation(self):

        self.world.tick()
        ts = self.world.wait_for_tick()
        self.clock.tick()

        self.frame = ts.frame_count
        while True:
            caarla_image = self.image_queue.get()

            if self.config["segmented"]:
                caarla_image.convert(cc.CityScapesPalette)
            self.observation_image = dc(get_cv_image(caarla_image))

            self.observation_image = cv2.resize(self.observation_image, (self.config["y_res"], self.config["x_res"]))

            if caarla_image.frame_number == ts.frame_count:
                break

        return self.observation_image

    def _get_actors(self):
        self._map = self.world.get_map()
        self.start_pose = self._map.get_spawn_points()[5]

        blue_print_library = self.world.get_blueprint_library()

        vehicle = self.world.spawn_actor(blue_print_library.filter('vehicle*')[5],
                                         self.source_transform)
        vehicle.set_simulate_physics(True)
        self.vehicle_actor = vehicle
        
        if self.config["segmented"]:
            camera = self.world.spawn_actor(blue_print_library.find(self.config["sensors"][1]),
                                            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=15)),
                                            attach_to=vehicle)
        else:
            camera = self.world.spawn_actor(blue_print_library.find(self.config["sensors"][0]),
                                            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=15)),
                                            attach_to=vehicle)
        self.actor_list.append(vehicle)
        self.actor_list.append(camera)
        collision_sensor = CollisionSensor(vehicle)
        lane_invasion_sensor = LaneInvasionSensor(vehicle)
        self.actor_list.append(collision_sensor)
        self.actor_list.append(lane_invasion_sensor)
        time.sleep(2)

    def _attach_image_queue_to_camera(self):

        camera = self.actor_list[1]
        camera.listen(self.image_queue.put)

    def _destroy_actors(self):
        for _ in range(0, len(self.actor_list)):
            a = self.actor_list.pop()
            a.destroy()

    def reset(self):
        self._destroy_actors()
        self._clear_episode_measurements()
        
        self.total_reward = 0
        self.prev_measurement = None
        
        self._get_actors()
        self._attach_image_queue_to_camera()
        self.frame = None
        self.source_transform, self.destination_transform = scenarios.get_fixed_short_straight_path_Town01()

        self.episode_measurements['num_collisions'] = self.actor_list[2].num_collisions
        self.episode_measurements['num_laneintersections'] = self.actor_list[3].num_laneintersections
        self.location = self.actor_list[0].get_location()
        self.episode_measurements['distance_to_goal'] = self.location.distance(self.destination_transform.location)
        self.episode_measurements['speed'] = get_speed_from_velocity(velocity=self.actor_list[0].get_velocity())

        self.prev_measurement = dc(self.episode_measurements)
        obs = {}
        obs['dist_to_target'] = np.array(
            [self.episode_measurements['distance_to_goal']])
        obs['image'] = np.random.random((88, 200, 3))
        obs['orientation'] = np.expand_dims(
	            self._get_orientation_measurements(), axis=0)
        return obs
        # print("CALLED RESET")
        self._reset()  # THIS CAUSES TROUBLE WITH PPO2

    def _is_game_over(self, collision, distance):
        if collision:
            print("Collision: Distance remaining: {}".format(distance))
            return True
        elif distance <= 4.0:
            print("Goal Reached: Distance remaining: {}".format(distance))
            return True
        else:
            return False
    
    def _compute_reward(self, name, prev_measurement, cur_measurement):
        if name == 'corl':
            reward = self._compute_reward_corl(prev_measurement, cur_measurement)
        elif name == 'cirl':
            reward = self._compute_reward_cirl(prev_measurement, cur_measurement)
        elif name == 'simplest':
            reward = self._compute_reward_simplest(prev_measurement, cur_measurement)
        elif name == "new":
            reward = self._compute_reward_new(prev_measurement, cur_measurement)
        return reward

    def _compute_reward_simplest(self, prev, current):
        
        # Distance reward
        distance_reward = 100.0 / (current['distance_to_goal'])
        self.episode_measurements['distance_reward'] = distance_reward
        
        # Steer penalty
        steer_reward = -50 if abs(current['control_steer']) > 0.3 else 0
        self.episode_measurements['steer_reward'] = steer_reward

        # Collision penalty
        no_collisions = (current["num_collisions"] - prev["num_collisions"])
        collision = no_collisions > 0
        collision_reward = -50 if collision else 0
        self.episode_measurements["collision_reward"] = collision_reward
        
        reward = distance_reward + steer_reward + collision_reward
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
        distance_reward = np.clip(prev_dist - cur_dist, -10.0, 10.0)
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

    def _compute_reward_new(self, prev, current):
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
    