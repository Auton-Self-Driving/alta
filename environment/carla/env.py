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
import server
import queue
import json
import numpy as np
import math

RETRIES_ON_ERROR=5

CARLA_PATH = os.environ.get("CARLA_PATH")
if CARLA_PATH == None:
    raise ValueError("Set $CARLA_PATH to directory that contains CarlaUE4.sh")

try:
    sys.path.append(glob.glob(CARLA_PATH+'/**/*%d.%d-%s.egg' % (
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

from agents.navigation.agent import *
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.local_planner import compute_connection, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import vector

# Dict storing basic environ config params
# NOTE: Doing this since it's more convenient to pass in a dict (compared to __init__ args)
# TODO: Split into server specific and client specific
DEFAULT_ENV = {
    "server_path" : CARLA_PATH,
    "server_binary" : CARLA_PATH + '/CarlaUE4.sh',
    "server_process" : None,
    # X Rendering Resolution
    "render_res_x" : 400,
    # Y Rendering Resolution
    "render_res_y" : 300,
    # Input X Res (Default set to Atari)
    "x_res" : 84,
    # Input Y Res (Default set to Atari)
    "y_res" : 84,
    "server_fps" : 10,
    "server_port" : None,
    "city_name" : "Town01",
    "frame_skip": 1, 
    "enable_planner" : True,
    "reward_function" : 'stub',
    "save_images_to_disk" : False,
    "record_sim": False,
    "write_data": False,
    # Print measurements to screen
    "print_obs" : True,
    "client" : None,
    "discrete_actions" : False,
    # Number of frames stacked together
    "framestack" : 1,
    "num_vehicles" : 0,
    "num_pedestrians" : 0,
    "max_steps" : 1000,
    "next_command": None,
    "verbose": True,
    "vehicle_type": 'vehicle.toyota.prius',
    "target_speed": 20,
    "sensors": ["sensor.camera.rgb"]
}

DISCRETE_ACTIONS = {
    # Coast
    0: [0.0, 0.0],
    # Forward
    1: [1.0, 0.0],
    # Brake
    2: [-0.5, 0.0],
    # Left
    3: [0.0, -0.5],
    # Right
    4: [0.0, 0.5],
    # Forward left
    5: [1.0, -0.5],
    # Forward right
    6: [1.0, 0.5],
    # Brake left
    7: [-0.5, -0.5],
    # Brake right
    8: [-0.5, 0.5]
}

episode_measurements = {
    "episode_id": None,
    "num_steps": None,
    "location": None,
    "forward_speed": None,
    "distance_to_goal": None
    # collision_vehicles
    # collision_pedestrians
    # collision_other
    # intersection_offroad
    # intersection_otherlane
    # next_command
}

CARLA_LOGS = os.path.expanduser("~/CARLA_LOGS/"+str(datetime.now()))

class CarlaEnv(gym.Env):
    def __init__(self, config=DEFAULT_ENV):
        self.config = config
        self.CarlaServer = None
        self.episode_measurements = episode_measurements
        self.server_port = config["server_port"]
        self.city_name = config["city_name"]
        # TODO: Check planner API from 0.9

        # if config["discrete_actions"]:
        #     self.action_space = Discrete(len(DISCRETE_ACTIONS))
        
        # 
        # image_space = Box(
        #     low=0,
        #     high=255,
        #     shape=(config["y_res"], 
        #     config["x_res"],
        #     3 * config["framestack"])
        # )

        self.episode_id = None
        self.client = None
        self.vehicle_actor = None
        self._world = None
        self._map = None
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = 0
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
        self._image_queue = None
        self.destination = None
        self.server_process = None
        self.CarlaServer = None
        self.target_speed = config['target_speed']
        self.actor_list = []
        self.image_data = None

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


    def _spawn_client(self, hostname='localhost', port_number=None):
        port_number = self.CarlaServer.server_port
        client = carla.Client(hostname, port_number)
        client.set_timeout(10.0)
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
        #NOTE: Only mapping to one action for now (target speed)
        speed = action

        self._local_planner.set_speed(speed)
        
        control = self._local_planner.run_step()

        # if(self.config['discrete_actions']):
        #     action = DISCRETE_ACTIONS[int(action)]
        #     throttle = float(np.clip(action[0], 0, 1))
        #     brake = float(np.abs(np.clip(action[0], -1, 0)))
        #     steer = float(np.clip(action[1], -1, 1))
        #     reverse = False
        #     hand_brake = False

        #Print actions
        if self.config['verbose']:
            print("steer", control.steer, "throttle", control.throttle, "brake", control.brake,
                  "reverse", control.reverse)

        #Store control for this step
        self.episode_measurements['control'] = {
            'steer': control.steer,
            'throttle': control.throttle,
            'brake': control.brake,
            'reverse': control.reverse,
            'hand_brake': control.hand_brake
        }

        #Send action to agent
        # control = carla.VehicleControl(
        #     throttle=throttle,
        #     steer=steer,
        #     brake=brake,
        #     hand_brake=False,
        #     reverse=False
        #     manual_gear_shift=False,
        #     gear=0
        # )
        self.vehicle_actor.apply_control(control)
        self.location = self.vehicle_actor.get_location()

        self.episode_measurements['distance_to_goal'] = self.location.distance(self.destination)

        sensor_image = self._read_data()

        self.episode_measurements['num_steps'] = self.num_steps

        reward = self._compute_reward(name=self.config['reward_function'], prev_measurement=self.prev_measurement,
        cur_measurement=self.episode_measurements)
        self.total_reward += reward
        self.episode_measurements['reward'] = reward
        self.episode_measurements['total_reward'] = self.total_reward
        #TODO: Define scenario file for consistent testing across episodes
        done = (self.num_steps > self.config['max_steps'])
        self.episode_measurements['done'] = done
        self.prev_measurement = self.episode_measurements

        if CARLA_LOGS:
            if not self.measurements_log:
                self.measurements_log = open(os.path.join(CARLA_LOGS,
                "measurements_{}.json".format(self.episode_id)), "w")
            self.measurements_log.write(json.dumps(episode_measurements))
            self.measurements_log.write("\n")

            if done:
                self.measurements_log.close()
                self.measurements_file = None
        #Only increment step after writing log (successful)
        self.num_steps += 1

        return (sensor_image, reward,
        done, self.episode_measurements)
    
    def reset(self):
        return self._reset()

    def destroy_all_existing_actors(self):
        
        # Delete all existing actors
        for actor in self.actor_list:
            try:
                actor.destroy()
            except Exception as e:
                print("Error during destroying actor {0}:{1}: {2}".format(actor.type_id, actor.id,traceback.format_exc()))
            


    def _reset(self):
        #TODO: Keep track of current location, and distance to goal (i.e. update eps meas params)
        
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
        self._map = self._world.get_map()

        blueprint_library = self._world.get_blueprint_library()
        try:
            vehicle_bp = blueprint_library.find(self.config['vehicle_type'])
        except Exception as e:
            print("Error during vehicle creation: {}".format(traceback.format_exc()))
        
        #Returns a list of carla.libcarla.Transform
        spawn_points = self._world.get_map().get_spawn_points()
        #carla.libcarla.Transform has attributes location, rotation
        spawn_point = random.choice(spawn_points)
        
        self.vehicle_actor = self._world.spawn_actor(vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle_actor)

        self.location = self.vehicle_actor.get_location()
        print('Spawned vehicle actor at', self.location)

        #TODO: Generalize this code to attach 'n' different sensors to the vehicle
        #Attach a sensor to the vehicle
        sensor = self.config['sensors'][0]
        camera = blueprint_library.find(sensor)
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_actor = self._world.spawn_actor(camera, camera_transform, attach_to=self.vehicle_actor)
        self.actor_list.append(self.camera_actor)

        #self._image_queue = queue.Queue()
        
        #Register callback to put images in the queue
        if(self.config['write_data']):
            self.camera_actor.listen(self._save_sensor_data)
        if(self.config['save_images_to_disk']):
            self.camera_actor.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame_number))
        elif(self.config['record_sim']):
            log_id = str(episode_measurements['episode_id'])
            self.client.start_recorder(log_id, self.vehicle_actor)

        #Attach planner to vehicle actor
        #TODO: Check how to give steering as input to PID? Target speed is present as input
        #TODO: Clean up destination init (pass in a location)
        if self.config["enable_planner"]:
            self._local_planner = LocalPlanner(self.vehicle_actor, opt_dict={'target_speed' : self.target_speed})
            self.destination = random.choice(spawn_points).location 
            self._set_destination(location=self.destination)
                
        # Get start and end positions (to figure out when to end the episode)
        # print("Start pos {}, End Pos {}".format(
        #     spawn_point.location, self.start_coord,
        #     self.scenario["end_pos_id"], self.end_coord))

    def _set_destination(self,location):
        """Generate waypoints and feed into local + global planner
        Parameters
        ----------
        location: Final destination waypoint
        """
        start_waypoint = self._map.get_waypoint(self.vehicle_actor.get_location())
        end_waypoint = self._map.get_waypoint(
            carla.Location(location.x, location.y, location.z))
        solution = []

        # Setting up global router
        dao = GlobalRoutePlannerDAO(self.vehicle_actor.get_world().get_map())
        grp = GlobalRoutePlanner(dao)
        grp.setup()

        # Obtain route plan
        x1 = start_waypoint.transform.location.x
        y1 = start_waypoint.transform.location.y
        x2 = end_waypoint.transform.location.x
        y2 = end_waypoint.transform.location.y
        route = grp.plan_route((x1, y1), (x2, y2))

        current_waypoint = start_waypoint
        route.append(RoadOption.VOID)
        for action in route:

            #   Generate waypoints to next junction
            wp_choice = current_waypoint.next(self._hop_resolution)
            while len(wp_choice) == 1:
                current_waypoint = wp_choice[0]
                solution.append((current_waypoint, RoadOption.LANEFOLLOW))
                wp_choice = current_waypoint.next(self._hop_resolution)

                #   Stop at destination
                if current_waypoint.transform.location.distance(
                    end_waypoint.transform.location) < self._hop_resolution: break
            if action == RoadOption.VOID: break

            #   Select appropriate path at the junction
            if len(wp_choice) > 1:

                # Current heading vector
                current_transform = current_waypoint.transform
                current_location = current_transform.location
                projected_location = current_location + \
                    carla.Location(
                        x=math.cos(math.radians(current_transform.rotation.yaw)),
                        y=math.sin(math.radians(current_transform.rotation.yaw)))
                v_current = vector(current_location, projected_location)
                direction = 0
                if action == RoadOption.LEFT:
                    direction = 1
                elif action == RoadOption.RIGHT:
                    direction = -1
                elif action == RoadOption.STRAIGHT:
                    direction = 0
                select_criteria = float('inf')

                #   Choose correct path
                for wp_select in wp_choice:
                    v_select = vector(
                        current_location, wp_select.transform.location)
                    cross = float('inf')
                    if direction == 0:
                        cross = abs(np.cross(v_current, v_select)[-1])
                    else:
                        cross = direction*np.cross(v_current, v_select)[-1]
                    if cross < select_criteria:
                        select_criteria = cross
                        current_waypoint = wp_select

                #   Generate all waypoints within the junction
                #   along selected path
                solution.append((current_waypoint, action))
                current_waypoint = current_waypoint.next(self._hop_resolution)[0]
                while current_waypoint.is_intersection:
                    solution.append((current_waypoint, action))
                    current_waypoint = current_waypoint.next(self._hop_resolution)[0]

        assert solution

        self._current_plan = solution
        self._local_planner.set_global_plan(self._current_plan)

    def _write_data(self, sensor_data):
        print("Received image from sensor at:", self.location)
        self._image_queue.put(sensor_data)

    def _save_sensor_data(self, sensor_data):
        self.image_data = sensor_data

    def _read_data(self):
        #TODO: Read data in from sensor callback and then call preprocess function
        #sensor data is Image object for all sensors (besides LIDAR)
        sensor_data = self._image_queue.get()
        print("Read image from queue at:", self.location)
        im_data = sensor_data.raw_data
        im_width = sensor_data.width
        im_height = sensor_data.height
        fov = sensor_data.fov
        im_processed = self._preprocess(im_data)
        return im_processed
    
    def _preprocess(self, image):
        #TODO: Add preprocessing steps here
        return image

    def _compute_reward(self, name, prev_measurement, cur_measurement):
        #TODO: Add dict functionality to call other reward functions
        reward = self._compute_reward_corl2017(prev_measurement, cur_measurement)
        return reward

    def _compute_reward_corl2017(self, prev, current):
        reward = 0.0

        cur_dist = current["distance_to_goal"]

        prev_dist = prev["distance_to_goal"]

        if self.config["verbose"]:
            print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

        # Distance travelled toward the goal in m
        reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)

        # Change in speed (km/h)
        reward += 0.05 * (current["forward_speed"] - prev["forward_speed"])

        # New collision damage
        reward -= .00002 * (
            current["collision_vehicles"] + current["collision_pedestrians"] +
            current["collision_other"] - prev["collision_vehicles"] -
            prev["collision_pedestrians"] - prev["collision_other"])

        # New sidewalk intersection
        reward -= 2 * (
            current["intersection_offroad"] - prev["intersection_offroad"])

        # New opposite lane intersection
        reward -= 2 * (
            current["intersection_otherlane"] - prev["intersection_otherlane"])

        return reward

if __name__ == "__main__":
    env = CarlaEnv()
    env.reset()
