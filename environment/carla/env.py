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
    "reward_function" : 'corl',
    "save_images_to_disk" : True,
    # Print measurements to screen
    "print_obs" : True,
    "client" : None,
    "discrete_actions" : True,
    # Number of frames stacked together
    "framestack" : 1,
    "num_vehicles" : 0,
    "num_pedestrians" : 0,
    "max_steps" : 1000,
    "next_command": None,
    "verbose": True
    "vehicle_type": 'vehicle.toyota.prius',
    "sensors": ["r"]
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
    4: [-0.5, 0.0],
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
    # episode ID
    # num_steps
    # x location
    # y location
    # x orientation
    # y orientation
    # forward speed
    # distance to goal
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
        self.server_port = config["server_port"]
        self.city_name = config["city_name"]
        # TODO: Check planner API from 0.9
        # if self.config["enable_planner"]:
        #     self.planner = Planner(self.city_name)
        
        if config["discrete_actions"]:
            self.action_space = Discrete(len(DISCRETE_ACTIONS))
        
        image_space = Box(
            low=0,
            high=255,
            shape=(config["y_res"], 
            config["x_res"],
            3 * config["framestack"])
        )

        self.episode_id = None
        self.client = None
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

    def spawn_client(self, hostname='localhost', port_number=None):
        
        return client

    def step(self, action):
        try:
            obs = self.step_(action)
            return obs
        except Exception:
            print("Error during step, terminating episode early",
        traceback.format_exc())

    def _step(self, action):
        if(self.config['discrete_actions']):
            action = DISCRETE_ACTIONS[int(action)]
        throttle = float(np.clip(action[0], 0, 1))
        brake = float(np.abs(np.clip(action[0], -1, 0)))
        steer = float(np.clip(action[1], -1, 1))
        reverse = False
        hand_brake = False

        #Print actions
        if self.config['verbose']:
            print("steer", steer, "throttle", throttle, "brake", brake,
                  "reverse", reverse)
        
        #Send action to agent
    
    def reset(self):
        pass

    def _reset(self):
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.prev_image = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None

        client = carla.Client(hostname, port_number)

        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        try:
            vehicle_bp = blueprint_library.find(self.config['vehicle_type'])
        except Exception as e:
            print("Error during vehicle creation: {}".format(traceback.format_exc()))
        
        #Returns a list of carla.libcarla.Transform
        spawn_points = world.get_map().get_spawn_points()
        #carla.libcarla.Transform has attributes location, rotation
        spawn_point = random.choice(spawn_points)
        
        vehicle_actor = world.spawn_actor(vehicle_bp, spawn_point)

        #TODO: Generalize this code to attach 'n' different sensors to the vehicle
        #Attach a sensor to the vehicle
        sensor = self.config['sensors'][0]
        camera = blueprint_library.find(sensor)
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera_actor = world.spawn_actor(camera, camera_transform, attach_to=vehicle_actor)
        if(self.config['save_images_to_disk']):
            camera_actor.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame_number))
        # Get start and end positions (to figure out when to end the episode)
        # print("Start pos {}, End Pos {}".format(
        #     spawn_point.location, self.start_coord,
        #     self.scenario["end_pos_id"], self.end_coord))



    def _read_data(self):
        pass