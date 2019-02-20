""" Environment file wrapper for CARLA """

import gym
from gym import error, utils
from gym.spaces import Box, Discrete, Tuple
from gym.utils import seeding

from datetime import datetime
import os
import glob
import sys

try:
    from carla.client import CarlaClient
    from carla.sensor import Camera
    from carla.settings import CarlaSettings
    from carla.planner.planner import Planner
except Exception as e:
    print("Failed to import Carla")
    raise e

# Dict storing basic environ config params
# NOTE: Doing this since it's more convenient to pass in a dict (compared to __init__ args)
DEFAULT_ENV = {
    # X Rendering Resolution
    "render_res_x" : 400,
    # Y Rendering Resolution
    "render_res_y" : 300,
    # Input X Res (Default set to Atari)
    "x_res" : 84,
    # Input Y Res (Default set to Atari)
    "y_res" : 84,
    "server_port" : 2000,
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
    "next_command": None
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
        if self.config["enable_planner"]:
            self.planner = Planner(self.city_name)
        
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


    def step(self, action):
        pass
    
    def reset(self):
        pass

    def _read_data(self):
        pass