""" Environment file wrapper for CARLA """

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import datetime
import os

try:
    from carla.client import CarlaClient
    from carla.sensor import Camera
    from carla.settings import CarlaSettings
    from carla.planner.planner import Planner
except Exception as e:
    print("Failed to import Carla")
    raise e

# Dict storing basic environ config params
ENV = {
    # X Rendering Resolution
    "render_res_x" : 400,
    # Y Rendering Resolution
    "render_res_y" : 400,
    # Input X Res (Default set to Atari)
    "x_res" : 84,
    # Input Y Res (Default set to Atari)
    "y_res" : 84,
    "port" : 2000,
    "city_name" : "Town01",
    "frame_skip": 1, 
    "enable_planner" : True,
    "reward_function" : 'corl',
    "save_images_to_disk" : True
}

CARLA_LOGS = os.path.expanduser("~/CARLA_LOGS/"+str(datetime.now()))

class CarlaEnv(gym.Env):
    def __init__(self, default_config=ENV):
        pass

    def step(self, action):
        pass
    
    def reset(self):
        pass
    