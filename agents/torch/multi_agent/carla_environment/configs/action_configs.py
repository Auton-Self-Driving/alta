from carla_environment.configs.base_config import BaseConfig
import numpy as np
from gym.spaces import Box, Discrete

def get_discrete_actions():
    steer = [-0.3, -0.1, 0.0, 0.1, 0.3]
    target_speed = [0, 20]

    # Dictionary of discrete (Target_Speed, Steer) actions
    action_space = {}

    n = 0
    for i in range(len(target_speed)):
        for j in range(len(steer)):
            action_space[n] = [target_speed[i], steer[j]]
            n = n+1

    action_space[n] = [20, -0.5]
    action_space[n+1] = [20, 0.5]
    return action_space

DISCRETE_ACTIONS = get_discrete_actions()

class BaseActionConfig(BaseConfig):
    def __init__(self):
        # What action space to use
        self.action_type = None

        # Gym Action Space
        self.action_space = None

        # Whether or not to use the brake when driving
        # If False, vehicle will not use brakes to decelarate
        self.enable_brake = None
        self.discrete_actions = None
        # Number of frames to skip between policy actions
        self.frame_skip = None
        # If true, the PID controller will calculate new commands at each skipped time step
        # If false, the same command (calculated from PID at the first step) will be used
        self.use_pid_in_frame_skip = None

        # "Speed Limit" of the vehicle
        self.target_speed = None

        # Bound for Steering value
        self.steering_scale = None

class MergedSpeedScaledTanhConfig(BaseActionConfig):
    def __init__(self):
        self.action_type = "merged_speed_scaled_tanh"
        self.action_space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
        self.enable_brake = True
        self.discrete_actions = False
        self.frame_skip = 1
        self.use_pid_in_frame_skip = True
        self.target_speed = 20
        self.steering_scale = 0.5

class MergedSpeedTanhConfig(BaseActionConfig):
    def __init__(self):
        self.action_type = "merged_speed_tanh"
        self.action_space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
        self.enable_brake = True
        self.discrete_actions = False
        self.frame_skip = 1
        self.use_pid_in_frame_skip = True
        self.target_speed = 20
        self.steering_scale = 0.5

class MergedSpeedScaledTanhSpeed50Config(BaseActionConfig):
    def __init__(self):
        self.action_type = "merged_speed_scaled_tanh"
        self.action_space = Box(low=np.array([-0.5, -1.0]), high=np.array([0.5, 1.0]))
        self.enable_brake = True
        self.discrete_actions = False
        self.frame_skip = 1
        self.use_pid_in_frame_skip = True
        self.target_speed = 50
        self.steering_scale = 0.5

class DiscreteConfig(BaseActionConfig):
    def __init__(self):
        self.action_type = "discrete"
        self.action_space = Discrete(len(DISCRETE_ACTIONS))
        self.enable_brake = True
        self.discrete_actions = DISCRETE_ACTIONS
        self.frame_skip = 1
        self.use_pid_in_frame_skip = True
        self.target_speed = 20
        self.steering_scale = 0.5