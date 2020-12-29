import torch
import torch.nn as nn
import gym
import numpy as np

from network import PPOActorCritic_Continuous
from carla_env import CarlaEnv
from config import ENV_CONFIG
from environment.carla_9_4.agents.navigation.agent import Agent

class _PPO_Individual_Agent(Agent):
    def __init__(self, vehicle, local_policy, glb_policy, rank=None,
        device='cuda:0', **kwargs):
        """A local indivial A3C agent.
        Args:
            vehicle: ego-vehicle in env (e.g. env.vehicle_actor)
            glb_net: network shared by all A3C agents, not required for MP
            rank: an integer for identification of this local agent,
                not required for MP
            device: the device for local and global net
            **kwargs: include proximity_threshold=10.0,
                traffic_light_proximity_threshold=10.0,
                vehicle_proximity_threshold=15.0
        """
        super().__init__(vehicle, **kwargs)
        self.local_policy = local_policy
        self.glb_policy = glb_policy
        self.reset_memory()
        self.rank = rank
        self.device = device
        self.done = False
        self.action = None
        self.id = vehicle.id
        self.type_id = vehicle.type_id
        self.vehicle_actor = vehicle
        self.num_total_steps = 0
        self.episode_reward = 0
        self.curr_reward = 0
        self.observation = None

    def select_action(self):
        prev_state = self.observation
        state_tensor = torch.from_numpy(prev_state).to(torch.float)
        action, logprob = self.local_policy.act(state_tensor)
        # update partial memory
        self.memory['state'].append(prev_state)
        self.memory['action'].append(action)
        self.memory['logprob'].append(logprob)
        return action

    def update_local_policy(self):
        self.local_policy.load_state_dict(self.global_policy.state_dict())

    def reset_memory(self):
        self.memory = {
            'action': [],
            'state': [],
            'logprob': [],
            'reward': [],
            'done': [],}


