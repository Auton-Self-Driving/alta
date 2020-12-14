"""Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for discrete action.
"""

import os
import torch

from a3c_network import Basic_Discrete
from a3c_env import CarlaEnv
from a3c_env_config import ENV_CONFIG
from a3c_agent import A3C_Collective_Agent


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["OMP_NUM_THREADS"] = '1'

env = CarlaEnv(ENV_CONFIG)

N_S = env.observation_space.shape[-1]
N_A = env.action_space.n
# print(N_S, N_A)
# from IPython import embed; embed()

glb_net = Basic_Discrete(N_S, N_A).to(ENV_CONFIG['device']) # global network
glb_optimizer = torch.optim.Adam(glb_net.parameters(), lr=1e-4, betas=(0.92, 0.999))

a3c_agent = A3C_Collective_Agent(env, glb_net, glb_optimizer, 
    num_agents=ENV_CONFIG['num_agents'], device=ENV_CONFIG['device'])

a3c_agent.learn()

