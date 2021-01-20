"""Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for discrete action.
"""

import os
import torch
import matplotlib.pyplot as plt

from network import Basic_Discrete
from carla_env import CarlaEnv
from config import ENV_CONFIG, A2C_CONFIG
from a2c_agent import A2C_Collective_Agent


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["OMP_NUM_THREADS"] = '1'
print('--------------------[PID {}]--------------------'.format(os.getpid()))

env = CarlaEnv(ENV_CONFIG)

N_S = env.observation_space.shape[-1]
N_A = env.action_space.n
# print(N_S, N_A)
# from IPython import embed; embed()

glb_net = Basic_Discrete(N_S, N_A).to(ENV_CONFIG['device']) # global network
glb_optimizer = torch.optim.Adam(glb_net.parameters(),
    lr=A2C_CONFIG['policy_lr'], betas=(0.92, 0.999))

a2c_agent = A2C_Collective_Agent(env, glb_net, glb_optimizer,
    num_agents=ENV_CONFIG['num_agents'],
    max_glb_num_steps=ENV_CONFIG['max_num_steps'],
    glb_update_freq=A2C_CONFIG['glb_update_freq'],
    verbose=ENV_CONFIG['verbose'],
)

if A2C_CONFIG['checkpoint']:
    ckpt = torch.load(A2C_CONFIG['checkpoint'], map_location='cpu')
    a2c_agent.resume(ckpt)

a2c_agent.learn()
env.close()

avg_reward = []
prev_reward = 0
for ep_idx, rw in enumerate(a2c_agent.glb_ep_reward_list):
    prev_reward += rw
    avg_reward.append(prev_reward / (ep_idx + 1))

plt.plot(range(1, len(avg_reward) + 1), avg_reward, label='global')
# for rk in range(len(a2c_agent.agent_reward_list)):
#     plt.plot(range(1, len(a2c_agent.agent_reward_list[rk]) + 1),
#         a2c_agent.agent_reward_list[rk], label='rank_{}'.format(rk))
plt.legend()
plt.xlabel('episode number')
plt.ylabel('reward')
plt.tight_layout()
plt.rcParams['savefig.dpi'] = 500
plt.show()
