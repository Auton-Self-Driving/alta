"""Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for discrete action.
"""

import os
import torch
import torch.multiprocessing as mp

from a3c_utils import SharedAdam, v_wrap, push_and_pull, record
from a3c_network import Basic_Discrete
from a3c_env import CarlaEnv
from a3c_env_config import ENV_CONFIG
from a3c_agent import A3C_MP_Agent

from environment.carla_9_4.agents.navigation.roaming_agent import RoamingAgent


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["OMP_NUM_THREADS"] = '1'

env = CarlaEnv(ENV_CONFIG)

N_S = env.observation_space.shape[-1]
N_A = env.action_space.n
print(N_S, N_A)
# from IPython import embed; embed()

glb_net = Basic_Discrete(N_S, N_A) # global network
glb_net.share_memory() # share the global parameters in multiprocessing
glb_optimizer = SharedAdam(glb_net.parameters(), lr=1e-4, betas=(0.92, 0.999))
glb_num_episodes = mp.Value('i', 0)
glb_episode_reward = mp.Value('d', 0)
glb_queue = mp.Queue()

# parallel training
workers = [A3C_MP_Agent(env, glb_net, glb_optimizer, glb_num_episodes, 
    glb_episode_reward, glb_queue, name='FOO') for i in range(1)]
# [w.start() for w in workers]
# res = [] # record episode reward to plot
# while True:
#     print('122 122 122')
#     r = glb_queue.get()
#     if r is not None:
#         res.append(r)
#     else:
#         break
# [w.join() for w in workers]

# print('*' * 80)
# print('FINISHED')
# print('*' * 80)

# from IPython import embed; embed()

obs = env.reset()
print('type(obs)', type(obs))
print(obs.shape)
agent = RoamingAgent(env.vehicle_actor)

num_episodes = 0
val_accuracy_total = []

for t in range(123123):

    # Take one step in env
    # control = agent.run_step()
    print('144', obs.dtype)
    obs = torch.from_numpy(obs).to(torch.float)
    print('146', obs.dtype)
    # control = gnet(obs)
    # print(control)
    # control = control.cpu().numpy()
    # print(control.shape)
    control = glb_net.choose_action(obs)
    print(control)
    new_obs, rew, done, eps_measurements = env.step(control)

    done = bool(done[0, 0])

    obs = new_obs
    print(obs)

    if done:
        num_episodes += 1
        obs = env.reset()
        agent = RoamingAgent(env.vehicle_actor)


