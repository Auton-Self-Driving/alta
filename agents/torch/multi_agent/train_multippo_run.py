"""Training multi-agent PPO algo
"""

import os
import time
import torch
import multiprocessing as mp
from threading import Thread
import matplotlib.pyplot as plt

from network import PPOActorCritic_Continuous
from carla_env import CarlaEnv
from config import ENV_CONFIG, PPO_CONFIG
from ppo_agent import MultiPPO_Collective_Agent


os.environ["OMP_NUM_THREADS"] = '1'
print('--------------------[PID {}]--------------------'.format(os.getpid()))


def create_env(list_, cfg, rank):
    list_.append(CarlaEnv(ENV_CONFIG), env_rank=rank)

# proc_list = []
# with mp.Manager() as mgr:
#     env_list = mgr.list()
#     for _ in range(ENV_CONFIG['num_envs']):
#         p = mp.Process(target=create_env, args=(env_list, ENV_CONFIG,))
#         proc_list.append(p)
#     for p in proc_list:
#         p.start()
#     for p in proc_list:
#         p.join()

thread_list, env_list = [], []
for rank in range(ENV_CONFIG['num_envs']):
    p = Thread(target=create_env, args=(env_list, ENV_CONFIG, rank))
    thread_list.append(p)
for p in thread_list:
    p.start()
for p in thread_list:
    p.join()

print(env_list)
# for _ in range(ENV_CONFIG['num_envs']):
#     env_list.append(CarlaEnv(ENV_CONFIG))
# time.sleep(20)

N_S = env_list[0].observation_space.shape[-1]
N_A = env_list[0].action_space.shape[-1]
# print(N_S, N_A)
# from IPython import embed; embed()

glb_policy = PPOActorCritic_Continuous(N_S, N_A).to(ENV_CONFIG['device']) # global network
# glb_policy.share_memory()
glb_optimizer = torch.optim.Adam(glb_policy.parameters(),
    lr=PPO_CONFIG['policy_lr'], betas=(0.92, 0.999))

dppo_agent = MultiPPO_Collective_Agent(env_list, glb_policy, glb_optimizer,
    num_agents=ENV_CONFIG['num_agents'],
    max_glb_num_steps=ENV_CONFIG['max_num_steps'],
    eps_clip=PPO_CONFIG['eps_clip'],
    grad_clip=PPO_CONFIG['grad_clip'],
    nesterov=PPO_CONFIG['nesterov'],
    glb_update_freq=PPO_CONFIG['glb_update_freq'],
    optim_epochs=PPO_CONFIG['optim_epochs'],
    focal_loss=PPO_CONFIG['focal_loss'],
    save_suffix=PPO_CONFIG['save_suffix'],
    verbose=ENV_CONFIG['verbose'],
)

# resume if necessary
if PPO_CONFIG['checkpoint']:
    ckpt = torch.load(PPO_CONFIG['checkpoint'], map_location='cpu')
    dppo_agent.resume(ckpt)

dppo_agent.tb_write_config('env_config',ENV_CONFIG)
dppo_agent.tb_write_config('ppo_config',PPO_CONFIG)
print('>>>PPO_CONFIG:{}\n>>>ENV_CONFIG:{}'.format(PPO_CONFIG, ENV_CONFIG))
dppo_agent.learn()

for env in env_list:
    env.close()

avg_reward = []
prev_reward = 0
for ep_idx, rw in enumerate(dppo_agent.glb_ep_reward_list):
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
