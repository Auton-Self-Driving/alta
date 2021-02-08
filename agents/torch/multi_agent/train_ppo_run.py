"""Training multi-agent PPO algo
"""

import os
import torch
import matplotlib.pyplot as plt

from network import PPOActorCritic_Continuous
from carla_env import CarlaEnv
from config import ENV_CONFIG, PPO_CONFIG
from ppo_agent import PPO_Collective_Agent


# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
os.environ["OMP_NUM_THREADS"] = '1'
print('--------------------[PID {}]--------------------'.format(os.getpid()))

env = CarlaEnv(ENV_CONFIG)

N_S = env.observation_space.shape[-1]
N_A = env.action_space.shape[-1]
# print(N_S, N_A)
# from IPython import embed; embed()

glb_policy = PPOActorCritic_Continuous(N_S, N_A).to(ENV_CONFIG['device']) # global network
glb_optimizer = torch.optim.Adam(glb_policy.parameters(),
    lr=PPO_CONFIG['policy_lr'], betas=(0.92, 0.999))

ppo_agent = PPO_Collective_Agent(env, glb_policy, glb_optimizer,
    num_agents=ENV_CONFIG['num_agents'],
    max_glb_num_steps=ENV_CONFIG['max_num_steps'],
    eps_clip=PPO_CONFIG['eps_clip'],
    grad_clip=PPO_CONFIG['grad_clip'],
    glb_update_freq=PPO_CONFIG['glb_update_freq'],
    optim_epochs=PPO_CONFIG['optim_epochs'],
    save_suffix=PPO_CONFIG['save_suffix'],
    verbose=ENV_CONFIG['verbose'],
)

# resume if necessary
if PPO_CONFIG['checkpoint']:
    ckpt = torch.load(PPO_CONFIG['checkpoint'], map_location='cpu')
    ppo_agent.resume(ckpt)

ppo_agent.tb_write_config('env_config',ENV_CONFIG)
ppo_agent.tb_write_config('ppo_config',PPO_CONFIG)
print('>>>PPO_CONFIG:{}\n>>>ENV_CONFIG:{}'.format(PPO_CONFIG, ENV_CONFIG))
ppo_agent.learn()
env.close()

avg_reward = []
prev_reward = 0
for ep_idx, rw in enumerate(ppo_agent.glb_ep_reward_list):
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
