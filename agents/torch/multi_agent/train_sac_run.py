"""Training multi-agent SAC algo
"""

import os
import torch
import matplotlib.pyplot as plt

from network import SoftQNetwork, PolicyNetwork
from carla_env import CarlaEnv
from config import ENV_CONFIG, SAC_CONFIG
from sac_agent import VanillaReplayBuffer, SAC_Collective_Agent


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["OMP_NUM_THREADS"] = '1'

env = CarlaEnv(ENV_CONFIG)

N_S = env.observation_space.shape[-1]
N_A = env.action_space.shape[-1]
# print(N_S, N_A)
# from IPython import embed; embed()

glb_policy = PolicyNetwork(N_S, N_A).to(ENV_CONFIG['device']) # policy network
policy_optimizer = torch.optim.Adam(glb_policy.parameters(), 
    lr=SAC_CONFIG['policy_lr'], betas=(0.92, 0.999))

glb_q1 = SoftQNetwork(N_S, N_A).to(ENV_CONFIG['device']) # q network
q1_optimizer = torch.optim.Adam(glb_q1.parameters(), 
    lr=SAC_CONFIG['q_lr'], betas=(0.92, 0.999))

glb_q2 = SoftQNetwork(N_S, N_A).to(ENV_CONFIG['device']) # q network
q2_optimizer = torch.optim.Adam(glb_q2.parameters(), 
    lr=SAC_CONFIG['q_lr'], betas=(0.92, 0.999))

replay_buffer = VanillaReplayBuffer(maxlen=SAC_CONFIG['buffer_len'])

log_alpha = torch.log(torch.tensor(1., dtype=torch.float, 
    device=ENV_CONFIG['device']))
log_alpha.requires_grad = True
alpha_optimizer = torch.optim.Adam((log_alpha,), 
    lr=SAC_CONFIG['alpha_lr'], betas=(0.92, 0.999))
target_entropy = SAC_CONFIG['target_entropy']

sac_agent = SAC_Collective_Agent(
    env, 
    glb_q1=glb_q1,
    q1_optimizer=q1_optimizer,
    glb_q2=glb_q2,
    q2_optimizer=q2_optimizer,
    glb_policy=glb_policy, 
    policy_optimizer=policy_optimizer,
    log_alpha=log_alpha,
    alpha_optimizer=alpha_optimizer,
    target_entropy=target_entropy,
    buffer=replay_buffer,
    tau=SAC_CONFIG['tau'],
    batch_size=SAC_CONFIG['batch_size'],
    q_update_freq=SAC_CONFIG['q_update_freq'],
    target_update_freq=SAC_CONFIG['target_update_freq'],
    num_agents=ENV_CONFIG['num_agents'],
    max_glb_num_steps=ENV_CONFIG['max_num_steps'],
    verbose=ENV_CONFIG['verbose'])

# resume if necessary
if SAC_CONFIG['checkpoint']:
    ckpt = torch.load(SAC_CONFIG['checkpoint'], map_location='cpu')
    sac_agent.resume(ckpt)

sac_agent.learn()
env.close()

avg_reward = []
prev_reward = 0
for ep_idx, rw in enumerate(sac_agent.glb_ep_reward_list):
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
