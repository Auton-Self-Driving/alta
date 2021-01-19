"""Test multi-agent algo
"""

import os
import torch
import matplotlib.pyplot as plt

from network import PPOActorCritic_Continuous, PolicyNetwork, SoftQNetwork
from carla_env import CarlaEnv
from config import ENV_CONFIG, TEST_CONFIG
from sac_agent import SAC_Collective_Agent, VanillaReplayBuffer
from ppo_agent import PPO_Collective_Agent


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["OMP_NUM_THREADS"] = '1'

# override config for testing
ENV_CONFIG.update(TEST_CONFIG)

env = CarlaEnv(ENV_CONFIG)
N_S = env.observation_space.shape[-1]
N_A = env.action_space.shape[-1]

if TEST_CONFIG['PPO']:
    glb_policy = PPOActorCritic_Continuous(N_S, N_A).to(ENV_CONFIG['device'])
    glb_optimizer = torch.optim.Adam(glb_policy.parameters(),
        lr=1e-3, betas=(0.92, 0.999))

    ppo_agent = PPO_Collective_Agent(env, glb_policy, glb_optimizer,
        num_agents=ENV_CONFIG['num_agents'],
        verbose=ENV_CONFIG['verbose'],
    )
    ckpt = torch.load(TEST_CONFIG['checkpoint'], map_location='cpu')
    ppo_agent.load(ckpt)
    ppo_agent.test()
else:
    glb_policy = PolicyNetwork(N_S, N_A).to(ENV_CONFIG['device']) # policy network
    policy_optimizer = torch.optim.Adam(glb_policy.parameters(),
        lr=1e-3, betas=(0.92, 0.999))

    glb_q1 = SoftQNetwork(N_S, N_A).to(ENV_CONFIG['device']) # q network
    q1_optimizer = torch.optim.Adam(glb_q1.parameters(),
        lr=1e-3, betas=(0.92, 0.999))

    glb_q2 = SoftQNetwork(N_S, N_A).to(ENV_CONFIG['device']) # q network
    q2_optimizer = torch.optim.Adam(glb_q2.parameters(),
        lr=1e-4, betas=(0.92, 0.999))

    replay_buffer = VanillaReplayBuffer(maxlen=1)

    log_alpha = torch.log(torch.tensor(1., dtype=torch.float,
        device=ENV_CONFIG['device']))
    log_alpha.requires_grad = True
    alpha_optimizer = torch.optim.Adam((log_alpha,),
        lr=1e-4, betas=(0.92, 0.999))
    target_entropy = -2.

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
        num_agents=ENV_CONFIG['num_agents'],
        max_glb_num_steps=ENV_CONFIG['max_num_steps'],
        verbose=ENV_CONFIG['verbose'])

    ckpt = torch.load(TEST_CONFIG['checkpoint'], map_location='cpu')
    sac_agent.load(ckpt)
    sac_agent.test()

env.close()
