"""Run Distributed Transfuser to collect offline data
"""

import os
import glob
import sys

CARLA_9_4_PATH = os.environ.get("CARLA_9_4_PATH")

try:
    sys.path.append(glob.glob(CARLA_9_4_PATH+ '/**/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

if CARLA_9_4_PATH == None:
    raise ValueError("Set $CARLA_9_4_PATH to directory that contains CarlaUE4.sh")

import carla

import os
import torch
import matplotlib.pyplot as plt

from network import PPOActorCritic_Continuous, PolicyNetwork, SoftQNetwork
from carla_env import CarlaEnv
from config import ENV_CONFIG, TEST_CONFIG
from sac_agent import SAC_Collective_Agent, VanillaReplayBuffer
from ppo_agent import PPO_Collective_Agent


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["OMP_NUM_THREADS"] = '1'
print('--------------------[PID {}]--------------------'.format(os.getpid()))

# override config for testing
ENV_CONFIG.update(TEST_CONFIG)
ENV_CONFIG['initial_town'] = ENV_CONFIG['city_name']


env = CarlaEnv(ENV_CONFIG)
N_S = env.observation_space.shape[-1]
N_A = env.action_space.shape[-1]

# print('testing checkpoint [{}]...'.format(TEST_CONFIG['checkpoint']))
print('testing config:\n{}'.format(TEST_CONFIG))
if TEST_CONFIG['PPO']:
    glb_policy = PPOActorCritic_Continuous(N_S, N_A,
        use_transformer=ENV_CONFIG['input_type']=='transformer'
    ).to(ENV_CONFIG['device'])
    glb_optimizer = torch.optim.Adam(glb_policy.parameters(),
        lr=1e-3, betas=(0.92, 0.999))

    ppo_agent = PPO_Collective_Agent(env, glb_policy, glb_optimizer,
        num_agents=ENV_CONFIG['num_agents'],
        verbose=ENV_CONFIG['verbose'],
    )
    ckpt = torch.load(TEST_CONFIG['checkpoint'], map_location='cpu')

    run_name = TEST_CONFIG['checkpoint'].split(".")[-2].split("/")[-1].split("_")
    run_name = '_'.join(run_name[1:-1]) + "/" + TEST_CONFIG["scenarios"]

    ppo_agent.load(ckpt, run_name)
    ppo_agent.test(videos=TEST_CONFIG['videos'])
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

    replay_buffer = VanillaReplayBuffer(maxlen=None)

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
    sac_agent.test(videos=TEST_CONFIG['videos'],
        save_buffer=TEST_CONFIG['save_buffer'])

env.close()

print('testing config:\n{}'.format(TEST_CONFIG))


