"""Training Distributed PPO algo
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
import time
import random
import torch
import torch.multiprocessing as mp
from threading import Thread
import matplotlib.pyplot as plt
import dist_utils as dist

from network import PPOActorCritic_Continuous
from carla_env import CarlaEnv
from config import ENV_CONFIG, DPPO_CONFIG
from dppo_agent import DPPO_Server_Agent, DPPO_Worker_Agent


os.environ["OMP_NUM_THREADS"] = '1'
print('--------------------[PID {}]--------------------'.format(os.getpid()))

print('>>>DPPO_CONFIG:{}\n>>>ENV_CONFIG:{}'.format(DPPO_CONFIG, ENV_CONFIG))

res = {'log_time': time.strftime('%b%d%I%M%p%S')}

def launch_server(rank, resources):
    os.environ['RANK'] = str(rank)

    # device = DPPO_CONFIG['device_list'][int(rank) % len(DPPO_CONFIG['device_list'])]
    device = ENV_CONFIG['device']

    # overriding carla device
    ENV_CONFIG['device'] = device
    # tmp_env = CarlaEnv(ENV_CONFIG, env_rank=rank)
    # N_S = tmp_env.observation_space.shape[-1]
    # N_A = tmp_env.action_space.shape[-1]
    # tmp_env.close()
    if ENV_CONFIG['input_type'] == 'wp_obs_info_speed_steer_ldist_goal_light':
        N_S, N_A = 8, 2
    elif ENV_CONFIG['input_type'] == 'wp_obs_info_speed_steer_ldist_light':
        N_S, N_A = 7, 2
    elif ENV_CONFIG['input_type'] == 'wp_obs_info_side_obs_info_speed_steer_ldist_light':
        N_S, N_A = 11, 2
    elif ENV_CONFIG['input_type'] == 'wp_obs_more_info_speed_steer_ldist_light':
        N_S, N_A = 15, 2
    elif ENV_CONFIG['input_type'] == 'wp_obs_more_info_speed_steer_ldist_light_ae':
        N_S, N_A = 15 + 8, 2
    else:
        N_S, N_A = 7, 2
    # print(N_S, N_A)
    # from IPython import embed; embed()

    glb_policy = PPOActorCritic_Continuous(N_S, N_A,
        use_transformer=ENV_CONFIG['input_type']=='transformer').to(device) # global network
    # glb_policy.share_memory()
    glb_optimizer = torch.optim.Adam(glb_policy.parameters(),
        lr=DPPO_CONFIG['policy_lr'], betas=(0.92, 0.999))

    server_agent = DPPO_Server_Agent(glb_policy, glb_optimizer,
        num_agents=ENV_CONFIG['num_agents'],
        max_glb_num_steps=ENV_CONFIG['max_num_steps'],
        eps_clip=DPPO_CONFIG['eps_clip'],
        grad_clip=DPPO_CONFIG['grad_clip'],
        glb_update_freq=DPPO_CONFIG['server_glb_update_freq'],
        optim_epochs=DPPO_CONFIG['worker_optim_epochs'],
        focal_loss=DPPO_CONFIG['focal_loss'],
        standard=DPPO_CONFIG['standard'],
        num_threads=DPPO_CONFIG['num_threads_per_server'],
        save_suffix=DPPO_CONFIG['save_suffix'],
        save_freq=DPPO_CONFIG['save_freq'],
        log_time=resources['log_time'],
        verbose=ENV_CONFIG['verbose'],
    )

    server_agent.tb_write_config('env_config',ENV_CONFIG)
    server_agent.tb_write_config('ppo_config',DPPO_CONFIG)


    # resume if necessary
    if DPPO_CONFIG['checkpoint']:
        ckpt = torch.load(DPPO_CONFIG['checkpoint'], map_location='cpu')
        _func = getattr(server_agent, DPPO_CONFIG['ckpt_mode'])
        _func(ckpt)

    server_agent.learn()


def launch_worker(rank, resources):
    os.environ['RANK'] = str(rank)

    device = DPPO_CONFIG['device_list'][(int(rank) - 1) % len(DPPO_CONFIG['device_list'])]

    # overriding carla device
    ENV_CONFIG['device'] = device
    env = CarlaEnv(ENV_CONFIG, env_rank=rank)

    N_S = env.observation_space.shape[-1]
    N_A = env.action_space.shape[-1]
    # from IPython import embed; embed()

    local_policy = PPOActorCritic_Continuous(N_S, N_A,
        use_transformer=ENV_CONFIG['input_type']=='transformer').to(device) # global network
    # glb_policy.share_memory()

    worker_agent = DPPO_Worker_Agent(env, local_policy,
        num_agents=ENV_CONFIG['num_agents'],
        max_glb_num_steps=ENV_CONFIG['max_num_steps'],
        eps_clip=DPPO_CONFIG['eps_clip'],
        grad_clip=DPPO_CONFIG['grad_clip'],
        grad_update_freq=DPPO_CONFIG['worker_grad_update_freq'],
        optim_epochs=DPPO_CONFIG['worker_optim_epochs'],
        focal_loss=DPPO_CONFIG['focal_loss'],
        standard=DPPO_CONFIG['standard'],
        save_suffix=DPPO_CONFIG['save_suffix'],
        log_time=resources['log_time'],
        verbose=ENV_CONFIG['verbose'],
    )
    worker_agent.learn()

    env.close()


dist.run_param_server(launch_server, launch_worker, DPPO_CONFIG['num_servers'],
    DPPO_CONFIG['num_workers'], res, dist.get_host_ip(), random.randint(10000, 60000))


