"""Training Distributed SAC algo on Slurm
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

import time
import random
import math
import torch
import torch.multiprocessing as mp
from threading import Thread
import matplotlib.pyplot as plt
import dist_utils as dist

from network import SoftQNetwork, PolicyNetwork
from carla_env import CarlaEnv
from config import ENV_CONFIG, DSAC_CONFIG
from sac_agent import DSAC_Server_Agent, DSAC_Worker_Agent


os.environ["OMP_NUM_THREADS"] = '1'
print('--------------------[PID {}]--------------------'.format(os.getpid()))

print('>>>DSAC_CONFIG:{}\n>>>ENV_CONFIG:{}'.format(DSAC_CONFIG, ENV_CONFIG))

res = {'log_time': time.strftime('%b%d%I%M%p%S')}

def launch_server(rank, gpu_id, resources):
    device = 'cuda:{}'.format(gpu_id)
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
    else:
        N_S, N_A = 7, 2

    glb_policy = PolicyNetwork(N_S, N_A).to(ENV_CONFIG['device']) # policy network
    policy_optimizer = torch.optim.Adam(glb_policy.parameters(),
        lr=DSAC_CONFIG['policy_lr'], betas=(0.92, 0.999))

    glb_q1 = SoftQNetwork(N_S, N_A).to(ENV_CONFIG['device']) # q network
    q1_optimizer = torch.optim.Adam(glb_q1.parameters(),
        lr=DSAC_CONFIG['q_lr'], betas=(0.92, 0.999))

    glb_q2 = SoftQNetwork(N_S, N_A).to(ENV_CONFIG['device']) # q network
    q2_optimizer = torch.optim.Adam(glb_q2.parameters(),
        lr=DSAC_CONFIG['q_lr'], betas=(0.92, 0.999))

    # log_alpha = torch.log(torch.tensor(DSAC_CONFIG['alpha'], dtype=torch.float,
    #     device=ENV_CONFIG['device']))
    log_alpha = torch.tensor(DSAC_CONFIG['log_alpha'], dtype=torch.float,
        device=ENV_CONFIG['device'])
    log_alpha.requires_grad = True
    alpha_optimizer = torch.optim.Adam((log_alpha,),
        lr=DSAC_CONFIG['alpha_lr'], betas=(0.92, 0.999))
    target_entropy = math.exp(DSAC_CONFIG['log_target_entropy'])

    server_agent = DSAC_Server_Agent(
        glb_q1=glb_q1,
        q1_optimizer=q1_optimizer,
        glb_q2=glb_q2,
        q2_optimizer=q2_optimizer,
        glb_policy=glb_policy,
        policy_optimizer=policy_optimizer,
        log_alpha=log_alpha,
        alpha_optimizer=alpha_optimizer,
        target_entropy=target_entropy,
        ent_autotune=DSAC_CONFIG['ent_autotune'],
        num_agents=ENV_CONFIG['num_agents'],
        max_glb_num_steps=ENV_CONFIG['max_num_steps'],
        buffer_len=DSAC_CONFIG['buffer_len'],
        num_threads=DSAC_CONFIG['num_threads_per_server'],
        tau=DSAC_CONFIG['tau'],
        batch_size=DSAC_CONFIG['batch_size'],
        q_update_freq=DSAC_CONFIG['q_update_freq'],
        target_update_freq=DSAC_CONFIG['target_update_freq'],
        train_after=DSAC_CONFIG['train_after'],
        save_suffix=DSAC_CONFIG['save_suffix'],
        log_time=resources['log_time'],
        verbose=ENV_CONFIG['verbose'])

    server_agent.tb_write_config('env_config',ENV_CONFIG)
    server_agent.tb_write_config('sac_config',DSAC_CONFIG)


    # resume if necessary
    if DSAC_CONFIG['checkpoint']:
        ckpt = torch.load(DSAC_CONFIG['checkpoint'], map_location='cpu')
        server_agent.resume(ckpt)

    server_agent.learn()


def launch_worker(rank, gpu_id, resources):
    # overriding carla device
    device = 'cuda:{}'.format(gpu_id)
    ENV_CONFIG['device'] = device
    env = CarlaEnv(ENV_CONFIG, env_rank=rank)
    N_S = env.observation_space.shape[-1]
    N_A = env.action_space.shape[-1]
    # from IPython import embed; embed()

    local_policy = PolicyNetwork(N_S, N_A).to(ENV_CONFIG['device']) # policy network
    # glb_policy.share_memory()

    worker_agent = DSAC_Worker_Agent(env, local_policy,
        num_agents=ENV_CONFIG['num_agents'],
        max_glb_num_steps=ENV_CONFIG['max_num_steps'],
        explore_before=DSAC_CONFIG['explore_before'],
        explore_mode=DSAC_CONFIG['explore_mode'],
        buffer_update_freq=DSAC_CONFIG['buffer_update_freq'],
        save_suffix=DSAC_CONFIG['save_suffix'],
        log_time=resources['log_time'],
        standard=DSAC_CONFIG['standard'],
        verbose=ENV_CONFIG['verbose'],
    )
    worker_agent.learn()

    env.close()


# overriding num workers based on the world size
DSAC_CONFIG['num_workers'] = int(os.environ['SLURM_NTASKS']) - DSAC_CONFIG['num_servers']

rank, gpu_id, world_size = dist.run_slurm_param_server(
    DSAC_CONFIG['num_servers'], DSAC_CONFIG['num_workers'])
    # port=random.randint(10000, 60000), method='spawn')
print('[slurm dsac run server {} worker {}]'.format(
    DSAC_CONFIG['num_servers'], DSAC_CONFIG['num_workers']), rank, gpu_id, world_size)
if rank < DSAC_CONFIG['num_servers']:
    launch_server(rank, gpu_id, res)
else:
    launch_worker(rank, gpu_id, res)


