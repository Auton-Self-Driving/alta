"""Training Distributed SAC algo
"""

import os
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

def launch_server(rank, resources):
    os.environ['RANK'] = str(rank)

    # device = DPPO_CONFIG['device_list'][int(rank) % len(DPPO_CONFIG['device_list'])]
    device = ENV_CONFIG['device']

    # overriding carla device
    ENV_CONFIG['device'] = device
    # N_S, N_A = 8, 2
    N_S, N_A = 7, 2
    # tmp_env = CarlaEnv(ENV_CONFIG, env_rank=rank)
    # N_S = tmp_env.observation_space.shape[-1]
    # N_A = tmp_env.action_space.shape[-1]
    # tmp_env.close()
    # print(N_S, N_A)
    # from IPython import embed; embed()

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


def launch_worker(rank, resources):
    os.environ['RANK'] = str(rank)

    device = DSAC_CONFIG['device_list'][(int(rank) - 1) % len(DSAC_CONFIG['device_list'])]

    # overriding carla device
    ENV_CONFIG['device'] = device
    env = CarlaEnv(ENV_CONFIG, env_rank=rank)

    N_S = env.observation_space.shape[-1]
    N_A = env.action_space.shape[-1]
    # print(N_S, N_A)
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


dist.run_param_server(launch_server, launch_worker, DSAC_CONFIG['num_servers'],
    DSAC_CONFIG['num_workers'], res, dist.get_host_ip(), random.randint(10000, 60000))


