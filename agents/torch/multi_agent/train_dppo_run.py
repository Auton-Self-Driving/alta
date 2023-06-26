"""Training Distributed PPO algo
"""

import os
import glob
import sys
import traceback

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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import time
import random
import torch
import argparse
import torch.multiprocessing as mp
from threading import Thread
import matplotlib.pyplot as plt
import dist_utils as dist

from network import PPOActorCritic_Continuous
from carla_env import CarlaEnv
from config import ENV_CONFIG, DPPO_CONFIG
from config_exp import override_configs
from dppo_agent import DPPO_Server_Agent, DPPO_Worker_Agent


os.environ["OMP_NUM_THREADS"] = '1'
# print('--------------------[PID {}]--------------------'.format(os.getpid()))

# print('>>>DPPO_CONFIG:{}\n>>>ENV_CONFIG:{}'.format(DPPO_CONFIG, ENV_CONFIG))

def get_state_action_dims(config):

    if config['input_type'] == 'wp_obs_info_speed_steer_ldist_goal_light':
        N_S = 8
    elif config['input_type'] == 'wp_obs_info_speed_steer_ldist_light':
        N_S = 7
    elif config['input_type'] == 'wp_obs_info_side_obs_info_speed_steer_ldist_light':
        N_S = 11
    elif config['input_type'] == 'wp_obs_more_info_steer_ldist_light': # No speed
        N_S = 14
    elif config['input_type'] == 'wp_obs_more_info_speed_steer_ldist_light':
        N_S = 15
    elif config['input_type'] == 'wp_2avg_obs_more_info_speed_steer_ldist_light':
        N_S = 16
    elif config['input_type'] == 'wp_360_obstacle_speed_steer':
        N_S = 24
    elif config['input_type'] == 'wp_list_obs_more_info_steer_ldist_light': # No speed
        N_S = 13 + config['num_waypoints'] 
    elif config['input_type'] == 'wp_list_obs_more_info_speed_steer_ldist_light':
        N_S = 14 + config['num_waypoints'] 
    else:
        N_S = 7

    if config['action_type'] == 'cubic_bezier_3dof':
        N_A = 4
    elif config['action_type'] == 'cubic_bezier_5dof':
        N_A = 6
    elif config['action_type'] == 'speed_wp':
        N_A = 3
    elif config['action_type'] == 'steer_only':
        N_A = 3
    else:
        N_A = 2

    return N_S, N_A


def launch_server(rank, resources):
    os.environ['RANK'] = str(rank)

    device = resources["agent_cfg"]['device_list'][0]

    # overriding carla device
    resources["env_cfg"]['device'] = device
    
    N_S, N_A = get_state_action_dims(resources["env_cfg"])
    
    glb_policy = PPOActorCritic_Continuous(N_S, N_A,
        use_transformer=resources["env_cfg"]['input_type']=='transformer', squash=resources["agent_cfg"]['squash']).to(device) # global network

    glb_optimizer = torch.optim.Adam(glb_policy.parameters(),
        lr=resources["agent_cfg"]['policy_lr'], betas=(0.92, 0.999))

    server_agent = DPPO_Server_Agent(glb_policy, glb_optimizer,
        num_agents=resources["env_cfg"]['num_agents'],
        max_glb_num_steps=resources["env_cfg"]['max_num_steps'],
        eps_clip=resources["agent_cfg"]['eps_clip'],
        grad_clip=resources["agent_cfg"]['grad_clip'],
        glb_update_freq=resources["agent_cfg"]['server_glb_update_freq'],
        glb_adaptive_freq=resources["agent_cfg"]['server_adaptive_freq'],
        optim_epochs=resources["agent_cfg"]['worker_optim_epochs'],
        focal_loss=resources["agent_cfg"]['focal_loss'],
        standard=resources["agent_cfg"]['standard'],
        push_grad=resources["agent_cfg"]['push_grad'],
        gamma=resources["agent_cfg"]['gamma'],
        num_threads=resources["agent_cfg"]['num_threads_per_server'],
        save_suffix=resources["agent_cfg"]['save_suffix'],
        save_freq=resources["agent_cfg"]['save_freq'],
        log_time=resources['log_time'],
        verbose=resources["env_cfg"]['verbose'],
    )

    server_agent.tb_write_config('env_config',resources["env_cfg"])
    server_agent.tb_write_config('ppo_config',resources["agent_cfg"])

    # Create checkpoint directory
    os.makedirs(os.path.join('./checkpoints', resources["agent_cfg"]['save_suffix'] if resources["agent_cfg"]['save_suffix'] else  "the_nameless_ones"),exist_ok=True)

    # resume if necessary
    if resources["agent_cfg"]['checkpoint']:
        ckpt = torch.load(os.path.join('./checkpoints', resources["agent_cfg"]['save_suffix'] if resources["agent_cfg"]['save_suffix'] else  "the_nameless_ones",resources["agent_cfg"]['checkpoint']), map_location='cpu')
        _func = getattr(server_agent, resources["agent_cfg"]['ckpt_mode'])
        _func(ckpt)

    server_agent.learn()

def launch_worker(rank, resources):

    os.environ['RANK'] = str(rank)

    device = resources["agent_cfg"]['device_list'][(int(rank) - 1) % len(resources["agent_cfg"]['device_list'])]

    # overriding carla device
    resources["env_cfg"]['device'] = device
    env = CarlaEnv(resources["env_cfg"], env_rank=rank)

    N_S = env.obs_manager.observation_space.shape[-1]
    N_A = env.obs_manager.action_space.shape[-1]

    local_policy = PPOActorCritic_Continuous(N_S, N_A,
        use_transformer=resources["env_cfg"]['input_type']=='transformer', squash=resources["agent_cfg"]['squash']).to(device) # global network


    worker_agent = DPPO_Worker_Agent(env, local_policy,
        num_agents=resources["env_cfg"]['num_agents'],
        max_glb_num_steps=resources["env_cfg"]['max_num_steps'],
        eps_clip=resources["agent_cfg"]['eps_clip'],
        grad_clip=resources["agent_cfg"]['grad_clip'],
        grad_update_freq=resources["agent_cfg"]['worker_grad_update_freq'],
        optim_epochs=resources["agent_cfg"]['worker_optim_epochs'],
        focal_loss=resources["agent_cfg"]['focal_loss'],
        standard=resources["agent_cfg"]['standard'],
        push_grad=resources["agent_cfg"]['push_grad'],
        save_suffix=resources["agent_cfg"]['save_suffix'],
        log_time=resources['log_time'],
        verbose=resources["env_cfg"]['verbose'],
    )
    worker_agent.learn()

    env.close()





parser = argparse.ArgumentParser(description='Run Options')

parser.add_argument('--exp_name', type=str, default='360deg_str_st_fs_1',
                    help='Name of experiment')
# parser.add_argument('--device_list', type=int, nargs='+',default=[1,2],
#                     help='GPU devices to run on')
args = parser.parse_args()  

env_cfg, agent_cfg = override_configs(args.exp_name,ENV_CONFIG, DPPO_CONFIG)

res = {'log_time': time.strftime('%b%d%I%M%p%S'), "env_cfg":env_cfg, "agent_cfg":agent_cfg}

dist.run_param_server(launch_server, launch_worker, agent_cfg['num_servers'],
    agent_cfg['num_workers'], res, dist.get_host_ip(), random.randint(10000, 60000))


