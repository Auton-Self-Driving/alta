"""Test multi-agent algo
"""

import os
import glob
import sys

CARLA_9_4_PATH = os.environ.get("CARLA_9_4_PATH")

try:
    sys.path.append(glob.glob(CARLA_9_4_PATH + '/**/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

if CARLA_9_4_PATH == None:
    raise ValueError("Set $CARLA_9_4_PATH to directory that contains CarlaUE4.sh")

import carla

import torch
import matplotlib.pyplot as plt
import argparse
import copy
from multiprocessing import Pool

from network import PPOActorCritic_Continuous, PolicyNetwork, SoftQNetwork
from carla_env import CarlaEnv
from config import ENV_CONFIG, TEST_CONFIG
from sac_agent import SAC_Collective_Agent, VanillaReplayBuffer
from ppo_agent import PPO_Collective_Agent



def test_policy(test_config):
    os.environ["OMP_NUM_THREADS"] = '1'
    print('--------------------[PID {}]--------------------'.format(os.getpid()))

    # override config for testing
    ENV_CONFIG.update(test_config)
    # ENV_CONFIG['initial_town'] = ENV_CONFIG['city_name']

    env = CarlaEnv(ENV_CONFIG)
    N_S = env.observation_space.shape[-1]
    N_A = env.action_space.shape[-1]

    print('testing config:\n{}'.format(test_config))

    if test_config['PPO']:

        glb_policy = PPOActorCritic_Continuous(N_S, N_A,
            use_transformer=ENV_CONFIG['input_type']=='transformer'
            ).to(ENV_CONFIG['device'])
        glb_optimizer = torch.optim.Adam(glb_policy.parameters(),
            lr=1e-3, betas=(0.92, 0.999))
        ppo_agent = PPO_Collective_Agent(env, glb_policy, glb_optimizer,
            num_agents=ENV_CONFIG['num_agents'],
            verbose=ENV_CONFIG['verbose'],
            )
        ckpt = torch.load(test_config['checkpoint'], map_location='cpu')
        print("Loaded weights from",test_config['checkpoint'])
        run_name = test_config['checkpoint'].split(".")[-2].split("/")[-1].split("_")
        run_name = '_'.join(run_name[1:-1]) + "/" + test_config["scenarios"]
        ppo_agent.load(ckpt, run_name)
        ppo_agent.test(videos=test_config['videos'])

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

        ckpt = torch.load(test_config['checkpoint'], map_location='cpu')
        sac_agent.load(ckpt)
        sac_agent.test(videos=test_config['videos'],
            save_buffer=test_config['save_buffer'])

    env.close()

    print('testing config:\n{}'.format(test_config))


def construct_config_update(args):

    ckpt_base_path = "./checkpoints/{}/".format(args.ckpt)
    ckpt_paths_list = [ glob.glob(ckpt_base_path+"*_{}_*".format(itr))[0] for itr in args.ckpt_iters]

    config_list = []

    for ckpt in ckpt_paths_list:
        for twn in args.test_towns:
            for scn in args.scenarios:

                # Skip if evaluations already exist
                eval_folder = '{}/{}/{}/{}/{}/'.format('./tests/evals',scn,twn,args.ckpt,ckpt.split('_')[-2])
                if os.path.exists(os.path.join(eval_folder,'25.png')):
                    continue

                config_list.append({})

                config_list[-1]['device'] = 'cuda:'+args.device

                config_list[-1]['scenarios'] = scn

                if scn == "no_crash_dense":
                    config_list[-1]['num_npc'] = 70
                elif scn == "no_crash_empty":
                    config_list[-1]['num_npc'] = 0            

                config_list[-1]['initial_town'] = twn

                config_list[-1]['checkpoint'] = ckpt

                config_list[-1]['num_episodes'] = args.num_eps

                if '7dim' in ckpt:
                    config_list[-1]['input_type'] = 'wp_obs_info_speed_steer_ldist_light'
                elif '15dim' in ckpt:
                    config_list[-1]['input_type'] = 'wp_obs_more_info_speed_steer_ldist_light'

    return config_list
                    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="No Crash Test Launcher")
    parser.add_argument("--ckpt", default=argparse.SUPPRESS)
    parser.add_argument("--ckpt_iters", nargs='*', default=argparse.SUPPRESS)
    parser.add_argument("--eval_suffix", default=argparse.SUPPRESS)
    parser.add_argument("--test_towns", nargs='*', default=argparse.SUPPRESS)
    parser.add_argument("--scenarios", nargs='*', default=argparse.SUPPRESS)
    parser.add_argument("--num_eps", type=int, default=25)
    parser.add_argument("--threads", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device", default='0', type=str)

    args = parser.parse_args()

    config_list = construct_config_update(args)

    for i in range(len(config_list)):
        test_config = copy.deepcopy(TEST_CONFIG)
        test_config.update(config_list[i])
        config_list[i] = test_config

    with Pool(args.threads) as p:
       p.map(test_policy, config_list)

# python test_run.py --ckpt 7dim_nocrach_dense_no_lane --ckpt_iters 4807135 6009340 7210465 8415674 --test_towns Town02 --scenarios no_crash_dense --num_eps 25 --threads 4 --device 2 

# python test_run.py --ckpt 15dim_nocrach_dense_no_lane_term_tanh_squashed --ckpt_iters 9624106 10829321 --test_towns Town02 --scenarios no_crash_dense --threads 2 --device 2

# python test_run.py --ckpt 14dim_nocrach_dense_no_lane_term_tanh_squashed --ckpt_iters 16220649 17424400 18625743 19828046 --test_towns Town02 --scenarios no_crash_dense --threads 4 --device 2

# python test_run.py --ckpt 15dim_nocrach_dense_no_lane_term_tanh_squashed_sp_30_wp_10 --ckpt_iters 9323607 8420271 7216397 6012595 4808944 3606154 2403630 --test_towns Town02 --scenarios no_crash_dense --num_eps 25 --threads 6 --device 2 


