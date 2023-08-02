"""Test multi-agent algo
"""

import os
import glob
import sys
import shutil

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

from network import PPOActorCritic_Continuous, PolicyNetwork, SoftQNetwork, PPOActorCritic_Mixed
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
    N_S = env.obs_manager.observation_space.shape[-1]
    N_A = env.obs_manager.action_space.shape[-1]

    print('testing config:\n{}'.format(test_config))

    if test_config['PPO']:
        if "disc_thrt" in test_config["action_type"]:
            glb_policy = PPOActorCritic_Mixed(N_S, N_A, ENV_CONFIG['discrete_spd_lvls'],
                use_transformer=ENV_CONFIG['input_type']=='transformer'
                ).to(ENV_CONFIG['device'])
        else:
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


def construct_config_update(args):

    ckpt_base_path = "./checkpoints/{}/".format(args.ckpt)

    ckpt_paths_list = [ glob.glob(ckpt_base_path+"*_{}_*".format(itr))[0] for itr in args.ckpt_iters]

    config_list = []

    for ckpt in ckpt_paths_list:
        for twn in args.test_towns:
            for scn in args.scenarios:

                # Skip if evaluations already exist
                eval_folder = '{}/{}/{}/{}/{}/'.format('./tests/evals',scn,twn,args.ckpt,ckpt.split('_')[-2])

                # TODO: Comment out
                if os.path.exists(eval_folder):
                    shutil.rmtree(eval_folder) 
                if os.path.exists(os.path.join(eval_folder,'1.png')):
                    continue

                config_list.append({})

                config_list[-1]['device'] = 'cuda:'+args.device

                config_list[-1]['num_episodes'] = args.num_eps
                
                config_list[-1]['scenarios'] = scn

                if scn == "no_crash_dense":
                    config_list[-1]['num_npc'] = 70
                elif scn == "no_crash_empty":
                    config_list[-1]['num_npc'] = 0
                elif scn in ["straight_overtake","straight_overtake_closeby"]:
                    config_list[-1]['num_npc'] = 1
                    config_list[-1]['enable_off_road_termination'] = False
                    config_list[-1]['enable_lane_invasion_termination'] = False
                else:
                    config_list[-1]['num_npc'] = 0      

                config_list[-1]['city_name'] = twn

                config_list[-1]['checkpoint'] = ckpt

                config_list[-1]['num_episodes'] = args.num_eps

                
                if '7dim' in ckpt:
                    config_list[-1]['input_type'] = 'wp_obs_info_speed_steer_ldist_light'
                elif '15dim' in ckpt:
                    config_list[-1]['input_type'] = 'wp_obs_more_info_speed_steer_ldist_light'
                elif '24dim' in ckpt or '360deg' in ckpt:
                    config_list[-1]['input_type'] = 'wp_360_obstacle_speed_steer'
                else:
                    config_list[-1]['input_type'] = 'wp_obs_info_speed_steer_ldist_light' # 7dim

                if "5dof" in ckpt:
                    if "disc_thrt" in ckpt:
                        config_list[-1]['action_type'] = 'cubic_bezier_5dof_disc_thrt'
                    else:
                        config_list[-1]['action_type'] = 'cubic_bezier_5dof'
                elif "wp" in ckpt:
                    if "steer_only" in ckpt:
                        pass
                    else:
                        config_list[-1]['action_type'] = 'speed_wp'
                # elif "str_steer_only" in ckpt:
                #     config_list[-1]['action_type'] = 'steer_only'

                if args.autopilot is not None:
                    config_list[-1]['autopilot_type'] = args.autopilot


                

    print(config_list)
    print("---------------------")

    return config_list
                    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="No Crash Test Launcher")
    parser.add_argument("--ckpt", default=argparse.SUPPRESS)
    parser.add_argument("--ckpt_iters", nargs='*', default=argparse.SUPPRESS)
    parser.add_argument("--autopilot", default=argparse.SUPPRESS)
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
# python test_run.py --ckpt 14dim_nocrach_dense_no_lane_term_tanh_squashed --ckpt_iters 17424400 --test_towns Town01 --scenarios curved t_junction left_right_curved right_curved straight_crowded --threads 5 --device 1
# python test_run.py --ckpt 16dim_nocrach_dense_no_lane_term_tanh_squashed --ckpt_iters 17424400 --test_towns Town02 --scenarios no_crash_dense --threads 8 --device 1
# python test_run.py --ckpt 24dim_10wp_nocrach_dense_no_lane_term_tanh_squashed --ckpt_iters 24057987 22254420 20750512 18642090 16839270 15034713 --test_towns Town02 --scenarios no_crash_dense --threads 4 --device 3

# python test_run.py --ckpt cubic_bezier3dof_long_straight --ckpt_iters 50005  --test_towns Town01 --scenarios straight --threads 1 --device 2
# python test_run.py --ckpt sp_throttle_straight --ckpt_iters 120088 150201 --test_towns Town01 --scenarios straight --threads 4 --device 2
# python test_run.py --ckpt sp_wp_straight --ckpt_iters 90111  --test_towns Town01 --scenarios straight --threads 1 --device 2
# python test_run.py --ckpt cubic_bezier5dof_straight --ckpt_iters 90183  --test_towns Town01 --scenarios straight --threads 1 --device 2
# python test_run.py --ckpt cubic_bezier5dof_st_ovrtk --ckpt_iters 30155  --test_towns Town01 --scenarios straight_overtake --threads 1 --device 2
# python test_run.py --ckpt sp_throttle_15dim_st_ovrtk --ckpt_iters 90157  --test_towns Town01 --scenarios straight_overtake --threads 1 --device 2
# python test_run.py --ckpt sp_wp_24dim_st_ovrtk_lanepen --ckpt_iters 150613 120568 30255 60320 90523  --test_towns Town01 --scenarios straight_overtake --threads 5 --device 2
# python test_run.py --ckpt sp_str_24dim_st_ovrtk_lanepen --ckpt_iters 300274 210133 120066 90046 --test_towns Town01 --scenarios straight_overtake --threads 5 --device 0
# python test_run.py --ckpt sp_str_24dim_st_frameskip_4 --ckpt_iters 60137 90249 120421 150458 302682 --test_towns Town01 --scenarios straight --threads 5 --device 1


# python test_run.py --num_eps 1 --ckpt 360deg_5dof_stovrtk_fs_1 --ckpt_iters 2283542 --test_towns Town01 --scenarios straight_overtake --threads 1 --device 3
# python test_run.py --num_eps 1 --ckpt 360deg_5dof_stovrtk_fs_4 --ckpt_iters 631242 --test_towns Town01 --scenarios straight_overtake --threads 1 --device 3
# python test_run.py --num_eps 1 --ckpt 360deg_str_steer_only_stovrtk_fs_4 --ckpt_iters 810414 --test_towns Town01 --scenarios straight_overtake --threads 1 --device 3
# python test_run.py --num_eps 1 --ckpt 360deg_5dof_steer_only_stovrtk_fs_4 --ckpt_iters 1080654 --test_towns Town01 --scenarios straight_overtake --threads 1 --device 0
# python test_run.py --num_eps 1 --ckpt 360deg_5dof_stovrtk_fs_12_gamma_50 --ckpt_iters 1110310 --test_towns Town01 --scenarios straight_overtake --threads 1 --device 0
# python test_run.py --num_eps 1 --ckpt 360deg_5dof_disc_thrt_stovrtk_fs_4 --ckpt_iters 960846 --test_towns Town01 --scenarios straight_overtake --threads 1 --device 0

# python test_run.py --num_eps 10 --ckpt 360deg_5dof_steer_only_stovrtk_fs_4 --ckpt_iters 1260731 --autopilot const_speed --test_towns Town01 --scenarios straight_overtake --threads 1 --device 1
# python test_run.py --num_eps 10 --ckpt 360deg_5dof_ppo_part_steer_final_stovrtk_fs_4 --ckpt_iters 1080676 --autopilot PPO_part_steer_final --test_towns Town01 --scenarios straight_overtake --threads 1 --device 1
# python test_run.py --num_eps 10 --ckpt 360deg_5dof_ppo_part_steer_interm_stovrtk_fs_4 --ckpt_iters 1051014 --autopilot PPO_part_steer_interm --test_towns Town01 --scenarios straight_overtake --threads 1 --device 1

# python test_run.py --num_eps 10 --ckpt 360deg_5dof_20_spd_stovrtk_fs_16 --ckpt_iters 120004 --autopilot const_speed --test_towns Town01 --scenarios straight_overtake_closeby --threads 1 --device 0


