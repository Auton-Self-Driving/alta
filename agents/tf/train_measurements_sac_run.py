from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, glob
sys.path.append('./../../')
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import ConfigManager

from ae.controller import AEController

import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time

import gym
from gym.spaces import Box, Discrete

import matplotlib
import matplotlib.pyplot as plt

import vis_module

from gym import wrappers
from datetime import datetime
import matplotlib.pyplot as plt
import tensorboard_logging as tf_log

# PPO specific
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import register_policy
from stable_baselines.common.misc_util import set_global_seeds
from sac_models import My_MlpPolicy_1layer, My_MlpPolicy_2layer, My_MlpPolicy_4layer
import traceback
import time
import csv
np.random.seed(5)
def test(model, env, dump_results=False, path='.', model_step=None):
    dummy_env = DummyVecEnv([lambda: env])
    success_episodes = 0
    collision_obs_episodes = 0
    collision_out_of_road_episodes = 0
    collision_lane_change_episodes = 0
    static_episodes = 0
    max_steps_episodes = 0
    runover_light_episodes = 0
    results = {}
    total_reward = 0
    #env.reset()
    for ind in range(env.config["num_episodes"]):
        print(ind, "of", env.config["num_episodes"])
        obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        obs[:] = env.reset(unseen=True, index=ind)
        done = False
        reward = 0
        
        while not done:
            actions = model.predict(obs, deterministic=True)[0]
            info = env.step(actions)
            reward += info[1][0][0]
            done = info[2]
            obs = np.expand_dims(info[0], axis=0)
        
        total_reward += reward
        if info[3]['termination_state'] == 'success':
            success_episodes += 1
            results[ind] = 1
        else:
            results[ind] = 0
            if info[3]['obs_collision']:
                collision_obs_episodes += 1
            elif info[3]['lane_change']:
                collision_lane_change_episodes += 1
            elif info[3]['out_of_road']:
                collision_out_of_road_episodes += 1
            elif info[3]['termination_state'] == 'runover_light':
                runover_light_episodes += 1
            elif info[3]['termination_state'] == 'static':
                static_episodes += 1
            elif info[3]['termination_state'] == 'max_steps':
                max_steps_episodes += 1

    env.reset()
    print("Results of train scenarios")
    print(results)
    print("# Success: {}, # Obstacle Collision: {}, # Lane-change Collision: {}, Out-of-road Collision: {}, Runover light: {}, Static: {}, Max_steps: {}".format(success_episodes,
                                collision_obs_episodes, collision_lane_change_episodes, collision_out_of_road_episodes, runover_light_episodes, static_episodes, max_steps_episodes))
    if dump_results:
        env.logger.log_scalar('test/term_success', success_episodes, model_step)
        env.logger.log_scalar('test/term_obstacle', collision_obs_episodes, model_step)
        env.logger.log_scalar('test/term_out_of_road', collision_out_of_road_episodes, model_step)
        env.logger.log_scalar('test/term_lane_change', collision_lane_change_episodes, model_step)
        env.logger.log_scalar('test/term_runover_light', runover_light_episodes, model_step)
        env.logger.log_scalar('test/term_static', static_episodes, model_step)
        env.logger.log_scalar('test/term_max_steps', max_steps_episodes, model_step)
        env.logger.log_scalar('test/total_reward', total_reward, model_step)

        with open(path + 'test_results.csv','a') as f:
                csvwriter = csv.writer(f, delimiter=',')
                csvwriter.writerow([model_step, success_episodes, total_reward, collision_obs_episodes,
                        collision_out_of_road_episodes, collision_lane_change_episodes, runover_light_episodes, static_episodes, max_steps_episodes])
    return total_reward, success_episodes, results

def get_scratch_dir(base_log_dir):
    return base_log_dir.split(base_log_dir.split("/home")[0])[1].replace("/home", "/home/scratch")

def run_sac(args, prefix, base_prefix, config):

    ALTA_LOGS = args.base_log_dir + base_prefix + prefix

    if ALTA_LOGS[-1] != '/':
        ALTA_LOGS += '/'

    if "/home/scratch" not in args.base_log_dir and os.path.exists('/home/scratch'):
        SCRATCH_DIR = os.path.join(get_scratch_dir(args.base_log_dir), prefix.split('_runid_')[0], prefix)
    else:
        SCRATCH_DIR = ALTA_LOGS

    if SCRATCH_DIR[-1] != '/':
        SCRATCH_DIR += '/'

    if not os.path.exists(ALTA_LOGS):
        os.makedirs(ALTA_LOGS)

    POLICY_PLOTS = ALTA_LOGS + 'policy_plots/'
    if not os.path.exists(POLICY_PLOTS):
        os.makedirs(POLICY_PLOTS)

    TF_MODELS = ALTA_LOGS+'tf-models/checkpoint/'
    if not os.path.exists(TF_MODELS):
        os.makedirs(TF_MODELS)

    FRAME_SKIP = 1
    SAVE_PATH = ALTA_LOGS + 'sac_weights'
    TB_LOGS_DIR = ALTA_LOGS+ 'tb/'
    MODEL_PATH = ALTA_LOGS + 'sac_measurements_weights' + '150000' + '.pkl'
    MAX_TRIALS = 5

    TEST = False

    print("Training begins")
    IMAGES_PATH = ALTA_LOGS+'images/'
    VIDEO_PATH = ALTA_LOGS+'videos/'
    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP, videos=config.config["videos"])

    vae = AEController(image_size=(128, 128, 5), learning_rate=args.ae_lr)
    
    # config = ConfigManager(algo="PPO")
    logger = tf_log.Logger(TB_LOGS_DIR)
    
    RETRIES_ON_ERROR = 5
    serverStartRetries = 0
    serverStarted = False
    while ((not serverStarted) and serverStartRetries < RETRIES_ON_ERROR):
        try:
            if args.task=="self-driving":
                print("Creating Carla Env")
                from my_sac import MY_SAC
                env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger, log_dir = ALTA_LOGS)
            else:
                print("Creating Mujoco Env")
                from my_sac_mujoco import MY_SAC
                env = gym.make(args.task)
            serverStarted = True
        
        except Exception as identifier:
            print(prefix, identifier)
            traceback.print_exc()
            if env is not None:
                env.close()
                serverStartRetries += 1
                time.sleep(10)

    # TODO: Handle training resume logic for SAC.
    try:
        # env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger)
        dummy_env = DummyVecEnv([lambda: env])
        '''if not args.train_vae and args.task == "self-driving":
            print("Loading pretrained AE!!!")
            vae.load(args.vae_model_path)
            env.set_vae(vae)'''
        if TEST:
            model = MY_SAC.load(MODEL_PATH, env)
            test(model, env, model_step=0)
        else:

            if args.network == "1_layer":
                policy = My_MlpPolicy_1layer
            elif args.network == "2_layer":
                policy = My_MlpPolicy_2layer
            elif args.network == "4_layer":
                policy = My_MlpPolicy_4layer
            else:
                print("specify either 1_layer or 2_layer as network input")
                env.close()
                print("exiting")
                return

            model = MY_SAC(config=config.config, policy=policy, env=dummy_env, learning_rate=args.lr,buffer_size=args.buffer_size,batch_size=config.config["batch_size"],learning_starts=5000,
                tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=False, verbose=1)
            
            model.learn(env, args.timesteps, 0, tb_log_name="SAC", save_file=SAVE_PATH, reset_num_timesteps=True, custom_logger = logger)
            model.save(SAVE_PATH)

    except Exception as identifier:
        with open(ALTA_LOGS + "error.txt", "w") as f:
            print(prefix, identifier)
            traceback.print_exc()
            f.write(str(identifier))

    finally:
        if env is not None:
            env.close()

if __name__ == '__main__':
    #run_ids = np.arange(5)+1
    run_ids = [1]
    base_log_dir = '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_dynamic-navigation/'

    config = ConfigManager(algo="SAC")
    config.config["videos"] = True
    config.config["carla_gpu"] = '2'
    config.config["code_gpu"]  = '2'
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(config.config["code_gpu"])
    config.config["testing"] = True
    config.config["test_fixed_spawn_points"] = True
    config.config["city_name"] = "Town01"
    config.config["input_type"] = "wp_obs_info_speed_steer_ldist_goal_light"
    config.config["num_npc"] = 70
    #config.config['spawn_points_fixed_idx'] = np.load(base_log_dir+'spawn_pt_order_2.npy')
    config.config["ent_coef"] = -1
    config.config["n_steps"] = 1
    config.config["gradient_steps_per_iteration"] = 1
    config.config["target_update_interval"] = 1
    config.config["task"] = "self-driving"
    config.config["network"] = "2_layer"
    config.config["num_npc_lower_threshold"] = 70
    config.config["num_episodes"] = 25
    config.config["train_freq"] = 1

    from my_sac import MY_SAC

    num_successes = []
    tot_rewards = []

    #env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger, log_dir = base_log_dir, base_prefix = base_prefix, prefix = prefix)
    set_global_seeds(5)

    base_prefix = 'algo_SAC_task_self-driving_input_8dim_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_nSteps_25_gdUpdFreq_100_tgtUpdInt_1_ent_0.005_dynamic_navigation_npc_70_cp-250.0-250.0_lp-250.0-250.0/'
    for run_id in run_ids:
        prefix = 'algo_SAC_task_self-driving_input_8dim_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_nSteps_25_gdUpdFreq_100_tgtUpdInt_1_ent_0.005_dynamic_navigation_npc_70_cp-250.0-250.0_lp-250.0-250.0_runid_run'+str(run_id)+'/'
    
        ALTA_LOGS = base_log_dir + base_prefix + prefix

        IMAGES_PATH = ALTA_LOGS+'images/'
        VIDEO_PATH = ALTA_LOGS+'videos/'
        vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, 1)

        config.config['spawn_points_fixed_idx'] = np.load(ALTA_LOGS+'spawn_pt_order.npy')

        env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=None, log_dir = ALTA_LOGS, base_prefix = None, prefix = None)
        dummy_env = DummyVecEnv([lambda: env])
        
        MODEL_PATH = ALTA_LOGS+'sac_weights4425000.pkl'    

        model = MY_SAC.load(MODEL_PATH, env)
        print('Starting evaluation on run id : '+str(run_id))
        tot_reward, success_episodes, _ = test(model, env, False, path = ALTA_LOGS, model_step=0)
        print(success_episodes)
        print(tot_reward)
        num_successes.append(success_episodes)
        tot_rewards.append(tot_reward)

    env.close()
    print(num_successes)
    print(tot_rewards)
