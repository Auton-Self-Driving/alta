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
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.policies import register_policy
from sac_models import My_MlpPolicy_1layer, My_MlpPolicy_2layer
import traceback
import time
import csv

def test(model, env, model_step, path=None):
    dummy_env = DummyVecEnv([lambda: env])
    # dummy_env = env
    success_episodes = 0
    results = {}
    total_reward = 0
    episode_timesteps = {}
    episode_time = {}
    saving_time = {}
    for ind in range(25):
        episode_timesteps[ind] = 0
        episode_time[ind] = 0
        saving_time[ind] = 0

    for ind in range(25):
        st = time.time()
        print("Episode number ", ind)
        obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        obs[:] = env.reset(unseen=True, index=ind)
        done = False
        reward = 0
        
        while not done:
            episode_timesteps[ind] +=1
            curr_st = time.time()
            action, _states = model.predict(obs, deterministic=True)
            info = env.step(action)
            reward += info[1]
            done = info[2]
            if done:
                saving_time[ind] += time.time()-curr_st
            else:
                episode_time[ind] += time.time()-curr_st
                saving_time[ind] += info[-1]['saving_time']
            obs = np.expand_dims(info[0], axis=0)
        
        total_reward += reward
        if info[3]['termination_state'] == 'success':
            success_episodes += 1
            results[ind] = reward
        else:
            results[ind] = reward
        print("Ended in ", time.time()-st)

    obs[:] = env.reset()
    print("Results of train scenarios")
    print(results)
    print(episode_timesteps)
    print(episode_time)
    print(saving_time)
    print("Step: {0} Total Success Episodes: {1}".format(model_step, success_episodes))

    with open(path + 'standalone_test_results.csv','a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([model_step, success_episodes, total_reward])

    return total_reward, results

def run_sac(args, prefix, base_prefix, config):

    ALTA_LOGS = args.base_log_dir + base_prefix + prefix
    POLICY_PLOTS = ALTA_LOGS + 'policy_plots/'
    if not os.path.exists(ALTA_LOGS):
        os.makedirs(ALTA_LOGS)

    TF_MODELS = ALTA_LOGS+'tf-models/checkpoint/'
    if not os.path.exists(TF_MODELS):
        os.makedirs(TF_MODELS)

    FRAME_SKIP = 1
    SAVE_PATH = ALTA_LOGS + 'sac_measurements_weights'
    TB_LOGS_DIR = ALTA_LOGS+ 'tb/'
    MODEL_PATH = ALTA_LOGS + 'sac_measurements_weights' + '150000' + '.pkl'
    MAX_TRIALS = 10

    TEST = False

    print("Training begins")
    IMAGES_PATH = ALTA_LOGS+'images/'
    VIDEO_PATH = ALTA_LOGS+'videos/'
    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)

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
                env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger, log_dir = args.base_log_dir, base_prefix = base_prefix, prefix = prefix)
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
        if not args.train_vae and args.task == "self-driving":
            print("Loading pretrained AE!!!")
            vae.load(args.vae_model_path)
            env.set_vae(vae)
        if TEST:
            model = MY_SAC.load(MODEL_PATH, env)
            test(model, env, model_step=0)
        else:

            if args.network == "1_layer":
                policy = My_MlpPolicy_1layer
            elif args.network == "2_layer":
                policy = My_MlpPolicy_2layer
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
    base_log_dir = '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_no-lane-sensor/'
    base_prefix = 'algo_SAC_task_self-driving_input_wp_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_n-steps_100_gradupd-per-iter_1_tgt-upd-int_1_ent-coef_0.005_cp-0.0-0.0_navigation/'
    prefix = 'algo_SAC_task_self-driving_input_wp_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_n-steps_100_gradupd-per-iter_1_tgt-upd-int_1_ent-coef_0.005_cp-0.0-0.0_navigation_runid_run10/'
    
    ALTA_LOGS = base_log_dir + base_prefix + prefix

    IMAGES_PATH = ALTA_LOGS+'images/'
    VIDEO_PATH = ALTA_LOGS+'videos/'
    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, 1)


    config = ConfigManager(algo="SAC")
    config.config["videos"] = True 
    config.config["carla_gpu"] = '3'
    config.config["code_gpu"]  = '3'
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(config.config["code_gpu"])
    config.config["testing"] = True
    config.config['spawn_points_fixed_idx'] = np.load(base_log_dir+'spawn_pt_order.npy')
    config.config["ent_coef"] = -1
    config.config["n_steps"] = 1
    config.config["gradient_steps_per_iteration"] = 1
    config.config["target_update_interval"] = 1
    config.config["task"] = "self-driving"

    logger = None
    env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger, log_dir = base_log_dir, base_prefix = base_prefix, prefix = prefix)
    dummy_env = DummyVecEnv([lambda: env])

    MODEL_PATH = ALTA_LOGS+'sac_measurements_weights230000.pkl'    

    from my_sac import MY_SAC
    model = MY_SAC.load(MODEL_PATH, env)
    tot_reward, ep_reward = test(model, env, model_step=0, path = ALTA_LOGS)
    print(ep_reward)
    print(tot_reward)

    env.close()
    
