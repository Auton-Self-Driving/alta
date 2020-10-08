# Idea create a CARLA environment that spawns multiple (100) agents of one type and keep them running 
# and count number of collisions and successes

import sys, os, glob
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))

from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import ConfigManager

import numpy as np
import math
import time
import vis_module
import traceback
import csv
import multiprocessing as mp
import matplotlib.pyplot as plt
# import logging

# logger = mp.log_to_stderr()
# logger.setLevel(mp.SUBDEBUG)

from datetime import datetime
import tensorboard_logging as tf_log

# PPO specific
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.policies import register_policy
from ppo import PPO, plot_policy_and_value_fns, test, plot_test_results
from models import Policy_1_layer, Policy_2_layer, CustomPolicy1, CustomPolicy2, CnnPolicy

def get_scratch_dir(base_log_dir):
    return base_log_dir.split(base_log_dir.split("/home")[0])[1].replace("/home", "/home/scratch")

def find_ext_format(MODEL_PATH):
    ext = None
    for fname in os.listdir(MODEL_PATH):
        if fname.endswith('.pkl'):
            ext = '.pkl'
        elif fname.endswith('.zip'):
            ext = '.zip'
        
        if ext is not None:
            break
    return ext

def plot_reward(timesteps, mean_reward, min_reward, max_reward, figname="mean_reward.png"):
    plt.figure(figsize=(11, 7))
    timesteps = timesteps / 1000000
    timesteps_interval = 0.5
    plt.plot(timesteps, mean_reward, label='WRL+',  color='orangered')
    plt.fill_between(timesteps, min_reward, max_reward, color='mistyrose')

    axes = plt.gca()
    plt.title("Reward")
    plt.xlabel('Timesteps (in M)', fontdict={'size' : 18})
    plt.ylabel('Total Reward', fontdict={'size' : 18})
    plt.xticks(list(np.arange(0, (math.ceil(timesteps[-1] / timesteps_interval) + 1) * timesteps_interval, timesteps_interval)), ('{}'.format(str(x)) for x in np.arange(0, (math.ceil(timesteps[-1] / timesteps_interval) + 1) * timesteps_interval, timesteps_interval)))
    plt.savefig(figname, dpi=200)

def plot_success(timesteps, mean_success, min_success, max_success, figname="mean_success.png"):
    plt.figure(figsize=(11, 7))
    timesteps = timesteps / 1000000
    timesteps_interval = 0.5
    plt.plot(timesteps, mean_success, label='WRL+',  color='orangered')
    plt.fill_between(timesteps, min_success, max_success, color='mistyrose')

    axes = plt.gca()
    plt.title("Success")
    plt.xlabel('Timesteps (in M)', fontdict={'size' : 18})
    plt.ylabel('Total Success', fontdict={'size' : 18})
    plt.xticks(list(np.arange(0, (math.ceil(timesteps[-1] / timesteps_interval) + 1) * timesteps_interval, timesteps_interval)), ('{}'.format(str(x)) for x in np.arange(0, (math.ceil(timesteps[-1] / timesteps_interval) + 1) * timesteps_interval, timesteps_interval)))
    plt.savefig(figname, dpi=200)

def launch_server(config, vis_wrapper, ALTA_LOGS, logger=None):
    RETRIES_ON_ERROR = 5
    serverStartRetries = 0
    serverStarted = False

    env = None
    while ((not serverStarted) and serverStartRetries < RETRIES_ON_ERROR):
        try:
            env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger, log_dir=ALTA_LOGS)
            serverStarted = True

        except Exception as identifier:
            traceback.print_exc()
            if env is not None:
                env.close()
                serverStartRetries += 1
                time.sleep(20)
    return env

def run_test_comparison(args, prefix, config):
    ALTA_LOGS = os.path.join(args.base_log_dir, prefix.split('_runid_')[0], prefix)
    # config.config['LOG_DIR'] = ALTA_LOGS
    if ALTA_LOGS[-1] != '/':
        ALTA_LOGS += '/'

    if "/home/scratch" not in args.base_log_dir and os.path.exists('/home/scratch'):
        SCRATCH_DIR = os.path.join(get_scratch_dir(args.base_log_dir), prefix.split('_runid_')[0], prefix)
    else:
        SCRATCH_DIR = ALTA_LOGS

    if SCRATCH_DIR[-1] != '/':
        SCRATCH_DIR += '/'
    
    POLICY_PLOTS = ALTA_LOGS + 'policy_plots/'
    if not os.path.exists(ALTA_LOGS):
        os.makedirs(ALTA_LOGS)

    TF_MODELS = ALTA_LOGS+'tf-models/checkpoint/'
    if not os.path.exists(TF_MODELS):
        os.makedirs(TF_MODELS)

    VIDEO_FRAME_SKIP = 1
    MODEL_PATH = os.path.join(ALTA_LOGS, 'models')
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    SAVE_PATH = os.path.join(MODEL_PATH, 'ppo2_weights')
    FORWARD_SEARCH_MODEL = os.path.join(MODEL_PATH, 'forward_search_model')
    TB_LOGS_DIR = ALTA_LOGS+ 'tb/'

    MAX_TRIALS = 5

    for i in range(MAX_TRIALS):
        try:
            # Create the environment
            logger = tf_log.Logger(TB_LOGS_DIR)
            # if os.path.exists(SAVE_PATH + ".zip"):
            #     print("Best model exists, Validating !!!!")
            if args.automatic_control:
                print('Enabled Testing Comparison with Automatic Control')
                
                # with open(ALTA_LOGS + "seed.txt", "r") as f:
                #     seed = int(f.readline())
                seed = 10
                print("Using the pre-initialized seed: {}".format(seed))
                set_global_seeds(seed)
                
                IMAGES_PATH = SCRATCH_DIR+'test_images'
                VIDEO_PATH = SCRATCH_DIR+'test_videos'
                vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, VIDEO_FRAME_SKIP, videos=config.config["videos"])

                if args.city_name == 'Town01':
                    spawn_points_fixed_idx = np.random.permutation(257)
                elif args.city_name == 'Town02':
                    spawn_points_fixed_idx = np.random.permutation(101)

                config.config['spawn_points_fixed_idx'] = list(spawn_points_fixed_idx)
                config.config['test_with_automatic_control'] = True
                # Sending logger as None so as to not affect existing validation plots
                # env = launch_server(config, vis_wrapper, ALTA_LOGS)
                env = launch_server(config, vis_wrapper, ALTA_LOGS)
    
                dummy_env = DummyVecEnv([lambda: env])
                
                env.reset()
                while True:
                    env.get_action_for_test_comparison()
                env.close()
            else:
                raise NotImplementedError
            break
        except Exception as e:
            with open(ALTA_LOGS + "error.txt", "w") as f:
                print("********** Code ERROR for prefix: {} **********".format(prefix))
                print(e)
                print(traceback.format_exc())
                f.write(str(e))
                f.write(traceback.format_exc())
        finally:
            env.close()
            time.sleep(120)