from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, glob
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import ConfigManager

import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time

import baselines.common.tf_util as U

# NOTE: not using baselines logger for now
# from baselines import logger
from baselines import deepq
from baselines.deepq.deepq import ActWrapper
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule
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
from my_sac import MY_SAC, plot_policy_and_value_fns, My_MlpPolicy_1layer, My_MlpPolicy_2layer
import traceback
# from models import CustomPolicy, CustomWPPolicy, Policy
# from stable_baselines.sac.policies import MlpPolicy


# def get_latest_model(log_dir=ALTA_LOGS, ext='*.pkl', sep='_'):
#     list_of_files = glob.glob(log_dir + ext)
#     latest_file = max(list_of_files, key=os.path.getctime)
#     latest_file = latest_file.split('.')[0]
#     ind = int(latest_file.split(sep)[1])
#     return ind, latest_file

def test(model, env, model_step):
    dummy_env = DummyVecEnv([lambda: env])
    # dummy_env = env
    success_episodes = 0
    results = {}
    total_reward = 0
    for ind in range(1):
        obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        obs[:] = env.reset(unseen=True, index=ind)
        done = False
        reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            info = env.step(action)
            reward += info[1]
            done = info[2]
            obs = np.expand_dims(info[0], axis=0)
        
        total_reward += reward
        if info[3]['termination_state'] == 'success':
            success_episodes += 1
            results[ind] = 1
        else:
            results[ind] = 0
    print("Results of train scenarios")
    print(results)
    print("Step: {0} Total Success Episodes: {1}".format(model_step, success_episodes))
    env.logger.log_scalar('test/success_episodes', success_episodes, model_step)
    env.logger.log_scalar('test/total_reward', total_reward, model_step)
    return total_reward, success_episodes

# def get_best_model(total_timesteps, save_file, env):
#     print("Searching for best model now!!!")
#     total_rewards = []
#     total_successes = []
#     for model_step in range(0, total_timesteps + 1, 40000):
#         model = MY_SAC.load(save_file + str(model_step))
#         print("Loading model file: {}".format(save_file + str(model_step)))
#         total_reward, success_episodes = test(model, env)
#         print(total_reward, success_episodes)
#         env.logger.log_scalar('test/success_episodes', success_episodes, model_step)
#         env.logger.log_scalar('test/total_reward', total_reward, model_step)
#         total_rewards.append(total_reward)
#         total_successes.append(success_episodes)
#     print("Rewards at intermediate training: {}".format(total_rewards))
#     print("Total success episodes: {}".format(total_successes))
#     m = max(total_successes)
#     max_inds = np.array([i for i, j in enumerate(total_successes) if j == m])
#     rewards = np.array(total_rewards)[max_inds]
#     ind = max_inds[np.argmax(rewards)]
#     print("Best model appears at index: {}".format(ind))
#     print("No of successes in best model: {}".format(total_successes[ind]))
#     print("Max no of successes: {}".format(m))
#     SAVE_PATH = save_file + str(4 * (ind + 1)) + "0000"
#     best_model = MY_SAC.load(SAVE_PATH, DummyVecEnv([lambda: env]))
    
#     with open(ALTA_LOGS + "best_model.txt", "w") as f:
#         f.write("Best model: {}\n".format(SAVE_PATH))
#         f.write("Best model appears at index: {}\n".format(ind))
#         f.write("No of successes in best model: {}\n".format(total_successes[ind]))
#         f.write("Max no of successes: {}\n".format(m))
#         f.write("Rewards at intermediate training: {}\n".format(total_rewards))
#         f.write("Total success episodes: {}\n".format(total_successes))
        
#     return best_model

def run_sac(args, prefix, base_prefix, config):

    # prefix = 'sac_nav_5_buf_1m_b_256_lr_3e_4_simple2_r_10_nn64_test1/'

    # ALTA_LOGS = '/zfsauton2/home/hiteshar/research/alta-logs/new_env/sac_runs1' + prefix
    
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
    
    # config = ConfigManager(algo="PPO")
    logger = tf_log.Logger(TB_LOGS_DIR)
    
    RETRIES_ON_ERROR = 5
    serverStartRetries = 0
    serverStarted = False
    while ((not serverStarted) and serverStartRetries < RETRIES_ON_ERROR):
        try:

            env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger)
            serverStarted = True
        
        except Exception as identifier:
            print(prefix, identifier)
            traceback.print_exc()
            if env is not None:
                env.close()
                serverStartRetries += 1
                time.sleep(10)

    
    try:
        # env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger)
        dummy_env = DummyVecEnv([lambda: env])

        if TEST:
            model = MY_SAC.load(MODEL_PATH, env)
            test(model, env, model_step=0)
        else:

            if args.layers == "1_layer":
                policy = My_MlpPolicy_1layer
            elif args.layers == "2_layer":
                policy = My_MlpPolicy_2layer
            else:
                print("specify either 1_layer or 2_layer as layers input")
                env.close()
                print("exiting")
                return

            model = MY_SAC(policy=policy, env=dummy_env, learning_rate=args.lr,buffer_size=args.buffer_size,batch_size=512,learning_starts=5000,
                tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=False, verbose=1)
            
            model.learn(env, args.timesteps, 0, tb_log_name="SAC", save_file=SAVE_PATH, reset_num_timesteps=True)
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
    
    # register_policy('CustomWPPolicy', CustomWPPolicy)
    steps = 1000000

    print("Training begins")
    IMAGES_PATH = ALTA_LOGS+'images/'
    VIDEO_PATH = ALTA_LOGS+'videos/'
    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
    
    config = ConfigManager(algo="PPO")
    logger = tf_log.Logger(TB_LOGS_DIR)
    env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger)
    dummy_env = DummyVecEnv([lambda: env])
    
    if TEST:
        model = MY_SAC.load(MODEL_PATH, env)
        test(model, env, model_step=0)
    else:

        model = MY_SAC(policy=My_MlpPolicy, env=dummy_env, learning_rate=3e-4,buffer_size=1000000,batch_size=256,
            tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=True, verbose=1)
        
        model.learn(env, steps, 0, tb_log_name="SAC", save_file=SAVE_PATH, reset_num_timesteps=True)
        model.save(SAVE_PATH)
    env.close()
    
