from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, glob
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

import vis_module

from gym import wrappers
from datetime import datetime
import matplotlib.pyplot as plt
import tensorboard_logging as tf_log

# PPO specific
from stable_baselines.ppo2.ppo2 import Runner
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.policies import register_policy
from ppo import PPO
from models import CustomPolicy, CustomWPPolicy, MlpPolicy

prefix = 'ppo_5/'

ALTA_LOGS = '/zfsauton2/home/tanmaya/projects/neurips/ppo_pid_wp_scenarios_straight/' + prefix
SCRATH_DIR = '/home/scratch/tanmaya/projects/neurips/ppo_pid_wp_scenarios_straight/' + prefix
if not os.path.exists(ALTA_LOGS):
    os.makedirs(ALTA_LOGS)

TF_MODELS = ALTA_LOGS+'tf-models/checkpoint/'
if not os.path.exists(TF_MODELS):
    os.makedirs(TF_MODELS)

FRAME_SKIP = 1
SAVE_PATH = ALTA_LOGS + 'ppo2_measurements_weights'
TB_LOGS_DIR = ALTA_LOGS+ 'tb/'

MAX_TRIALS = 100

if __name__ == '__main__':
    
    register_policy('CustomWPPolicy', CustomWPPolicy)
    steps = 0
    
    for i in range(MAX_TRIALS):
        try:
            # Create the environment
            config = ConfigManager(algo="PPO")
            logger = tf_log.Logger(TB_LOGS_DIR)
            
            with open(ALTA_LOGS + "seed.txt", "r") as f:
                millis = int(f.readline())
            print("Using the pre-initialized seed: {}".format(millis))
            set_global_seeds(millis)

            IMAGES_PATH = ALTA_LOGS+'random_images/'
            VIDEO_PATH = ALTA_LOGS+'random_videos/'
            vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
            
            env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger)
            dummy_env = DummyVecEnv([lambda: env])
            
            model = PPO(policy=MlpPolicy, env=dummy_env, n_steps=500, nminibatches=4, verbose=1, learning_rate=2e-4, 
                        tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=True)

            random_model = model.learn(steps, 0, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=True, seed=millis)
            random_model.save(SAVE_PATH + "0")
            
            random_model = PPO.load(SAVE_PATH + "random", dummy_env)
            success_episodes = 0
            results = {}
            with open(ALTA_LOGS + "random_model" + ".txt", "w") as f:
                total_reward = 0
                for ind in range(25):
                    obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
                    obs[:] = env.reset(unseen=True, index=ind)
                    done = False
                    reward = 0
                    
                    while not done:
                        actions = random_model.step(obs, deterministic=True)[0]
                        info = env.step(actions)
                        reward += info[1]
                        done = info[2]
                        obs = np.expand_dims(info[0], axis=0)
                    
                    print("Termination State: {}".format(info[3]['termination_state']))
                    f.write("Termination State: {}".format(info[3]['termination_state']))
                    total_reward += reward
                    if info[3]['termination_state'] == 'success':
                        success_episodes += 1
                        results[ind] = 1
                    else:
                        results[ind] = 0
                print("Total Reward: {}".format(total_reward))
                print("Task Name: {}".format(config.config["scenarios"]))
                print("Town Name: {}".format(config.config["city_name"]))
                print("Results of random model scenarios")
                print(results)
                print("Total Success Episodes: {}".format(success_episodes))
                f.write("\n")
                f.write("Total Reward: {}\n".format(total_reward))
                f.write("Task Name: {}\n".format(config.config["scenarios"]))
                f.write("Town Name: {}\n".format(config.config["city_name"]))
                f.write("Results of test scenarios\n")
                # f.write(results)
                f.write("Total Success Episodes: {}\n".format(str(success_episodes)))
            break
        except Exception as e:
            print(e)
            env.close()
            time.sleep(120)