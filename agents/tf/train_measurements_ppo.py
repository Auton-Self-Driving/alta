from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

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
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from ppo import PPO

prefix = 'ppo_wp_straight_merged_speed_simple6_w_bias_lr_25e4_1/'

ALTA_LOGS = '/zfsauton2/home/tanmaya/projects/alta-logs/ppo_pid_scenarios_straight/'
if not os.path.exists(ALTA_LOGS):
    os.makedirs(ALTA_LOGS)

TF_MODELS = ALTA_LOGS+prefix+'tf-models/checkpoint/'
if not os.path.exists(TF_MODELS):
    os.makedirs(TF_MODELS)

FRAME_SKIP = 1
SAVE_PATH = ALTA_LOGS + prefix + 'ppo2_measurements_weights110000'
TB_LOGS_DIR = ALTA_LOGS+prefix+str(datetime.now())

def train():
    model = PPO(policy=MlpPolicy, env=dummy_env, n_steps=500, nminibatches=4, verbose=1, learning_rate=5e-5, 
                tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=True)
    model.learn(1000000, tb_log_name="PPO2", save_file=SAVE_PATH)
    return model

if __name__ == '__main__':

    # Create the environment
    config = ConfigManager(algo="PPO")
    logger = tf_log.Logger(TB_LOGS_DIR)
    
    if os.path.exists(SAVE_PATH + ".pkl"):
        IMAGES_PATH = ALTA_LOGS+prefix+'test_images_' + config.config["city_name"] + '_5/'
        VIDEO_PATH = ALTA_LOGS+prefix+'test_videos_' + config.config["city_name"] + '_5/'
        vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
        
        env = CarlaEnv(config.config, vis_wrapper, logger)
        dummy_env = DummyVecEnv([lambda: env])
        
        print("Task: test")
        model = PPO.load(SAVE_PATH, dummy_env)
        success_episodes = 0
        results = {}
        for ind in range(25):
            obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
            obs[:] = env.reset(unseen=True, index=ind)
            done = False
            while not done:
                actions = model.step(obs, deterministic=False)[0]
                info = env.step(actions)
                done = info[2]
                obs = np.expand_dims(info[0], axis=0)
            
            print("Termination State: {}".format(info[3]['termination_state']))
            if info[3]['termination_state'] == 'success':
                success_episodes += 1
                results[ind] = 1
            else:
                results[ind] = 0
        print("Task Name: {}".format(config.config["scenarios"]))
        print("Town Name: {}".format(config.config["city_name"]))
        print("Results of test scenarios")
        print(results)
        print("Total Success Episodes: {}".format(success_episodes))
    else:
        IMAGES_PATH = ALTA_LOGS+prefix+'images/'
        VIDEO_PATH = ALTA_LOGS+prefix+'videos/'
        vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
        
        env = CarlaEnv(config.config, vis_wrapper, logger)
        dummy_env = DummyVecEnv([lambda: env])

        model = train()
        model.save(SAVE_PATH)