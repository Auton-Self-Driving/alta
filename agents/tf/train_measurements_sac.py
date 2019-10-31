from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

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

# SAC specific
# from stable_baselines.ppo2.ppo2 import PPO2, Runner
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC

prefix = 'sac_nav_dist_10_buf_2000k_batch_256_lr_3e_4_1/'

ALTA_LOGS = '/zfsauton2/home/hiteshar/research/alta-logs/'
if not os.path.exists(ALTA_LOGS):
    os.makedirs(ALTA_LOGS)

TF_MODELS = ALTA_LOGS+prefix+'tf-models/checkpoint/'
if not os.path.exists(TF_MODELS):
    os.makedirs(TF_MODELS)

IMAGES_PATH = ALTA_LOGS+prefix+'images/'
VIDEO_PATH = ALTA_LOGS+prefix+'videos/'
FRAME_SKIP = 4
SAVE_PATH = ALTA_LOGS + prefix + 'sac_weights'
TB_LOGS_DIR = ALTA_LOGS+prefix+str(datetime.now())

def train():
    model = SAC(policy=MlpPolicy, env=env, learning_rate=3e-4,buffer_size=2000000,batch_size=256,
                 tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=True, verbose=1)
    model.learn(1000000, log_interval=10)
    return model

if __name__ == '__main__':

    # Create the environment
    config = ConfigManager(algo="PPO")
    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
    logger = tf_log.Logger(TB_LOGS_DIR)
    env = CarlaEnv(config.config, vis_wrapper, None, logger)
    env = DummyVecEnv([lambda: env])
    
    
    if os.path.exists(SAVE_PATH + ".pkl"):
        print("Task: test")
        model = SAC.load(SAVE_PATH, env)
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        while True:
            actions = model.step(obs)[0]
            obs = env.step(actions)[0]

    else:

        model = train()
        model.save(SAVE_PATH)