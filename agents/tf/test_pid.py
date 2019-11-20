import sys, os
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))


from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import ConfigManager
import itertools
import numpy as np
import tensorflow as tf
import time

import vis_module
from gym import wrappers

from datetime import datetime

import matplotlib.pyplot as plt
import tensorboard_logging as tf_log



def test_pid_method(args, prefix, config):
    
    ALTA_LOGS = args.base_log_dir + prefix
    if not os.path.exists(ALTA_LOGS):
        os.makedirs(ALTA_LOGS)

    IMAGES_PATH = ALTA_LOGS + 'images/'
    VIDEO_PATH = ALTA_LOGS + 'videos/'
    IMAGES_PATH_VAE = ALTA_LOGS + 'images_VAE/'
    VIDEO_PATH_VAE = ALTA_LOGS + 'videos_VAE/'
    
    VIDEO_FRAME_SKIP = 1
    
    TF_MODELS = ALTA_LOGS+'tf-models/checkpoint/'
    if not os.path.exists(TF_MODELS):
        os.makedirs(TF_MODELS)

    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, frame_skip=VIDEO_FRAME_SKIP)
    vis_wrapper_vae = vis_module.vis(IMAGES_PATH_VAE, VIDEO_PATH_VAE, frame_skip=VIDEO_FRAME_SKIP)
    
    TB_LOGS_DIR = ALTA_LOGS+ 'tb/'
    logger = tf_log.Logger(TB_LOGS_DIR)

    env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger, log_dir=ALTA_LOGS)
    obs = env.reset()

    
    num_episodes = 0
    val_accuracy_total = []

    # actions = np.array([[0,10]]*400 + [[0,0]] * 200 + [[0,10]]*400 + [[0,20]]* 1000)
    actions = np.array([[0,20]]*250 + [[0,0]] * 200 + [[0,20]]* 1000)
    actions = np.array([[0,20]]*200 + [[0,0]] * 200 + [[0,20]]* 200+ [[0,0]] * 200 + [[0,20]]* 1000)
    actions = np.array([[0,10]]*200 + [[0,0]] * 200 + [[0,10]]*200 + [[0,0]] * 200 + [[0,10]]* 1000)
    actions = np.array([[0,10]]*50 + [[0,0]] * 50 + [[0,10]]*50 + [[0,0]] * 200 + [[0,10]]* 1000)
    actions = np.array([[0,10]]*1000)
    done = False
    i = 0
    while not done:
        action = actions[i]
        i = i + 1
        # steer, target_speed
        # action = np.array([0, 1])
        new_obs, rew, done, eps_measurements = env.step(action)
