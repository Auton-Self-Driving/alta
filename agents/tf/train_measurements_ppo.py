from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, glob
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

# PPO specific
from stable_baselines.ppo2.ppo2 import Runner
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.policies import register_policy
from ppo import PPO
from models import CustomPolicy, CustomWPPolicy, MlpPolicy

prefix = 'ppo_new_lr_5e5/'

ALTA_LOGS = '/zfsauton2/home/tanmaya/projects/alta-logs/new_ppo_pid_scenarios_straight_1layer/' + prefix
if not os.path.exists(ALTA_LOGS):
    os.makedirs(ALTA_LOGS)

TF_MODELS = ALTA_LOGS+'tf-models/checkpoint/'
if not os.path.exists(TF_MODELS):
    os.makedirs(TF_MODELS)

FRAME_SKIP = 1
SAVE_PATH = ALTA_LOGS + 'ppo2_measurements_weights'
TB_LOGS_DIR = ALTA_LOGS+ 'tb/'

def get_latest_model(log_dir=ALTA_LOGS, ext='*.pkl', sep='_'):
    list_of_files = glob.glob(log_dir + ext)
    latest_file = max(list_of_files, key=os.path.getctime)
    latest_file = latest_file.split('.')[0]
    ind = int(latest_file.split(sep)[1])
    return ind, latest_file

if __name__ == '__main__':

    # Create the environment
    config = ConfigManager(algo="PPO")
    logger = tf_log.Logger(TB_LOGS_DIR)
    
    if os.path.exists(SAVE_PATH + ".pkl"):
        with open(ALTA_LOGS + "seed.txt", "r") as f:
            seed = int(f.readline())
        print("Using the pre-initialized seed: {}".format(seed))
        set_global_seeds(seed)

        IMAGES_PATH = ALTA_LOGS+'test_images_' + config.config["city_name"] + '/'
        VIDEO_PATH = ALTA_LOGS+'test_videos_' + config.config["city_name"] + '/'
        vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
        
        env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger)
        dummy_env = DummyVecEnv([lambda: env])
        
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
        IMAGES_PATH = ALTA_LOGS+'images/'
        VIDEO_PATH = ALTA_LOGS+'videos/'
        vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
        
        env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger)
        dummy_env = DummyVecEnv([lambda: env])
        register_policy('CustomWPPolicy', CustomWPPolicy)
        
        model = PPO(policy=CustomWPPolicy, env=dummy_env, n_steps=500, nminibatches=4, verbose=1, learning_rate=5e-5, 
                tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=True)
        steps = 1000000
        if any(fname.endswith('.pkl') for fname in os.listdir(ALTA_LOGS)):
            with open(ALTA_LOGS + "seed.txt", "r") as f:
                seed = int(f.readline())
            print("Using the pre-initialized seed: {}".format(seed))
            set_global_seeds(seed)
            completed_steps, latest_model = get_latest_model(log_dir=ALTA_LOGS, ext='*.pkl', sep='hts')
            completed_episodes, _ = get_latest_model(log_dir=ALTA_LOGS + 'videos/', ext='*.mp4', sep='log_')
            print("Loading Latest model!!!")
            model.load(latest_model, dummy_env)
            print("Model: {} loaded successfully".format(latest_model))
            env.total_steps = completed_steps
            env.episode_num = completed_episodes
            _, best_model = model.learn(steps, completed_steps, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=False, seed=seed)    
        else:
            dt = datetime.now()
            millis = dt.microsecond
            print(millis)
            with open(ALTA_LOGS + "seed.txt", "w") as f:
                f.write(str(millis))
            _, best_model = model.learn(steps, 0, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=True, seed=millis)
        
        best_model.save(SAVE_PATH)