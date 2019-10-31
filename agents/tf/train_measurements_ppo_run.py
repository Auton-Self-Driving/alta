from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, glob
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))

from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import ConfigManager

import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
import traceback

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
from stable_baselines.ppo2.ppo2 import Runner
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.policies import register_policy
from ppo import PPO, plot_policy_and_value_fns, test
from models import CustomPolicy, CustomWPPolicy, Policy_1_layer, Policy_2_layer

def run_ppo(args, prefix, config):
    ALTA_LOGS = os.path.join(args.base_log_dir, prefix)
    # config.config['LOG_DIR'] = ALTA_LOGS
    
    def get_latest_model(log_dir=ALTA_LOGS, ext='*.pkl', sep='_'):
        list_of_files = glob.glob(log_dir + ext)
        latest_file = max(list_of_files, key=os.path.getctime)
        latest_file = latest_file.split('{}'.format(ext[1:]))[0]
        ind = int(latest_file.split(sep)[1])
        return ind, latest_file
    
    # prefix = 'ppo_entcoeff_01_logstd_23_w_tanh_reward_10_nav_5_1/'

    POLICY_PLOTS = ALTA_LOGS + 'policy_plots/'
    if not os.path.exists(ALTA_LOGS):
        os.makedirs(ALTA_LOGS)

    TF_MODELS = ALTA_LOGS+'tf-models/checkpoint/'
    if not os.path.exists(TF_MODELS):
        os.makedirs(TF_MODELS)

    FRAME_SKIP = 1
    SAVE_PATH = ALTA_LOGS + 'ppo2_measurements_weights'
    TB_LOGS_DIR = ALTA_LOGS+ 'tb/'

    MAX_TRIALS = 5
    
    register_policy('CustomWPPolicy', CustomWPPolicy)
    steps = args.timesteps
    
    for i in range(MAX_TRIALS):
        try:
            # Create the environment
            logger = tf_log.Logger(TB_LOGS_DIR)
            if os.path.exists(SAVE_PATH + ".pkl"):
                print("Best model exists, Validating !!!!")
                with open(ALTA_LOGS + "seed.txt", "r") as f:
                    seed = int(f.readline())
                print("Using the pre-initialized seed: {}".format(seed))
                set_global_seeds(seed)

                IMAGES_PATH = ALTA_LOGS+'final_images_' + config.config["city_name"] + '/'
                VIDEO_PATH = ALTA_LOGS+'final_videos_' + config.config["city_name"] + '/'
                vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
                
                env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger)
                dummy_env = DummyVecEnv([lambda: env])
                
                # with open(ALTA_LOGS + "best_model.txt", "r") as f:
                #     lines = [line for line in f.readlines()]
                #     ind = int(lines[1].split("index: ")[1])
                #     print("Index is: {}".format(ind))
                    
                #     path = SAVE_PATH + str(4 * (ind + 1)) + "0000"
                #     print(path)
                #     best_model = PPO.load(SAVE_PATH, DummyVecEnv([lambda: env]))
                #     best_model.save(SAVE_PATH)
                
                model = PPO.load(SAVE_PATH, dummy_env)
                success_episodes = 0
                results = {}
                with open(ALTA_LOGS + config.config["scenarios"] + config.config["city_name"] + ".txt", "w") as f:
                    total_reward, success_episodes, results = test(model, env)
                    # for ind in range(25):
                    #     obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
                    #     obs[:] = env.reset(unseen=True, index=ind)
                    #     done = False
                    #     while not done:
                    #         actions = model.step(obs, deterministic=True)[0]
                    #         print(actions.shape)
                    #         info = env.step(actions)
                    #         done = info[2]
                    #         obs = np.expand_dims(info[0], axis=0)
                        
                    #     print("Termination State: {}".format(info[3]['termination_state']))
                    #     if info[3]['termination_state'] == 'success':
                    #         success_episodes += 1
                    #         results[ind] = 1
                    #     else:
                    #         results[ind] = 0
                    print("Task Name: {}".format(config.config["scenarios"]))
                    print("Town Name: {}".format(config.config["city_name"]))
                    print("Results of test scenarios")
                    print(results)
                    print("Total Success Episodes: {}".format(success_episodes))
                    f.write("Task Name: {}".format(config.config["scenarios"]))
                    f.write("Town Name: {}".format(config.config["city_name"]))
                    f.write("Results of test scenarios")
                    # f.write(results)
                    f.write("Total Success Episodes: {}".format(str(success_episodes)))
            # elif os.path.exists(SAVE_PATH + str(steps) + ".pkl"):
            #     print("All models exist. Finding best model !!")
            #     IMAGES_PATH = ALTA_LOGS+'best_model_images/'
            #     VIDEO_PATH = ALTA_LOGS+'best_model_videos/'
            #     vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
            #     env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger)
            #     best_model = get_best_model(steps, SAVE_PATH, env)
            #     best_model.save(SAVE_PATH)
            else:
                print("Training begins")
                IMAGES_PATH = ALTA_LOGS+'images/'
                VIDEO_PATH = ALTA_LOGS+'videos/'
                vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
                
                env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger)
                dummy_env = DummyVecEnv([lambda: env])
                
                if args.layers == "1_layer":
                    policy = Policy_1_layer
                elif args.layers == "2_layer":
                    policy = Policy_2_layer
                else:
                    print("specify either 1_layer or 2_layer as layers input")
                    env.close()
                    print("exiting")
                    return
                
                model = PPO(policy=policy, env=dummy_env, n_steps=500, nminibatches=4, verbose=1, learning_rate=args.lr, 
                        tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=False, ent_coef=args.ent_coef)
                if any(fname.endswith('.pkl') for fname in os.listdir(ALTA_LOGS)):
                    with open(ALTA_LOGS + "seed.txt", "r") as f:
                        seed = int(f.readline())
                    print("Using the pre-initialized seed: {}".format(seed))
                    set_global_seeds(seed)
                    completed_steps, latest_model = get_latest_model(log_dir=ALTA_LOGS, ext='*.pkl', sep='hts')
                    env.total_steps = completed_steps
                    if config.config["videos"]:
                        completed_episodes, _ = get_latest_model(log_dir=ALTA_LOGS + 'videos/', ext='*.mp4', sep='log_')
                        env.episode_num = completed_episodes
                    print("Loading Latest model!!!")
                    model = PPO.load(latest_model, dummy_env)
                    print("Model: {} loaded successfully".format(latest_model))
                    best_model = model.learn(steps, completed_steps, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=False, seed=seed)    
                else:
                    dt = datetime.now()
                    millis = dt.microsecond
                    print(millis)
                    with open(ALTA_LOGS + "seed.txt", "w") as f:
                        f.write(str(millis))
                    # random_model = model.learn(0, 0, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=True, seed=millis)
                    # random_model.save(SAVE_PATH + "0")
                    # plot_policy_and_value_fns(random_model, 0, POLICY_PLOTS)
                    best_model = model.learn(steps, 0, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=True, seed=millis)
                
                best_model.save(SAVE_PATH)
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