from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, glob
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
from my_sac import MY_SAC, plot_policy_and_value_fns
from sac_models import My_MlpPolicy_1layer, My_MlpPolicy_2layer, My_MlpPolicy_3layer
import traceback

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
            env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger, log_dir = args.base_log_dir, base_prefix = base_prefix, prefix = prefix)
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
        if not args.train_vae:
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
            elif args.network == "3_layer":
                policy = My_MlpPolicy_3layer
            else:
                print("specify either 1_layer or 2_layer or 3_layer as network input")
                env.close()
                print("exiting")
                return
            bs=config.config["batch_size"]
            model = MY_SAC(config=config.config, policy=policy, env=dummy_env, learning_rate=args.lr,buffer_size=args.buffer_size,batch_size=bs,learning_starts=5000,
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
    
