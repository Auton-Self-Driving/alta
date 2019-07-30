import sys

import sys, os
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

from stable_baselines.common.vec_env import DummyVecEnv
from models import VAEController, CustomPolicy
# from stable_baselines import logger

from ppo_with_vae import PPOWithVAE
from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import ConfigManager

import vis_module

from stable_baselines.ppo2.ppo2 import PPO2
from stable_baselines.common.policies import register_policy
import numpy as np
from datetime import datetime
import tensorboard_logging as tf_log

vae = VAEController()
# change
prefix = 'debug/'

PATH_MODEL_VAE = prefix + ".json"
ALTA_LOGS = '/zfsauton2/home/tanmaya/projects/alta-logs/ppo_pid_vae_scenarios_straight/' + prefix + '/'
if not os.path.exists(ALTA_LOGS):
    os.makedirs(ALTA_LOGS)

# logger.configure(folder=ALTA_LOGS)

PATH_MODEL_PPO2 = ALTA_LOGS + prefix + 'ppo2_measurements_weights'
TF_MODELS = ALTA_LOGS+prefix+'tf-models/checkpoint/'
if not os.path.exists(TF_MODELS):
    os.makedirs(TF_MODELS)

IMAGES_PATH = ALTA_LOGS+prefix+'images/'
VIDEO_PATH = ALTA_LOGS+prefix+'videos/'
FRAME_SKIP = 4
TB_LOGS_DIR = ALTA_LOGS+prefix+str(datetime.now())

def train():
    # Register the policy, it will check that the name is not already taken
    register_policy('CustomPolicy', CustomPolicy)
    model = PPOWithVAE(policy=CustomPolicy, env=env, n_steps=600, nminibatches=10, verbose=1,
                       tensorboard_log=ALTA_LOGS, full_tensorboard_log=False, learning_rate=1e-6)
    model.learn(1000000, vae=vae, tb_log_name='PPO2')
    return model

if __name__ == '__main__':
    
    # Create the environment
    config = ConfigManager(algo="PPO")
    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
    logger = tf_log.Logger(TB_LOGS_DIR)
    env = CarlaEnv(config.config, vis_wrapper, logger)
    env = DummyVecEnv([lambda: env])
 
    if os.path.exists(PATH_MODEL_PPO2 + ".pkl"):
        print("Task: test")
        vae.load(PATH_MODEL_VAE)
        env.env_method('set_vae', *[vae])
        model = PPOWithVAE.load(PATH_MODEL_PPO2, env)
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        while True:
            actions = model.step(obs)[0]
            obs = env.step(actions)[0]


    else:

        env.env_method('set_vae', *[vae])
        model = train()
        model.save(PATH_MODEL_PPO2)
        vae.save(PATH_MODEL_VAE)
