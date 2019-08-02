from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
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

import vis_module

from gym import wrappers

from datetime import datetime

# from models import CoRLModel, VAEModel
from atari_model import AtariModel

import matplotlib.pyplot as plt

import tensorboard_logging as tf_log

from vae.controller import VAEController
from environment.carla_9_4.agents.navigation.roaming_agent import RoamingAgent

prefix = 'vae_training/'

ALTA_LOGS = '/zfsauton2/home/hiteshar/research/alta-logs/'
# ALTA_LOGS = '/home/hitesh/research/alta-logs/'
if not os.path.exists(ALTA_LOGS):
    os.makedirs(ALTA_LOGS)

TF_MODELS = ALTA_LOGS+prefix+'tf-models/checkpoint/'
if not os.path.exists(TF_MODELS):
    os.makedirs(TF_MODELS)

IMAGES_PATH = ALTA_LOGS+prefix+'images/'
VIDEO_PATH = ALTA_LOGS+prefix+'videos/'
IMAGES_PATH_VAE = ALTA_LOGS+prefix+'images_VAE/'
VIDEO_PATH_VAE = ALTA_LOGS+prefix+'videos_VAE/'
FRAME_SKIP = 4
TOTAL_TIMESTEPS = 1000000
VAE_WEIGHTS_PATH = ALTA_LOGS + 'ppo_vae_right_rgb.json'
VAE_TRAINING_STEPS = 10000

def get_vae_observation(vae, observation_image):
        ob = vae.encode(observation_image)
        return ob

def get_and_add_vae_observation(vae, observation_image):
        vae.buffer_append(observation_image)
        ob = vae.encode(observation_image)
        return ob

if __name__ == '__main__':

    # Create the environment
    config = ConfigManager(algo="VAE_seg")
    env = CarlaEnv(config.config)
    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
    vis_wrapper_vae = vis_module.vis(IMAGES_PATH_VAE, VIDEO_PATH_VAE, FRAME_SKIP)
    # NOTE: not using Monitor for now. integrate later
    # env = wrappers.Monitor(env, '/tmp/deepq'+str(datetime.now()), force=True)
    logger = tf_log.Logger(ALTA_LOGS+prefix+str(datetime.now()))
    print('-'*50)
    print('Launched environment!')
    print('-'*50)

    vae = VAEController()

    obs = env.reset()
    print('-'*50)
    print('Received observation of shape:', obs['image'].shape)
    print('-'*50)

    agent = RoamingAgent(env.vehicle_actor)
    

    num_episodes = 0
    num_done = 0
    for t in range(TOTAL_TIMESTEPS):
        encoded_image = get_and_add_vae_observation(vae, obs['image'])
        vis_wrapper.save_image(obs['image'], t)
        decoded_image = vae.decode(encoded_image)[0].astype(np.uint8)
        vis_wrapper_vae.save_image(decoded_image, t)

        control = agent.run_step()
        new_obs, rew, done, eps_measurements = env.step(control)
        
        
        new_encoded_image = get_vae_observation(vae, new_obs['image'])
        # Store transition in the replay buffer.
        # Read only sensor image part of the observation (sensor_image, [measurements_array])
        rew = float(rew[0, 0])
        done = bool(done[0, 0])

        obs = new_obs
        if done:
            num_episodes += 1
            num_done += 1
            # print('-'*50)
            # print('Timesteps:', t)
            # print('-'*50)
            # logger.log_scalar('episodes/train/dist_to_target', eps_measurements['distance_to_goal'], num_episodes)
            # logger.log_scalar('episodes/train/reward', eps_measurements['total_reward'], num_episodes)
            # logger.log_scalar('timesteps/train/dist_to_target', eps_measurements['distance_to_goal'], t)
            # logger.log_scalar('timesteps/train/reward', eps_measurements['total_reward'], t)
            print('-'*50)
            print('Generating video')
            print('-'*50)
            vis_wrapper.generate_video(num_episodes)
            vis_wrapper.remove_images()
            vis_wrapper_vae.generate_video(num_episodes)
            vis_wrapper_vae.remove_images()
            obs = env.reset()
            agent = RoamingAgent(env.vehicle_actor)
        
        if (t > 1 and t % 500 == 0):
            vae.optimize()
        if(t > 100 and t % 1000 == 0):
            vae.save(TF_MODELS+'vae-model-'+str(t)+'.json') 
