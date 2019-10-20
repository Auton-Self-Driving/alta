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

from ae.controller import VAEController
import ae.util as util
from environment.carla_9_4.agents.navigation.roaming_agent import RoamingAgent

prefix = 'ae_v125_z512_kl_0_crop_z20_sem_7/'

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
TOTAL_TIMESTEPS = 200000
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

    vae = VAEController(image_size=(160, 80, 13))

    obs = env.reset()
    print('-'*50)
    print('Received observation of shape:', obs['image'].shape)
    print('-'*50)

    agent = RoamingAgent(env.vehicle_actor)
    

    num_episodes = 0
    num_done = 0
    for t in range(TOTAL_TIMESTEPS):

        semantic_image = obs['semantic_image']
        # print("semantic_image", semantic_image, np.max(semantic_image), np.min(semantic_image))
        semantic_image_rgb = util.convert_to_rgb(np.expand_dims(semantic_image, axis=-1)).astype(np.uint8)

        print("type of obs[image]", type(obs['image']))
        print("type of semantic_image_rgb", type(semantic_image_rgb))

        # print("semantic_image_rgb", semantic_image_rgb, np.max(semantic_image_rgb), np.min(semantic_image_rgb))
        # print("semantic_image", np.shape(semantic_image))
        semantic_image_onehot = util.convert_to_one_hot(semantic_image)
        # print("semantic_image_onehot", np.shape(semantic_image_onehot))
        # print("semantic_image_onehot", semantic_image_onehot[0,0,:])
        encoded_image = get_and_add_vae_observation(vae, semantic_image_onehot)
        # print("encoded_image", np.shape(encoded_image))
        vis_wrapper.save_image(semantic_image_rgb, t)
        decoded_image_onehot = vae.decode(encoded_image)[0].astype(np.uint8)
        # print("decoded_image_onehot", np.shape(decoded_image_onehot))
        decoded_image = util.convert_from_one_hot(decoded_image_onehot)
        # print("decoded_image", np.shape(decoded_image))

        decoded_image = util.convert_to_rgb(decoded_image).astype(np.uint8)
        # decoded_image = np.tile(decoded_image, (1,1,3))
        vis_wrapper_vae.save_image(decoded_image, t)

        control = agent.run_step()
        new_obs, rew, done, eps_measurements = env.step(control)
        
        
        # new_encoded_image = get_vae_observation(vae, new_obs['image'])
        # Store transition in the replay buffer.
        # Read only sensor image part of the observation (sensor_image, [measurements_array])
        rew = float(rew[0, 0])
        done = bool(done[0, 0])

        obs = new_obs
        if done:
            num_episodes += 1
            num_done += 1
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
            train_loss_avg, accuracy_avg, confusion_matrix_final, train_step = vae.optimize()
            logger.log_scalar('timesteps/vae_train/train_loss', train_loss_avg, t)
            logger.log_scalar('timesteps/vae_train/accuracy_avg', accuracy_avg, t)
            logger.log_scalar('timesteps/vae_train/global_step', train_step, t)
            logger.log_scalar('timesteps/vae_train/train_loss_global_step', train_loss_avg, train_step)
            print("loss and accuracy")
            print(t, train_loss_avg, accuracy_avg, confusion_matrix_final, train_step)
            model_params, model_shapes, model_names = vae.ae.get_model_params()
            for (i, model_param) in enumerate(model_params):
                model_name = model_names[i]
                model_param_all = (np.ravel(np.array(model_param)))
                logger.log_histogram('timesteps/ae_train/model_parameters_' + model_name, model_param_all, train_step)
        if(t > 100 and t % 5000 == 0):
            vae.save(TF_MODELS+'vae-model-'+str(t)+'.json') 
