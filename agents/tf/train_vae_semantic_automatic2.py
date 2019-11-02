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
# import tensorflow.contrib.layers as layers
import time

import baselines.common.tf_util as U

# NOTE: not using baselines logger for now
# from baselines import logger
# from baselines import deepq
# from baselines.deepq.deepq import ActWrapper
# from baselines.deepq.replay_buffer import ReplayBuffer
# from baselines.deepq.utils import ObservationInput
# from baselines.common.schedules import LinearSchedule

import vis_module

from gym import wrappers

from datetime import datetime

# from models import CoRLModel, VAEModel
# from atari_model import AtariModel

import matplotlib.pyplot as plt

import tensorboard_logging as tf_log

from ae.controller import AEController
import ae.util as util
import ae.plot_cm as plot_cm
from environment.carla_9_4.agents.navigation.roaming_agent import RoamingAgent
import csv
import traceback

prefix = 'ae_v125_sem_lr_5e3_nn_16_32_32_32_c5_fs_10_test/'

ALTA_LOGS = '/zfsauton2/home/hiteshar/research/alta-logs/test/ae'
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
FRAME_SKIP = 10
TOTAL_TIMESTEPS = 2000000
VAL_TIMESTEPS = 100000
VAE_WEIGHTS_PATH = ALTA_LOGS + 'ppo_vae_right_rgb.json'
NUM_CLASSES  = 5
Accuracy_File = ALTA_LOGS+prefix+"acurracy_town1.csv"
confusion_matrix_file = ALTA_LOGS+prefix+"cm_town1.txt"


def get_vae_observation(vae, observation_image):
        ob = vae.encode(observation_image)
        return ob

def get_and_add_vae_observation(vae, observation_image):
        vae.buffer_append(observation_image)
        ob = vae.encode(observation_image)
        return ob

if __name__ == '__main__':

    # Create the environment
    config = ConfigManager(algo="AE")
    env = CarlaEnv(config.config)
    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, frame_skip=1)
    vis_wrapper_vae = vis_module.vis(IMAGES_PATH_VAE, VIDEO_PATH_VAE, frame_skip    =1)
    # NOTE: not using Monitor for now. integrate later
    # env = wrappers.Monitor(env, '/tmp/deepq'+str(datetime.now()), force=True)
    logger = tf_log.Logger(ALTA_LOGS+prefix+str(datetime.now()))
    print('-'*50)
    print('Launched environment!')
    print('-'*50)

    vae = AEController(image_size=(160, 80, 5), learning_rate=5e-3, batch_size=64)

    obs = env.reset()
    print('-'*50)
    print('Received observation of shape:', obs['image'].shape)
    print('-'*50)

    agent = RoamingAgent(env.vehicle_actor)
    

    num_episodes = 0
    num_done = 0
    val_accuracy_total = []
    for t in range(TOTAL_TIMESTEPS):

        if (t % FRAME_SKIP == 0):

            semantic_image = obs['semantic_image']
            semantic_image = util.reduce_classes(semantic_image)
            # semantic_image_rgb = util.convert_to_rgb(np.expand_dims(semantic_image, axis=-1)).astype(np.uint8)
            semantic_image_rgb = util.convert_to_rgb(semantic_image, reduced_classes=True).astype(np.uint8)

            print("type of obs[image]", type(obs['image']))
            print("type of semantic_image_rgb", type(semantic_image_rgb))

            semantic_image_onehot = util.convert_to_one_hot(semantic_image, num_classes=5)
            encoded_image = get_and_add_vae_observation(vae, semantic_image_onehot)
            vis_wrapper.save_image(semantic_image_rgb, t)
            decoded_image_onehot = vae.decode(encoded_image)[0]
            decoded_image = util.convert_from_one_hot(decoded_image_onehot)
            
            decoded_image = util.convert_to_rgb(decoded_image, reduced_classes=True).astype(np.uint8)
            # decoded_image = np.tile(decoded_image, (1,1,3))
            vis_wrapper_vae.save_image(decoded_image , t)

        control = agent.run_step()
        try:
            new_obs, rew, done, eps_measurements = env.step(control)
        except Exception as identifier:
            print(identifier)
            traceback.print_exc()
        
        
        # new_encoded_image = get_vae_observation(vae, new_obs['image'])
        # Store transition in the replay buffer.
        # Read only sensor image part of the observation (sensor_image, [measurements_array])
        # rew = float(rew[0, 0])
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

        if (t > 1 and t % (500 * FRAME_SKIP) == 0):
            # train_loss_avg, accuracy_avg, confusion_matrix_final, train_step = vae.optimize()
            train_loss_avg, accuracy_avg, confusion_matrix_final, train_step, my_accuracy_avg, my_confusion_matrix_final, my_confusion_matrix_normalized, my_confusion_matrix_normalized_final = vae.optimize()
            logger.log_scalar('timesteps/vae_train/train_loss', train_loss_avg, t)
            logger.log_scalar('timesteps/vae_train/accuracy_avg', accuracy_avg, t)
            # logger.log_scalar('timesteps/vae_train/global_step', train_step, t)
            # logger.log_scalar('timesteps/vae_train/train_loss_global_step', train_loss_avg, train_step)
            logger.log_scalar('timesteps/vae_train/my_accuracy_avg', my_accuracy_avg, t)
            print("loss and accuracy")
            print(t, train_loss_avg, accuracy_avg, confusion_matrix_final, train_step)
            print(t, my_accuracy_avg, my_confusion_matrix_final, my_confusion_matrix_normalized_final)
            model_params, model_shapes, model_names = vae.ae.get_model_params()
            for (i, model_param) in enumerate(model_params):
                model_name = model_names[i]
                model_param_all = (np.ravel(np.array(model_param)))
                logger.log_histogram('timesteps/ae_train/model_parameters_' + model_name, model_param_all, train_step)
        if(t > 100 and t % (1000* FRAME_SKIP) == 0):
            vae.save(TF_MODELS+'vae-model-'+str(t)+'.json')

        if(t > 10 and t % (1000 * FRAME_SKIP) == 0):
                
            # Validation
            num_episodes += 1
            print('-'*50)
            print('Generating video')
            print('-'*50)
            vis_wrapper.generate_video(num_episodes)
            vis_wrapper.remove_images()
            vis_wrapper_vae.generate_video(num_episodes)
            vis_wrapper_vae.remove_images()

            obs = env.reset()
            agent = RoamingAgent(env.vehicle_actor)
            
            confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
            val_accuracy_array = []
            for valT in range(VAL_TIMESTEPS):

                if (valT % FRAME_SKIP == 0):
                    semantic_image = obs['semantic_image']
                    semantic_image = util.reduce_classes(semantic_image)
                    semantic_image_rgb = util.convert_to_rgb(semantic_image, reduced_classes=True).astype(np.uint8)

                    semantic_image_onehot = util.convert_to_one_hot(semantic_image, num_classes=5)
                    encoded_image = get_vae_observation(vae, semantic_image_onehot)
                    decoded_image_onehot = vae.decode(encoded_image)[0]
                    decoded_image = util.convert_from_one_hot(decoded_image_onehot)
                    
                    decoded_image_rgb = util.convert_to_rgb(decoded_image, reduced_classes=True).astype(np.uint8)
                    
                    my_decoded_image = np.expand_dims(np.argmax(decoded_image_onehot, axis=-1), axis=-1)
                    decoded_image_onehot_bad = decoded_image_onehot.astype(np.uint8)
                    input_labels_flattened, output_labels_flattened = np.reshape(semantic_image, (-1)), np.reshape(decoded_image, (-1))
                    my_accuracy = np.mean(np.equal(input_labels_flattened, output_labels_flattened))

                    
                    for i in range(np.size(input_labels_flattened)):
                        input_label = input_labels_flattened[i]
                        output_label = output_labels_flattened[i]
                        confusion_matrix[input_label][output_label] += 1
                    
                    val_accuracy_array.append(my_accuracy)
                
                control = agent.run_step()
                new_obs, rew, done, eps_measurements = env.step(control)
                
                
                # rew = float(rew[0, 0])
                done = bool(done[0, 0])

                obs = new_obs
                if done:
                    obs = env.reset()
                    agent = RoamingAgent(env.vehicle_actor)

            val_accuracy_array = np.array(val_accuracy_array)
            val_accuracy_avg = np.mean(val_accuracy_array)
            print(t, np.size(val_accuracy_array), np.mean(val_accuracy_array))

            confusion_matrix_normalized =  confusion_matrix / np.sum(confusion_matrix, axis=1).reshape((-1, 1))
            
            
            logger.log_scalar('timesteps/vae_train/town1_accuracy_avg', val_accuracy_avg, t)

            # class_names = [i[0] for i in util.REDUCED_SEMANTIC_COLOR_MAP]
            # class_names = [util.REDUCED_SEMANTIC_COLOR_MAP[i][0] for i in range(5)]
            # cm_image_png, cm_image = plot_cm.get_cm_image(confusion_matrix)
            # logger.log_images("CM", [cm_image_png],t)
            with open(Accuracy_File,'a') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([t, val_accuracy_avg])
            with open(confusion_matrix_file, 'a') as f:
                f.write(str(t))
                f.write(str(confusion_matrix))
                f.write("\n normalized\n")
                f.write(str(confusion_matrix_normalized))

            val_accuracy_total.append(val_accuracy_avg)
            obs = env.reset()
            agent = RoamingAgent(env.vehicle_actor)

    val_accuracy_total = np.array(val_accuracy_total)
    best_val_accuracy = np.max(val_accuracy_total)
    best_val_accuracy_index = np.argmax(val_accuracy_total)
    
    print("best_val_accuracy, best_val_accuracy_index")
    print(best_val_accuracy, best_val_accuracy_index)
    with open(confusion_matrix_file, 'a') as f:
        f.write("best_val_accuracy, best_val_accuracy_index")
        f.write(str(best_val_accuracy))
        f.write(str(best_val_accuracy_index))
        