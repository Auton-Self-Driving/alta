from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

from ae.controller import AEController
from vae_semantic.controller import VAEController
import ae.util as util
import ae.plot_cm as plot_cm
from environment.carla_9_4.agents.navigation.roaming_agent import RoamingAgent
import csv
import traceback

def get_vae_observation(vae, observation_image):
        ob = vae.encode(observation_image)
        return ob

def get_and_add_vae_observation(vae, observation_image):
        vae.buffer_append(observation_image)
        ob = vae.encode(observation_image)
        return ob

def train_vae_ae(args, prefix, config):
    
    env = CarlaEnv(config.config)

    ALTA_LOGS = args.base_log_dir + prefix
    if not os.path.exists(ALTA_LOGS):
        os.makedirs(ALTA_LOGS)

    IMAGES_PATH = ALTA_LOGS + 'images/'
    VIDEO_PATH = ALTA_LOGS + 'videos/'
    IMAGES_PATH_VAE = ALTA_LOGS + 'images_VAE/'
    VIDEO_PATH_VAE = ALTA_LOGS + 'videos_VAE/'
    CM_PATH = ALTA_LOGS + 'CM_images/'
    FRAME_SKIP = 10
    TOTAL_TIMESTEPS = 200000 * FRAME_SKIP
    VAL_TIMESTEPS = 1000 * FRAME_SKIP
    NUM_CLASSES  = 5
    Accuracy_File = ALTA_LOGS+"acurracy_town1.csv"
    confusion_matrix_file = ALTA_LOGS+"cm_town1.txt"
    VIDEO_FRAME_SKIP = 1
    plot_param_histogram = True
    TRAIN_FREQ = 500 * FRAME_SKIP
    VAL_FREQ = 5000 * FRAME_SKIP

    TF_MODELS = ALTA_LOGS+'tf-models/checkpoint/'
    if not os.path.exists(TF_MODELS):
        os.makedirs(TF_MODELS)

    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, frame_skip=VIDEO_FRAME_SKIP)
    vis_wrapper_vae = vis_module.vis(IMAGES_PATH_VAE, VIDEO_PATH_VAE, frame_skip=VIDEO_FRAME_SKIP)
    
    TB_LOGS_DIR = ALTA_LOGS+ 'tb/'
    logger = tf_log.Logger(ALTA_LOGS)

    if args.algo == "AE":
        model = AEController(z_size=args.vae_zsize, image_size=(160, 80, 5), learning_rate=args.lr, batch_size=args.batch_size)
    elif args.algo == "VAE":
        model = VAEController(z_size=args.vae_zsize, image_size=(160, 80, 5), learning_rate=args.lr, batch_size=args.batch_size, kl_tolerance=0.5)

    obs = env.reset()
    agent = RoamingAgent(env.vehicle_actor)

    num_episodes = 0
    val_accuracy_total = []

    for t in range(TOTAL_TIMESTEPS):

        if (t % FRAME_SKIP == 0):

            semantic_image = obs['semantic_image']
            semantic_image = util.reduce_classes(semantic_image)
            semantic_image_rgb = util.convert_to_rgb(semantic_image, reduced_classes=True).astype(np.uint8)

            semantic_image_onehot = util.convert_to_one_hot(semantic_image, num_classes=5)
            encoded_image = get_and_add_vae_observation(model, semantic_image_onehot)
            vis_wrapper.save_image(semantic_image_rgb, t)

            decoded_image_onehot = model.decode(encoded_image)[0]
            decoded_image = util.convert_from_one_hot(decoded_image_onehot)
            decoded_image = util.convert_to_rgb(decoded_image, reduced_classes=True).astype(np.uint8)
            vis_wrapper_vae.save_image(decoded_image , t)

        # Take one step in env
        control = agent.run_step()
        new_obs, rew, done, eps_measurements = env.step(control)
        
        done = bool(done[0, 0])

        obs = new_obs
        if done:
            num_episodes += 1
            vis_wrapper.generate_video(num_episodes)
            vis_wrapper.remove_images()
            vis_wrapper_vae.generate_video(num_episodes)
            vis_wrapper_vae.remove_images()
            obs = env.reset()
            agent = RoamingAgent(env.vehicle_actor)

        if (t > 1 and t % (TRAIN_FREQ) == 0):
            
            if args.algo == "AE":
                train_loss_avg, accuracy_avg, confusion_matrix_final, train_step, my_accuracy_avg, my_confusion_matrix_final, my_confusion_matrix_normalized, my_confusion_matrix_normalized_final = model.optimize()
            elif args.algo == "VAE":
                train_loss_avg, entropy_loss_avg, kl_loss_avg, accuracy_avg, my_accuracy_avg, confusion_matrix_final, confusion_matrix_final, train_step = model.optimize()

                logger.log_scalar('timesteps/train/entropy_loss', entropy_loss_avg, t)
                logger.log_scalar('timesteps/train/kl_loss', kl_loss_avg, t)
            
            logger.log_scalar('timesteps/train/train_loss', train_loss_avg, t)
            logger.log_scalar('timesteps/train/accuracy_avg', accuracy_avg, t)
            logger.log_scalar('timesteps/train/my_accuracy_avg', my_accuracy_avg, t)
            logger.log_scalar('timesteps/train/global_step', train_step, t)
            # print("loss and accuracy")
            # print(t, train_loss_avg, accuracy_avg, confusion_matrix_final, train_step)
            # print(t, my_accuracy_avg, my_confusion_matrix_final, my_confusion_matrix_normalized_final)
            
            if plot_param_histogram:

                if args.algo == "AE":
                    model_params, model_shapes, model_names = model.ae.get_model_params()
                elif args.algo == "VAE":
                    model_params, model_shapes, model_names = model.vae.get_model_params()

                for (i, model_param) in enumerate(model_params):
                    model_name = model_names[i]
                    model_param_all = (np.ravel(np.array(model_param)))
                    logger.log_histogram('timesteps/train/model_parameters_' + model_name, model_param_all, train_step)
        
        # Saving model
        if(t > 1 and t % VAL_FREQ == 0):
            model.save(TF_MODELS+'model-'+str(t)+'.json')

        # Validation
        if(t % VAL_FREQ == 0):

            # Terminating current episode for validation
            num_episodes += 1
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
                    encoded_image = get_vae_observation(model, semantic_image_onehot)
                    decoded_image_onehot = model.decode(encoded_image)[0]
                    decoded_image = util.convert_from_one_hot(decoded_image_onehot)
                    
                    decoded_image_rgb = util.convert_to_rgb(decoded_image, reduced_classes=True).astype(np.uint8)
                    
                    input_labels_flattened, output_labels_flattened = np.reshape(semantic_image, (-1)), np.reshape(decoded_image, (-1))
                    my_accuracy = np.mean(np.equal(input_labels_flattened, output_labels_flattened))
                    
                    for i in range(np.size(input_labels_flattened)):
                        input_label = input_labels_flattened[i]
                        output_label = output_labels_flattened[i]
                        confusion_matrix[input_label][output_label] += 1
                    
                    val_accuracy_array.append(my_accuracy)
                
                control = agent.run_step()
                new_obs, rew, done, eps_measurements = env.step(control)
                
                done = bool(done[0, 0])

                obs = new_obs
                if done:
                    obs = env.reset()
                    agent = RoamingAgent(env.vehicle_actor)

            val_accuracy_array = np.array(val_accuracy_array)
            val_accuracy_avg = np.mean(val_accuracy_array)
            # print(t, np.size(val_accuracy_array), np.mean(val_accuracy_array))

            eps = 1e-8
            normalization = np.sum(confusion_matrix, axis=1).reshape((-1, 1)) + eps
            confusion_matrix_normalized =  confusion_matrix / normalization
            
            logger.log_scalar('timesteps/train/town1_accuracy_avg', val_accuracy_avg, t)

            plot_cm.save_cm(confusion_matrix_normalized, CM_PATH , t)
            with open(Accuracy_File,'a') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([t, val_accuracy_avg])
            with open(confusion_matrix_file, 'a') as f:
                f.write("\n")
                f.write(str(t))
                f.write("\n")
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
        f.write("\n best_val_accuracy, best_val_accuracy_index \n")
        f.write(str(best_val_accuracy))
        f.write(",")
        f.write(str(best_val_accuracy_index))
    
if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    prefix = 'ae_v125_sem_lr_5e3_nn_16_32_32_32_c5_fs_10_test3/'

    class Args:
        base_log_dir = '/zfsauton2/home/hiteshar/research/alta-logs/test/ae'
        algo = "VAE"
        lr = 5e-4
        vae_zsize = 512
        batch_size = 64
    
    args = Args()
    config = ConfigManager(algo="AE")

    train_vae_ae(args, prefix, config)