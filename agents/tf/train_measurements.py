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

from models import CoRLModel, MeasurementsModel
from atari_model import AtariModel

import matplotlib.pyplot as plt

import tensorboard_logging as tf_log

prefix = 'dqn_measure_wp_lr_5e6_g_95_straight_corl10000_run1/'

ALTA_LOGS = '/media/hdd/hiteshar/alta-logs/'
if not os.path.exists(ALTA_LOGS):
    os.makedirs(ALTA_LOGS)

TF_MODELS = ALTA_LOGS+prefix+'tf-models/checkpoint/'
if not os.path.exists(TF_MODELS):
    os.makedirs(TF_MODELS)

IMAGES_PATH = ALTA_LOGS+prefix+'images/'
VIDEO_PATH = ALTA_LOGS+prefix+'videos/'
FRAME_SKIP = 4

if __name__ == '__main__':
    with U.make_session():
        # Create the environment
        config = ConfigManager(algo="DQN")
        env = CarlaEnv(config.config)
        vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
        # NOTE: not using Monitor for now. integrate later
        # env = wrappers.Monitor(env, '/tmp/deepq'+str(datetime.now()), force=True)
        logger = tf_log.Logger(ALTA_LOGS+prefix+str(datetime.now()))
        print('-'*50)
        print('Launched environment!')
        print('-'*50)
        observation_space = Box(low=-4.0, high=4.0, shape=(1,), dtype=np.float32)

        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(observation_space, name=name),
            q_func=CoRLModel,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-6),
            gamma=0.95,
            double_q=True
        )

        act_params = {
                'make_obs_ph': lambda name: ObservationInput(observation_space, name=name),
                'q_func': CoRLModel,
                'num_actions': env.action_space.n
                }

        print('-'*50)
        print('Built model!')
        print('-'*50)
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=5000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        obs = env.reset()
        print('-'*50)
        print('Received observation of shape:', obs['orientation'].shape)
        print('-'*50)
        num_episodes = 0
        num_done = 0
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs['orientation'], update_eps=exploration.value(t))[0]
            new_obs, rew, done, eps_measurements = env.step(action)
            vis_wrapper.save_image(obs['image'], t)
            # Store transition in the replay buffer.
            # Read only sensor image part of the observation (sensor_image, [measurements_array])
            rew = float(rew[0, 0])
            done = bool(done[0, 0])
            replay_buffer.add(obs['orientation'], action, rew, new_obs['orientation'], float(done))
            obs = new_obs
            if done:
                num_episodes += 1
                num_done += 1
                print('-'*50)
                print('Timesteps:', t)
                print('-'*50)
                logger.log_scalar('episodes/train/dist_to_target', eps_measurements['distance_to_goal'], num_episodes)
                logger.log_scalar('episodes/train/reward', eps_measurements['total_reward'], num_episodes)
                logger.log_scalar('timesteps/train/dist_to_target', eps_measurements['distance_to_goal'], t)
                logger.log_scalar('timesteps/train/reward', eps_measurements['total_reward'], t)
                print('-'*50)
                print('Generating video')
                print('-'*50)
                vis_wrapper.generate_video(num_episodes)
                vis_wrapper.remove_images()
                obs = env.reset()
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if(num_done % 10 == 0):
                    print('-'*50)
                    print('Launching validation step on seen')
                    print('-'*50)
                    validation_done = None
                    while(validation_done != True):
                        # Take action and update exploration to the newest value
                        action = act(obs['orientation'], update_eps=0)[0]
                        new_obs, rew, done, eps_measurements = env.step(action)
                        # Store transition in the replay buffer.
                        # Read only sensor image part of the observation (sensor_image, [measurements_array])
                        rew = float(rew[0, 0])
                        done = bool(done[0, 0])
                        obs = new_obs
                        # plt.imsave('img'+str(t).zfill(4)+'.png', obs)
                        if done:
                            logger.log_scalar('episodes/val/dist_to_target', eps_measurements['distance_to_goal'], num_episodes)
                            logger.log_scalar('episodes/val/reward', eps_measurements['total_reward'], num_episodes)
                            logger.log_scalar('timesteps/val/dist_to_target', eps_measurements['distance_to_goal'], t)
                            logger.log_scalar('timesteps/val/reward', eps_measurements['total_reward'], t)
                            validation_done = True

                    print('-'*50)
                    print('Launching validation step on unseen')
                    print('-'*50)
                    obs = env.reset(unseen=True)
                    validation_done = None
                    while(validation_done != True):
                        # Take action and update exploration to the newest value
                        action = act(obs['orientation'], update_eps=0)[0]
                        new_obs, rew, done, eps_measurements = env.step(action)
                        # Store transition in the replay buffer.
                        # Read only sensor image part of the observation (sensor_image, [measurements_array])
                        rew = float(rew[0, 0])
                        done = bool(done[0, 0])
                        obs = new_obs
                        # plt.imsave('img'+str(t).zfill(4)+'.png', obs)
                        if done:
                            logger.log_scalar('episodes/val_unseen/dist_to_target', eps_measurements['distance_to_goal'], num_episodes)
                            logger.log_scalar('episodes/val_unseen/reward', eps_measurements['total_reward'], num_episodes)
                            logger.log_scalar('timesteps/val_unseen/dist_to_target', eps_measurements['distance_to_goal'], t)
                            logger.log_scalar('timesteps/val_unseen/reward', eps_measurements['total_reward'], t)
                            obs = env.reset()
                            validation_done = True

                # Update target network periodically
                if(t > 1000):
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(64)
                    td_error = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                if(num_done % 100 == 0 and t > 1000):
                    print('-'*50)
                    print('Saving model (checkpoint)!')
                    print('-'*50)
                    wrapped_act = ActWrapper(act, act_params)
                    wrapped_act.save(TF_MODELS+'corl-carla-model-'+str(t)+'.pkl')
            if(t % 1000 == 0):
                update_target()