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

import baselines.common.tf_util as U

# NOTE: not using baselines logger for now
# from baselines import logger
from baselines import deepq
from baselines.deepq.deepq import ActWrapper
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule

from gym import wrappers

from datetime import datetime

from models import CoRLModel
from atari_model import AtariModel

import matplotlib.pyplot as plt

import tensorboard_logging as tf_log

if __name__ == '__main__':
    MODEL_SAVE_DIR = '/media/hdd/shubhand/'
    with U.make_session():
        # Create the environment
        config = ConfigManager(algo="DQN")
        env = CarlaEnv(config.config)
        # NOTE: not using Monitor for now. integrate later
        # env = wrappers.Monitor(env, '/tmp/deepq'+str(datetime.now()), force=True)
        logger = tf_log.Logger('./tf-logs/'+str(datetime.now()))
        print('-'*50)
        print('Launched environment!')
        print('-'*50)
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=CoRLModel,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            gamma=0.95,
            double_q=True
        )

        act_params = {
                'make_obs_ph': lambda name: ObservationInput(env.observation_space, name=name),
                'q_func': CoRLModel,
                'num_actions': env.action_space.n
                }

        print('-'*50)
        print('Built model!')
        print('-'*50)
        # Create the replay buffer
        replay_buffer = ReplayBuffer(30000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        obs = env.reset()
        print('-'*50)
        print('Received observation of shape:', obs['image'].shape)
        print('-'*50)
        num_episodes = 0
        num_done = 0
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs['image'], update_eps=exploration.value(t))[0]
            new_obs, rew, done, eps_measurements = env.step(action)
            # Store transition in the replay buffer.
            # Read only sensor image part of the observation (sensor_image, [measurements_array])
            rew = float(rew[0, 0])
            done = bool(done[0, 0])
            replay_buffer.add(obs['image'], action, rew, new_obs['image'], float(done))
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
                obs = env.reset()
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if(num_done % 10 == 0):
                    print('-'*50)
                    print('Launching validation step')
                    print('-'*50)
                    validation_done = None
                    while(validation_done != True):
                        # Take action and update exploration to the newest value
                        action = act(obs['image'], update_eps=exploration.value(0))[0]
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
                            obs = env.reset()
                            validation_done = True
                            is_solved = (eps_measurements['distance_to_goal'] < 2.0)
                            if is_solved:
                                print('-'*50)
                                print('Solved!')
                                print('-'*50)
                                print('-'*50)
                                print('Saving model (completed goal)!')
                                print('-'*50)
                                wrapped_act = ActWrapper(act, act_params)
                                wrapped_act.save(MODEL_SAVE_DIR+'tf-models/trained/corl-carla-model-'+str(t)+'.pkl')
                # Update target network periodically
                if(t > 1000):
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(64)
                    td_error = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                if(t % 1000 == 0 and t > 0):
                    print('-'*50)
                    print('Saving model (checkpoint)!')
                    print('-'*50)
                    wrapped_act = ActWrapper(act, act_params)
                    wrapped_act.save(MODEL_SAVE_DIR+'tf-models/checkpoint/corl-carla-model-'+str(t)+'.pkl')
                    update_target()
