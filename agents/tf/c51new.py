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
import math

# import tensorflow.contrib.layers as layers
import time

# import baselines.common.tf_util as U

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
import random


import matplotlib.pyplot as plt

import tensorboard_logging as tf_log
from collections import deque




class C51_Agent:
    def __init__(self, 
        sess, 
        model, 
        learning_rate, 
        state_dim, 
        action_dim, 
        n_atoms, 
        v_min, 
        v_max, 
        batch_size = 32, 
        tau = 0.01, 
        k = 1, 
        gamma=0.99, 
        render=False):

        self.learning_rate = learning_rate
        self.state_size = state_dim
        self.action_size = action_dim
        self.model = model
        self.sess = sess
        self.batch_size = batch_size
        self.gamma = gamma
        self.quantile_embedding_dim = 128

        self.num_support = 51
        self.V_max = 100
        self.V_min = -5
        self.dz = float(self.V_max - self.V_min) / (self.num_support - 1)
        self.z = [self.V_min + i * self.dz for i in range(self.num_support)]
        
        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.dqn_Y = tf.placeholder(tf.float32, [None, 1])
        self.Y = tf.placeholder(tf.float32, [None, self.num_support])
        self.M = tf.placeholder(tf.float32, [None, self.num_support])
        self.tau = tf.placeholder(tf.float32, [None, self.num_support])

        self.main_network, self.main_action_support, self.main_params = self._build_network('main')
        self.target_network, self.target_action_support, self.target_params = self._build_network('target')

        if self.model == 'IQN':
            expand_dim_action = tf.expand_dims(self.action, -1)
            main_support = tf.reduce_sum(self.main_network * expand_dim_action, axis=1)

            theta_loss_tile = tf.tile(tf.expand_dims(main_support, axis=2), [1, 1, self.num_support])
            logit_valid_tile = tf.tile(tf.expand_dims(self.Y, axis=1), [1, self.num_support, 1])
            Huber_loss = tf.losses.huber_loss(logit_valid_tile, theta_loss_tile, reduction=tf.losses.Reduction.NONE)
            tau = self.tau
            inv_tau = 1 - tau
            tau = tf.tile(tf.expand_dims(tau, axis=1), [1, self.num_support, 1])
            inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, self.num_support, 1])
            error_loss = logit_valid_tile - theta_loss_tile

            Loss = tf.where(tf.less(error_loss, 0.0), inv_tau * Huber_loss, tau * Huber_loss)
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(Loss, axis=2), axis=1))

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        elif self.model == 'DQN':
            self.Q_s_a = self.main_network * self.action
            self.Q_s_a = tf.expand_dims(tf.reduce_sum(self.Q_s_a, axis=1), -1)
            self.loss = tf.losses.mean_squared_error(self.dqn_Y, self.Q_s_a)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        elif self.model == 'C51':
            self.z_space = tf.tile(tf.reshape(self.z, [1, 1, self.num_support]), [self.batch_size, self.action_size, 1])
            self.z_space_with_target_action_support = self.target_action_support * self.z_space
            expand_dim_action = tf.expand_dims(self.action, -1)
            self.Q_s_a = self.main_network * expand_dim_action
            self.Q_s_a = tf.reduce_sum(self.Q_s_a, axis=1)
            self.loss = - tf.reduce_mean(tf.reduce_sum(tf.multiply(self.M, tf.log(self.Q_s_a + 1e-20)), axis=1))
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        elif self.model == 'QRDQN':
            self.theta_s_a = self.main_network
            expand_dim_action = tf.expand_dims(self.action, -1)
            theta_s_a = tf.reduce_sum(self.main_network * expand_dim_action, axis=1)

            theta_loss_tile = tf.tile(tf.expand_dims(theta_s_a, axis=2), [1, 1, self.num_support])
            logit_valid_tile = tf.tile(tf.expand_dims(self.Y, axis=1), [1, self.num_support, 1])

            Huber_loss = tf.losses.huber_loss(logit_valid_tile, theta_loss_tile, reduction=tf.losses.Reduction.NONE)
            tau = tf.reshape(tf.range(1e-10, 1, 1 / self.num_support), [1, self.num_support])
            inv_tau = 1.0 - tau

            tau = tf.tile(tf.expand_dims(tau, axis=1), [1, self.num_support, 1])
            inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, self.num_support, 1])

            error_loss = logit_valid_tile - theta_loss_tile
            Loss = tf.where(tf.less(error_loss, 0.0), inv_tau * Huber_loss, tau * Huber_loss)
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(Loss, axis=2), axis=1))

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.assign_ops = []
        for v_old, v in zip(self.target_params, self.main_params):
            self.assign_ops.append(tf.assign(v_old, v))

    def train(self, minibatch):
        # minibatch = random.sample(memory, self.batch_size)
        state_stack = [mini[0] for mini in minibatch]
        next_state_stack = [mini[1] for mini in minibatch]
        action_stack = [mini[2] for mini in minibatch]
        reward_stack = [mini[3] for mini in minibatch]
        done_stack = [mini[4] for mini in minibatch]
        done_stack = [int(i) for i in done_stack]

        # state_stack = sample[0] #np.asarray([x[0] for x in sample])[:, 0, :]
        # action_stack = sample[1]#np.asarray([x[1] for x in sample])
        # reward_stack = sample[2]#np.asarray([x[3] for x in sample])
        # next_state_stack = sample[3]# np.asarray([x[2] for x in sample])[:, 0, :]
        
        # done_stack = sample[4]#np.asarray([x[4] for x in sample])


        if self.model == 'IQN':
            t = np.random.rand(self.batch_size, self.num_support)
            Q_next_state = self.sess.run(self.target_network, feed_dict={self.state: next_state_stack, self.tau: t})
            next_action = np.argmax(np.mean(Q_next_state, axis=2), axis=1)
            Q_next_state_next_action = [Q_next_state[i, action, :] for i, action in enumerate(next_action)]
            T_theta = [reward + (1-done)*self.gamma*Q for reward, Q, done in zip(reward_stack, Q_next_state_next_action, done_stack)]
            return self.sess.run([self.train_op, self.loss], feed_dict={self.state: state_stack, self.action: action_stack, self.tau:t, self.Y: T_theta})


        elif self.model == 'DQN':
            Q_next_state = self.sess.run(self.target_network, feed_dict={self.state: next_state_stack})
            next_action = np.argmax(Q_next_state, axis=1)

            Q_next_state_next_action = [s[a] for s, a in zip(Q_next_state, next_action)]
            T_theta = [[reward + (1-done)*self.gamma * Q] for reward, Q, done in zip(reward_stack, Q_next_state_next_action, done_stack)]
            return self.sess.run([self.train_op, self.loss],
                                 feed_dict={self.state: state_stack, self.action: action_stack, self.dqn_Y: T_theta})

        elif self.model == 'C51':
            Q_next_state = self.sess.run(self.z_space_with_target_action_support, feed_dict={self.state: next_state_stack})
            next_action = np.argmax(np.sum(Q_next_state, axis=2), axis=1)
            prob_next_state = self.sess.run(self.target_network, feed_dict={self.state: next_state_stack})
            # import ipdb; ipdb.set_trace()
            prob_next_state_action = [prob_next_state[i, action, :] for i, action in enumerate(next_action)]

            m_prob = np.zeros([self.batch_size, self.num_support])

            for i in range(self.batch_size):
                for j in range(self.num_support):
                    Tz = np.fmin(self.V_max, np.fmax(self.V_min, reward_stack[i] + (1 - done_stack[i]) * 0.99 * (self.V_min + j * self.dz)))
                    bj = (Tz - self.V_min) / self.dz

                    lj = np.floor(bj).astype(int)
                    uj = np.ceil(bj).astype(int)

                    blj = bj - lj
                    buj = uj - bj

                    m_prob[i, lj] += (done_stack[i] + (1 - done_stack[i]) * (prob_next_state_action[i][j])) * buj
                    m_prob[i, uj] += (done_stack[i] + (1 - done_stack[i]) * (prob_next_state_action[i][j])) * blj

            m_prob = m_prob / m_prob.sum(axis=1, keepdims=1)

            return self.sess.run([self.train_op, self.loss],
                                 feed_dict={self.state: state_stack, self.action: action_stack, self.M: m_prob})

        elif self.model == 'QRDQN':
            Q_next_state = self.sess.run(self.target_network, feed_dict={self.state: next_state_stack})
            next_action = np.argmax(np.mean(Q_next_state, axis=2), axis=1)
            Q_next_state_next_action = [Q_next_state[i, action, :] for i, action in enumerate(next_action)]
            Q_next_state_next_action = np.sort(Q_next_state_next_action)
            T_theta = [np.ones(self.num_support) * reward if done else reward + self.gamma * Q for reward, Q, done in
                       zip(reward_stack, Q_next_state_next_action, done_stack)]
            return self.sess.run([self.train_op, self.loss],
                                 feed_dict={self.state: state_stack, self.action: action_stack, self.Y: T_theta})



    def _build_network(self, name):
        with tf.variable_scope(name):
            if self.model == 'DQN':
                layer_1 = tf.layers.dense(inputs=self.state, units=64, activation=tf.nn.relu, trainable=True)
                layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.nn.relu, trainable=True)
                layer_3 = tf.layers.dense(inputs=layer_2, units=64, activation=tf.nn.relu,
                                          trainable=True)
                net = tf.layers.dense(inputs=layer_3, units=self.action_size, activation=None)
                net_action = net

            elif self.model == 'C51':
                layer_1 = tf.layers.dense(inputs=self.state, units=64, activation=tf.nn.relu, trainable=True)
                layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.nn.relu, trainable=True)
                layer_3 = tf.layers.dense(inputs=layer_2, units=self.action_size * self.num_support, activation=None,
                                          trainable=True)

                net_pre = tf.reshape(layer_3, [-1, self.action_size, self.num_support])
                net = tf.nn.softmax(net_pre, axis=2)
                net_action = net

            elif self.model == 'QRDQN':
                layer_1 = tf.layers.dense(inputs=self.state, units=64, activation=tf.nn.relu, trainable=True)
                layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.nn.relu, trainable=True)
                layer_3 = tf.layers.dense(inputs=layer_2, units=64, activation=tf.nn.relu,
                                          trainable=True)
                layer_4 = tf.layers.dense(inputs=layer_3, units=self.action_size * self.num_support, activation=None,
                                          trainable=True)
                net = tf.reshape(layer_4, [-1, self.action_size, self.num_support])
                net_action = net

            elif self.model == 'IQN':
                state_tile = tf.tile(self.state, [1, self.num_support])
                state_reshape = tf.reshape(state_tile, [-1, self.state_size])
                state_net = tf.layers.dense(inputs=state_reshape, units=self.quantile_embedding_dim, activation=tf.nn.selu)

                tau = tf.reshape(self.tau, [-1, 1])
                pi_mtx = tf.constant(np.expand_dims(np.pi * np.arange(0, 64), axis=0), dtype=tf.float32)
                cos_tau = tf.cos(tf.matmul(tau, pi_mtx))
                phi = tf.layers.dense(inputs=cos_tau, units=self.quantile_embedding_dim, activation=tf.nn.relu)

                net = tf.multiply(state_net, phi)
                net = tf.layers.dense(inputs=net, units=512, activation=tf.nn.relu)
                net = tf.layers.dense(inputs=net, units=128, activation=tf.nn.relu)
                net = tf.layers.dense(inputs=net, units=self.action_size, activation=None)

                net_action = tf.transpose(tf.split(net, 1, axis=0), perm=[0, 2, 1])

                net = tf.transpose(tf.split(net, self.batch_size, axis=0), perm=[0, 2, 1])

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return net, net_action, params


    def choose_action(self, state):
        if self.model == 'DQN':
            result = self.sess.run(self.main_network, feed_dict={self.state: [state]})[0]
            action = np.argmax(result)

        elif self.model == 'C51':
            Q = self.sess.run(self.main_action_support, feed_dict={self.state: [state]})
            z_space = np.repeat(np.expand_dims(self.z, axis=0), self.action_size, axis=0)
            Q_s_a = np.sum(Q[0] * z_space, axis=1)
            action = np.argmax(Q_s_a)

        elif self.model == 'QRDQN':
            Q = self.sess.run(self.main_network, feed_dict={self.state: [state]})
            Q_s_a = np.mean(Q[0], axis=1)
            action = np.argmax(Q_s_a)

        elif self.model == 'IQN':
            t = np.random.rand(1, self.num_support)
            Q_s_a = self.sess.run(self.main_action_support, feed_dict={self.state: [state], self.tau: t})
            Q_s_a = Q_s_a[0]
            Q_a = np.sum(Q_s_a, axis=1)
            action = np.argmax(Q_a)
        return action


            
    
def test(agent, env, logger, model_step, path=None):
    # dummy_env = DummyVecEnv([lambda: env])
    # dummy_env = env
    success_episodes = 0
    e_obs_collision = 0
    e_out_of_road = 0
    e_lane_change = 0
    e_runover_light = 0
    e_static = 0
    e_max_steps = 0
    e_max_steps_obstacle = 0
    e_max_steps_light = 0
    e_unexpected_collision = 0
    e_unknown = 0
    results = {}
    total_reward = 0


    for ind in range(5):
        # obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        obs = env.reset(unseen=True, index=ind).squeeze()
        done = False
        reward = 0
    
        rewards = []
        validation_ep_index = '0'
        while not done:
            action = agent.choose_action(obs) # np.array([agent.get_action(obs.reshape([1, 1, agent.state_dim[1]]), 0)])
            info = env.step(np.array([action]))

            reward += info[1][0]
            done = info[2]
            obs = np.expand_dims(info[0], axis=0).squeeze()
            rewards.append(info[1][0])
            
            if done:
                validation_ep_index = info[3]['val_ep_idx']
        
        total_reward += reward
        if info[3]['termination_state'] == 'success':
            success_episodes += 1
            results[ind] = 1
        else:
            results[ind] = 0
            if info[3]['termination_state'] == 'obs_collision':
                e_obs_collision += 1
            elif info[3]['termination_state'] == 'out_of_road':
                e_out_of_road += 1
            elif info[3]['termination_state'] == 'lane_invasion':
                e_lane_change += 1
            elif info[3]['termination_state'] == 'runover_light':
                e_runover_light += 1
            elif info[3]['termination_state'] == 'static':
                e_static += 1
            elif info[3]['termination_state'] == 'max_steps':
                e_max_steps += 1
            elif info[3]['termination_state'] == 'max_steps_obstacle':
                e_max_steps_obstacle += 1
            elif info[3]['termination_state'] == 'max_steps_light':
                e_max_steps_light += 1
            elif info[3]['termination_state'] == 'unexpected_collision':
                e_unexpected_collision += 1
            else:
                e_unknown += 1
        

    # Reset env after testing
    env.reset()
    print("Results of train scenarios")
    print(results)
    print("Step: {0} Total Success Episodes: {1}".format(model_step, success_episodes))
    logger.log_scalar('test/success_episodes', success_episodes, model_step)
    logger.log_scalar('test/total_reward', total_reward, model_step)
    logger.log_scalar('test/e_obs_collision', e_obs_collision, model_step)
    logger.log_scalar('test/e_out_of_road', e_out_of_road, model_step)
    logger.log_scalar('test/e_lane_change', e_lane_change, model_step)
    logger.log_scalar('test/e_runover_light', e_runover_light, model_step)
    logger.log_scalar('test/e_static', e_static, model_step)
    logger.log_scalar('test/e_max_steps', e_max_steps, model_step)
    logger.log_scalar('test/e_unexpected_collision', e_unexpected_collision, model_step)
    logger.log_scalar('test/e_unknown', e_unknown, model_step)
    # logger.log_scalar('test/last_td_error', last_td_error, model_step)

    # with open(path + 'test_results.csv','a') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow([model_step, success_episodes, total_reward[0],
    #         e_obs_collision,  e_out_of_road, e_lane_change,
    #         e_runover_light, e_static, e_max_steps, e_max_steps_obstacle, e_max_steps_light])

    return total_reward, success_episodes

def plot_test_results(total_successes, total_rewards, total_updates, path):
    fig, (ax1, ax2)  = plt.subplots(1, 2)
    fig.suptitle('Test Results v/s training timesteps')

    ax1.plot(total_updates, np.array(total_successes), color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Success Episodes')
    ax2.plot(total_updates, np.array(total_rewards), color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    ax2.set_xlabel('Total Reward')
    ax2.set_ylabel('time')
    
    ax1.grid(True)
    ax2.grid(True)
    
    plt.grid(True)
    plt.savefig(path + 'test_results.png')
    plt.close()


def run_c51(args, prefix, config): 
    ALTA_LOGS = os.path.join(args.base_log_dir, prefix) #os.path.join(args.base_log_dir, prefix.split('_runid_')[0], prefix)
    if ALTA_LOGS[-1] != '/':
        ALTA_LOGS += '/'

    SCRATCH_DIR = ALTA_LOGS
    if SCRATCH_DIR[-1] != '/':
        SCRATCH_DIR += '/'

    if not os.path.exists(ALTA_LOGS):
        os.makedirs(ALTA_LOGS)

    TF_MODELS = ALTA_LOGS+'tf-models/checkpoint/'
    if not os.path.exists(TF_MODELS):
        os.makedirs(TF_MODELS)

    FRAME_SKIP = 5
    SAVE_PATH = ALTA_LOGS + 'dqn_measurements_weights'
    TB_LOGS_DIR = ALTA_LOGS+'tb/'

    steps = args.timesteps

    print('ALTA_LOGS: ', ALTA_LOGS)

    

    try:
            # Create the environment
        logger = tf_log.Logger(TB_LOGS_DIR)
        
        
        print("Training begins")
        IMAGES_PATH = SCRATCH_DIR+'images/'
        VIDEO_PATH = SCRATCH_DIR+'videos/'
        IMAGES_PATH_VAE = SCRATCH_DIR+'vae_images/'
        VIDEO_PATH_VAE = SCRATCH_DIR+'vae_videos/'
        
        vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP, videos=config.config["videos"])
        vis_wrapper_vae = None

        RETRIES_ON_ERROR = 1
        serverStartRetries = 0
        serverStarted = False
        env = None
        while ((not serverStarted) and serverStartRetries < RETRIES_ON_ERROR):
            try:

                env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, vis_wrapper_vae=vis_wrapper_vae, logger=logger, log_dir=ALTA_LOGS)
                serverStarted = True
            
            except Exception as identifier:
                print(prefix, identifier, serverStartRetries)
                traceback.print_exc()
                if env is not None:
                    env.close()
                    serverStartRetries += 1
                    time.sleep(20)
            print('server started: ', serverStarted)
        
        # dummy_env = DummyVecEnv([lambda: env])

        # replay_buffer = ReplayBuffer(args.buffer_size)
        exploration_fraction = 0.1 
        replay_buffer = deque(maxlen=args.buffer_size)
        exploration = LinearSchedule(schedule_timesteps=exploration_fraction*steps, initial_p=1.0, final_p=0.05)

        # Initialize the parameters and copy them to the target network.
        # U.initialize()
        n_atoms = 51
        v_min = -5
        v_max = 100
        learning_rate = args.lr


        state_dim = env.observation_space.shape[1]
        action_dim = env.action_space.n

        sess = tf.Session()

        model = 'C51'
        agent = C51_Agent(sess, model, learning_rate, state_dim, action_dim, n_atoms, v_min, v_max)
        sess.run(tf.global_variables_initializer())
        sess.run(agent.assign_ops)



        



        obs = env.reset().squeeze()
        # print(agent.get_action(obs.reshape([1, 1, state_dim[1]]), 0))
        # import ipdb; ipdb.set_trace()

        num_episodes = 0 
        num_done = 0
        PRINT_FREQ = 100
        TEST_FREQ = 50000
        SAVE_FREQ = 500000
        learning_starts = 1000
        batch_size = 32


        total_reward_list = []
        success_episodes_list = []
        t_list = []
        to_test = False 

        # total_reward, success_episodes = test(agent, env, logger, 0)
        # t_list.append(0)
        # print('TESTING: t %d | rew %d | success %d ' % (t, total_reward, success))
        # total_reward_list.append(total_reward)
        # success_episodes_list.append(success_episodes)
        # plot_test_results(success_episodes_list, total_reward_list, np.array(t_list), ALTA_LOGS)



        for t in range(steps): 
            if np.random.rand() < exploration.value(t):
                action = np.random.choice(action_dim)
            else:
                action = agent.choose_action(obs)

            # action = np.array([agent.get_action(obs.reshape([1, 1, state_dim[1]]), exploration.value(t))])#[0]
            new_obs, rew, done, eps_measurements = env.step(np.array([action]))

            rew = float(rew[0])
            done = bool(done[0])
            # import ipdb; ipdb.set_trace()
            action_one_hot = np.zeros([action_dim])
            action_one_hot[action] = 1
            # replay_buffer.add(obs, action_one_hot, rew, new_obs.squeeze(), float(done))
            replay_buffer.append([obs, new_obs.squeeze(), action_one_hot, rew, int(done)])
            obs = new_obs.squeeze()


            if t > learning_starts: 
                _, td_error = agent.train(random.sample(replay_buffer, batch_size))#obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # if (t + 1) % SAVE_FREQ == 0: 
                #     agent.model.save(ALTA_LOGS + 'model_%d.h5' % (t))

                # if (t + 1) % args.target_freq == 0: 
                #     agent.copy_target()

                if (t+1) % TEST_FREQ == 0: 
                    to_test = True

                if (t+1) % PRINT_FREQ == 0: 
                    print('TRAINING: t %d | rew %d | loss %f' % (t, rew, td_error))

            # if done:
                
            #     num_episodes += 1
            #     num_done += 1

            #     if num_episodes % TEST_FREQ == 0: 
            #         total_reward, success_episodes = test(agent, env, logger, t)
            #         total_reward_list.append(total_reward)
            #         success_episodes_list.append(success_episodes)
            #         plot_test_results(success_episodes_list, total_reward_list, num_episodes, ALTA_LOGS)

                


                
                if done: 
                    num_episodes += 1

                    if to_test: 
                        t_list.append(t)
                        total_reward, success_episodes = test(agent, env, logger, t)
                        print('TESTING: t %d | rew %d | success %d ' % (t, total_reward, success_episodes))
                        total_reward_list.append(total_reward)
                        success_episodes_list.append(success_episodes)
                        plot_test_results(success_episodes_list, total_reward_list, np.array(t_list), ALTA_LOGS)
                        to_test = False 


                    
                    logger.log_scalar('train/episodes/dist_to_target', eps_measurements['distance_to_goal'], num_episodes)
                    logger.log_scalar('train/episodes/reward', eps_measurements['total_reward'], num_episodes)
                    logger.log_scalar('train/timesteps/dist_to_target', eps_measurements['distance_to_goal'], t)
                    logger.log_scalar('train/timesteps/reward', eps_measurements['total_reward'], t)

                    obs = env.reset().squeeze()
                    done = False 
                    
                
               

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
    
    






   

