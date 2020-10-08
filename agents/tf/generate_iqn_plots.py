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
# from baselines.deepq.deepq import ActWrapper
from baselines.deepq.replay_buffer import ReplayBuffer
# from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule

import vis_module

from gym import wrappers

from datetime import datetime
import random


import matplotlib.pyplot as plt

import tensorboard_logging as tf_log
from collections import deque

from iqn import C51_Agent


def compute_discounted_returns(rewards, gamma):
    returns = np.zeros_like(rewards)
    n = np.size(rewards)
    
    returns[-1] = rewards[-1]
    for i in range(n-2, 0, -1):
        returns[i] = rewards[i] + gamma* returns[i+1]

    return returns

def plot_q_values(q_values_matrix, actions, action_q_values, returns, ind, model_step, path): 
    n = len(actions)
    # if model_step == 0: 
    #     print(np.array(q_values_matrix).shape)
    fig, (ax1, ax2, ax3)  = plt.subplots(3, 1, figsize=(12, 12))

    ax1.matshow(np.transpose(np.array(q_values_matrix)), cmap=plt.cm.Blues, aspect='auto')
    ax1.set_ylabel('action')

    ax2.plot(range(n), actions)
    ax2.set_ylabel('action')

    ax3.plot(range(n), action_q_values, label='q values')
    ax3.plot(range(n), returns, label='returns')
    ax3.legend()
    ax3.set_ylabel('returns')
    ax3.set_xlabel('t')


    plt.savefig(path + 'qvalues_step_%s_ind_%s.png' % (str(model_step), str(ind)))

def plot_samples(samples, f_samples, actions, ind, model_step, path): 
    # import ipdb; ipdb.set_trace()
    

    samples = np.array(samples)
    f_samples = np.array(f_samples)
    # print(samples.shape, f_samples.shape)
    (traj_length, num_actions, num_atoms) = samples.shape
    fig, axes = plt.subplots(num_actions, traj_length, figsize=(len(range(traj_length)), 6))
    for a in range(num_actions): 
        for t in range(traj_length): 
            # index = a * traj_length + t + 1 
            # import ipdb; ipdb.set_trace()
            ax = axes[a, t] 
            uniform = f_samples[t, 0, :]
            returns = samples[t, a, :] 

            color = 'blue'
            if actions[t] == a: 
                color = 'red'

            for s in range(num_atoms): 
                ax.plot([uniform[s], uniform[s]], [0, returns[s]], linewidth=2, c=color)
            #ax.set_ylim([-5 / 105., 100 / 105.])
            ax.set_ylim([-5, 100])
            ax.set_xlim([0, 1])
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            ax.get_xaxis().set_ticklabels([])
            ax.get_yaxis().set_ticklabels([])
    plt.savefig(path + 'samples_step_%s_ind_%s.png' % (str(model_step), str(ind)))
    plt.close()



# def test(agent, env, model_step, path=None):
#     # dummy_env = DummyVecEnv([lambda: env])
#     # dummy_env = env
#     success_episodes = 0
#     e_obs_collision = 0
#     e_out_of_road = 0
#     e_lane_change = 0
#     e_runover_light = 0
#     e_static = 0
#     e_max_steps = 0
#     e_max_steps_obstacle = 0
#     e_max_steps_light = 0
#     e_unexpected_collision = 0
#     e_unknown = 0
#     results = {}
#     total_reward = 0


#     for ind in range(4):
#         # obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
#         obs = env.reset(unseen=True, index=ind).squeeze()
#         done = False
#         reward = 0
    
#         rewards = []
#         q_values_matrix = []
#         actions = [] 
#         action_q_values = []
#         validation_ep_index = '0'
#         while not done:
#             action, Q_a = agent.choose_action(obs) # np.array([agent.get_action(obs.reshape([1, 1, agent.state_dim[1]]), 0)])
#             info = env.step(np.array([action]))
            
#             q_values_matrix.append(Q_a / np.sum(np.abs(Q_a)))
#             actions.append(action)
#             action_q_values.append(Q_a[action])

#             reward += info[1][0]
#             done = info[2]
#             obs = np.expand_dims(info[0], axis=0).squeeze()
#             rewards.append(info[1][0])
            
#             if done:
#                 validation_ep_index = info[3]['val_ep_idx']
        
#         total_reward += reward
#         if info[3]['termination_state'] == 'success':
#             success_episodes += 1
#             results[ind] = 1
#         else:
#             results[ind] = 0

#         returns = compute_discounted_returns(rewards, agent.gamma)

#         plot_q_values(q_values_matrix, actions, action_q_values, returns, ind, model_step, path)
        

#     # Reset env after testing
#     env.reset()
#     print("Results of train scenarios")
#     print(results)
#     print("Step: {0} Total Success Episodes: {1}".format(model_step, success_episodes))

#     # logger.log_scalar('test/last_td_error', last_td_error, model_step)

#     # with open(path + 'test_results.csv','a') as f:
#     #     writer = csv.writer(f, delimiter=',')
#     #     writer.writerow([model_step, success_episodes, total_reward[0],
#     #         e_obs_collision,  e_out_of_road, e_lane_change,
#     #         e_runover_light, e_static, e_max_steps, e_max_steps_obstacle, e_max_steps_light])

#     return total_reward, success_episodes


def test(agent, env, model_step, path=None, num_test=5):
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
    Q_action_total = np.zeros([num_test])
    returns_total = np.zeros([num_test])


    for ind in range(num_test):
        # obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        obs = env.reset(unseen=True, index=ind).squeeze()
        done = False
        reward = 0
    
        rewards = []
        q_values_matrix = []
        actions = [] 
        action_q_values = []
        samples = []
        f_samples = []
        validation_ep_index = '0'
        while not done:
            # Q_s_a is (actions, atoms)
            action, Q_a, f, Q_s_a = agent.choose_action(obs) # np.array([agent.get_action(obs.reshape([1, 1, agent.state_dim[1]]), 0)])
            info = env.step(np.array([action]))
            
            q_values_matrix.append(Q_a / np.sum(np.abs(Q_a)))
            actions.append(action)
            action_q_values.append(Q_a[action])
            samples.append(Q_s_a)
            f_samples.append(f)

            reward += info[1][0]
            done = info[2]
            obs = np.expand_dims(info[0], axis=0).squeeze()
            rewards.append(info[1][0])
            
        
        total_reward += reward

        returns = compute_discounted_returns(rewards, agent.gamma)
        Q_action_total[ind] = np.mean(action_q_values)
        returns_total[ind] = np.mean(returns)

        plot_q_values(q_values_matrix, actions, action_q_values, returns, ind, model_step, path)
        plot_samples(samples, f_samples, actions, ind, model_step, path)
        

    # Reset env after testing
    env.reset()

    # logger.log_scalar('test/last_td_error', last_td_error, model_step)

    # with open(path + 'test_results.csv','a') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow([model_step, success_episodes, total_reward[0],
    #         e_obs_collision,  e_out_of_road, e_lane_change,
    #         e_runover_light, e_static, e_max_steps, e_max_steps_obstacle, e_max_steps_light])
    print('model step: ', model_step)
    print('success:', success_episodes)

    return total_reward, success_episodes, Q_action_total, returns_total

def plot_test_results(total_successes, total_rewards, total_updates, path):
    total_successes.reverse() 
    total_rewards.reverse()
    total_updates.reverse()

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

def plot_average_q(q_action_mean_list, q_action_var_list, returns_mean_list, returns_var_list, total_updates, path): 
    q_action_mean_list.reverse()
    q_action_var_list.reverse()
    returns_mean_list.reverse()
    returns_var_list.reverse()
    total_updates.reverse()
    q_action_mean_list = np.array(q_action_mean_list)
    q_action_var_list = np.array(q_action_var_list)
    returns_mean_list = np.array(returns_mean_list)
    returns_var_list = np.array(returns_var_list)
    plt.figure() 
    plt.plot(total_updates, q_action_mean_list, label='q values')
    plt.fill_between(total_updates, q_action_mean_list - q_action_var_list, q_action_mean_list + q_action_var_list, alpha=0.2)

    plt.plot(total_updates, returns_mean_list, label='returns')
    plt.fill_between(total_updates, returns_mean_list - returns_var_list, returns_mean_list + returns_var_list, alpha=0.2)

    plt.legend()
    plt.xlabel('t')
    plt.savefig(path + 'qvalues.png')
    plt.close()




def run_iqn_plots(args, prefix, config): 
    ALTA_LOGS = os.path.join(args.base_log_dir, prefix) #os.path.join(args.base_log_dir, prefix.split('_runid_')[0], prefix)
    if ALTA_LOGS[-1] != '/':
        ALTA_LOGS += '/'

    SCRATCH_DIR = ALTA_LOGS
    if SCRATCH_DIR[-1] != '/':
        SCRATCH_DIR += '/'

    # if not os.path.exists(ALTA_LOGS):
    #     os.makedirs(ALTA_LOGS)

    TF_MODELS = ALTA_LOGS+'checkpoint/'
    # if not os.path.exists(TF_MODELS):
    #     os.makedirs(TF_MODELS)

    FRAME_SKIP = args.frame_skip 
    SAVE_PATH = ALTA_LOGS + 'dqn_measurements_weights'
    TB_LOGS_DIR = ALTA_LOGS+'tb_copy/'

    steps = args.timesteps

    print('ALTA_LOGS: ', ALTA_LOGS)

    

    try:
            # Create the environment
        logger = tf_log.Logger(TB_LOGS_DIR)
        
        
        IMAGES_PATH = SCRATCH_DIR+'images_plots/'
        
        VIDEO_PATH = SCRATCH_DIR+'videos_plots/'
        IMAGES_PATH_VAE = SCRATCH_DIR+'vae_images/'
        VIDEO_PATH_VAE = SCRATCH_DIR+'vae_videos/'
        QVALUES_PATH = SCRATCH_DIR+'qvalue_plots_test/'

        if not os.path.exists(IMAGES_PATH):
            os.makedirs(IMAGES_PATH)
        if not os.path.exists(VIDEO_PATH):
            os.makedirs(VIDEO_PATH)
        if not os.path.exists(QVALUES_PATH):
            os.makedirs(QVALUES_PATH)
        
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
        
        
        env.reset()
        state_dim = env.observation_space.shape[1]
        action_dim = env.action_space.n
        learning_rate = args.lr
        # num_atoms = args.num_atoms
        num_atoms = args.num_atoms
        alpha = args.alpha
        print('num atoms: ', num_atoms)
        if args.network == '2_layer': 
            f_arch = [128, 128]
        elif args.network == '3_layer': 
            f_arch = [256, 128, 64]
        else: 
            f_arch = [128, 128]

        
        sess = tf.Session()

        model = 'IQN'
        agent = C51_Agent(sess, 
            model, 
            learning_rate, 
            state_dim, 
            action_dim, 
            num_support=num_atoms, 
            alpha=alpha, 
            f_arch = f_arch,
            )
        sess.run(tf.global_variables_initializer())
        # sess.run(agent.assign_ops)
        # agent.main_network.save(TF_MODELS)


        # tf.reset_default_graph()
        t_list = []
        total_reward_list = []
        success_episodes_list = []
        q_action_mean_list = []
        q_action_var_list = []
        returns_mean_list = []
        returns_var_list = []


        saver = tf.train.Saver()
        #times = ['399999', '499999', '599999', '699999', '799999','899999', '999999', '1099999']
        times = ['0', '99999', '199999', '299999', '399999', '499999', '599999', '699999', '799999','899999', '999999', '1099999']#, '1199999','1299999', '1399999', '1499999']
        #times = ['799999']
        times.reverse()
        
        for t in times: 
            tf.reset_default_graph()
            path = TF_MODELS + 'iter_' + t + '/model_' + t + '.ckpt'
            print('path: ', path)
            saver.restore(sess, path)
            sess.run(agent.assign_ops)
            t_list.append(t)

            # test(agent, env, t, path=QVALUES_PATH)
            total_reward, success_episodes, Q_action_total, returns_total = test(agent, env, t, path=QVALUES_PATH)
            total_reward_list.append(total_reward)
            success_episodes_list.append(success_episodes)
            plot_test_results(success_episodes_list, total_reward_list, t_list, QVALUES_PATH)

            q_action_mean_list.append(np.mean(Q_action_total))
            q_action_var_list.append(np.std(Q_action_total))
            returns_mean_list.append(np.mean(returns_total))
            returns_var_list.append(np.std(returns_total))
            plot_average_q(q_action_mean_list, q_action_var_list, returns_mean_list, returns_var_list, t_list, QVALUES_PATH)

            # if len(t_list) == 5: 
            #     import ipdb; ipdb.set_trace()
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
    

