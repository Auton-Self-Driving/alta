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
import pickle 
from iqn import C51_Agent
from agents.tf.ae.models import ConvAutoEncoder 

def test(agent, env, model_step, num_test=2):
    # dummy_env = DummyVecEnv([lambda: env])
    # dummy_env = env
    # success_episodes = 0
    # e_obs_collision = 0
    # e_out_of_road = 0
    # e_lane_change = 0
    # e_runover_light = 0
    # e_static = 0
    # e_max_steps = 0
    # e_max_steps_obstacle = 0
    # e_max_steps_light = 0
    # e_unexpected_collision = 0
    # e_unknown = 0
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
            # action, Q_a, f, Q_s_a = agent.choose_action(obs) # np.array([agent.get_action(obs.reshape([1, 1, agent.state_dim[1]]), 0)])
            action = 4
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

        # plot_q_values(q_values_matrix, actions, action_q_values, returns, ind, model_step, path)
        # plot_samples(samples, f_samples, actions, ind, model_step, path)
        

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




def run_iqn_vae(args, prefix, config): 

    ALTA_LOGS = os.path.join(args.base_log_dir, prefix) #os.path.join(args.base_log_dir, prefix.split('_runid_')[0], prefix)
    if ALTA_LOGS[-1] != '/':
        ALTA_LOGS += '/'

    SCRATCH_DIR = ALTA_LOGS
    if SCRATCH_DIR[-1] != '/':
        SCRATCH_DIR += '/'

    if not os.path.exists(ALTA_LOGS):
        os.makedirs(ALTA_LOGS)

    TF_MODELS = ALTA_LOGS+'checkpoint/'
    if not os.path.exists(TF_MODELS):
        os.makedirs(TF_MODELS)

    FRAME_SKIP = 5
    SAVE_PATH = ALTA_LOGS + 'dqn_measurements_weights'
    TB_LOGS_DIR = ALTA_LOGS+'tb/'

    steps = args.timesteps

    

    try:
            # Create the environment
        logger = tf_log.Logger(TB_LOGS_DIR)
        
        
        print("Training begins")
        IMAGES_PATH = SCRATCH_DIR+'images/'
        VIDEO_PATH = SCRATCH_DIR+'videos/'
        IMAGES_PATH_VAE = SCRATCH_DIR+'vae_images/'
        VIDEO_PATH_VAE = SCRATCH_DIR+'vae_videos/'
        QVALUES_PATH = SCRATCH_DIR+'qvalue_plots/'

        REPLAY_PATH = SCRATCH_DIR + 'replay/'

        if not os.path.exists(IMAGES_PATH):
            os.makedirs(IMAGES_PATH)
        if not os.path.exists(VIDEO_PATH):
            os.makedirs(VIDEO_PATH)
        if not os.path.exists(QVALUES_PATH):
            os.makedirs(QVALUES_PATH)
            os.makedirs(REPLAY_PATH)
        
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
        
        state_dim = env.observation_space.shape[1]
        action_dim = env.action_space.n
        learning_rate = args.lr
        num_atoms = args.num_atoms
        alpha = args.alpha
        if args.network == '2_layer': 
            f_arch = [128, 128]
        elif args.network == '3_layer': 
            f_arch = [256, 128, 64]
        else: 
            f_arch = [128, 128]


        print('LOADING VAE: ')
        ae = ConvAutoEncoder()
        ae.load_json('./trained_models/ae_model_rl_trained.json')
        env.set_vae(ae)

       
        print('LOADING MODEL: ')
        # sess = tf.Session()
        # model = 'IQN'
        # agent = C51_Agent(sess, 
        #     model, 
        #     learning_rate, 
        #     state_dim, 
        #     action_dim, 
        #     num_support=num_atoms, 
        #     alpha=alpha, 
        #     f_arch = f_arch,
        #     )

        # sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver(max_to_keep=50)

        # tf.reset_default_graph()
        # agent_path = '/zfsauton2/home/audreyh/alta-logs/iqn/9_6/algo_IQN_input_wp_obs_info_speed_steer_ldist_light_scenario_straight_npcs_150_lr_0.0001_alpha_1.0_network_2_layer_targetfreq_16000_rewardnorm_105_lightpenalty_3.0_frameskip_5_runid_target_16000_seed_1/checkpoint/iter_599999/model_599999.ckpt'
        # saver.restore(sess, agent_path)
        # sess.run(agent.assign_ops)

        
        # import ipdb; ipdb.set_trace()


        
        # print(agent.get_action(obs.reshape([1, 1, state_dim[1]]), 0))
        # import ipdb; ipdb.set_trace()

        num_episodes = 0 
        num_done = 0
        PRINT_FREQ = 100
        TEST_FREQ = 500
        SAVE_FREQ = 100000
        TARGET_FREQ = args.target_freq
        learning_starts = 1000
        batch_size = 32


        total_reward_list = []
        success_episodes_list = []
        q_action_mean_list = []
        q_action_var_list = []
        returns_mean_list = []
        returns_var_list = []
        t_list = []
        to_test = False 

        
        
        
        # save_path = saver.save(sess, TF_MODELS + 'iter_%s/model_%s.ckpt' % (str(0), str(0)))
        # total_reward, success_episodes, Q_action_total, returns_total = test(agent, env, logger, 0, path=QVALUES_PATH, num_test=1)
        # print('TESTING: t %d | rew %d | success %f ' % (0, total_reward, success_episodes))
        # obs = env.reset().squeeze()

        total_reward, success_episodes, Q_action_total, returns_total = test(None, env, 0)



        # for t in range(steps): 
        #     # if np.random.rand() < exploration.value(t):
        #     #     action = np.random.choice(action_dim)
        #     #     # if t == 1: 
        #     #     #     print('PRACTICE SAVING')
        #     #     #     with open(REPLAY_PATH + 'buffer_' + str(1) + '.pickle', 'wb') as handle:
        #     #     #         pickle.dump(replay_buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #     # else:
        #     #     action, _, _, _ = agent.choose_action(obs)
        #     action = 4

        #     # action = np.array([agent.get_action(obs.reshape([1, 1, state_dim[1]]), exploration.value(t))])#[0]
        #     new_obs, rew, done, eps_measurements = env.step(np.array([action]))

            # rew = float(rew[0])
            # done = bool(done[0])
            # # import ipdb; ipdb.set_trace()
            # action_one_hot = np.zeros([action_dim])
            # action_one_hot[action] = 1
            # # replay_buffer.add(obs, action_one_hot, rew, new_obs.squeeze(), float(done))
            # replay_buffer.append([obs, new_obs.squeeze(), action_one_hot, rew, int(done)])
            # obs = new_obs.squeeze()


            # if t > learning_starts: 
            #     _, td_error = agent.train(random.sample(replay_buffer, batch_size))#obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

            #     if (t+1) % PRINT_FREQ == 0: 
            #         print('TRAINING: t %d | rew %d | loss %f ' % (t, rew, td_error))
                
            #     if (t+1) % SAVE_FREQ == 0: 
            #         save_path = saver.save(sess, TF_MODELS + 'iter_%s/model_%s.ckpt' % (str(t), str(t)))
            #         with open(REPLAY_PATH + 'buffer_' + str(t) + '.pickle', 'wb') as handle:
            #             pickle.dump(replay_buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #     if (t+1) % args.target_freq == 0: 
            #         sess.run(agent.assign_ops)


                
            #     if done: 
            #         num_episodes += 1

            #         if num_episodes % TEST_FREQ == 0: #to_test: 
            #             t_list.append(t)
            #             total_reward, success_episodes, Q_action_total, returns_total = test(agent, env, logger, t, path=QVALUES_PATH)
            #             print('TESTING: t %d | rew %d | success %f ' % (t, total_reward, success_episodes))
            #             total_reward_list.append(total_reward)
            #             success_episodes_list.append(success_episodes)
            #             plot_test_results(success_episodes_list, total_reward_list, np.array(t_list), ALTA_LOGS)

            #             q_action_mean_list.append(np.mean(Q_action_total))
            #             q_action_var_list.append(np.std(Q_action_total))
            #             returns_mean_list.append(np.mean(returns_total))
            #             returns_var_list.append(np.std(returns_total))
            #             plot_average_q(np.array(q_action_mean_list), np.array(q_action_var_list), np.array(returns_mean_list), np.array(returns_var_list), t_list, ALTA_LOGS)
            #             to_test = False 


                    
            #         logger.log_scalar('train/episodes/dist_to_target', eps_measurements['distance_to_goal'], num_episodes)
            #         logger.log_scalar('train/episodes/reward', eps_measurements['total_reward'], num_episodes)
            #         logger.log_scalar('train/timesteps/dist_to_target', eps_measurements['distance_to_goal'], t)
            #         logger.log_scalar('train/timesteps/reward', eps_measurements['total_reward'], t)

            #         obs = env.reset().squeeze()
            #         done = False 
            # if t > 10**7: 
            #     break
                    
                
               

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
    
    

