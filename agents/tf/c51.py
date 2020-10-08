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
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Input
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
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

def make_model(input_dim, output_dim, num_atoms, lr=0.0001):         
    state_input = Input(shape=input_dim)
    layer = Dense(256, activation='relu')(state_input)
    layer = Dense(256, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)

    distribution_list = []
    for i in range(output_dim):
        distribution_list.append(Dense(num_atoms, activation='softmax')(layer))

    model = Model(inputs=state_input, outputs=distribution_list)

    optimizer = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer)

    return model 


class C51_Agent():

    
    def __init__(self, 
        state_dim, 
        action_dim, 
        n_atoms, 
        v_min, 
        v_max, 
        load_model='', 
        batch_size=64, 
        tau = 0.01, 
        k=1, 
        gamma=0.99, 
        render=False):

        
        
        self.action_dim = action_dim
        self.state_dim = state_dim

        # import ipdb; ipdb.set_trace()
        
        
        self.num_atoms = n_atoms 
        self.v_min = v_min 
        self.v_max = v_max
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = np.array([self.v_min + i * self.delta_z for i in range(self.num_atoms)])
        
        self.model = make_model(self.state_dim, self.action_dim, self.num_atoms)
        self.target_model = make_model(self.state_dim, self.action_dim, self.num_atoms)
        
        
        self.batch_size = batch_size
        
        
        self.gamma = gamma
        
        self.tau = 0.01
    
    

    def get_action(self, state, epsilon=0):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= epsilon:
            #print("----------Random Action----------")
            action_idx = random.randrange(self.action_dim)
        else:
            action_idx = self.get_optimal_action(state)

        return action_idx

    def get_optimal_action(self, state):
        """Get optimal action for a state
        """
        z = np.array(self.model.predict(state)).squeeze() # Return a list [1x51, 1x51, 1x51]

        # z_concat = np.vstack(z)
        q = np.dot(z, self.z)
        # q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1) 

        # Pick action with the biggest Q value
        action_idx = np.argmax(q)
        
        return action_idx
    
    
    def q_loss(self, sample): 
        
        states_old = sample[0] #np.asarray([x[0] for x in sample])[:, 0, :]
        actions = sample[1]#np.asarray([x[1] for x in sample])
        rewards = sample[2]#np.asarray([x[3] for x in sample])
        next_states = sample[3]# np.asarray([x[2] for x in sample])[:, 0, :]
        
        done = sample[4]#np.asarray([x[4] for x in sample])
        m_prob = np.zeros((self.action_dim, self.batch_size, 1, self.num_atoms))
        
        
        z = self.model.predict(next_states) # Return a list [32x51, 32x51, 32x51]
        z_ = np.array(self.target_model.predict(next_states)).squeeze() # Return a list [32x51, 32x51, 32x51]

        # Get Optimal Actions for the next states (from distribution z)
        # optimal_action_idxs = []
        # z_concat = np.vstack(z)
        # q = np.sum(np.multiply(z_concat, self.z), axis=1) # length (num_atoms x num_actions)
        # q = q.reshape((self.batch_size, self.action_dim), order='F')
        q = np.dot(np.array(z).squeeze(), self.z)
        optimal_action_idxs = np.argmax(q, axis=0)
        dummy_index1 = range(self.batch_size)
        dummy_index2 = np.zeros([self.batch_size]).astype(int)

        # import ipdb; ipdb.set_trace()

        for j in range(self.num_atoms): 

            future = np.asarray([0 if done[i] else self.z[j] for i in range(self.batch_size)])
            Tz = np.minimum(self.v_max, np.maximum(self.v_min, rewards + self.gamma * future))
            bj = (Tz - self.v_min) / self.delta_z 
            m_l = np.floor(bj)
            m_u = np.ceil(bj)
            mult = np.asarray([1 if done[i] else z_[optimal_action_idxs[i], i, j] for i in range(self.batch_size)])
            m_prob[np.ix_(actions, dummy_index1, dummy_index2, m_l.astype(int))] += np.multiply(mult, m_u - bj)
            m_prob[np.ix_(actions, dummy_index1, dummy_index2, m_u.astype(int))] += np.multiply(mult, bj - m_l)
            # m_prob[actions, :, :, m_l.astype(int)] += np.multiply(mult, m_u - bj)
            # m_prob[actions, :, :, m_u.astype(int)] += np.multiply(mult, b_j - m_l)
            # import ipdb; ipdb.set_trace()



        # Project Next State Value Distribution (of optimal action) to Current State
        # for i in range(self.batch_size):
        #     if done[i]: # Terminal State
        #         # Distribution collapses to a single point
        #         Tz = min(self.v_max, max(self.v_min, rewards[i]))
        #         bj = (Tz - self.v_min) / self.delta_z 
        #         m_l, m_u = math.floor(bj), math.ceil(bj)
        #         m_prob[actions[i]][i, 0, int(m_l)] += (m_u - bj)
        #         m_prob[actions[i]][i, 0, int(m_u)] += (bj - m_l)
        #     else:
        #         for j in range(self.num_atoms):
        #             Tz = min(self.v_max, max(self.v_min, rewards[i] + self.gamma * self.z[j]))
        #             bj = (Tz - self.v_min) / self.delta_z 
        #             m_l, m_u = math.floor(bj), math.ceil(bj)
        #             m_prob[actions[i]][i, 0, int(m_l)] += z_[optimal_action_idxs[i], i, j] * (m_u - bj)
        #             m_prob[actions[i]][i, 0, int(m_u)] += z_[optimal_action_idxs[i], i, j] * (bj - m_l)

        train_m_prob = [m_prob[a] for a in range(self.action_dim)]
        # import ipdb; ipdb.set_trace() 
        history = self.model.fit(states_old, train_m_prob, epochs=1, verbose=0)
        
        
        return  history.history['loss'][-1]
        
    def copy_target(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)
    
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)): 
            target_weights[i] = self.tau * weights[i] + (1. - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)


            
    
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
        obs = env.reset(unseen=True, index=ind)
        done = False
        reward = 0
    
        rewards = []
        validation_ep_index = '0'
        while not done:
            action = np.array([agent.get_action(obs.reshape([1, 1, agent.state_dim[1]]), 0)])
            info = env.step(action)

            reward += info[1][0]
            done = info[2]
            obs = np.expand_dims(info[0], axis=0)
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
    ALTA_LOGS = args.base_log_dir #os.path.join(args.base_log_dir, prefix.split('_runid_')[0], prefix)
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

        replay_buffer = ReplayBuffer(args.buffer_size)
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.05)

        # Initialize the parameters and copy them to the target network.
        # U.initialize()
        n_atoms = 51
        v_min = -5
        v_max = 100

        state_dim = env.observation_space.shape
        action_dim = env.action_space.n


        

        # total_reward, success_episodes = test(agent, env, logger, 0)

        obs = env.reset()
        # print(agent.get_action(obs.reshape([1, 1, state_dim[1]]), 0))
        # import ipdb; ipdb.set_trace()

        num_episodes = 0 
        num_done = 0
        TEST_FREQ = 50000
        SAVE_FREQ = 500000
        batch_size = 64
        learning_starts = 64

        agent = C51_Agent(state_dim, action_dim, n_atoms, v_min, v_max)
        agent.copy_target()

        total_reward_list = []
        success_episodes_list = []
        t_list = []
        to_test = False 



        for t in range(steps): 

            action = np.array([agent.get_action(obs.reshape([1, 1, state_dim[1]]), exploration.value(t))])#[0]
            new_obs, rew, done, eps_measurements = env.step(np.array(action))

            rew = float(rew[0])
            done = bool(done[0])
            replay_buffer.add(obs, action[0], rew, new_obs, float(done))
            obs = new_obs


            if t > learning_starts: 
                td_error = agent.q_loss(replay_buffer.sample(batch_size))#obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                if (t + 1) % SAVE_FREQ == 0: 
                    agent.model.save(ALTA_LOGS + 'model_%d.h5' % (t))

                if (t + 1) % args.target_freq == 0: 
                    agent.copy_target()

                if (t+1) % TEST_FREQ == 0: 
                    to_test = True

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
                        print('TESTING: t %d | rew %d | success %d ' % (t, total_reward, success))
                        total_reward_list.append(total_reward)
                        success_episodes_list.append(success_episodes)
                        plot_test_results(success_episodes_list, total_reward_list, np.array(t_list), ALTA_LOGS)

                    
                    logger.log_scalar('train/episodes/dist_to_target', eps_measurements['distance_to_goal'], num_episodes)
                    logger.log_scalar('train/episodes/reward', eps_measurements['total_reward'], num_episodes)
                    logger.log_scalar('train/timesteps/dist_to_target', eps_measurements['distance_to_goal'], t)
                    logger.log_scalar('train/timesteps/reward', eps_measurements['total_reward'], t)

                    obs = env.reset()
                    done = False 
                    to_test = False 

                
               

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
    
    






   

if __name__ == '__main__':

 # with U.make_session():
        # Create the environment
    config = ConfigManager(algo="DQN")
    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
    # NOTE: not using Monitor for now. integrate later
    # env = wrappers.Monitor(env, '/tmp/deepq'+str(datetime.now()), force=True)
    logger = tf_log.Logger('./tf-logs/'+ prefix +str(datetime.now()))
    env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger, log_dir=ALTA_LOGS)
    # import ipdb; ipdb.set_trace()

    print('-'*50)
    print('Launched environment!')
    print('-'*50)
    # Create all the functions necessary to train the model
    # act, train, update_target, debug = deepq.build_train(
    #     make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
    #     q_func=CoRLModel,
    #     num_actions=env.action_space.n,
    #     optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
    #     gamma=0.95,
    #     double_q=True
    # )

    # act_params = {
    #         'make_obs_ph': lambda name: ObservationInput(env.observation_space, name=name),
    #         'q_func': CoRLModel,
    #         'num_actions': env.action_space.n
    #         }

    # print('-'*50)
    # print('Built model!')
    # print('-'*50)
    # Create the replay buffer
    replay_buffer = ReplayBuffer(100000)
    # Create the schedule for exploration starting from 1 (every action is random) down to
    # 0.02 (98% of actions are selected according to values predicted by the model).
    exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.05)

    # Initialize the parameters and copy them to the target network.
    # U.initialize()
    n_atoms = 51
    v_min = -5
    v_max = 100
    agent = C51_Agent(env.observation_space.shape, env.action_space.n, n_atoms, v_min, v_max)
    agent.copy_target()
    t = 0
    agent.model.save(MODEL_SAVE_DIR + 'model_%s.h5' % (str(t)))

    obs = env.reset()
    # import ipdb; ipdb.set_trace()
    # print('-'*50)
    # print('Received observation of shape:', obs['image'].shape)
    # print('-'*50)
    num_episodes = 0
    num_done = 0
    for t in itertools.count():
        # import ipdb; ipdb.set_trace()
        # Take action and update exploration to the newest value
        action = np.array([agent.get_action(obs, exploration.value(t))])#[0]
        new_obs, rew, done, eps_measurements = env.step(np.array(action))
        # vis_wrapper.save_image(obs['image'], t)
        # import ipdb; ipdb.set_trace()

        # Store transition in the replay buffer.
        # Read only sensor image part of the observation (sensor_image, [measurements_array])
        rew = float(rew[0])
        done = bool(done[0])
        replay_buffer.add(obs, action[0], rew, new_obs, float(done))
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
            # vis_wrapper.generate_video(num_episodes)
            # vis_wrapper.remove_images()
            obs = env.reset()
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if(num_done % 10 == 0):
                print('-'*50)
                print('Launching validation step on seen')
                print('-'*50)
                validation_done = None
                while(validation_done != True):
                    # Take action and update exploration to the newest value
                    action = np.array([agent.get_action(obs, exploration.value(t))])
                    new_obs, rew, done, eps_measurements = env.step(action)
                    # Store transition in the replay buffer.
                    # Read only sensor image part of the observation (sensor_image, [measurements_array])
                    rew = float(rew[0])
                    done = bool(done[0])
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
                    action = np.array([agent.get_action(obs, exploration.value(t))])
                    new_obs, rew, done, eps_measurements = env.step(action)
                    # Store transition in the replay buffer.
                    # Read only sensor image part of the observation (sensor_image, [measurements_array])
                    rew = float(rew[0])
                    done = bool(done[0])
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
                # obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(64)
                td_error = agent.q_loss(replay_buffer.sample(64))#obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
            if(t % 1000 == 0 and t > 0):
                print('-'*50)
                print('Saving model (checkpoint)!')
                print('-'*50)
                agent.model.save(MODEL_SAVE_DIR + 'model_%s.h5' % (str(t)))

                # wrapped_act = ActWrapper(act, act_params)
                # wrapped_act.save(MODEL_SAVE_DIR+'tf-models/checkpoint/corl-carla-model-'+ prefix +str(t)+'.pkl')
                agent.target_train()
