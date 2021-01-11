import sys
import time
import multiprocessing
from collections import deque
import warnings

import numpy as np
import tensorflow as tf

from stable_baselines import SAC
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.policies import SACPolicy
from stable_baselines import logger
import matplotlib.pyplot as plt
import matplotlib
import os
import csv
import time

import ipdb
trace = ipdb.set_trace

def test(model, env, dump_results=False, path='.', model_step=None):
    dummy_env = DummyVecEnv([lambda: env])
    success_episodes = 0
    collision_obs_episodes = 0
    collision_out_of_road_episodes = 0
    collision_lane_change_episodes = 0
    static_episodes = 0
    max_steps_episodes = 0
    runover_light_episodes = 0
    results = {}
    total_reward = 0
    #env.reset()

    '''saved_scenarios = env.base_dir+"/testing_scenarios/"
    if not os.path.exists(saved_scenarios):
        os.makedirs(saved_scenarios)'''

    for ind in range(env.config["num_episodes"]):
        obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        obs[:] = env.reset(unseen=True, index=ind)
        done = False
        reward = 0
        
        while not done:
            actions = model.predict(obs, deterministic=True)[0]
            info = env.step(actions)
            reward += info[1][0][0]
            done = info[2]
            obs = np.expand_dims(info[0], axis=0)
        
        total_reward += reward
        if info[3]['termination_state'] == 'success':
            success_episodes += 1
            results[ind] = 1
        else:
            results[ind] = 0
            if info[3]['obs_collision']:
                collision_obs_episodes += 1
            elif info[3]['lane_change']:
                collision_lane_change_episodes += 1
            elif info[3]['out_of_road']:
                collision_out_of_road_episodes += 1
            elif info[3]['termination_state'] == 'runover_light':
                runover_light_episodes += 1
            elif info[3]['termination_state'] == 'static':
                static_episodes += 1
            elif info[3]['termination_state'] == 'max_steps':
                max_steps_episodes += 1
        #env.client.stop_recorder()

    env.reset()
    print("Results of train scenarios")
    print(results)
    print("# Success: {}, # Obstacle Collision: {}, # Lane-change Collision: {}, Out-of-road Collision: {}, Runover light: {}, Static: {}, Max_steps: {}".format(success_episodes,
                                collision_obs_episodes, collision_lane_change_episodes, collision_out_of_road_episodes, runover_light_episodes, static_episodes, max_steps_episodes))
    if dump_results:
        env.logger.log_scalar('test/term_success', success_episodes, model_step)
        env.logger.log_scalar('test/term_obstacle', collision_obs_episodes, model_step)
        env.logger.log_scalar('test/term_out_of_road', collision_out_of_road_episodes, model_step)
        env.logger.log_scalar('test/term_lane_change', collision_lane_change_episodes, model_step)
        env.logger.log_scalar('test/term_runover_light', runover_light_episodes, model_step)
        env.logger.log_scalar('test/term_static', static_episodes, model_step)
        env.logger.log_scalar('test/term_max_steps', max_steps_episodes, model_step)
        env.logger.log_scalar('test/total_reward', total_reward, model_step)

        with open(path + 'test_results.csv','a') as f:
                csvwriter = csv.writer(f, delimiter=',')
                csvwriter.writerow([model_step, success_episodes, total_reward, collision_obs_episodes,
                        collision_out_of_road_episodes, collision_lane_change_episodes, runover_light_episodes, static_episodes, max_steps_episodes])
    return total_reward, success_episodes, results

def plot_policy_and_value_fns(model, ind, path, inp_type):
    if not os.path.exists(path):
        os.makedirs(path)

    if inp_type == "wp":
        observations = np.arange(-2, 2, 0.01).reshape((-1, 1))
    else:
        return

    det_actions = []
    stoch_actions = []
    var_actions = []
    values = []
    means = []

    for i in range(observations.shape[0]):
        obs = observations[np.newaxis, i, :]
        act = model.predict(obs, deterministic=True)[0]
        # print("Action", act)
        # act[0, 1] = (act[0, 1] + 10)
        det_actions.append(act)
        mean, std = model.predict_proba_step(obs)
        # print("mean, std",mean, std)
        act= model.predict(obs, deterministic=False)[0]
        # act[0, 1] = (act[0, 1] + 10)
        stoch_actions.append(act)
        var_actions.append(std)
        # values.append(mean)
        means.append(mean)
        
    det_actions = np.array(det_actions).reshape((-1, 2))
    stoch_actions = np.array(stoch_actions).reshape((-1, 2))
    # var_actions = np.exp(np.array(var_actions).reshape((-1, 2)))
    var_actions = np.array(var_actions).reshape((-1, 2))
    # values = np.array(values).reshape((-1, 1))
    means = np.array(means).reshape((-1,2))

    fig, axs = plt.subplots(4, 2, figsize=(12, 12))
    fig.suptitle('Policy plots for {} model'.format(ind))

    axs[0, 0].plot(observations, det_actions[:, 0], color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[0, 0].set_xlabel('Waypoint orientation')
    axs[0, 0].set_ylabel('Deterministic - Steer')
    axs[0, 1].plot(observations, det_actions[:, 1], color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[0, 1].set_xlabel('Waypoint orientation')
    axs[0, 1].set_ylabel('Deterministic - Target Speed')
    
    axs[1, 0].plot(observations, var_actions[:, 0], color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[1, 0].set_xlabel('Waypoint orientation')
    axs[1, 0].set_ylabel('Std Deviation - Steer')
    axs[1, 1].plot(observations, var_actions[:, 1], color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[1, 1].set_xlabel('Waypoint orientation')
    axs[1, 1].set_ylabel('Std Deviation - Target Speed')
    
    axs[2, 0].plot(observations, means[:, 0], color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[2, 0].set_xlabel('Waypoint orientation')
    axs[2, 0].set_ylabel('Mean - Steer')
    axs[2, 1].plot(observations, means[:, 1], color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[2, 1].set_xlabel('Waypoint orientation')
    axs[2, 1].set_ylabel('Mean - Target Speed')

    axs[3, 0].plot(observations, stoch_actions[:, 0], color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[3, 0].set_xlabel('Waypoint orientation')
    axs[3, 0].set_ylabel('Stochastic - Steer')
    axs[3, 1].plot(observations, stoch_actions[:, 1], color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[3, 1].set_xlabel('Waypoint orientation')
    axs[3, 1].set_ylabel('Stochastic - Target Speed')

    axs[0,0].grid(True)
    axs[0,1].grid(True)
    axs[1,0].grid(True)
    axs[1,1].grid(True)
    axs[2,0].grid(True)
    axs[2,1].grid(True)
    axs[3,0].grid(True)
    axs[3,1].grid(True)

    plt.grid(True)
    plt.savefig(path + 'policy_{}.png'.format(ind))
    plt.close()
    
    # plt.figure()
    # plt.plot(observations, values, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    # plt.xlabel('Waypoint orientation')
    # plt.ylabel('Mean')
    # fig.suptitle('Mean plots for {} model'.format(ind))  
    # plt.savefig(path + 'value_{}.png'.format(ind))

def get_save_best_model(total_rewards, total_successes, model_file_names, path):
    print("Rewards at intermediate training: {}".format(total_rewards))
    print("Total success episodes: {}".format(total_successes))
    m = max(total_successes)
    max_inds = np.array([i for i, j in enumerate(total_successes) if j == m])
    rewards = np.array(total_rewards)[max_inds]
    ind = max_inds[np.argmax(rewards)]
    print("Best model appears at index: {}".format(ind))
    print("No of successes in best model: {}".format(total_successes[ind]))
    print("Max no of successes: {}".format(m))
    best_file_name = model_file_names[ind]

    best_model = MY_SAC.load(best_file_name)

    with open(path + "best_model.txt", "w") as f:
        f.write("Best model: {}\n".format(best_file_name))
        f.write("Best model appears at index: {}\n".format(ind))
        f.write("No of successes in best model: {}\n".format(total_successes[ind]))
        f.write("Max no of successes: {}\n".format(m))
        f.write("Rewards at intermediate training: {}\n".format(total_rewards))
        f.write("Total success episodes: {}\n".format(total_successes))

    return best_model

class MY_SAC(SAC):
    '''def __init__(self, config = None, **kwargs):
        if config is None:
            print("Using config from env")
            config = kwargs.get('env').config
        if config["ent_coef"]==-1:
            config["ent_coef"] = 'auto'
        super(MY_SAC, self).__init__(ent_coef = config["ent_coef"], train_freq = config["train_freq"], \
                                        learning_starts = config["learning_starts"], **kwargs)
        self.config = config
        self.n_steps = config["n_steps"]
        self.gradient_steps = config["gradient_steps_per_iteration"]
        self.target_update_interval = config["target_update_interval"]
        #self.ent_coef = config["ent_coef"]
        self.st = time.time()'''

    def __init__(self, config = None, **kwargs):
        super(MY_SAC, self).__init__(ent_coef = 5e-2, train_freq = 512, \
                                        learning_starts = 100000, **kwargs)
        self.n_steps = 25
        self.gradient_steps = 100
        self.target_update_interval = 100
        #self.ent_coef = config["ent_coef"]
        self.st = time.time()
    

    def learn(self, env, total_timesteps, trained_timesteps, callback=None, seed=None,
              log_interval=100, tb_log_name="SAC", reset_num_timesteps=True, custom_logger = None, save_file="sac_weights"):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        total_rewards = []
        total_successes = []
        model_file_names = []

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn(seed)

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]

            obs = self.env.reset()
            rv_img = self.env.envs[0].recent_image_obs
            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            n_updates = 0
            infos_values = []

            mini_ep_info = []

            step = trained_timesteps
            #for step in range(trained_timesteps, total_timesteps):
            st = time.time()
            while(step<total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                if step % 75000==0 and not self.config["testing"] and step>0:
                    print("Starting Validation...")
                    self.save(save_file + str(step))
                    st1 = time.time()
                    plot_policy_and_value_fns(self, step, save_file.split('sac_me')[0] + 'policy_plots/', self.config["input_type"])
                    '''total_reward, success_episodes,_ = test(self, env, dump_results=True, path=save_file.split('models')[0], model_step=step)
                    total_rewards.append(total_reward)
                    total_successes.append(success_episodes)'''
                    model_file_names.append(save_file + str(step))
                    print('*'*100, time.time()-st1)
                    print('*'*100, time.time()-self.st)
                    print("Ending Validation...")
                    self.st = time.time()

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy.
                if self.num_timesteps < self.learning_starts:
                    if self.env.envs[0].expert is not None:
                        rv_img = rv_img.reshape((1, -1))
                        rv_img = self.env.envs[0].preproc_img(rv_img).astype(np.float32)
                        rv_img = np.expand_dims(rv_img, axis = 0)                        
                        with tf.device('gpu:0'):
                            action = self.env.envs[0].expert.predict([rv_img, obs[:,-4:-1]])[0]
                            action = action.flatten()
                            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                    else:
                        action = self.env.action_space.sample()
                    # No need to rescale when sampling random action
                    rescaled_action = action
                else:
                    action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    # Rescale from [-1, 1] to the correct bounds
                    #rescaled_action = action * np.abs(self.action_space.low)
                    rescaled_action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(rescaled_action)
                rv_img = self.env.envs[0].recent_image_obs

                # Store transition in the replay buffer.
                #self.replay_buffer.add(obs, action, reward, new_obs, float(done))
                mini_ep_info.append([obs, action, reward/(self.n_steps*1.0), new_obs, float(done)])
                obs = new_obs

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, self.num_timesteps)

                ##############################################
                # N-step Rewards
                ##############################################
                #(Need to optimize)
                if(len(mini_ep_info) >= self.n_steps):
                    mc_reward = 0
                    mc_next_obs = mini_ep_info[-1][3]
                    gamma_next_s = 1.0
                    interrupt = 0

                    for i in range(len(mini_ep_info))[::-1]:
                        if(mini_ep_info[i][-1]):
                            #mc_reward = 0
                            #mc_next_obs = mini_ep_info[i][0]
                            mc_reward = mini_ep_info[i][2]
                            mc_next_obs = mini_ep_info[i][3]
                            gamma_next_s = 1.0
                            gamma_bs = 1.0
                            interrupt = 1
                            #self.replay_buffer.add(mini_ep_info[i][0], mini_ep_info[i][1], mini_ep_info[i][2], mini_ep_info[i][3],gamma_bs)
                        else:
                            mc_reward = mini_ep_info[i][2]+self.gamma*mc_reward
                            gamma_next_s = gamma_next_s*self.gamma
                            gamma_bs = 1.0-(gamma_next_s/self.gamma)

                    if(interrupt):
                        gamma_bs = 1.0

                    '''if self.num_timesteps < self.learning_starts:
                        mc_reward=5.0'''

                    # if self.num_timesteps < self.learning_starts:
                    if(mini_ep_info[0][-1]):					# We are looking at terminal state
                        self.replay_buffer.add(mini_ep_info[0][0], mini_ep_info[0][1], mini_ep_info[0][2], mini_ep_info[0][3],1.0)
                    else:
                        self.replay_buffer.add(mini_ep_info[0][0], mini_ep_info[0][1], mc_reward, mc_next_obs, gamma_bs)

                    mini_ep_info.pop(0)

                ##############################################
                # Training step
                ##############################################
                if step % 10000 == 0:
                    print("Training at tstep :", step)
                    print("Took", (time.time()-st)/60, "mins")

                if step % self.train_freq == 0:

                    np.save(env.log_dir+'replay_buffer.npy', np.array(self.replay_buffer._storage))

                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        if self.num_timesteps < self.batch_size:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        # Update policy and critics (q functions)
                        mb_infos_vals.append(self._train_step(step, writer, current_lr))
                        # Update target network
                        if (step + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)
                    #st = time.time()

                episode_rewards[-1] += reward
                if done:
                    if not isinstance(self.env, VecEnv):
                        obs, rv_img = self.env.reset()
                    episode_rewards.append(0.0)

                if len(episode_rewards[-501:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-501:-1])), 1)

                num_episodes = len(episode_rewards)
                self.num_timesteps += 1
                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 500 episode reward", mean_reward)
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []

                step +=1

            best_model = get_save_best_model(total_rewards, total_successes, model_file_names, path= save_file.split('sac_me')[0])            
            return best_model
    
    def predict_proba_step(self, observation, state=None, mask=None):
        observation = np.array(observation)
        

        observation = observation.reshape((-1,) + self.observation_space.shape)
        act_mean, std = self.policy_tf.proba_step(observation)
        
        return act_mean, std