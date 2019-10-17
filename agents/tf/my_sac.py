import sys
import time
import multiprocessing
from collections import deque
import warnings

import numpy as np
import tensorflow as tf

from stable_baselines import SAC
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.a2c.utils import find_trainable_variables, total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.policies import SACPolicy
from stable_baselines import logger
import matplotlib.pyplot as plt
import matplotlib
import os


def test(model, env, model_step):
    dummy_env = DummyVecEnv([lambda: env])
    # dummy_env = env
    success_episodes = 0
    results = {}
    total_reward = 0
    for ind in range(5):
        obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        obs[:] = env.reset(unseen=True, index=ind)
        done = False
        reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            info = env.step(action)
            reward += info[1]
            done = info[2]
            obs = np.expand_dims(info[0], axis=0)
        
        total_reward += reward
        if info[3]['termination_state'] == 'success':
            success_episodes += 1
            results[ind] = 1
        else:
            results[ind] = 0
    print("Results of train scenarios")
    print(results)
    print("Step: {0} Total Success Episodes: {1}".format(model_step, success_episodes))
    env.logger.log_scalar('test/success_episodes', success_episodes, model_step)
    env.logger.log_scalar('test/total_reward', total_reward, model_step)
    return total_reward, success_episodes

def plot_policy_and_value_fns(model, ind, path):
    if not os.path.exists(path):
        os.makedirs(path)
    observations = np.arange(-2, 2, 0.01).reshape((-1, 1))
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


class MY_SAC(SAC):

    def learn(self, env, total_timesteps, trained_timesteps, callback=None, seed=None,
              log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, save_file="sac_weights"):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

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
            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            n_updates = 0
            infos_values = []

            for step in range(trained_timesteps, total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy.
                if self.num_timesteps < self.learning_starts:
                    action = self.env.action_space.sample()
                    # No need to rescale when sampling random action
                    rescaled_action = action
                else:
                    action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(rescaled_action)

                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, reward, new_obs, float(done))
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

                if step % self.train_freq == 0:
                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        if self.num_timesteps < self.batch_size or self.num_timesteps < self.learning_starts:
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

                episode_rewards[-1] += reward
                if done:
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                self.num_timesteps += 1
                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
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

                if step % 10000 == 0:
                    self.save(save_file + str(step))
                    plot_policy_and_value_fns(self, step, save_file.split('sac_me')[0] + 'policy_plots/')
                    test(self, env, step)
                    
                        
            return self

    def predict_proba_step(self, observation, state=None, mask=None):
        observation = np.array(observation)
        

        observation = observation.reshape((-1,) + self.observation_space.shape)
        act_mean, std = self.policy_tf.proba_step(observation)
        
        return act_mean, std    