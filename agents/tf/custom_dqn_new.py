from stable_baselines import DQN

# Imports from dqn.py
from functools import partial
import tensorflow as tf
import numpy as np
import gym

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.deepq.build_graph import build_train
from stable_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from custom_replay_buffer import Custom_ReplayBuffer, Custom_PrioritizedReplayBuffer
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger

from stable_baselines.common.vec_env import DummyVecEnv
import csv, os
import matplotlib.pyplot as plt

def compute_discounted_returns(rewards, gamma):
    returns = np.zeros_like(rewards)
    n = np.size(rewards)
    
    returns[-1] = rewards[-1]
    for i in range(n-2, 0, -1):
        returns[i] = rewards[i] + gamma* returns[i+1]

    return returns


def test(model, env, model_step, path=None):
    dummy_env = DummyVecEnv([lambda: env])
    # dummy_env = env
    success_episodes = 0
    e_obs_collision = 0
    e_out_of_road = 0
    e_lane_change = 0
    e_runover_light = 0
    e_static = 0
    e_max_steps = 0
    e_unexpected_collision = 0
    e_unknown = 0
    results = {}
    total_reward = 0
    for ind in range(25):
        obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        obs[:] = env.reset(unseen=True, index=ind)
        done = False
        reward = 0
        
        q_values_matrix = []
        q_values_matrix_normalized = []
        rewards = []
        action_q_values = []
        actions_taken = []
        validation_ep_index = '0'
        while not done:
            action, q_values, actions_proba = model.predict(obs, deterministic=True)
            q_values_matrix.append(q_values[0])
            q_values_matrix_normalized.append(actions_proba[0])
            action_q_values.append(q_values[0][action])
            actions_taken.append(action)

            info = env.step(action)
            # print(info)
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
            elif info[3]['termination_state'] == 'unexpected_collision':
                e_unexpected_collision += 1
            else:
                e_unknown += 1

        action_q_values = np.array(action_q_values)
        actions_taken = np.array(actions_taken)
        returns = compute_discounted_returns(np.array(rewards), gamma=0.99)
        last_td_error = action_q_values.reshape(-1)[-1] - returns.reshape(-1)[-1] 
        
        # plot q_values for this validation episode
        plot_q_values(np.array(q_values_matrix), np.array(q_values_matrix_normalized),
        validation_ep_index, returns, action_q_values, actions_taken, path)
        

    # Reset env after testing
    env.reset()
    print("Results of train scenarios")
    print(results)
    print("Step: {0} Total Success Episodes: {1}".format(model_step, success_episodes))
    env.logger.log_scalar('test/success_episodes', success_episodes, model_step)
    env.logger.log_scalar('test/total_reward', total_reward, model_step)
    env.logger.log_scalar('test/e_obs_collision', e_obs_collision, model_step)
    env.logger.log_scalar('test/e_out_of_road', e_out_of_road, model_step)
    env.logger.log_scalar('test/e_lane_change', e_lane_change, model_step)
    env.logger.log_scalar('test/e_runover_light', e_runover_light, model_step)
    env.logger.log_scalar('test/e_static', e_static, model_step)
    env.logger.log_scalar('test/e_max_steps', e_max_steps, model_step)
    env.logger.log_scalar('test/e_unexpected_collision', e_unexpected_collision, model_step)
    env.logger.log_scalar('test/e_unknown', e_unknown, model_step)
    env.logger.log_scalar('test/last_td_error', last_td_error, model_step)

    with open(path + 'test_results.csv','a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([model_step, success_episodes, total_reward[0],
            e_obs_collision,  e_out_of_road, e_lane_change,
            e_runover_light, e_static, e_max_steps])

    return total_reward, success_episodes

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

    best_model = Custom_DQN.load(best_file_name)

    with open(path + "best_model.txt", "w") as f:
        f.write("Best model: {}\n".format(best_file_name))
        f.write("Best model appears at index: {}\n".format(ind))
        f.write("No of successes in best model: {}\n".format(total_successes[ind]))
        f.write("Max no of successes: {}\n".format(m))
        f.write("Rewards at intermediate training: {}\n".format(total_rewards))
        f.write("Total success episodes: {}\n".format(total_successes))

    return best_model

def plot_q_values(q_values_matrix, q_values_matrix_normalized, validation_ep_index,
                returns, action_q_values, actions_taken, path):
    path = os.path.join(path, 'Validation_qvalue_plots_buffer')
    if not os.path.exists(path):
        os.makedirs(path)
    # Transpose the matrix
    q_values_matrix = q_values_matrix.T
    q_values_matrix_normalized = q_values_matrix_normalized.T

    figure_name = os.path.join(path, validation_ep_index + '_qvalues.png')
    # fig, ax = plt.subplots()
    fig, (ax1, ax2, ax3, ax4, ax5)  = plt.subplots(5, 1, figsize=(12, 12))
    fig.suptitle('Q Values {}'.format(validation_ep_index))

    ax1.matshow(q_values_matrix, cmap=plt.cm.Blues, aspect='auto')
    ax2.matshow(q_values_matrix_normalized, cmap=plt.cm.Blues, aspect='auto')
    ax5.hist(q_values_matrix.reshape(-1), bins=20, density=True)
    # row, col = np.shape(q_values_matrix)
    # for i in range(row):
    #     for j in range(col):
    #         c = q_values_matrix[i,j]
    #         ax.text(j, i, str(c), va='center', ha='center')

    returns = returns.reshape(-1)
    action_q_values = action_q_values.reshape(-1)
    n = np.size(returns)
    ax3.plot(np.arange(n), np.array(returns),'b')
    ax3.plot(np.arange(n), np.array(action_q_values),'g')
    ax3.legend('r', 'q', loc='best')
    ax3.set_xlim(left=0, right=n)

    actions_taken = actions_taken.reshape(-1)
    ax4.plot(np.arange(n), np.array(actions_taken),'b')
    ax4.set_xlim(left=0, right=n)

    ax3.set_ylabel('returns(blue)')
    ax4.set_ylabel('actions')
    ax5.set_ylabel('Q histogram')

    ax1.set_ylabel('Actions')
    
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Actions (Normalized))')
    
    plt.savefig(figure_name)
    plt.close()

    # saving arrays    
    fpath_array = os.path.join(path, validation_ep_index + '_qvalues_array')
    np.savez(fpath_array, q_values_matrix= q_values_matrix, q_values_matrix_normalized=q_values_matrix_normalized,
             returns=returns, action_q_values=action_q_values, actions_taken=actions_taken)


def plot_test_results(total_successes, total_rewards, total_updates, path):
    fig, (ax1, ax2)  = plt.subplots(1, 2)
    fig.suptitle('Test Results v/s training timesteps')

    ax1.plot(np.array(total_updates), np.array(total_successes), color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Success Episodes')
    ax2.plot(np.array(total_updates), np.array(total_rewards), color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    ax2.set_xlabel('Total Reward')
    ax2.set_ylabel('Timesteps')
    
    ax1.grid(True)
    ax2.grid(True)
    
    plt.grid(True)
    plt.savefig(path + 'test_results.png')
    plt.close()
class Custom_DQN(DQN):
    '''
    Custom_DQN class with addition of testing within training in DQN class.
    '''

    def learn(self, env, total_timesteps, callback=None, log_interval=100, tb_log_name="DQN",
                reset_num_timesteps=True, replay_wrapper=None, save_file="dqn_weights", num_opt_epochs=5):

            new_tb_log = self._init_num_timesteps(reset_num_timesteps)

            # Custom: Arrays for storing info across training
            total_rewards = []
            total_successes = []
            model_file_names = []
            total_updates = []

            with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                    as writer:
                self._setup_learn()

                # Create the replay buffer
                if self.prioritized_replay:
                    self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                    if self.prioritized_replay_beta_iters is None:
                        prioritized_replay_beta_iters = total_timesteps
                    else:
                        prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                    self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                        initial_p=self.prioritized_replay_beta0,
                                                        final_p=1.0)
                else:
                    self.replay_buffer = ReplayBuffer(self.buffer_size)
                    self.beta_schedule = None

                if replay_wrapper is not None:
                    assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                    self.replay_buffer = replay_wrapper(self.replay_buffer)

                # Create the schedule for exploration starting from 1.
                self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                                initial_p=self.exploration_initial_eps,
                                                final_p=self.exploration_final_eps)

                episode_rewards = [0.0]
                episode_successes = []
                obs = self.env.reset()
                reset = True
                self.episode_reward = np.zeros((1,))

                for _ in range(total_timesteps):
                    if callback is not None:
                        # Only stop training if return value is False, not when it is None. This is for backwards
                        # compatibility with callbacks that have no return statement.
                        if callback(locals(), globals()) is False:
                            break
                    
                    # Custom: model test and save logic
                    MODEL_SAVE_FREQ = 10000
                    if self.num_timesteps % MODEL_SAVE_FREQ == 0:
                        
                        # save less frequently than testing
                        if (self.num_timesteps % (MODEL_SAVE_FREQ * 50)) == 0:
                            self.save(save_file + str(self.num_timesteps))
                        
                        total_reward, success_episodes = test(self, env, self.num_timesteps, save_file.split('dqn_me')[0])
                        total_rewards.append(total_reward)
                        total_successes.append(success_episodes)
                        model_file_names.append(save_file + str(self.num_timesteps))
                        total_updates.append(self.num_timesteps)                    
                        plot_test_results(total_successes, total_rewards, total_updates, save_file.split('dqn_me')[0])
                    
                    # Take action and update exploration to the newest value
                    kwargs = {}
                    if not self.param_noise:
                        update_eps = self.exploration.value(self.num_timesteps)
                        update_param_noise_threshold = 0.
                    else:
                        update_eps = 0.
                        # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                        # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                        # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                        # for detailed explanation.
                        update_param_noise_threshold = \
                            -np.log(1. - self.exploration.value(self.num_timesteps) +
                                    self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                        kwargs['reset'] = reset
                        kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                        kwargs['update_param_noise_scale'] = True
                    with self.sess.as_default():
                        action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                    env_action = action
                    reset = False
                    new_obs, rew, done, info = self.env.step(env_action)
                    # Store transition in the replay buffer.
                    self.replay_buffer.add(obs, action, rew, new_obs, float(done))
                    
                    obs = new_obs

                    # if writer is not None:
                    #     ep_rew = np.array([rew]).reshape((1, -1))
                    #     ep_done = np.array([done]).reshape((1, -1))
                    #     self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                    #                                                     self.num_timesteps)

                    episode_rewards[-1] += rew
                    if done:
                        maybe_is_success = info.get('is_success')
                        if maybe_is_success is not None:
                            episode_successes.append(float(maybe_is_success))
                        if not isinstance(self.env, VecEnv):
                            obs = self.env.reset()
                        episode_rewards.append(0.0)
                        reset = True

                    # Do not train if the warmup phase is not over
                    # or if there are not enough samples in the replay buffer
                    can_sample = self.replay_buffer.can_sample(self.batch_size)
                    if can_sample and self.num_timesteps > self.learning_starts \
                            and self.num_timesteps % self.train_freq == 0:

                        # Running training optimizations for num_opt_epochs


                        for i_opt in range(num_opt_epochs):

                            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                            # pytype:disable=bad-unpacking
                            if self.prioritized_replay:
                                assert self.beta_schedule is not None, \
                                    "BUG: should be LinearSchedule when self.prioritized_replay True"
                                experience = self.replay_buffer.sample(self.batch_size,
                                                                    beta=self.beta_schedule.value(self.num_timesteps))
                                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                                # (obses_t, actions, rewards, obses_tp1, dones, infos, _, weights, batch_idxes) = experience
                            else:
                                obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                                # obses_t, actions, rewards, obses_tp1, dones, infos, _ = self.replay_buffer.sample(self.batch_size)
                                weights, batch_idxes = np.ones_like(rewards), None
                            # pytype:enable=bad-unpacking

                            summary = None
                            run_metadata = None
                            if writer is not None:
                                # run loss backprop with summary, but once every 100 steps save the metadata
                                # (memory, compute time, ...)
                                if (1 + self.num_timesteps) % 10000 == 0:
                                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                    run_metadata = tf.RunMetadata()
                                    summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                        dones, weights, sess=self.sess, options=run_options,
                                                                        run_metadata=run_metadata)
                                    # writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                                else:
                                    summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                        dones, weights, sess=self.sess)
                                # Removing summary from here and adding it after all optimizations in this training step
                                # writer.add_summary(summary, self.num_timesteps)
                            else:
                                _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                                sess=self.sess)

                            if self.prioritized_replay:
                                new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                                assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
                                self.replay_buffer.update_priorities(batch_idxes, new_priorities)
                        
                        # Adding summary after all optimizations in this training step
                        if writer is not None and summary is not None:
                            writer.add_summary(summary, self.num_timesteps)

                            if (1 + self.num_timesteps) % 10000 == 0 and run_metadata is not None:
                                writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)

                    if can_sample and self.num_timesteps > self.learning_starts and \
                            self.num_timesteps % self.target_network_update_freq == 0:
                        # Update target network periodically.
                        self.update_target(sess=self.sess)

                    if len(episode_rewards[-101:-1]) == 0:
                        mean_100ep_reward = -np.inf
                    else:
                        mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                    num_episodes = len(episode_rewards)
                    if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                        logger.record_tabular("steps", self.num_timesteps)
                        logger.record_tabular("episodes", num_episodes)
                        if len(episode_successes) > 0:
                            logger.logkv("success rate", np.mean(episode_successes[-100:]))
                        logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                        logger.record_tabular("% time spent exploring",
                                            int(100 * self.exploration.value(self.num_timesteps)))
                        logger.dump_tabular()

                    self.num_timesteps += 1

            # Custom: get and save best model
            best_model = get_save_best_model(total_rewards, total_successes, model_file_names, path= save_file.split('dqn_me')[0])
            return best_model
            # return self
    
    def learn_new(self, env, total_timesteps, callback=None, log_interval=100, tb_log_name="DQN",
                reset_num_timesteps=True, replay_wrapper=None, save_file="dqn_weights", num_opt_epochs=5):

            '''
            This method includes saving additional information of termination_state_code, time_to_termination
            for each transition in replay buffer. This is the only difference from learn() method.
            '''
            
            new_tb_log = self._init_num_timesteps(reset_num_timesteps)

            # Custom: Arrays for storing info across training
            total_rewards = []
            total_successes = []
            model_file_names = []
            total_updates = []

            with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                    as writer:
                self._setup_learn()

                # Create the replay buffer if not already created /loaded from existing model
                if self.replay_buffer is None:
                    if self.prioritized_replay:
                        self.replay_buffer = Custom_PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                        if self.prioritized_replay_beta_iters is None:
                            prioritized_replay_beta_iters = total_timesteps
                        else:
                            prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                        self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                            initial_p=self.prioritized_replay_beta0,
                                                            final_p=1.0)
                    else:
                        self.replay_buffer = Custom_ReplayBuffer(self.buffer_size)
                        self.beta_schedule = None

                if replay_wrapper is not None:
                    assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                    self.replay_buffer = replay_wrapper(self.replay_buffer)

                # Create the schedule for exploration starting from 1.
                self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                                initial_p=self.exploration_initial_eps,
                                                final_p=self.exploration_final_eps)

                episode_rewards = [0.0]
                episode_successes = []
                obs = self.env.reset()
                reset = True
                self.episode_reward = np.zeros((1,))

                episode_t = 0
                exp_list = []

                for _ in range(total_timesteps):
                    if callback is not None:
                        # Only stop training if return value is False, not when it is None. This is for backwards
                        # compatibility with callbacks that have no return statement.
                        if callback(locals(), globals()) is False:
                            break
                    
                    # Custom: model test and save logic
                    MODEL_SAVE_FREQ = 10000
                    if self.num_timesteps % MODEL_SAVE_FREQ == 0:
                        
                        # save less frequently than testing
                        if (self.num_timesteps % (MODEL_SAVE_FREQ * 50)) == 0:
                            self.save(save_file + str(self.num_timesteps))
                        
                        total_reward, success_episodes = test(self, env, self.num_timesteps, save_file.split('dqn_me')[0])
                        total_rewards.append(total_reward)
                        total_successes.append(success_episodes)
                        model_file_names.append(save_file + str(self.num_timesteps))
                        total_updates.append(self.num_timesteps)                    
                        plot_test_results(total_successes, total_rewards, total_updates, save_file.split('dqn_me')[0])
                    
                    # Take action and update exploration to the newest value
                    kwargs = {}
                    if not self.param_noise:
                        update_eps = self.exploration.value(self.num_timesteps)
                        update_param_noise_threshold = 0.
                    else:
                        update_eps = 0.
                        # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                        # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                        # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                        # for detailed explanation.
                        update_param_noise_threshold = \
                            -np.log(1. - self.exploration.value(self.num_timesteps) +
                                    self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                        kwargs['reset'] = reset
                        kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                        kwargs['update_param_noise_scale'] = True
                    with self.sess.as_default():
                        action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                    env_action = action
                    reset = False
                    new_obs, rew, done, info = self.env.step(env_action)
                    
                    exp_t = (obs, action, rew, new_obs, float(done), info['termination_state_code'])
                    exp_list.append(exp_t)
                    obs = new_obs
                    episode_t  = episode_t  + 1

                    # if writer is not None:
                    #     ep_rew = np.array([rew]).reshape((1, -1))
                    #     ep_done = np.array([done]).reshape((1, -1))
                    #     self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                    #                                                     self.num_timesteps)

                    episode_rewards[-1] += rew
                    if done:
                        maybe_is_success = info.get('is_success')
                        if maybe_is_success is not None:
                            episode_successes.append(float(maybe_is_success))
                        if not isinstance(self.env, VecEnv):
                            obs = self.env.reset()
                        episode_rewards.append(0.0)
                        reset = True

                        time_to_termination = 0
                        for i in range(episode_t -1, -1, -1):
                            obs, action, rew, new_obs, done, termination_state_code = exp_list[i]
                            self.replay_buffer.add(obs, action, rew, new_obs, done, termination_state_code, time_to_termination)
                            time_to_termination = time_to_termination + 1
                        
                        episode_t = 0
                        exp_list = []


                    # Do not train if the warmup phase is not over
                    # or if there are not enough samples in the replay buffer
                    can_sample = self.replay_buffer.can_sample(self.batch_size)
                    if can_sample and self.num_timesteps > self.learning_starts \
                            and self.num_timesteps % self.train_freq == 0:

                        # Running training optimizations for num_opt_epochs


                        for i_opt in range(num_opt_epochs):

                            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                            # pytype:disable=bad-unpacking
                            if self.prioritized_replay:
                                assert self.beta_schedule is not None, \
                                    "BUG: should be LinearSchedule when self.prioritized_replay True"
                                experience = self.replay_buffer.sample(self.batch_size,
                                                                    beta=self.beta_schedule.value(self.num_timesteps))
                                # (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                                (obses_t, actions, rewards, obses_tp1, dones, infos, _, weights, batch_idxes) = experience
                            else:
                                obses_t, actions, rewards, obses_tp1, dones, infos, _ = self.replay_buffer.sample(self.batch_size)
                                weights, batch_idxes = np.ones_like(rewards), None
                                
                            # pytype:enable=bad-unpacking

                            summary = None
                            run_metadata = None
                            if writer is not None:
                                # run loss backprop with summary, but once every 100 steps save the metadata
                                # (memory, compute time, ...)
                                if (1 + self.num_timesteps) % 10000 == 0:
                                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                    run_metadata = tf.RunMetadata()
                                    summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                        dones, weights, sess=self.sess, options=run_options,
                                                                        run_metadata=run_metadata)
                                    # writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                                else:
                                    summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                        dones, weights, sess=self.sess)
                                # Removing summary from here and adding it after all optimizations in this training step
                                # writer.add_summary(summary, self.num_timesteps)
                            else:
                                _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                                sess=self.sess)

                            if self.prioritized_replay:
                                new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                                assert isinstance(self.replay_buffer, Custom_PrioritizedReplayBuffer)
                                self.replay_buffer.update_priorities(batch_idxes, new_priorities)
                        
                        # Adding summary after all optimizations in this training step
                        if writer is not None and summary is not None:
                            writer.add_summary(summary, self.num_timesteps)

                            if (1 + self.num_timesteps) % 10000 == 0 and run_metadata is not None:
                                writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)

                    if can_sample and self.num_timesteps > self.learning_starts and \
                            self.num_timesteps % self.target_network_update_freq == 0:
                        # Update target network periodically.
                        self.update_target(sess=self.sess)

                    if len(episode_rewards[-101:-1]) == 0:
                        mean_100ep_reward = -np.inf
                    else:
                        mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                    num_episodes = len(episode_rewards)
                    if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                        logger.record_tabular("steps", self.num_timesteps)
                        logger.record_tabular("episodes", num_episodes)
                        if len(episode_successes) > 0:
                            logger.logkv("success rate", np.mean(episode_successes[-100:]))
                        logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                        logger.record_tabular("% time spent exploring",
                                            int(100 * self.exploration.value(self.num_timesteps)))
                        logger.dump_tabular()

                    self.num_timesteps += 1

            # Custom: get and save best model
            best_model = get_save_best_model(total_rewards, total_successes, model_file_names, path= save_file.split('dqn_me')[0])
            return best_model

    def learn_new_buffer(self, env, total_timesteps, callback=None, log_interval=100, tb_log_name="DQN",
                reset_num_timesteps=True, replay_wrapper=None, save_file="dqn_weights", num_opt_epochs=5):

            '''

            This learn method includes special sampling logic for replay buffer,
            which samples transitions with less time_to_termination more often.

            '''
            
            new_tb_log = self._init_num_timesteps(reset_num_timesteps)

            # Custom: Arrays for storing info across training
            total_rewards = []
            total_successes = []
            model_file_names = []
            total_updates = []

            with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                    as writer:
                self._setup_learn()

                # Create the replay buffer if not already created /loaded from existing model
                if self.replay_buffer is None:
                    if self.prioritized_replay:
                        self.replay_buffer = Custom_PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                        if self.prioritized_replay_beta_iters is None:
                            prioritized_replay_beta_iters = total_timesteps
                        else:
                            prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                        self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                            initial_p=self.prioritized_replay_beta0,
                                                            final_p=1.0)
                    else:
                        self.replay_buffer = Custom_ReplayBuffer(self.buffer_size)
                        self.beta_schedule = None

                if replay_wrapper is not None:
                    assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                    self.replay_buffer = replay_wrapper(self.replay_buffer)

                # Create the schedule for exploration starting from 1.
                self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                                initial_p=self.exploration_initial_eps,
                                                final_p=self.exploration_final_eps)

                episode_rewards = [0.0]
                episode_successes = []
                obs = self.env.reset()
                reset = True
                self.episode_reward = np.zeros((1,))

                episode_t = 0
                exp_list = []

                for _ in range(total_timesteps):
                    if callback is not None:
                        # Only stop training if return value is False, not when it is None. This is for backwards
                        # compatibility with callbacks that have no return statement.
                        if callback(locals(), globals()) is False:
                            break
                    
                    # Custom: model test and save logic
                    MODEL_SAVE_FREQ = 10000
                    if self.num_timesteps % MODEL_SAVE_FREQ == 0:
                        
                        # save less frequently than testing
                        if (self.num_timesteps % (MODEL_SAVE_FREQ * 50)) == 0:
                            self.save(save_file + str(self.num_timesteps))
                        
                        total_reward, success_episodes = test(self, env, self.num_timesteps, save_file.split('dqn_me')[0])
                        total_rewards.append(total_reward)
                        total_successes.append(success_episodes)
                        model_file_names.append(save_file + str(self.num_timesteps))
                        total_updates.append(self.num_timesteps)                    
                        plot_test_results(total_successes, total_rewards, total_updates, save_file.split('dqn_me')[0])
                    
                    # Take action and update exploration to the newest value
                    kwargs = {}
                    if not self.param_noise:
                        update_eps = self.exploration.value(self.num_timesteps)
                        update_param_noise_threshold = 0.
                    else:
                        update_eps = 0.
                        # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                        # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                        # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                        # for detailed explanation.
                        update_param_noise_threshold = \
                            -np.log(1. - self.exploration.value(self.num_timesteps) +
                                    self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                        kwargs['reset'] = reset
                        kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                        kwargs['update_param_noise_scale'] = True
                    with self.sess.as_default():
                        action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                    env_action = action
                    reset = False
                    new_obs, rew, done, info = self.env.step(env_action)
                    
                    exp_t = (obs, action, rew, new_obs, float(done), info['termination_state_code'])
                    exp_list.append(exp_t)
                    obs = new_obs
                    episode_t  = episode_t  + 1

                    # if writer is not None:
                    #     ep_rew = np.array([rew]).reshape((1, -1))
                    #     ep_done = np.array([done]).reshape((1, -1))
                    #     self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                    #                                                     self.num_timesteps)

                    episode_rewards[-1] += rew
                    if done:
                        maybe_is_success = info.get('is_success')
                        if maybe_is_success is not None:
                            episode_successes.append(float(maybe_is_success))
                        if not isinstance(self.env, VecEnv):
                            obs = self.env.reset()
                        episode_rewards.append(0.0)
                        reset = True

                        time_to_termination = 0
                        for i in range(episode_t -1, -1, -1):
                            obs, action, rew, new_obs, done, termination_state_code = exp_list[i]
                            self.replay_buffer.add(obs, action, rew, new_obs, done, termination_state_code, time_to_termination)
                            time_to_termination = time_to_termination + 1
                        
                        episode_t = 0
                        exp_list = []


                    # Do not train if the warmup phase is not over
                    # or if there are not enough samples in the replay buffer
                    can_sample = self.replay_buffer.can_sample(self.batch_size)
                    if can_sample and self.num_timesteps > self.learning_starts \
                            and self.num_timesteps % self.train_freq == 0:

                        # Running training optimizations for num_opt_epochs


                        for i_opt in range(num_opt_epochs):

                            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                            # pytype:disable=bad-unpacking
                            if self.prioritized_replay:
                                assert self.beta_schedule is not None, \
                                    "BUG: should be LinearSchedule when self.prioritized_replay True"
                                experience = self.replay_buffer.sample(self.batch_size,
                                                                    beta=self.beta_schedule.value(self.num_timesteps))
                                # (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                                (obses_t, actions, rewards, obses_tp1, dones, infos, _, weights, batch_idxes) = experience
                            else:
                                # obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                                concatenate = False
                                if len(self.replay_buffer._time_to_termination_idx[0]) > 64:
                                    batch_size_t = 384
                                    batch_size_t0 = 64
                                    batch_size_t1 = 32
                                    batch_size_t2 = 32
                                    concatenate = True
                                else:
                                    batch_size_t = 512
                                    batch_size_t0 = 0
                                    batch_size_t1 = 0
                                    batch_size_t2 = 0
                                    concatenate = False

                                obses_t_t, actions_t, rewards_t, obses_tp1_t, dones_t, infos_t, _ = self.replay_buffer.sample(batch_size_t)
                                weights_t, batch_idxes_t = np.ones_like(rewards_t), None

                                obses_t_t0, actions_t0, rewards_t0, obses_tp1_t0, dones_t0, _, _ = self.replay_buffer.sample_time_to_termination(batch_size_t0, [0])
                                weights_t0, batch_idxes_t0 = np.ones_like(rewards_t0), None

                                obses_t_t1, actions_t1, rewards_t1, obses_tp1_t1, dones_t1, _, _ = self.replay_buffer.sample_time_to_termination(batch_size_t1, [1])
                                weights_t1, batch_idxes_t1 = np.ones_like(rewards_t1), None

                                obses_t_t2, actions_t2, rewards_t2, obses_tp1_t2, dones_t2, _, _ = self.replay_buffer.sample_time_to_termination(batch_size_t2, [2])
                                weights_t2, batch_idxes_t2 = np.ones_like(rewards_t2), None
                                
                                # import pdb
                                # pdb.set_trace()
                                if concatenate:
                                    obses_t = np.concatenate((obses_t_t, obses_t_t0, obses_t_t1, obses_t_t2))
                                    actions = np.concatenate((actions_t, actions_t0, actions_t1, actions_t2))
                                    rewards = np.concatenate((rewards_t, rewards_t0, rewards_t1, rewards_t2))
                                    obses_tp1 = np.concatenate((obses_tp1_t, obses_tp1_t0, obses_tp1_t1, obses_tp1_t2))
                                    dones = np.concatenate((dones_t, dones_t0, dones_t1, dones_t2))
                                    
                                    weights = np.concatenate((weights_t, weights_t0, weights_t1, weights_t2))
                                    batch_idxes = None
                                else:
                                    obses_t = obses_t_t
                                    actions = actions_t
                                    rewards = rewards_t
                                    obses_tp1 = obses_tp1_t
                                    dones = dones_t
                                    weights = weights_t
                                    batch_idxes = None
                                
                            # pytype:enable=bad-unpacking

                            summary = None
                            run_metadata = None
                            if writer is not None:
                                # run loss backprop with summary, but once every 100 steps save the metadata
                                # (memory, compute time, ...)
                                if (1 + self.num_timesteps) % 10000 == 0:
                                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                    run_metadata = tf.RunMetadata()
                                    summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                        dones, weights, sess=self.sess, options=run_options,
                                                                        run_metadata=run_metadata)
                                    # writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                                else:
                                    summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                        dones, weights, sess=self.sess)
                                # Removing summary from here and adding it after all optimizations in this training step
                                # writer.add_summary(summary, self.num_timesteps)
                            else:
                                _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                                sess=self.sess)

                            if self.prioritized_replay:
                                new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                                assert isinstance(self.replay_buffer, Custom_PrioritizedReplayBuffer)
                                self.replay_buffer.update_priorities(batch_idxes, new_priorities)
                        
                        # Adding summary after all optimizations in this training step
                        if writer is not None and summary is not None:
                            writer.add_summary(summary, self.num_timesteps)

                            if (1 + self.num_timesteps) % 10000 == 0 and run_metadata is not None:
                                writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)

                    if can_sample and self.num_timesteps > self.learning_starts and \
                            self.num_timesteps % self.target_network_update_freq == 0:
                        # Update target network periodically.
                        self.update_target(sess=self.sess)

                    if len(episode_rewards[-101:-1]) == 0:
                        mean_100ep_reward = -np.inf
                    else:
                        mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                    num_episodes = len(episode_rewards)
                    if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                        logger.record_tabular("steps", self.num_timesteps)
                        logger.record_tabular("episodes", num_episodes)
                        if len(episode_successes) > 0:
                            logger.logkv("success rate", np.mean(episode_successes[-100:]))
                        logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                        logger.record_tabular("% time spent exploring",
                                            int(100 * self.exploration.value(self.num_timesteps)))
                        logger.dump_tabular()

                    self.num_timesteps += 1

            # Custom: get and save best model
            best_model = get_save_best_model(total_rewards, total_successes, model_file_names, path= save_file.split('dqn_me')[0])
            return best_model

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            actions, q_values, actions_proba = self.step_model.step(observation, deterministic=deterministic)

        if not vectorized_env:
            actions = actions[0]

        return actions, q_values, actions_proba
    
    def save(self, save_path, cloudpickle=False):
        # params
        data = {
            "double_q": self.double_q,
            "param_noise": self.param_noise,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "prioritized_replay": self.prioritized_replay,
            "prioritized_replay_eps": self.prioritized_replay_eps,
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,
            "prioritized_replay_alpha": self.prioritized_replay_alpha,
            "prioritized_replay_beta0": self.prioritized_replay_beta0,
            "prioritized_replay_beta_iters": self.prioritized_replay_beta_iters,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "replay_buffer" : self.replay_buffer
        }

        # np.savez(save_path + '_buffer', replay_buffer=np.array(self.replay_buffer._storage))
        # np.savez_compressed(save_path + '_buffer_compressed', replay_buffer=np.array(self.replay_buffer._storage))
        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

    def learn_from_buffer(self, env, total_timesteps, callback=None, log_interval=100, tb_log_name="DQN",
                reset_num_timesteps=True, replay_wrapper=None, save_file="dqn_weights",
                sample_done=False, sample_termination_state_list=[1], sample_time_to_termination_time_list=None):

            '''

            This method is used to learn only from buffer (supervised training).
            It was written for debugging purposes.
            '''

            new_tb_log = self._init_num_timesteps(reset_num_timesteps)

            # Custom: Arrays for storing info across training
            total_rewards = []
            total_successes = []
            model_file_names = []
            total_updates = []

            with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                    as writer:
                # self._setup_learn()

                # self.replay_buffer = Custom_ReplayBuffer(self.replay_buffer._maxsize, self.replay_buffer._storage, self.replay_buffer._next_idx)
                for _ in range(total_timesteps):
                    
                    # Custom: model test and save logic
                    MODEL_SAVE_FREQ = 1000
                    if self.num_timesteps % MODEL_SAVE_FREQ == 0:
                        
                        # save less frequently than testing
                        # if self.num_timesteps > 0 and (self.num_timesteps % MODEL_SAVE_FREQ * 5) == 0:
                        #     self.save(save_file + str(self.num_timesteps))
                        
                        total_reward, success_episodes = test(self, env, self.num_timesteps, save_file.split('dqn_me')[0])
                        total_rewards.append(total_reward)
                        total_successes.append(success_episodes)
                        model_file_names.append(save_file + str(self.num_timesteps))
                        total_updates.append(self.num_timesteps)                    
                        plot_test_results(total_successes, total_rewards, total_updates, save_file.split('dqn_me')[0])
                    

                    # Currently hard-coded. Change if needed
                    self.batch_size = 100

                    # Do not train if the warmup phase is not over
                    # or if there are not enough samples in the replay buffer
                    can_sample = self.replay_buffer.can_sample(self.batch_size)
                    if can_sample:

                        # Running training optimizations for num_opt_epochs

                        # obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                        # obses_t, actions, rewards, obses_tp1, dones, infos = self.replay_buffer.sample(self.batch_size)
                        # import pdb
                        # pdb.set_trace()

                        if sample_done:
                            obses_t, actions, rewards, obses_tp1, dones, term_states, time_to_terminations = self.replay_buffer.sample_done(self.batch_size)
                        elif sample_termination_state_list is not None:
                            obses_t, actions, rewards, obses_tp1, dones, term_states, time_to_terminations = self.replay_buffer.sample_done_term_state(self.batch_size, sample_termination_state_list)
                        elif sample_time_to_termination_time_list is not None:
                            obses_t, actions, rewards, obses_tp1, dones, term_states, time_to_terminations = self.replay_buffer.sample_time_to_termination(self.batch_size, sample_time_to_termination_time_list)
                        else:
                            assert(False)

                        weights, batch_idxes = np.ones_like(rewards), None
                        # pytype:enable=bad-unpacking

                        summary = None
                        run_metadata = None
                        if writer is not None:
                            # run loss backprop with summary, but once every 100 steps save the metadata
                            # (memory, compute time, ...)
                            if (1 + self.num_timesteps) % 10000 == 0:
                                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                run_metadata = tf.RunMetadata()
                                summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                    dones, weights, sess=self.sess, options=run_options,
                                                                    run_metadata=run_metadata)
                                # writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                            else:
                                summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                    dones, weights, sess=self.sess)
                            # Removing summary from here and adding it after all optimizations in this training step
                            # writer.add_summary(summary, self.num_timesteps)
                        else:
                            _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                            sess=self.sess)

                        # Adding summary after all optimizations in this training step
                        if writer is not None and summary is not None:
                            writer.add_summary(summary, self.num_timesteps)

                            if (1 + self.num_timesteps) % 10000 == 0 and run_metadata is not None:
                                writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)                    
                    
                    if self.num_timesteps % 100 == 0:
                        env.logger.log_scalar('test_buffer/td_error', np.mean(td_errors), self.num_timesteps)
                    
                    print("timesteps, td_error", self.num_timesteps, np.mean(td_errors))
                    self.num_timesteps += 1

            return self
   
