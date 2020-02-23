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
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger

from stable_baselines.common.vec_env import DummyVecEnv
import csv
import matplotlib.pyplot as plt

def test(model, env, model_step, path=None):
    dummy_env = DummyVecEnv([lambda: env])
    # dummy_env = env
    success_episodes = 0
    collision_episodes = 0
    results = {}
    total_reward = 0
    for ind in range(2):
        obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        obs[:] = env.reset(unseen=True, index=ind)
        done = False
        reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            info = env.step(action)
            # print(info)
            reward += info[1][0]
            done = info[2]
            obs = np.expand_dims(info[0], axis=0)
        
        total_reward += reward
        if info[3]['termination_state'] == 'success':
            success_episodes += 1
            results[ind] = 1
        else:
            results[ind] = 0
    
    # Reset env after testing
    env.reset()
    print("Results of train scenarios")
    print(results)
    print("Step: {0} Total Success Episodes: {1}".format(model_step, success_episodes))
    env.logger.log_scalar('test/success_episodes', success_episodes, model_step)
    env.logger.log_scalar('test/total_reward', total_reward, model_step)

    with open(path + 'test_results.csv','a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([model_step, success_episodes, total_reward])

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
                reset_num_timesteps=True, replay_wrapper=None, save_file="dqn_weights"):

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

                    if writer is not None:
                        ep_rew = np.array([rew]).reshape((1, -1))
                        ep_done = np.array([done]).reshape((1, -1))
                        self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                                                                        self.num_timesteps)

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
                        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                        # pytype:disable=bad-unpacking
                        if self.prioritized_replay:
                            assert self.beta_schedule is not None, \
                                "BUG: should be LinearSchedule when self.prioritized_replay True"
                            experience = self.replay_buffer.sample(self.batch_size,
                                                                beta=self.beta_schedule.value(self.num_timesteps))
                            (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                        else:
                            obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                            weights, batch_idxes = np.ones_like(rewards), None
                        # pytype:enable=bad-unpacking

                        if writer is not None:
                            # run loss backprop with summary, but once every 100 steps save the metadata
                            # (memory, compute time, ...)
                            if (1 + self.num_timesteps) % 100 == 0:
                                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                run_metadata = tf.RunMetadata()
                                summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                    dones, weights, sess=self.sess, options=run_options,
                                                                    run_metadata=run_metadata)
                                writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                            else:
                                summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                    dones, weights, sess=self.sess)
                            writer.add_summary(summary, self.num_timesteps)
                        else:
                            _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                            sess=self.sess)

                        if self.prioritized_replay:
                            new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                            assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
                            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

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