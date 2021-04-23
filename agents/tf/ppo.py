"""PPO optimizer with VAE dimensionality reduction of the input images"""

import time
import numpy as np
from mpi4py import MPI
import sys
import gym
from collections import deque
import os
import sys
import csv
import multiprocessing
from collections import deque, OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from stable_baselines.common.policies import RecurrentActorCriticPolicy
from stable_baselines import logger
from stable_baselines.common import explained_variance, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.ppo2.ppo2 import PPO2, Runner
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.a2c.utils import total_episode_reward_logger


# change
PATH_MODEL_VAE = "ppo_vae_turn_rgb3_test.json"
PATH_MODEL_PPO2 = "ppo_carla_turn_rgb_corl3_test"

def make_carla_env():
    """Import the package for carla Env, this packge calls the __init__ that registers the environment.Did this just to
    be consistent with gym"""
    sys.path.append('/home/frcvision1/Final/My_Environments/')
    import Carla

    host = 'localhost'
    port = 2000
    env = gym.make('CarlaEnv-v0')
    env = DummyVecEnv([lambda: env])
    return env, host, port


def get_schedule_fn(value_schedule):
    """
    Transform (if needed) learning rate and clip range
    to callable.

    :param value_schedule: (callable or float)
    :return: (function)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, float):
        value_schedule = constfn(value_schedule)
    else:
        assert callable(value_schedule)
    return value_schedule


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def constfn(val):
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: (float)
    :return: (function)
    """

    def func(_):
        return val

    return func


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)

def plot_policy_and_value_fns(model, ind, path):
    if not os.path.exists(path):
        os.makedirs(path)
    observations = np.arange(-2, 2, 0.01).reshape((-1, 1))
    det_actions = []
    stoch_actions = []
    var_actions = []
    values = []

    for i in range(observations.shape[0]):
        obs = observations[np.newaxis, i, :]
        act, value, _, _, _, _ = model.predict(obs, deterministic=True)
        act[0, 1] = (act[0, 1] + 1) * 10.0
        det_actions.append(act)
        act, _, _, _, logstd, _ = model.predict(obs, deterministic=False)
        act[0, 1] = (act[0, 1] + 1) * 10.0
        stoch_actions.append(act)
        var_actions.append(logstd)
        values.append(value)

    det_actions = np.array(det_actions).reshape((-1, 2))
    stoch_actions = np.array(stoch_actions).reshape((-1, 2))
    var_actions = np.exp(np.array(var_actions).reshape((-1, 2)))
    values = np.array(values).reshape((-1, 1))

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
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

    axs[2, 0].plot(observations, stoch_actions[:, 0], color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[2, 0].set_xlabel('Waypoint orientation')
    axs[2, 0].set_ylabel('Stochastic - Steer')
    axs[2, 1].plot(observations, stoch_actions[:, 1], color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    axs[2, 1].set_xlabel('Waypoint orientation')
    axs[2, 1].set_ylabel('Stochastic - Target Speed')

    axs[0,0].grid(True)
    axs[0,1].grid(True)
    axs[1,0].grid(True)
    axs[1,1].grid(True)
    axs[2,0].grid(True)
    axs[2,1].grid(True)

    plt.grid(True)
    plt.savefig(path + 'policy_{}.png'.format(ind))
    plt.close()

    plt.figure()
    plt.plot(observations, values, color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Waypoint orientation')
    plt.ylabel('Value')
    fig.suptitle('Valuex plots for {} model'.format(ind))
    plt.savefig(path + 'value_{}.png'.format(ind))

def plot_test_results(total_successes, total_rewards, total_updates, path):
    fig, (ax1, ax2)  = plt.subplots(1, 2)
    fig.suptitle('Test Results v/s training timesteps')

    ax1.plot(np.array(total_updates) / 1000, np.array(total_successes), color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    ax1.set_xlabel('Timesteps (k)')
    ax1.set_ylabel('Success Episodes')
    ax2.plot(np.array(total_updates) / 1000, np.array(total_rewards), color='#bd83ce', linestyle='-', linewidth=2, markersize=8)
    ax2.set_xlabel('Total Reward')
    ax2.set_ylabel('Timesteps (k)')

    ax1.grid(True)
    ax2.grid(True)

    plt.grid(True)
    plt.savefig(path + 'test_results.png')
    plt.close()

class OverideRunner(Runner):

    def run(self, logger=None):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs, logstd, mean = self.model.step(self.obs, self.states, self.dones)
            # import pdb; pdb.set_trace()
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)

            if logger is not None:
                steer_mean = mean[0][0]
                steer_std = np.exp(logstd[0][0])

                tspeed_mean = mean[0][1]
                tspeed_std = np.exp(logstd[0][1])

                total_steps = infos[0]['total_steps']

                # logger.log_scalar('train/steer_mean', steer_mean, total_steps)
                # logger.log_scalar('train/steer_std', steer_std, total_steps)
                # logger.log_scalar('train/tspeed_mean', tspeed_mean, total_steps)
                # logger.log_scalar('train/tspeed_std', tspeed_std, total_steps)

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward

def test(model, env, dump_results=False, path='.', model_step=None):
    dummy_env = DummyVecEnv([lambda: env])
    success_episodes = 0
    collision_obs_episodes = 0
    collision_lane_change_episodes = 0
    collision_out_of_road_episodes = 0
    collision_unexpected_episodes = 0
    runover_light_episodes = 0
    max_steps_episodes = 0
    max_steps_obs_episodes = 0
    max_steps_light_episodes = 0
    static_episodes = 0
    unknown_episodes = 0
    results = {}
    total_reward = 0
    env.reset()
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
            if info[3]['termination_state'] == 'obs_collision':
                collision_obs_episodes += 1
            elif info[3]['termination_state'] == 'lane_invasion':
                collision_lane_change_episodes += 1
            elif info[3]['termination_state'] == 'out_of_road':
                collision_out_of_road_episodes += 1
            elif info[3]['termination_state'] == 'unexpected_collision':
                collision_unexpected_episodes += 1
            elif info[3]['termination_state'] == 'runover_light':
                runover_light_episodes += 1
            elif info[3]['termination_state'] == 'max_steps':
                max_steps_episodes += 1
            elif info[3]['termination_state'] == 'max_steps_obstacle':
                max_steps_obs_episodes += 1
            elif info[3]['termination_state'] == 'max_steps_light':
                max_steps_light_episodes += 1
            elif info[3]['termination_state'] == 'static':
                static_episodes += 1
            else:
                unknown_episodes += 1

    success_episodes += max_steps_obs_episodes
    env.reset()
    print("Results of test scenarios")
    print(results)
    print("# Success: {}, # Obstacle Collision: {}, # Lane-change Collision: {}, Out-of-road Collision: {}, Unexpected Collision: {}, Runover light: {}, Max_steps: {}, Max_steps Obstacle: {}, Max_steps Traffic Light: {}, Static: {}, Unknown: {}".format(success_episodes,
                collision_obs_episodes, collision_lane_change_episodes, collision_out_of_road_episodes, collision_unexpected_episodes, runover_light_episodes, max_steps_episodes, max_steps_obs_episodes, max_steps_light_episodes, static_episodes, unknown_episodes))
    data = [model_step, success_episodes, total_reward, collision_obs_episodes, collision_lane_change_episodes, collision_out_of_road_episodes, collision_unexpected_episodes,
                                    runover_light_episodes, max_steps_episodes, max_steps_obs_episodes, max_steps_light_episodes, static_episodes, unknown_episodes]
    if dump_results and env.logger is not None:
        env.logger.log_scalar('test/term_success', success_episodes, model_step)
        env.logger.log_scalar('test/term_obstacle', collision_obs_episodes, model_step)
        env.logger.log_scalar('test/term_lane_change', collision_lane_change_episodes, model_step)
        env.logger.log_scalar('test/term_out_of_road', collision_out_of_road_episodes, model_step)
        env.logger.log_scalar('test/term_unexpected', collision_unexpected_episodes, model_step)
        env.logger.log_scalar('test/term_runover_light', runover_light_episodes, model_step)
        env.logger.log_scalar('test/term_max_steps', max_steps_episodes, model_step)
        env.logger.log_scalar('test/term_max_steps_obs', max_steps_obs_episodes, model_step)
        env.logger.log_scalar('test/term_max_steps_light', max_steps_light_episodes, model_step)
        env.logger.log_scalar('test/term_static', static_episodes, model_step)
        env.logger.log_scalar('test/term_unknown', unknown_episodes, model_step)
        env.logger.log_scalar('test/total_reward', total_reward, model_step)

        with open(path + 'test_results.csv','a') as f:
                csvwriter = csv.writer(f, delimiter=',')
                csvwriter.writerow(data)
    return total_reward, success_episodes, results, data

def get_best_model(env, total_rewards, total_successes, model_file_names, path):
    print("Searching for best model now!!!")
    print("Rewards at intermediate training: {}".format(total_rewards))
    print("Total success episodes: {}".format(total_successes))
    m = max(total_successes)
    max_inds = np.array([i for i, j in enumerate(total_successes) if j == m])
    rewards = np.array(total_rewards)[max_inds]
    ind = max_inds[np.argmax(rewards)]
    print("Best model appears at index: {}".format(ind))
    print("No of successes in best model: {}".format(total_successes[ind]))
    print("Max no of successes: {}".format(m))
    best_model = PPO.load(model_file_names[ind], DummyVecEnv([lambda: env]))

    with open(path + "best_model.txt", "w") as f:
        f.write("Best model: {}\n".format(path))
        f.write("Best model appears at index: {}\n".format(ind))
        f.write("No of successes in best model: {}\n".format(total_successes[ind]))
        f.write("Max no of successes: {}\n".format(m))
        f.write("Rewards at intermediate training: {}\n".format(total_rewards))
        f.write("Total success episodes: {}\n".format(total_successes))

    return best_model

class PPO(PPO2):
    """A modification to the PPO algorithm to save models more often"""

    def learn(self, total_timesteps, trained_timesteps, env, callback=None, seed=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True, save_file="default", policy_plots=False, vae=None, train_vae=False, validation_interval=40000,
              disable_greedy_best=False):
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(0)

            runner = OverideRunner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam)
            self.episode_reward = np.zeros((self.n_envs,))

            ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()

            nupdates = total_timesteps // self.n_batch

            print("No of updates: {}".format(nupdates))
            print("Total timesteps : {}".format(total_timesteps))
            print("Batch size: {}".format(self.n_batch))
            total_rewards = []
            total_successes = []
            model_file_names = []
            total_updates = []
            for update in range((trained_timesteps // self.n_batch), nupdates + 1):
                if update == (trained_timesteps // self.n_batch):
                    self.save(save_file + str(update * self.n_batch))
                    if policy_plots:
                        plot_policy_and_value_fns(self, update * self.n_batch, save_file.split('models')[0] + 'policy_plots/')
                    total_reward, success_episodes, _, _ = test(self, env, dump_results=True, path=save_file.split('models')[0], model_step=update * self.n_batch)
                    total_rewards.append(total_reward)
                    total_successes.append(success_episodes)
                    model_file_names.append(save_file + str(update * self.n_batch))
                    total_updates.append(update * self.n_batch)
                    continue
                # if (update * self.n_batch) <= trained_timesteps:
                #     continue
                assert self.n_batch % self.nminibatches == 0
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / nupdates
                # frac = 1.0
                lr_now = self.learning_rate(frac)
                cliprangenow = self.cliprange(frac)
                # true_reward is the reward without discount
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run(logger=env.logger)
                ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
                    inds = np.arange(self.n_batch)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, batch_size):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_batch + epoch_num *
                                                                            self.n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_loss_vals.append(self._train_step(lr_now, cliprangenow, *slices, writer=writer,
                                                                 update=timestep))
                    self.num_timesteps += (self.n_batch * self.noptepochs) // batch_size * update_fac
                    if (update * self.n_batch) % validation_interval == 0:
                        self.save(save_file + str(update * self.n_batch))
                        if policy_plots:
                            plot_policy_and_value_fns(self, update * self.n_batch, save_file.split('models')[0] + 'policy_plots/')
                        total_reward, success_episodes, _, _ = test(self, env, dump_results=True, path=save_file.split('models')[0], model_step=update * self.n_batch)
                        total_rewards.append(total_reward)
                        total_successes.append(success_episodes)
                        model_file_names.append(save_file + str(update * self.n_batch))
                        total_updates.append(update * self.n_batch)
                        plot_test_results(total_successes, total_rewards, total_updates, save_file.split('models')[0])
                else:  # recurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_envs + epoch_num *
                                                                            self.n_envs + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, cliprangenow, *slices, update=timestep,
                                                                 writer=writer, states=mb_states))
                    self.num_timesteps += (self.n_envs * self.noptepochs) // envs_per_batch * update_fac

                if train_vae:
                    """Optimize the VAE"""
                    time_start = time.time()
                    vae.optimize()
                    print("Time to optimize the AE: ", time.time() - time_start)
                    if (update * self.n_batch) % validation_interval == 0:
                        base_path = save_file.split('models')[0] + 'ae_weights/'
                        if not os.path.exists(base_path):
                            os.makedirs(base_path)
                        vae.save(os.path.join(base_path, 'ae_' + str(update * self.n_batch)))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if writer is not None:
                    self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                      true_reward.reshape((self.n_envs, self.n_steps)),
                                                                      masks.reshape((self.n_envs, self.n_steps)),
                                                                      writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("nupdates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break
            if not disable_greedy_best:
                return get_best_model(env, total_rewards, total_successes, model_file_names, save_file.split('models')[0])
            else:
                return self

    def predict(self, observation, state=None, mask=None, deterministic=False):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions, _, states, _, _, _ = self.step(observation, state, mask, deterministic=deterministic)

        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            clipped_actions = clipped_actions[0]

        return clipped_actions, states

    def get_optimizer_weights(self):
        # Get Adam's Optimizer variables
        optimizer_weights = []
        print(self.params)
        print(self.trainer.get_slot_names())
        for var in self.params:
            for name in self.trainer.get_slot_names():
                print(var, name)
                slot_var = self.trainer.get_slot(var, name)
                if slot_var is not None:
                    optimizer_weights.append(slot_var)

        # Get Adam's Beta weights
        print("Get extra beta parameters")
        beta1, beta2 = self.trainer._get_beta_accumulators()
        print(beta1, beta2)
        if beta1 is not None and beta2 is not None:
            optimizer_weights.extend([beta1, beta2])

        return optimizer_weights

    def get_all_params(self):
        raise NotImplementedError
        return self.all_params

    def get_all_parameters(self):
        """
        Get current model parameters as dictionary of variable name -> ndarray.
        :return: (OrderedDict) Dictionary of variable name -> ndarray of model's parameters.
        """
        parameters = self.params + self.get_optimizer_weights()
        parameter_values = self.sess.run(parameters)
        return_dictionary = OrderedDict((param.name, value) for param, value in zip(parameters, parameter_values))
        print(return_dictionary)
        return return_dictionary

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update,
                    writer, states=None, cliprange_vf=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        :param cliprange_vf: (float) Clipping factor for the value function
        """
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions,
                  self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.clip_range_vf_ph] = cliprange_vf

        if states is None:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
        else:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train], td_map)

        # Retrieve Adam optimizer's variables (or states)
        # self.all_params = self.get_all_parameters()
        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def save(self, save_path, pid=None, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
            "cliprange_vf": self.cliprange_vf,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            # "n_cpu_tf_sess": self.n_cpu_tf_sess,
            # "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            # "all_params": self.all_params
        }

        params_to_save = self.get_parameters()

        if pid is not None:
            save_path = save_path + "_{}".format(pid)
            model_path = save_path.rsplit('/', 1)[0]
            self.saver.save(self.sess, os.path.join(model_path, 'policy-model-ckpt-{}'.format(pid)))

        # self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
        self._save_to_file(save_path, data=data, params=params_to_save)

    def load_optimizer_state(self, load_path, data, pid=None):
        if pid is not None:
            model_path = load_path.rsplit('/', 1)[0]
            load_ckpt_path = os.path.join(model_path, 'policy-model-ckpt-{}'.format(pid))
            self.saver.restore(self.sess, load_ckpt_path)

    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, pid=None, **kwargs):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Environment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        if pid is not None:
            load_path = load_path + "_{}".format(pid)

        data, params = cls._load_from_file(load_path, custom_objects=custom_objects)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=None, _init_setup_model=False)

        model.__dict__.update(data)
        model.__dict__.update(kwargs)

        model.set_env(env)
        model.setup_model()

        if pid is not None:
            model.load_optimizer_state(load_path, data, pid=pid)

        model.load_parameters(params)

        return model

    def setup_model(self):
        with SetVerbosity(self.verbose):

            # assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                            #    "an instance of common.policies.ActorCriticPolicy."

            self.n_batch = self.n_envs * self.n_steps

            n_cpu = multiprocessing.cpu_count()
            if sys.platform == 'darwin':
                n_cpu //= 2

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    assert self.n_envs % self.nminibatches == 0, "For recurrent policies, "\
                        "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch_step, reuse=False, **self.policy_kwargs)
                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              reuse=True, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                    vpred = train_model.value_flat
                    vpredclipped = self.old_vpred_ph + tf.clip_by_value(
                        train_model.value_flat - self.old_vpred_ph, - self.clip_range_ph, self.clip_range_ph)
                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpredclipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                      self.clip_range_ph), tf.float32))
                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('loss', loss)

                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()
                        grads = tf.gradients(loss, self.params)
                        if self.max_grad_norm is not None:
                            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                        grads = list(zip(grads, self.params))
                        # print(grads)
                        # for var in self.params:
                        #     print(var.name)
                        if self.full_tensorboard_log:
                            for var in self.params:
                                tf.summary.histogram(var.name, var)
                            for grad, var in grads[:-2]:
                                tf.summary.histogram(var.name + '/gradient', grad)
                    # grads = tf.gradients(loss, self.params)
                    # if self.max_grad_norm is not None:
                    #     grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    # grads = list(zip(grads, self.params))
                self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train = self.trainer.apply_gradients(grads)

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    tf.summary.scalar('old_neglog_action_probabilty', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        tf.summary.histogram('clip_range', self.clip_range_ph)
                        tf.summary.histogram('old_neglog_action_probabilty', self.old_neglog_pac_ph)
                        tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

                self.train_model = train_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101
                # self.all_params = self.get_all_parameters()

                self.summary = tf.summary.merge_all()
                self.saver = tf.train.Saver()
