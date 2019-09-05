"""PPO optimizer with VAE dimensionality reduction of the input images"""

import time
import numpy as np
from mpi4py import MPI
from stable_baselines import logger
from stable_baselines.common.base_class import SetVerbosity, TensorboardWriter
from stable_baselines.ppo2.ppo2 import PPO2, Runner
from stable_baselines.common.vec_env import DummyVecEnv
import sys
import gym
from collections import deque
from stable_baselines.common.math_util import explained_variance
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


class PPO(PPO2):
    """A modification to the PPO algorithm to save models more often"""
    
    def get_best_model(self, total_timesteps, save_file, env):
        print("Searching for best model now!!!")
        nupdates = total_timesteps // self.n_batch
        total_rewards = []
        total_successes = []
        for update in range(1, nupdates + 1):
            if (update * self.n_batch) % 10000 == 0:
                self.load(save_file + str(update * self.n_batch))
                print("Loading model file: {}".format(save_file + str(update * self.n_batch)))
                total_reward, success_episodes = self.test(env)
                print(total_reward, success_episodes)
                env.logger.log_scalar('test/success_episodes', success_episodes, update * self.n_batch)
                env.logger.log_scalar('test/total_reward', total_reward, update * self.n_batch)
                total_rewards.append(total_reward)
                total_successes.append(success_episodes)
        print("Rewards at intermediate training: {}".format(total_rewards))
        print("Total success episodes: {}".format(total_successes))
        m = max(total_successes)
        max_inds = np.array([i for i, j in enumerate(total_successes) if j == m])
        total_rewards = np.array(total_rewards)[max_inds]
        ind = max_inds[np.argmax(total_rewards)]
        print("Best model appears at index: {}".format(ind))
        print("No of successes in best model: {}".format(total_successes[ind]))
        print("Max no of successes: {}".format(m))
        SAVE_PATH = save_file + str(ind + 1) + "0000"
        best_model = self.load(SAVE_PATH, DummyVecEnv([lambda: env]))
        
        return best_model
    
    def learn(self, total_timesteps, trained_timesteps, env, callback=None, seed=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True, save_file="default"):
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)

            runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam)
            self.episode_reward = np.zeros((self.n_envs,))

            ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()

            nupdates = total_timesteps // self.n_batch
            
            print("No of updates: {}".format(nupdates))
            print("Total timesteps : {}".format(total_timesteps))
            print("Batch size: {}".format(self.n_batch))
            # total_rewards = []
            # total_successes = []
            for update in range((trained_timesteps // self.n_batch) + 1, nupdates + 1):
                # if (update * self.n_batch) <= trained_timesteps:
                #     continue
                assert self.n_batch % self.nminibatches == 0
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / nupdates
                lr_now = self.learning_rate(frac)
                cliprangenow = self.cliprange(frac)
                # true_reward is the reward without discount
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run()
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
                    if (update * self.n_batch) % 10000 == 0:
                        self.save(save_file + str(update * self.n_batch))
                        # total_reward, success_episodes = self.test(env)
                        # env.logger.log_scalar('test/success_episodes', success_episodes, update * self.n_batch)
                        # env.logger.log_scalar('test/total_reward', total_reward, update * self.n_batch)
                        # total_rewards.append(total_reward)
                        # total_successes.append(success_episodes)
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
            best_model = self.get_best_model(total_timesteps, save_file, env)
            
            return self, best_model
        
    def test(self, env):
        dummy_env = DummyVecEnv([lambda: env])
        success_episodes = 0
        results = {}
        total_reward = 0
        for ind in range(1):
            obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
            obs[:] = env.reset(unseen=True, index=ind)
            done = False
            reward = 0
            
            while not done:
                actions = self.step(obs, deterministic=True)[0]
                info = env.step(actions)
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
        print("Total Success Episodes: {}".format(success_episodes))
        return total_reward, success_episodes

        
class PPOWithVAE(PPO2):
    """A modification to the PPO algorithm to put in VAE optimization step"""

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True, vae=None):
        #  making the learning rate and clip range callable here.

        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        new_tb_log = self._init_num_timesteps(reset_num_timesteps=reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)

            runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam)
            self.episode_reward = np.zeros((self.n_envs,))

            ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()

            nupdates = total_timesteps // self.n_batch

            for update in range(1, nupdates + 1):
                assert self.n_batch % self.nminibatches == 0
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / nupdates
                lr_now = self.learning_rate(frac)
                cliprangenow = self.cliprange(frac)

                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run()
                ep_info_buf.extend(ep_infos)
                mb_loss_vals = []

                if states is None:
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
                    self.save(PATH_MODEL_PPO2)
                    """Optimize the VAE"""
                    time_start = time.time()
                    # vae.optimize()
                    # vae.save(PATH_MODEL_VAE)
                    # print("Time to optimize the VAE: ", time.time() - time_start)

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
                    logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                    logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()
            return self
