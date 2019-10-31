from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, glob
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import ConfigManager

import numpy as np
import time
import vis_module

from datetime import datetime
import tensorboard_logging as tf_log

# PPO specific
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.policies import register_policy
from ppo import PPO, plot_policy_and_value_fns
from models import CustomPolicy, CustomWPPolicy, Policy

prefix = 'ppo_entcoeff_01_logstd_23_w_tanh_reward_10_nav_5_1/'

ALTA_LOGS = '/zfsauton2/home/hiteshar/research/alta-logs/ppo_tanh_runs1/' + prefix
POLICY_PLOTS = ALTA_LOGS + 'policy_plots/'
if not os.path.exists(ALTA_LOGS):
    os.makedirs(ALTA_LOGS)

TF_MODELS = ALTA_LOGS+'tf-models/checkpoint/'
if not os.path.exists(TF_MODELS):
    os.makedirs(TF_MODELS)

FRAME_SKIP = 1
SAVE_PATH = ALTA_LOGS + 'ppo2_measurements_weights'
TB_LOGS_DIR = ALTA_LOGS+ 'tb/'

MAX_TRIALS = 100

def get_latest_model(log_dir=ALTA_LOGS, ext='*.pkl', sep='_'):
    list_of_files = glob.glob(log_dir + ext)
    latest_file = max(list_of_files, key=os.path.getctime)
    latest_file = latest_file.split('.')[0]
    ind = int(latest_file.split(sep)[1])
    return ind, latest_file

def test(model, env):
    dummy_env = DummyVecEnv([lambda: env])
    success_episodes = 0
    results = {}
    total_reward = 0
    for ind in range(25):
        obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        obs[:] = env.reset(unseen=True, index=ind)
        done = False
        reward = 0
        
        while not done:
            actions = model.step(obs, deterministic=True)[0]
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

def get_best_model(total_timesteps, save_file, env):
    print("Searching for best model now!!!")
    total_rewards = []
    total_successes = []
    for model_step in range(0, total_timesteps + 1, 40000):
        model = PPO.load(save_file + str(model_step))
        print("Loading model file: {}".format(save_file + str(model_step)))
        total_reward, success_episodes = test(model, env)
        print(total_reward, success_episodes)
        env.logger.log_scalar('test/success_episodes', success_episodes, model_step)
        env.logger.log_scalar('test/total_reward', total_reward, model_step)
        total_rewards.append(total_reward)
        total_successes.append(success_episodes)
    print("Rewards at intermediate training: {}".format(total_rewards))
    print("Total success episodes: {}".format(total_successes))
    m = max(total_successes)
    max_inds = np.array([i for i, j in enumerate(total_successes) if j == m])
    rewards = np.array(total_rewards)[max_inds]
    ind = max_inds[np.argmax(rewards)]
    print("Best model appears at index: {}".format(ind))
    print("No of successes in best model: {}".format(total_successes[ind]))
    print("Max no of successes: {}".format(m))
    path = save_file + str(4 * (ind + 1)) + "0000"
    best_model = PPO.load(path, DummyVecEnv([lambda: env]))
    
    with open(ALTA_LOGS + "best_model.txt", "w") as f:
        f.write("Best model: {}\n".format(path))
        f.write("Best model appears at index: {}\n".format(ind))
        f.write("No of successes in best model: {}\n".format(total_successes[ind]))
        f.write("Max no of successes: {}\n".format(m))
        f.write("Rewards at intermediate training: {}\n".format(total_rewards))
        f.write("Total success episodes: {}\n".format(total_successes))
        
    return best_model

if __name__ == '__main__':
    
    register_policy('CustomWPPolicy', CustomWPPolicy)
    steps = 5000000
    
    for i in range(MAX_TRIALS):
        try:
            # Create the environment
            config = ConfigManager(algo="PPO")
            logger = tf_log.Logger(TB_LOGS_DIR)
            if os.path.exists(SAVE_PATH + ".pkl"):
                print("Best model exists, Validating !!!!")
                with open(ALTA_LOGS + "seed.txt", "r") as f:
                    seed = int(f.readline())
                print("Using the pre-initialized seed: {}".format(seed))
                set_global_seeds(seed)

                IMAGES_PATH = ALTA_LOGS+'final_images_' + config.config["city_name"] + '/'
                VIDEO_PATH = ALTA_LOGS+'final_videos_' + config.config["city_name"] + '/'
                vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
                
                env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger)
                dummy_env = DummyVecEnv([lambda: env])
                
                model = PPO.load(SAVE_PATH, dummy_env)
                success_episodes = 0
                results = {}
                with open(ALTA_LOGS + config.config["scenarios"] + config.config["city_name"] + ".txt", "w") as f:
                    for ind in range(25):
                        obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
                        obs[:] = env.reset(unseen=True, index=ind)
                        done = False
                        while not done:
                            actions = model.step(obs, deterministic=True)[0]
                            print(actions.shape)
                            info = env.step(actions)
                            done = info[2]
                            obs = np.expand_dims(info[0], axis=0)
                        
                        print("Termination State: {}".format(info[3]['termination_state']))
                        f.write("Termination State: {}".format(info[3]['termination_state']))
                        if info[3]['termination_state'] == 'success':
                            success_episodes += 1
                            results[ind] = 1
                        else:
                            results[ind] = 0
                    print("Task Name: {}".format(config.config["scenarios"]))
                    print("Town Name: {}".format(config.config["city_name"]))
                    print("Results of test scenarios")
                    print(results)
                    print("Total Success Episodes: {}".format(success_episodes))
                    f.write("Task Name: {}".format(config.config["scenarios"]))
                    f.write("Town Name: {}".format(config.config["city_name"]))
                    f.write("Results of test scenarios")
                    # f.write(results)
                    f.write("Total Success Episodes: {}".format(str(success_episodes)))
            elif os.path.exists(SAVE_PATH + str(steps) + ".pkl"):
                print("All models exist. Finding best model !!")
                env = CarlaEnv(config=config.config)
                best_model = get_best_model(steps, SAVE_PATH, env)
                best_model.save(SAVE_PATH)
            else:
                print("Training begins")
                IMAGES_PATH = ALTA_LOGS+'images/'
                VIDEO_PATH = ALTA_LOGS+'videos/'
                vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
                
                env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger)
                dummy_env = DummyVecEnv([lambda: env])
                
                model = PPO(policy=Policy, env=dummy_env, n_steps=500, nminibatches=4, verbose=1, learning_rate=2e-4, 
                        tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=True, ent_coef=0.005)
                if any(fname.endswith('.pkl') for fname in os.listdir(ALTA_LOGS)):
                    with open(ALTA_LOGS + "seed.txt", "r") as f:
                        seed = int(f.readline())
                    print("Using the pre-initialized seed: {}".format(seed))
                    set_global_seeds(seed)
                    completed_steps, latest_model = get_latest_model(log_dir=ALTA_LOGS, ext='*.pkl', sep='hts')
                    env.total_steps = completed_steps
                    if config.config["videos"]:
                        completed_episodes, _ = get_latest_model(log_dir=ALTA_LOGS + 'videos/', ext='*.mp4', sep='log_')
                        env.episode_num = completed_episodes
                    print("Loading Latest model!!!")
                    model = PPO.load(latest_model, dummy_env)
                    print("Model: {} loaded successfully".format(latest_model))
                    _ = model.learn(steps, completed_steps, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=False, seed=seed, policy_plots=True)    
                else:
                    dt = datetime.now()
                    millis = dt.microsecond
                    print(millis)
                    with open(ALTA_LOGS + "seed.txt", "w") as f:
                        f.write(str(millis))
                    random_model = model.learn(0, 0, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=True, seed=millis, policy_plots=True)
                    random_model.save(SAVE_PATH + "0")
                    plot_policy_and_value_fns(random_model, 0, POLICY_PLOTS)
                    _ = model.learn(steps, 0, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=True, seed=millis, policy_plots=True)
                
                best_model = get_best_model(steps, SAVE_PATH, env)
                best_model.save(SAVE_PATH)
            break
        except Exception as e:
            print(e)
            env.close()
            time.sleep(120)
