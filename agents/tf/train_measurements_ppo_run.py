import sys, os, glob
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))

from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import ConfigManager

import numpy as np
import math
import time
import vis_module
import traceback
import csv
import multiprocessing as mp
import matplotlib.pyplot as plt
# import logging

# logger = mp.log_to_stderr()
# logger.setLevel(mp.SUBDEBUG)

from datetime import datetime
import tensorboard_logging as tf_log

# PPO specific
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.policies import register_policy
from ppo import PPO, plot_policy_and_value_fns, test, plot_test_results
from models import Policy_1_layer, Policy_2_layer, CustomPolicy1, CustomPolicy2, CnnPolicy

def get_scratch_dir(base_log_dir):
    return base_log_dir.split(base_log_dir.split("/home")[0])[1].replace("/home", "/home/scratch")

def find_ext_format(MODEL_PATH):
    ext = None
    for fname in os.listdir(MODEL_PATH):
        if fname.endswith('.pkl'):
            ext = '.pkl'
        elif fname.endswith('.zip'):
            ext = '.zip'
        
        if ext is not None:
            break
    return ext

def plot_reward(timesteps, mean_reward, min_reward, max_reward, figname="mean_reward.png"):
    plt.figure(figsize=(11, 7))
    timesteps = timesteps / 1000000
    timesteps_interval = 0.5
    plt.plot(timesteps, mean_reward, label='WRL+',  color='orangered')
    plt.fill_between(timesteps, min_reward, max_reward, color='mistyrose')

    axes = plt.gca()
    plt.title("Reward")
    plt.xlabel('Timesteps (in M)', fontdict={'size' : 18})
    plt.ylabel('Total Reward', fontdict={'size' : 18})
    plt.xticks(list(np.arange(0, (math.ceil(timesteps[-1] / timesteps_interval) + 1) * timesteps_interval, timesteps_interval)), ('{}'.format(str(x)) for x in np.arange(0, (math.ceil(timesteps[-1] / timesteps_interval) + 1) * timesteps_interval, timesteps_interval)))
    plt.savefig(figname, dpi=200)
    plt.clf()
    plt.close()

def plot_success(timesteps, mean_success, min_success, max_success, figname="mean_success.png"):
    plt.figure(figsize=(11, 7))
    timesteps = timesteps / 1000000
    timesteps_interval = 0.5
    plt.plot(timesteps, mean_success, label='WRL+',  color='orangered')
    plt.fill_between(timesteps, min_success, max_success, color='mistyrose')

    axes = plt.gca()
    plt.title("Success")
    plt.xlabel('Timesteps (in M)', fontdict={'size' : 18})
    plt.ylabel('Total Success', fontdict={'size' : 18})
    plt.xticks(list(np.arange(0, (math.ceil(timesteps[-1] / timesteps_interval) + 1) * timesteps_interval, timesteps_interval)), ('{}'.format(str(x)) for x in np.arange(0, (math.ceil(timesteps[-1] / timesteps_interval) + 1) * timesteps_interval, timesteps_interval)))
    plt.savefig(figname, dpi=200)
    plt.clf()
    plt.close()

def model_learn(total_timesteps, trained_timesteps, ALTA_LOGS, save_file, validation_interval, disable_greedy_best, config, vis_wrapper, pid, callback=None, log_interval=1, tb_log_name="PPO2", reset_num_timesteps=True, policy_plots=False, vae=None, train_vae=False):
    env = launch_server(config, vis_wrapper, ALTA_LOGS)
    dummy_env = DummyVecEnv([lambda: env])
    print("Carla env created")

    model = PPO.load(save_file, env=dummy_env, pid=pid)
    print("Model object created")

    model = model.learn(total_timesteps, trained_timesteps, env, tb_log_name="PPO2", save_file=save_file, reset_num_timesteps=True, policy_plots=False, validation_interval=validation_interval, disable_greedy_best=disable_greedy_best)
    total_reward, success_episodes, results, _ = test(model, env)

    pid = os.getpid()
    model.save(save_file, pid=pid)
    env.close()

    return [model.get_parameters(), pid, total_reward, success_episodes, results]

def launch_server(config, vis_wrapper, ALTA_LOGS, logger=None):
    RETRIES_ON_ERROR = 5
    serverStartRetries = 0
    serverStarted = False

    env = None
    while ((not serverStarted) and serverStartRetries < RETRIES_ON_ERROR):
        try:
            env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger, log_dir=ALTA_LOGS)
            serverStarted = True

        except Exception as identifier:
            traceback.print_exc()
            if env is not None:
                env.close()
                serverStartRetries += 1
                time.sleep(20)
    return env

def run_ppo(args, prefix, config):
    ALTA_LOGS = os.path.join(args.base_log_dir, prefix.split('_runid_')[0], prefix)
    # config.config['LOG_DIR'] = ALTA_LOGS
    if ALTA_LOGS[-1] != '/':
        ALTA_LOGS += '/'

    if "/home/scratch" not in args.base_log_dir and os.path.exists('/home/scratch'):
        SCRATCH_DIR = os.path.join(get_scratch_dir(args.base_log_dir), prefix.split('_runid_')[0], prefix)
    else:
        SCRATCH_DIR = ALTA_LOGS

    if SCRATCH_DIR[-1] != '/':
        SCRATCH_DIR += '/'
    
    POLICY_PLOTS = ALTA_LOGS + 'policy_plots/'
    if not os.path.exists(ALTA_LOGS):
        os.makedirs(ALTA_LOGS)

    TF_MODELS = ALTA_LOGS+'tf-models/checkpoint/'
    if not os.path.exists(TF_MODELS):
        os.makedirs(TF_MODELS)

    VIDEO_FRAME_SKIP = 1
    MODEL_PATH = os.path.join(ALTA_LOGS, 'models')
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    SAVE_PATH = os.path.join(MODEL_PATH, 'ppo2_weights')
    FORWARD_SEARCH_MODEL = os.path.join(MODEL_PATH, 'forward_search_model')
    TB_LOGS_DIR = ALTA_LOGS+ 'tb/'

    MAX_TRIALS = 5
    
    def get_latest_model(log_dir=MODEL_PATH, ext='*.zip', sep='_'):
        list_of_files = glob.glob(os.path.join(log_dir, ext))
        latest_file = max(list_of_files, key=os.path.getctime)
        latest_file = latest_file.split('{}'.format('.' + ext.split('.')[1]))[0]
        ind = int(latest_file.split(sep)[1])
        return ind, latest_file

    def get_completed_episodes(log_dir=ALTA_LOGS, ext='*.zip', sep1='_step_', sep2='_ind_'):
        list_of_files = glob.glob(os.path.join(log_dir, ext))
        latest_file = max(list_of_files, key=os.path.getctime)
        latest_file = latest_file.split('{}'.format('.' + ext.split('.')[1]))[0]
        ind = int(latest_file.split(sep1)[1].split(sep2)[0])
        return ind, latest_file

    
    for i in range(MAX_TRIALS):
        try:
            # Create the environment
            logger = tf_log.Logger(TB_LOGS_DIR)
            # if os.path.exists(SAVE_PATH + ".zip"):
            #     print("Best model exists, Validating !!!!")
            if args.test:
                print('Testing Begins')
                np.random.seed(10)
                if args.city_name == 'Town01':
                    spawn_points_fixed_idx = np.array([np.random.permutation(257) for i in range(args.test_trails)])
                elif args.city_name == 'Town02':
                    spawn_points_fixed_idx = np.array([np.random.permutation(101) for i in range(args.test_trails)])

                # with open(ALTA_LOGS + "seed.txt", "r") as f:
                #     seed = int(f.readline())
                # print("Using the pre-initialized seed: {}".format(seed))
                # set_global_seeds(seed)

                rewards = []
                successes = []
                for test_idx in range(args.test_trails):
                    IMAGES_PATH = SCRATCH_DIR+'test_images_' + config.config["city_name"] + config.config['scenarios'] + '_run_' + str(test_idx) + '/'
                    VIDEO_PATH = SCRATCH_DIR+'test_videos_' + config.config["city_name"] + config.config['scenarios'] +  '_run_' + str(test_idx) + '/'
                    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, VIDEO_FRAME_SKIP, videos=config.config["videos"])

                    config.config['spawn_points_fixed_idx'] = list(spawn_points_fixed_idx[test_idx])

                    # Sending logger as None so as to not affect existing validation plots
                    # env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=None, log_dir=ALTA_LOGS)
                    env = launch_server(config, vis_wrapper, ALTA_LOGS)

                    dummy_env = DummyVecEnv([lambda: env])
                    model = PPO.load(args.agent_model_path, env=dummy_env)

                    with open(ALTA_LOGS + 'test_results_' + config.config["city_name"] +  config.config['scenarios'] +  '_run_' + str(test_idx) + ".txt", "a") as f:
                        total_reward, success_episodes, results, data = test(model, env)
                        collision_obs_episodes, collision_lane_change_episodes, collision_out_of_road_episodes, collision_unexpected_episodes, \
                            runover_light_episodes, max_steps_episodes, max_steps_obs_episodes, max_steps_light_episodes, static_episodes, unknown_episodes = data[3:]

                        print("Task Name: {}".format(config.config["scenarios"]))
                        print("Town Name: {}".format(config.config["city_name"]))
                        # print("Results of test scenarios")
                        # print(results)
                        print("Total Success Episodes: {}".format(success_episodes))
                        f.write("Task Name: {}\n".format(config.config["scenarios"]))
                        f.write("Town Name: {}\n".format(config.config["city_name"]))
                        f.write("Results of test scenarios\n")
                        f.write(str(results))
                        f.write("Total Success: {}, Collision Obstacle: {}, Collision LaneChange: {}, Collision OutOfRoad: {}, Collision Unexpected: {}, Runover Light: {}, Max Steps: {}, Max StepsObstacle: {}, Max StepsLight: {}, Static: {}, Unknown: {}\n".format(success_episodes,
                                    collision_obs_episodes, collision_lane_change_episodes, collision_out_of_road_episodes, collision_unexpected_episodes, runover_light_episodes, max_steps_episodes, max_steps_obs_episodes, max_steps_light_episodes, static_episodes, unknown_episodes))
                        f.write("Total Collisions: {}, Static Collisions: {}, Vehicle Collisions:{}\n".format(env.total_collisions, env.static_collisions, env.vehicle_collisions))
                        f.write("Traffic Light Violations: {}\n".format(env.traffic_light_violations))
                        f.write("Total Distance: {}\n".format(env.total_distance))
                        f.write("Spawn Points Permutation: {}\n".format(str(env.config['spawn_points_fixed_idx'])))
                    rewards.append(total_reward)
                    successes.append(success_episodes)
                    env.close()
                rewards = np.array(rewards)
                successes = np.array(successes)
                with open(ALTA_LOGS + 'final_test_results_' + config.config["city_name"]+  config.config['scenarios'] + ".txt", "a") as f:
                    f.write("Task Name: {}\n".format(config.config["scenarios"]))
                    f.write("Town Name: {}\n".format(config.config["city_name"]))
                    f.write("Model path used for testing: {}\n".format(args.agent_model_path))
                    f.write("Results of final testing\n")
                    f.write("Rewards: {}\n".format(" ".join(map(str, rewards))))
                    f.write("Success: {}\n".format(" ".join(map(str, successes))))
                    f.write("Avg Success: {}\n".format(np.mean(successes)))
                    f.write("Std Success: {}\n".format(np.std(successes)))
                    f.write("Total Successes: {}\n".format(np.sum(successes)))
            elif args.validation:
                print('Validation Begins')
                with open(ALTA_LOGS + "seed.txt", "r") as f:
                    seed = int(f.readline())
                print("Using the pre-initialized seed: {}".format(seed))
                set_global_seeds(seed)

                spawn_points_fixed_idx = np.load(ALTA_LOGS + "spawn_pt_order.npy")

                rewards = []
                successes = []

                IMAGES_PATH = SCRATCH_DIR+'val_images/'
                VIDEO_PATH = SCRATCH_DIR+'val_videos/'

                vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, VIDEO_FRAME_SKIP, videos=config.config["videos"])

                config.config['spawn_points_fixed_idx'] = list(spawn_points_fixed_idx)

                # Sending logger as None so as to not affect existing validation plots
                # env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=None, log_dir=ALTA_LOGS)
                env = launch_server(config, vis_wrapper, ALTA_LOGS)
                dummy_env = DummyVecEnv([lambda: env])

                rewards = []
                successes = []
                updates = []
                ext = find_ext_format(MODEL_PATH)
                model_files = [os.path.join(ALTA_LOGS, model) for model in os.listdir(ALTA_LOGS) if model.endswith(ext)]
                model_files = sorted(model_files, key=os.path.getctime)

                update = 0
                for model_file in model_files[:-1]:
                    model = PPO.load(model_file, env=dummy_env, seed=seed)
                    total_reward, success_episodes, results, _ = test(model, env)
                    print("Model: {}, Success: {}, Reward: {}".format(model_file, success_episodes, total_reward))
                    rewards.append(total_reward)
                    successes.append(success_episodes)
                    updates.append(update)
                    plot_test_results(successes, rewards, updates, ALTA_LOGS)
                    with open(ALTA_LOGS + 'test_results2.csv','a') as f:
                        csvwriter = csv.writer(f, delimiter=',')
                        csvwriter.writerow([update, success_episodes, total_reward])
                    update += args.validation_interval
                env.close()
            else:
                print("Training begins")
                IMAGES_PATH = SCRATCH_DIR+'images/'
                VIDEO_PATH = SCRATCH_DIR+'videos/'
                vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, VIDEO_FRAME_SKIP, videos=config.config["videos"])
                
                # env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger, log_dir=ALTA_LOGS)
                env = launch_server(config, vis_wrapper, ALTA_LOGS, logger=logger)

                dummy_env = DummyVecEnv([lambda: env])
                
                if args.network == "1_layer":
                    policy = Policy_1_layer
                elif args.network == "2_layer":
                    policy = Policy_2_layer
                elif args.network == "CustomPolicy1":
                    policy = CustomPolicy1
                elif args.network == "CustomPolicy2":
                    policy = CustomPolicy2
                elif args.network == "CNN":
                    policy = CnnPolicy
                else:
                    print("specify either 1_layer, 2_layer CustomPolicy1, CustomPolicy2 or CnnPolicy as network input")
                    env.close()
                    print("exiting")
                    return
                
                if not args.enable_search and any(fname.endswith('.pkl') or fname.endswith('.zip') for fname in os.listdir(MODEL_PATH)):
                    ext = find_ext_format(MODEL_PATH)
                    with open(ALTA_LOGS + "seed.txt", "r") as f:
                        seed = int(f.readline())
                    print("Using the pre-initialized seed: {}".format(seed))
                    set_global_seeds(seed)
                    completed_steps, latest_model = get_latest_model(log_dir=MODEL_PATH, ext='*[0-9]' + ext, sep='hts')
                    env.total_steps = completed_steps
                    completed_episodes, _ = get_completed_episodes(log_dir=ALTA_LOGS + 'val_episode_info_plots/', ext='*.png', sep1='_TrainEp_', sep2='_step_')
                    env.episode_num = completed_episodes
                    print(env.total_steps, env.episode_num)
                    print("Completed episodes: {}".format(completed_episodes))
                    print("Loading Latest model!!!")
                    model = PPO.load(latest_model, env=dummy_env, seed=seed)
                    print("Model: {} loaded successfully".format(latest_model))
                    if not args.enable_search:
                        best_model = model.learn(args.timesteps, completed_steps, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=False, policy_plots=False, validation_interval=args.validation_interval)
                    else:
                        raise NotImplementedError
                else:
                    dt = datetime.now()
                    millis = dt.microsecond
                    print(millis)
                    with open(ALTA_LOGS + "seed.txt", "w") as f:
                        f.write(str(millis))
                    if args.agent_model_path is None:
                        model = PPO(policy=policy, env=dummy_env, n_steps=args.n_steps, nminibatches=args.no_minibatches, verbose=1, learning_rate=args.lr,
                            tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=False, ent_coef=args.ent_coef, noptepochs=args.no_epochs, cliprange=args.clip, seed=millis)
                    else:
                        model = PPO.load(args.agent_model_path, env=dummy_env, seed=millis)
                        print("Loading pretrained agent from: {}".format(args.agent_model_path))
                    if not args.enable_search:
                        best_model = model.learn(args.timesteps, 0, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=True, policy_plots=False, validation_interval=args.validation_interval)
                    else:
                        timesteps = []
                        total_reward = []
                        total_success = []
                        if os.path.exists(os.path.join(ALTA_LOGS, 'forward_search_train_stats.npz')):
                            train_stats = np.load(os.path.join(ALTA_LOGS, 'forward_search_train_stats.npz'))
                            pid = int(train_stats['pid'])
                            completed_steps = int(train_stats['completed_steps'])
                            timesteps = list(train_stats['timesteps'])
                            total_reward = list(train_stats['total_reward'])
                            total_success = list(train_stats['total_success'])
                            model = PPO.load(FORWARD_SEARCH_MODEL, env=dummy_env, pid=pid)
                            print("Loading forward search model with pid:{}, completed_steps:{}".format(pid, completed_steps))
                        else:
                            pid = os.getpid()
                            completed_steps = 0

                        epochs = args.timesteps // args.pop_train_interval
                        model.save(FORWARD_SEARCH_MODEL, pid=pid)

                        print("Running forward search with population size: {}, epochs: {}, PID:{}".format(args.pop_size, epochs, pid))

                        for epoch in range(completed_steps // args.pop_train_interval, epochs):
                            with mp.get_context("spawn").Pool(args.pop_size) as pool:
                                pooled_results = pool.starmap(model_learn,
                                                    ((args.pop_train_interval, 0, ALTA_LOGS, FORWARD_SEARCH_MODEL, args.validation_interval, args.disable_greedy_best, config, None, pid)
                                                        for _ in range(args.pop_size)))

                            pooled_results = np.array(pooled_results)
                            models_parameters = pooled_results[:, 0]
                            process_ids = pooled_results[:, 1]
                            rewards = pooled_results[:, 2]
                            successes = pooled_results[:, 3]

                            # for idx in range(pooled_results.shape[0]):
                            #     _, pid, total_reward, success_episodes, results = pooled_results[idx]
                            #     print(pid, total_reward, success_episodes, results)

                            max_success = max(successes)
                            max_inds = np.array([i for i, j in enumerate(successes) if j == max_success])
                            rewards = np.array(rewards)[max_inds]
                            max_reward = np.amax(rewards)
                            ind = max_inds[np.argmax(rewards)]
                            print("Epoch:{} Best child index from population: {}, Total Reward:{}, Total Success:{}".format(epoch + 1, ind, max_reward, max_success))

                            model = PPO.load(FORWARD_SEARCH_MODEL, env=dummy_env, pid=process_ids[ind])
                            model.load_parameters(models_parameters[ind], exact_match=True)
                            model.save(FORWARD_SEARCH_MODEL, pid=pid)

                            timesteps.append((epoch + 1) * args.pop_train_interval)
                            total_reward.append(max_reward)
                            total_success.append(max_success)

                            plot_reward(np.array(timesteps), np.array(total_reward), np.zeros_like(total_reward), np.zeros_like(total_reward), figname=os.path.join(ALTA_LOGS, 'fsepoch_reward_{}.png'.format(epoch + 1)))
                            plot_success(np.array(timesteps), np.array(total_success), np.zeros_like(total_success), np.zeros_like(total_success), figname=os.path.join(ALTA_LOGS, 'fsepoch_success_{}.png'.format(epoch + 1)))

                            with open(os.path.join(ALTA_LOGS, 'forward_search.csv'), 'a') as f:
                                csvwriter = csv.writer(f, delimiter=',')
                                csvwriter.writerow([epoch + 1, ind, max_success, max_reward, pooled_results[:, 1:4]])

                            np.savez_compressed(os.path.join(ALTA_LOGS, 'forward_search_train_stats.npz'), timesteps=timesteps, total_reward=total_reward, total_success=total_success, pid=pid, completed_steps=((epoch + 1) * args.pop_train_interval))

                        best_model = model
                best_model.save(SAVE_PATH)
            break
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