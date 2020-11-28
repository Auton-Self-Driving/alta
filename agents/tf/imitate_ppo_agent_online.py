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
import pickle

from PIL import Image
from AuxNet.controller import AuxNetController
from ae.util import *

from datetime import datetime
import tensorboard_logging as tf_log

# PPO specific
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.policies import register_policy
from ppo import PPO, plot_policy_and_value_fns, plot_test_results
from models import Policy_1_layer, Policy_2_layer, CustomPolicy1, CustomPolicy2, CnnPolicy

import ipdb
import random

st = ipdb.set_trace

z_dim = 3
image_size = (128, 128, 3)
rv_img_sz = list(image_size)
rv_img_sz[-1] *=1
rv_img_sz = tuple(rv_img_sz)
DATA_DIR = '/project_data/schneider/mayank/'

def test(model, env, image_size, dump_results=False, path='.', model_step=None, rv=False):
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
        sys.stdout.write("Testing - Iter : %d of %d \r"%(ind, env.config["num_episodes"]))
        #obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        obs, rv_img = env.reset(unseen=True, index=ind)
        done = False
        reward = 0
        
        while not done:
            img = obs[:,:-8]
            rv_img = rv_img.reshape((1, -1))
            obs = obs[:,-8:]
            if z_dim == 3:
                manual_states = np.hstack([obs[:, :1], obs[:,3:5]])
            elif z_dim == 5:
                manual_states = np.hstack([obs[:, :1], obs[:,3:5], obs[:,6:]])
            #actions = model.predict(obs, deterministic=True)[0]
            if rv:
                rv_img = np.expand_dims(preproc_img(rv_img, image_size), axis = 0)
                actions = model.predict(rv_img, manual_states)
            else:
                img = np.expand_dims(preproc_img(img, image_size), axis = 0)
                actions = model.predict(img, manual_states)
            info = env.step(actions)
            reward += info[1][0][0]
            done = info[2]
            obs = info[0]
            rv_img = info[-1]

            #obs = np.expand_dims(info[0], axis=0)
        
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

    return total_reward, success_episodes, results, data

def collect_data(model, env, dump_results=False, path='.', num_ep=10, model_step=None, save = False):
    dummy_env = DummyVecEnv([lambda: env])
    env.reset()
    data = []
    #for ind in range(env.config["num_episodes"]):
    for ind in range(num_ep):
        print("\n\n########################")
        print("###   Episode :", ind, "   ")
        print("########################\n\n")
        #obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        #st()
        #obs[:] = env.reset(unseen=True, index=ind)
        #obs = env.reset(unseen=True, index=ind)
        obs, rv_img  = env.reset()
        done = False
        reward = 0
        
        prev_manual = None
        while not done:
            img = obs[:,:-8]
            rv_img = rv_img.reshape((1, -1))
            obs = obs[:,-8:]
            if z_dim == 3:
                manual_states = np.hstack([obs[:, :1], obs[:,3:5]])
            elif z_dim == 5:
                manual_states = np.hstack([obs[:, :1], obs[:,3:5], obs[:,6:]])
            if manual_states.ndim>1:
                manual_states = manual_states.reshape((-1,))
            actions = model.predict(obs, deterministic=True)[0]
            #sys.stdout.write("Waypoint: {%.5f}, Speed: {%.5f}, Steer: {%.5f}, GoalDist: {%.5f}, Light: {%.5f}, Expert Speed: {%.5f}, Expert Steer: {%.5f}\r"%(manual_states[0],manual_states[1],manual_states[2],0,0,actions[1],actions[0]))
            to_append = tuple([img, rv_img, manual_states, actions])
            #if prev_manual is None or not (prev_manual-manual_states).all():
            data.append(to_append)
            info = env.step(actions)
            reward += info[1][0][0]
            done = info[2]
            obs = info[0]
            rv_img = info[-1]
            prev_manual = manual_states
            #obs = np.expand_dims(info[0], axis=0)
        
    env.reset()
    data = np.asarray(data)
    if save:
        fl = open(DATA_DIR+'imitation_data_front_rgb3.p', 'wb')
        pickle.dump(data, fl)
        fl.close()
    return data

def collect_data_agent(expert_model, agent_model, env, dump_results=False, path='.', num_ep=10, model_step=None, save = False, iter=0):
    dummy_env = DummyVecEnv([lambda: env])
    env.reset()
    data = []
    #for ind in range(env.config["num_episodes"]):
    for ind in range(num_ep):
        print("\n\n########################")
        print("###   Episode :", ind, "   ")
        print("########################\n\n")
        #obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        #st()
        #obs[:] = env.reset(unseen=True, index=ind)
        #obs = env.reset(unseen=True, index=ind)
        obs, rv_img  = env.reset()
        done = False
        reward = 0
        
        prev_manual = None
        while not done:
            img = obs[:,:-8]
            rv_img = rv_img.reshape((1, -1))
            obs = obs[:,-8:]
            if z_dim == 3:
                manual_states = np.hstack([obs[:, :1], obs[:,3:5]])
            elif z_dim == 5:
                manual_states = np.hstack([obs[:, :1], obs[:,3:5], obs[:,6:]])
            if manual_states.ndim>1:
                manual_states = manual_states.reshape((-1,))
            expert_actions = expert_model.predict(obs, deterministic=True)[0]
            #sys.stdout.write("Waypoint: {%.5f}, Speed: {%.5f}, Steer: {%.5f}, GoalDist: {%.5f}, Light: {%.5f}, Expert Speed: {%.5f}, Expert Steer: {%.5f}\r"%(manual_states[0],manual_states[1],manual_states[2],0,0,expert_actions[1],expert_actions[0]))
            processed_img = np.expand_dims(preproc_img(rv_img, image_size), axis = 0)
            processed_manual_states = np.expand_dims(manual_states, 0)
            actions = agent_model.priv_imitator.predict(processed_img, processed_manual_states)
            to_append = tuple([img, rv_img, manual_states, expert_actions])
            #if prev_manual is None or not np.allclose(prev_manual,manual_states):
            data.append(to_append)

            info = env.step(actions)
            reward += info[1][0][0]
            done = info[2]
            obs = info[0]
            rv_img = info[-1]
            prev_manual = manual_states
            #obs = np.expand_dims(info[0], axis=0)
        
    env.reset()
    data = np.asarray(data)
    if save:
        fl = open(DATA_DIR+'imitation_agent_collected_iter_'+str(iter)+'.p', 'wb')
        pickle.dump(data, fl)
        fl.close()
    return data


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

def preproc_img(im, im_size, to_rgb = False):
    img = im.reshape(im_size)
    if to_rgb:
        img = convert_from_one_hot(img)
        rgb = convert_to_rgb(img, reduced_classes=True)
        rgb_img = Image.fromarray(rgb.astype(np.uint8), 'RGB').convert('RGBA')
        return rgb_img
    return img/255

def get_env(args, config, ALTA_LOGS, test_idx, SCRATCH_DIR):
    IMAGES_PATH = SCRATCH_DIR+'test_images_' + config.config["city_name"] + config.config['scenarios'] + '_run_' + str(test_idx) + '/'
    VIDEO_PATH = SCRATCH_DIR+'test_videos_' + config.config["city_name"] + config.config['scenarios'] +  '_run_' + str(test_idx) + '/'
    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, 1, videos=config.config["videos"])

    if args.city_name == 'Town01':
        spawn_points_fixed_idx = np.array([np.random.permutation(255) for i in range(args.test_trails)])
    elif args.city_name == 'Town02':
        spawn_points_fixed_idx = np.array([np.random.permutation(101) for i in range(args.test_trails)])

    config.config['spawn_points_fixed_idx'] = list(spawn_points_fixed_idx[test_idx])

    # Sending logger as None so as to not affect existing validation plots
    # env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=None, log_dir=ALTA_LOGS)
    env = launch_server(config, vis_wrapper, ALTA_LOGS)

    dummy_env = DummyVecEnv([lambda: env]) 

    return env, dummy_env   

def imitate_ppo(args, prefix, config):
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

    try:
        # Create the environment
        logger = tf_log.Logger(TB_LOGS_DIR)
        # if os.path.exists(SAVE_PATH + ".zip"):
        #     print("Best model exists, Validating !!!!")
        test_idx = 0
        if args.dataset_path is None:
            print("Starting env")
            env, dummy_env = get_env(args, config, ALTA_LOGS, test_idx, SCRATCH_DIR)
            print("env done")
            model = PPO.load(args.agent_model_path, env=dummy_env)

            env.config['input_type'] = 'wp_bev_rv_obs_info_speed_steer_ldist_goal_light'
            with open(ALTA_LOGS + 'test_results_' + config.config["city_name"] +  config.config['scenarios'] +  '_run_' + str(test_idx) + ".txt", "a") as f:
                if args.test:
                    print("Testing")
                    Imitator = AuxNetController(z_size = z_dim, image_size = image_size, buffer_size = 1, gt_size = 2, epoch_per_optimization=50)
                    Imitator.load(DATA_DIR+'front_dagger_iter_2.json')
                    priv_imitator = Imitator.priv_imitator
                    total_reward, success_episodes, results, data = test(priv_imitator, env, image_size = image_size, rv=True)
                    collision_obs_episodes, collision_lane_change_episodes, collision_out_of_road_episodes, collision_unexpected_episodes, \
                        runover_light_episodes, max_steps_episodes, max_steps_obs_episodes, max_steps_light_episodes, static_episodes, unknown_episodes = data[3:]
                    return
                else:
                    print("Generating data")
                    data = collect_data(model, env, num_ep = 20, save=True)
            env.close()
        else:
            print("Loading data", end='\r')
            data = pickle.load(open('/zfsauton2/home/mayankgu/ResnetWP_alta/alta/agents/tf/run_scripts/ppo/imitate_ppo/imitation_data_front_rgb3.p', 'rb'))
            #data2 = pickle.load(open('/zfsauton2/home/mayankgu/ResnetWP_alta/alta/agents/tf/run_scripts/ppo/imitate_ppo/imitation_agent_collected_iter_1.p', 'rb'))
            #data3 = pickle.load(open('/zfsauton2/home/mayankgu/ResnetWP_alta/alta/agents/tf/run_scripts/ppo/imitate_ppo/imitation_agent_collected_iter_0.p', 'rb'))
            #data = list(data1)+list(data2)+list(data3)
            

            print("Data Loaded")

        print("Starting env")
        env, dummy_env = get_env(args, config, ALTA_LOGS, test_idx, SCRATCH_DIR)
        env.config['input_type'] = 'wp_bev_rv_obs_info_speed_steer_ldist_goal_light'
        print("env done")

        Imitator = AuxNetController(z_size = z_dim, image_size = image_size, frame_stack = 1, buffer_size = 4*len(data), gt_size = 2, epoch_per_optimization=25)
        #Imitator.load('front_dagger_iter_1.json')
        Imitator.buffer = []
        for iter in range(5):
            print("Preprocessing data")
            for data_idx in range(len(data)):
                img = preproc_img(data[data_idx][1], rv_img_sz)
                manual_state = data[data_idx][-2]
                gt = data[data_idx][-1]
                Imitator.buffer.append(tuple([img, manual_state, gt]))

            Imitator.save_every_epoch = True
            Imitator.model_filepath = DATA_DIR+'front_dagger_iter_'+str(iter)+'.json'
            print("Training begins")
            train_loss_hist,_ = Imitator.optimize()
            print(train_loss_hist)

            # Collect Data using agent and label using expert - DAGGER.
            env.reset()

            model = PPO.load(args.agent_model_path, env=dummy_env)
            new_data = collect_data_agent(model, Imitator, env, num_ep = 10, save=True, iter=iter)
            
            data = list(data)+list(new_data)
            random.shuffle(data)

        # Testing
        env.reset()
        priv_imitator = Imitator.priv_imitator
        total_reward, success_episodes, results, test_data = test(priv_imitator, env, image_size = rv_img_sz, rv=True)
        collision_obs_episodes, collision_lane_change_episodes, collision_out_of_road_episodes, collision_unexpected_episodes, \
            runover_light_episodes, max_steps_episodes, max_steps_obs_episodes, max_steps_light_episodes, static_episodes, unknown_episodes = test_data[3:]

        
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