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
import keras.backend as K
import matplotlib.pyplot as plt
import pickle

from PIL import Image

from AuxNet.resnet_controller import ResnetController
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
import tensorflow as tf
from time import time

st = ipdb.set_trace

z_dim = 3
image_size = (224, 224, 3)
rv_img_sz = list(image_size)
rv_img_sz[-1] *=1
rv_img_sz = tuple(rv_img_sz)
DATA_DIR = '/home/scratch/mayankgu/'

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
        sys.stdout.write("\nTesting - Iter : %d of %d \n"%(ind, env.config["num_episodes"]))
        #obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        obs, rv_img = env.reset(unseen=True, index=ind)
        done = False
        reward = 0
        cnt=0
        while not done:
            img = obs[:,:-8]
            rv_img = rv_img.reshape((1, -1))
            obs = obs[:,-8:]
            if z_dim == 3:
                manual_states = np.hstack([obs[:, :1], obs[:,3:5]])
            elif z_dim == 5:
                manual_states = np.hstack([obs[:, :1], obs[:,3:5], obs[:,6:]])
            if rv:
                rv_img = np.expand_dims(preproc_img(rv_img, image_size), axis = 0)
                with tf.device('gpu:0'):
                    actions = model.predict([rv_img, manual_states])
            else:
                img = np.expand_dims(preproc_img(img, image_size), axis = 0)
                with tf.device('gpu:0'):
                    actions = model.predict([img, manual_states])
            manual_states = manual_states[0]
            cnt+=1
            sys.stdout.write("Iter %d : Waypoint: {%.5f}, Speed: {%.5f}, Steer: {%.5f}, GoalDist: {%.5f}, Light: {%.5f}\r"%(cnt, manual_states[0],manual_states[1],manual_states[2],obs[:,6],obs[:,7]))
            #sys.stdout.flush()
            info = env.step(actions)
            reward += info[1][0][0]
            done = info[2]
            obs = info[0]
            rv_img = info[-1]
            
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

    for ind in range(num_ep):
        print("\n\n########################")
        print("###   Episode :", ind, "   ")
        print("########################\n\n")

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
                            
            sys.stdout.write("Waypoint: {%.5f}, Speed: {%.5f}, Steer: {%.5f}, GoalDist: {%.5f}, Light: {%.5f}, Expert Speed: {%.5f}, Expert Steer: {%.5f}\r"%(manual_states[0],manual_states[1],manual_states[2],obs[:,6],obs[:,7],actions[1],actions[0]))
            
            rv_img_ = preproc_img(rv_img, rv_img_sz).astype(np.float32)
            manual_states = manual_states.astype(np.float32)
            
            to_append = [[rv_img_, manual_states], actions]
            
            #if prev_manual is None or not (prev_manual-manual_states).all():
            data.append(to_append)
            info = env.step(actions)
            reward += info[1][0][0]
            done = info[2]
            obs = info[0]
            rv_img = info[-1]
            prev_manual = manual_states
        
    env.reset()
    data = np.asarray(data)
    if save:
        fl = open(DATA_DIR+'imitation_data_front_rgb3_val.p', 'wb')
        pickle.dump(data, fl)
        fl.close()
    return data

def collect_data_agent(expert_model, imitator, env, dump_results=False, path='.', num_ep=10, model_step=None, save = False, iter=0):
    dummy_env = DummyVecEnv([lambda: env])
    env.reset()
    data = []
    
    sess = K.get_session()

    for ind in range(num_ep):
        print("\n\n########################")
        print("###   Episode :", ind, "   ")
        print("########################\n\n")
    
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
            sys.stdout.write("Waypoint: {%.5f}, Speed: {%.5f}, Steer: {%.5f}, GoalDist: {%.5f}, Light: {%.5f}, Expert Speed: {%.5f}, Expert Steer: {%.5f}\r"%(manual_states[0],manual_states[1],manual_states[2],obs[:,6],obs[:,7],expert_actions[1],expert_actions[0]))
            rv_img_ = preproc_img(rv_img, rv_img_sz).astype(np.float32)
            manual_states = manual_states.astype(np.float32)
            
            processed_img = np.expand_dims(rv_img_, axis = 0)
            processed_manual_states = np.expand_dims(manual_states, 0)
            with tf.device('gpu:0'):
                actions = imitator.model.predict([processed_img, processed_manual_states])

            to_append = [[rv_img_, manual_states], expert_actions]
            #if prev_manual is None or not np.allclose(prev_manual,manual_states):
            data.append(to_append)
            info = env.step(actions)
            reward += info[1][0][0]
            done = info[2]
            obs = info[0]
            rv_img = info[-1]
            prev_manual = manual_states
        
    env.reset()
    data = np.asarray(data)
    if save:
        fl = open(DATA_DIR+'imitation_agent_collected_iter_'+str(iter)+'.p', 'wb')
        pickle.dump(data, fl)
        fl.close()
    return data


def get_scratch_dir(base_log_dir):
    return base_log_dir.split(base_log_dir.split("/home")[0])[1].replace("/home", "/home/scratch")

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
    
    try:
        logger = tf_log.Logger(TB_LOGS_DIR)
    
        test_idx = 0
        if args.dataset_path is None:
            env, dummy_env = get_env(args, config, ALTA_LOGS, test_idx, SCRATCH_DIR)
            model = PPO.load(args.agent_model_path, env=dummy_env)

            env.config['input_type'] = 'wp_bev_rv_obs_info_speed_steer_ldist_goal_light'
            with open(ALTA_LOGS + 'test_results_' + config.config["city_name"] +  config.config['scenarios'] +  '_run_' + str(test_idx) + ".txt", "a") as f:
                if args.test:
                    print("Testing")
                    Imitator = ResnetController(zdim=z_dim)
                    Imitator.load(os.path.join(DATA_DIR,'DAGGER_iter_2.ckpt'))
                    priv_imitator = Imitator.model
                    total_reward, success_episodes, results, data = test(priv_imitator, env, image_size = image_size, rv=True)
                    collision_obs_episodes, collision_lane_change_episodes, collision_out_of_road_episodes, collision_unexpected_episodes, \
                        runover_light_episodes, max_steps_episodes, max_steps_obs_episodes, max_steps_light_episodes, static_episodes, unknown_episodes = data[3:]
                    return
                else:
                    print("Generating data")
                    data = collect_data(model, env, num_ep = 7, save=True)
                    return
            env.close()
        else:
            print("Loading data", end='\r')
            data = pickle.load(open(os.path.join(DATA_DIR, 'imitation_data_front_rgb3.p'), 'rb'))
            data_val = pickle.load(open(os.path.join(DATA_DIR, 'imitation_data_front_rgb3_val.p'), 'rb'))
            print("Data Loaded")

        np.random.shuffle(data)
        print("Starting env")
        env, dummy_env = get_env(args, config, ALTA_LOGS, test_idx, SCRATCH_DIR)
        env.config['input_type'] = 'wp_bev_rv_obs_info_speed_steer_ldist_goal_light'
        print("env done")

        Imitator = ResnetController(zdim=z_dim, val_data=data_val)
        expert_model = PPO.load(args.agent_model_path, env=dummy_env)

        ### DELETE LATER
        '''
        Imitator.load(os.path.join(DATA_DIR,'DAGGER_iter_0.ckpt'))
        env.reset()
        new_data = collect_data_agent(expert_model, Imitator, env, num_ep = 10, save=True, iter=0)            
        data = list(data)+list(new_data)
        random.shuffle(data)

        # Testing
        env.reset()
        priv_imitator = Imitator.model
        total_reward, success_episodes, results, test_data = test(priv_imitator, env, image_size = rv_img_sz, rv=True)
        collision_obs_episodes, collision_lane_change_episodes, collision_out_of_road_episodes, collision_unexpected_episodes, \
            runover_light_episodes, max_steps_episodes, max_steps_obs_episodes, max_steps_light_episodes, static_episodes, unknown_episodes = test_data[3:]
        ### DELETE LATER - END
        '''
        
        
        for iter in range(5):
            Imitator.buffer = data
            
            print("Training begins")
            Imitator.optimize(iter)
            
            env.reset()

            # DAGGER
            new_data = collect_data_agent(expert_model, Imitator, env, num_ep = 10, save=True, iter=iter)
            
            data = list(data)+list(new_data)
            random.shuffle(data)

            # Testing
            env.reset()
            priv_imitator = Imitator.model
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