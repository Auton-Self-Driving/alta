import sys, os, time, glob
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from stable_baselines.common.vec_env import DummyVecEnv
from models import CustomPolicy, MlpPolicy
from ae.controller import AEController
# from stable_baselines import logger

from ppo import PPOWithVAE, PPO
from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import ConfigManager

import vis_module

from stable_baselines.ppo2.ppo2 import PPO2
from stable_baselines.common.policies import register_policy
from stable_baselines.common.misc_util import set_global_seeds
import numpy as np
from datetime import datetime
import tensorboard_logging as tf_log

vae = AEController()
# change
prefix = 'vae_dim1_custom_net_lr_5e4/'

PATH_MODEL_VAE = "vae-model-30000.json"
ALTA_LOGS = '/zfsauton2/home/tanmaya/projects/alta-logs/ppo_pid_reduced_vae_scenarios_single_straight_steer_only/' + prefix
if not os.path.exists(ALTA_LOGS):
    os.makedirs(ALTA_LOGS)

# logger.configure(folder=ALTA_LOGS)

SAVE_PATH = ALTA_LOGS + 'ppo2_measurements_weights'
TF_MODELS = ALTA_LOGS+'tf-models/checkpoint/'
if not os.path.exists(TF_MODELS):
    os.makedirs(TF_MODELS)

FRAME_SKIP = 1
TB_LOGS_DIR = ALTA_LOGS+'tb/'

def get_latest_model(log_dir=ALTA_LOGS, ext='*.pkl', sep='_'):
    list_of_files = glob.glob(log_dir + ext)
    latest_file = max(list_of_files, key=os.path.getctime)
    latest_file = latest_file.split('.')[0]
    ind = int(latest_file.split(sep)[1])
    return ind, latest_file

if __name__ == '__main__':
    
    # Create the environment
    config = ConfigManager(algo="PPO")
    logger = tf_log.Logger(TB_LOGS_DIR)
 
    if os.path.exists(SAVE_PATH + ".pkl"):
        with open(ALTA_LOGS + "seed.txt", "r") as f:
            seed = int(f.readline())
        print("Using the pre-initialized seed: {}".format(seed))
        set_global_seeds(seed)

        IMAGES_PATH = ALTA_LOGS+'test_images_' + config.config["city_name"] + '/'
        VIDEO_PATH = ALTA_LOGS+'test_videos_' + config.config["city_name"] + '/'
        IMAGES_PATH_VAE = ALTA_LOGS+'test_vae_images_' + config.config["city_name"] + '/'
        VIDEO_PATH_VAE = ALTA_LOGS+'test_vae_videos_' + config.config["city_name"] + '/'
        
        vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
        vis_wrapper_vae = vis_module.vis(IMAGES_PATH_VAE, VIDEO_PATH_VAE, FRAME_SKIP)
        
        env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, vis_wrapper_vae=vis_wrapper_vae, logger=logger)
        dummy_env = DummyVecEnv([lambda: env])
        
        vae.load(PATH_MODEL_VAE)
        env.set_vae(vae)
        
        model = PPO.load(SAVE_PATH, dummy_env)
        success_episodes = 0
        results = {}
        for ind in range(1):
            obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
            obs[:] = env.reset(unseen=True, index=ind)
            done = False
            while not done:
                actions = model.step(obs, deterministic=False)[0]
                info = env.step(actions)
                done = info[2]
                obs = np.expand_dims(info[0], axis=0)
            
            print("Termination State: {}".format(info[3]['termination_state']))
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
    else:
        IMAGES_PATH = ALTA_LOGS+'images/'
        VIDEO_PATH = ALTA_LOGS+'videos/'
        IMAGES_PATH_VAE = ALTA_LOGS+'vae_images_' + config.config["city_name"] + '/'
        VIDEO_PATH_VAE = ALTA_LOGS+'vae_videos_' + config.config["city_name"] + '/'
        
        vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP)
        vis_wrapper_vae = vis_module.vis(IMAGES_PATH_VAE, VIDEO_PATH_VAE, FRAME_SKIP)

        env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, vis_wrapper_vae=vis_wrapper_vae, logger=logger)
        dummy_env = DummyVecEnv([lambda: env])
        vae.load(PATH_MODEL_VAE)
        env.set_vae(vae)
        # Register the policy, it will check that the name is not already taken
        register_policy('CustomPolicy', CustomPolicy)
        
        model = PPO(policy=CustomPolicy, env=dummy_env, n_steps=500, nminibatches=10, verbose=1,
                       tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=True, learning_rate=5e-4)
        steps = 1000000
        if any(fname.endswith('.pkl') for fname in os.listdir(ALTA_LOGS)):
            with open(ALTA_LOGS + "seed.txt", "r") as f:
                seed = int(f.readline())
            print("Using the pre-initialized seed: {}".format(seed))
            set_global_seeds(seed)
            completed_steps, latest_model = get_latest_model(log_dir=ALTA_LOGS, ext='*.pkl', sep='hts')
            completed_episodes, _ = get_latest_model(log_dir=ALTA_LOGS + 'videos/', ext='*.mp4', sep='log_')
            print("Loading Latest model!!!")
            model.load(latest_model, dummy_env)
            print("Model: {} loaded successfully".format(latest_model))
            env.total_steps = completed_steps
            env.episode_num = completed_episodes
            _, best_model = model.learn(steps, completed_steps, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=False, seed=seed)    
        else:
            dt = datetime.now()
            millis = dt.microsecond
            print(millis)
            with open(ALTA_LOGS + "seed.txt", "w") as f:
                f.write(str(millis))
            _, best_model = model.learn(steps, 0, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=True, seed=millis)
        
        best_model.save(SAVE_PATH)