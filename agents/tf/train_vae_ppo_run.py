import sys, os, time, glob
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))

from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import ConfigManager

import numpy as np
import time
import vis_module
import traceback

from datetime import datetime
import tensorboard_logging as tf_log

from ae.controller import AEController

# PPO specific
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.policies import register_policy
from ppo import PPO, test
from models import Policy_1_layer, Policy_2_layer, CustomPolicy1, CustomPolicy2

def get_scratch_dir(base_log_dir):
    return base_log_dir.split(base_log_dir.split("/home")[0])[1].replace("/home", "/home/scratch")

def run_ppo_vae(args, prefix, config):
    ALTA_LOGS = os.path.join(args.base_log_dir, prefix.split('_runid_')[0], prefix)
    SCRATCH_DIR = os.path.join(get_scratch_dir(args.base_log_dir), prefix.split('_runid_')[0], prefix)
    
    vae = AEController(image_size=(128, 128, 5), learning_rate=args.ae_lr)
    
    if not os.path.exists(ALTA_LOGS):
        os.makedirs(ALTA_LOGS)

    TF_MODELS = ALTA_LOGS+'tf-models/checkpoint/'
    if not os.path.exists(TF_MODELS):
        os.makedirs(TF_MODELS)

    FRAME_SKIP = 1
    SAVE_PATH = ALTA_LOGS + 'ppo2_measurements_weights'
    TB_LOGS_DIR = ALTA_LOGS+'tb/'

    MAX_TRIALS = 5
    
    # Register the policy, it will check that the name is not already taken
    register_policy('CustomPolicy1', CustomPolicy1)
    register_policy('CustomPolicy2', CustomPolicy2)
    steps = args.timesteps

    def get_latest_model(log_dir=ALTA_LOGS, ext='*.pkl', sep='_'):
        list_of_files = glob.glob(log_dir + ext)
        latest_file = max(list_of_files, key=os.path.getctime)
        latest_file = latest_file.split('{}'.format(ext[1:]))[0]
        ind = int(latest_file.split(sep)[1])
        return ind, latest_file


    for i in range(MAX_TRIALS):
        try:
            # Create the environment
            logger = tf_log.Logger(TB_LOGS_DIR)
            if os.path.exists(SAVE_PATH + ".pkl"):
                print("Best model exists, Validating !!!!")
                with open(ALTA_LOGS + "seed.txt", "r") as f:
                    seed = int(f.readline())
                print("Using the pre-initialized seed: {}".format(seed))
                set_global_seeds(seed)

                IMAGES_PATH = SCRATCH_DIR+'final_images_' + config.config["city_name"] + '/'
                VIDEO_PATH = SCRATCH_DIR+'final_videos_' + config.config["city_name"] + '/'
                IMAGES_PATH_VAE = SCRATCH_DIR+'final_vae_images_' + config.config["city_name"] + '/'
                VIDEO_PATH_VAE = SCRATCH_DIR+'final_vae_videos_' + config.config["city_name"] + '/'
                
                vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP, videos=config.config["videos"])
                vis_wrapper_vae = vis_module.vis(IMAGES_PATH_VAE, VIDEO_PATH_VAE, FRAME_SKIP, videos=config.config["videos"])
                
                env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, vis_wrapper_vae=vis_wrapper_vae, logger=logger, log_dir=ALTA_LOGS)
                dummy_env = DummyVecEnv([lambda: env])
                if not args.train_vae:
                    print("Loading pretrained AE!!!")
                    vae.load(args.vae_model_path)
                env.set_vae(vae)

                model = PPO.load(SAVE_PATH, dummy_env)
                with open(ALTA_LOGS + config.config["scenarios"] + config.config["city_name"] + ".txt", "w") as f:
                    total_reward, success_episodes, results = test(model, env)
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
            else:
                print("Training begins")
                IMAGES_PATH = SCRATCH_DIR+'images/'
                VIDEO_PATH = SCRATCH_DIR+'videos/'
                IMAGES_PATH_VAE = SCRATCH_DIR+'vae_images/'
                VIDEO_PATH_VAE = SCRATCH_DIR+'vae_videos/'
                
                vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP, videos=config.config["videos"])
                vis_wrapper_vae = vis_module.vis(IMAGES_PATH_VAE, VIDEO_PATH_VAE, FRAME_SKIP, videos=config.config["videos"])

                env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, vis_wrapper_vae=vis_wrapper_vae, logger=logger, log_dir=ALTA_LOGS)
                dummy_env = DummyVecEnv([lambda: env])
                if not args.train_vae:
                    print("Loading pretrained AE!!!")
                    vae.load(args.vae_model_path)
                env.set_vae(vae)
                
                if args.network == "1_layer":
                    policy = Policy_1_layer
                elif args.network == "2_layer":
                    policy = Policy_2_layer
                elif args.network == "CustomPolicy1":
                    policy = CustomPolicy1
                elif args.network == "CustomPolicy2":
                    policy = CustomPolicy2
                else:
                    print("specify either 1_layer, 2_layer CustomPolicy1 or CustomPolicy2 as network input")
                    env.close()
                    print("exiting")
                    return
                
                if args.use_pretrained_agent:
                    print("Loading pretrained agent!!!")
                    model = PPO.load(args.agent_model_path, dummy_env)
                else:
                    model = PPO(policy=policy, env=dummy_env, n_steps=500, nminibatches=4, verbose=1, learning_rate=args.lr,
                            tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=False, ent_coef=args.ent_coef)
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
                    best_model = model.learn(steps, completed_steps, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=False, seed=seed, vae=vae, train_vae=(args.train_vae or args.finetune_vae))    
                else:
                    dt = datetime.now()
                    millis = dt.microsecond
                    print(millis)
                    with open(ALTA_LOGS + "seed.txt", "w") as f:
                        f.write(str(millis))
                    best_model = model.learn(steps, 0, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=True, seed=millis, vae=vae, train_vae=(args.train_vae or args.finetune_vae))
                
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