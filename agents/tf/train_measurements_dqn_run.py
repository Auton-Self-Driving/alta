import sys, os, glob

from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import ConfigManager

import numpy as np
import time
import vis_module
import traceback

from datetime import datetime
import tensorboard_logging as tf_log

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
# from custom_dqn import Custom_DQN
from custom_dqn_new import Custom_DQN

def get_scratch_dir(base_log_dir):
    return base_log_dir.split(base_log_dir.split("/home")[0])[1].replace("/home", "/home/scratch")


# TODO: Modify that it works for episodes too.
def get_latest_model(log_dir, ext='*.pkl', sep='_'):
        list_of_files = glob.glob(log_dir + ext)
        latest_file = max(list_of_files, key=os.path.getctime)
        latest_file = latest_file.split('{}'.format(ext[1:]))[0]
        ind = int(latest_file.split(sep)[1])
        return ind, latest_file

def run_dqn(args, prefix, config):

    ALTA_LOGS = os.path.join(args.base_log_dir, prefix.split('_runid_')[0], prefix)
    if ALTA_LOGS[-1] != '/':
        ALTA_LOGS += '/'

    # if os.path.exists('/home/scratch'):
    #     SCRATCH_DIR = os.path.join(get_scratch_dir(args.base_log_dir), prefix.split('_runid_')[0], prefix)
    # else:
    #     SCRATCH_DIR = ALTA_LOGS

    SCRATCH_DIR = ALTA_LOGS
    if SCRATCH_DIR[-1] != '/':
        SCRATCH_DIR += '/'

    if not os.path.exists(ALTA_LOGS):
        os.makedirs(ALTA_LOGS)

    TF_MODELS = ALTA_LOGS+'tf-models/checkpoint/'
    if not os.path.exists(TF_MODELS):
        os.makedirs(TF_MODELS)

    FRAME_SKIP = 1
    SAVE_PATH = ALTA_LOGS + 'dqn_measurements_weights'
    TB_LOGS_DIR = ALTA_LOGS+'tb/'

    MAX_TRIALS = 1
    steps = args.timesteps
    

    for i in range(MAX_TRIALS):
        try:
            # Create the environment
            logger = tf_log.Logger(TB_LOGS_DIR)
            
            if args.test:
                print('Testing Begins')
                
                # We want to keep same seed for testing across agents
                np.random.seed(10)
                if args.city_name == 'Town01':
                    spawn_points_fixed_idx = np.array([np.random.permutation(257) for i in range(args.test_trails)])
                elif args.city_name == 'Town02':
                    spawn_points_fixed_idx = np.array([np.random.permutation(101) for i in range(args.test_trails)])
                
                rewards = []
                successes = []
                for test_idx in range(args.test_trails):
                    IMAGES_PATH = SCRATCH_DIR+'test_images1_' + config.config["city_name"] + config.config['scenarios'] + '_run_' + str(test_idx) + '/'
                    VIDEO_PATH = SCRATCH_DIR+'test_videos1_' + config.config["city_name"] + config.config['scenarios'] +  '_run_' + str(test_idx) + '/'
                    IMAGES_PATH_VAE = SCRATCH_DIR+'test_vae_images1_' + config.config["city_name"] + config.config['scenarios'] +  '_run_' + str(test_idx) + '/'
                    VIDEO_PATH_VAE = SCRATCH_DIR+'test_vae_videos1_' + config.config["city_name"] +  config.config['scenarios'] + '_run_' + str(test_idx) + '/'

                    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP, videos=config.config["videos"])
                    vis_wrapper_vae = vis_module.vis(IMAGES_PATH_VAE, VIDEO_PATH_VAE, FRAME_SKIP, videos=config.config["videos"])

                    config.config['spawn_points_fixed_idx'] = list(spawn_points_fixed_idx[test_idx])
                    
                    # Sending logger as None so as to not affect existing validation plots
                    env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, vis_wrapper_vae=vis_wrapper_vae, logger=None, log_dir=ALTA_LOGS)
                    dummy_env = DummyVecEnv([lambda: env])

                    model = DQN.load(args.agent_model_path, dummy_env)
                    with open(ALTA_LOGS + 'test_results1_' + config.config["city_name"] +  config.config['scenarios'] +  '_run_' + str(test_idx) + ".txt", "w") as f:
                        #TODO: Add test() method
                        # total_reward, success_episodes, results = test(model, env)
                        total_reward, success_episodes, results = 0, 0, 0
                        print("Task Name: {}".format(config.config["scenarios"]))
                        print("Town Name: {}".format(config.config["city_name"]))
                        print("Results of test scenarios")
                        print(results)
                        print("Total Success Episodes: {}".format(success_episodes))
                        f.write("Task Name: {}\n".format(config.config["scenarios"]))
                        f.write("Town Name: {}\n".format(config.config["city_name"]))
                        f.write("Results of test scenarios\n")
                        f.write(str(results))
                        f.write("Total Success Episodes: {}\n".format(str(success_episodes)))
                        f.write("Spawn Points Permutation: {}\n".format(str(env.config['spawn_points_fixed_idx'])))
                    rewards.append(total_reward)
                    successes.append(success_episodes)
                    env.close()
                rewards = np.array(rewards)
                successes = np.array(successes)
                with open(ALTA_LOGS + 'final_test_results1_' + config.config["city_name"]+  config.config['scenarios'] + ".txt", "w") as f:
                    f.write("Task Name: {}\n".format(config.config["scenarios"]))
                    f.write("Town Name: {}\n".format(config.config["city_name"]))
                    f.write("Model path used for testing: {}\n".format(args.agent_model_path))
                    f.write("Results of final testing\n")
                    f.write("Rewards: {}\n".format(" ".join(map(str, rewards))))
                    f.write("Success: {}\n".format(" ".join(map(str, successes))))
                    f.write("Avg Success: {}\n".format(np.mean(successes)))
                    f.write("Std Success: {}\n".format(np.std(successes)))
            else:
                print("Training begins")
                IMAGES_PATH = SCRATCH_DIR+'images/'
                VIDEO_PATH = SCRATCH_DIR+'videos/'
                IMAGES_PATH_VAE = SCRATCH_DIR+'vae_images/'
                VIDEO_PATH_VAE = SCRATCH_DIR+'vae_videos/'
                
                vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP, videos=config.config["videos"])
                vis_wrapper_vae = None

                env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, vis_wrapper_vae=vis_wrapper_vae, logger=logger, log_dir=ALTA_LOGS)
                dummy_env = DummyVecEnv([lambda: env])

                policy = MlpPolicy

                # TODO: Need to add resume training logic if replay buffer gets saved.
                
                dt = datetime.now()
                millis = dt.microsecond
                print(millis)
                with open(ALTA_LOGS + "seed.txt", "w") as f:
                    f.write(str(millis))
                if args.agent_model_path is None:
                    # model = DQN(policy=policy, env=dummy_env, learning_rate=args.lr, buffer_size=args.buffer_size, exploration_fraction=0.1,
                    #             exploration_final_eps=0.02, batch_size=32, prioritized_replay=False, param_noise=False,
                    #             tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=False)
                    model = Custom_DQN(policy=policy, env=dummy_env, learning_rate=args.lr, buffer_size=args.buffer_size, exploration_fraction=0.025,
                                exploration_final_eps=0.05, batch_size=256, prioritized_replay=args.prioritized_replay, param_noise=args.param_noise,
                                tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=args.full_tensorboard_log)
                else:
                    model = Custom_DQN.load(args.agent_model_path, dummy_env)
                    print("Loading pretrained agent from: {}".format(args.agent_model_path))
                # best_model = model.learn(steps, seed=millis)
                best_model = model.learn(env, steps, tb_log_name="DQN", save_file=SAVE_PATH)
                
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