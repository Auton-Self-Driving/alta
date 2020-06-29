import sys, os, glob
sys.path.append('./../../')
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))

from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import ConfigManager

import numpy as np
import time
import vis_module
import traceback

from datetime import datetime
import tensorboard_logging as tf_log

# PPO specific
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.policies import register_policy
from models import Policy_1_layer, Policy_2_layer, CustomPolicy1, CustomPolicy2
import gym
import ipdb
st = ipdb.set_trace
np.random.seed(5)
def test(model, env, dump_results=False, path='.', model_step=None):
    dummy_env = DummyVecEnv([lambda: env])
    success_episodes = 0
    collision_obs_episodes = 0
    collision_out_of_road_episodes = 0
    collision_lane_change_episodes = 0
    static_episodes = 0
    max_steps_episodes = 0
    runover_light_episodes = 0
    results = {}
    total_reward = 0
    #env.reset()
    saved_scenarios = env.base_dir+"/testing_scenarios"
    if not os.path.exists(saved_scenarios):
        os.makedirs(saved_scenarios)

    for ind in range(env.config["num_episodes"]):
        print(ind)
        obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        obs[:] = env.reset(unseen=True, index=ind, saved_scenarios = saved_scenarios)
        done = False
        reward = 0
        
        while not done:
            actions = model.step(obs, deterministic=True)[0]
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
            if info[3]['obs_collision']:
                collision_obs_episodes += 1
            elif info[3]['lane_change']:
                collision_lane_change_episodes += 1
            elif info[3]['out_of_road']:
                collision_out_of_road_episodes += 1
            elif info[3]['termination_state'] == 'runover_light':
                runover_light_episodes += 1
            elif info[3]['termination_state'] == 'static':
                static_episodes += 1
            elif info[3]['termination_state'] == 'max_steps':
                max_steps_episodes += 1
        #env.destroy_all_existing_actors()
        

    #env.client.stop_recorder()
    env.reset()
    print("Results of train scenarios")
    print(results)
    print("# Success: {}, # Obstacle Collision: {}, # Lane-change Collision: {}, Out-of-road Collision: {}, Runover light: {}, Static: {}, Max_steps: {}".format(success_episodes,
                                collision_obs_episodes, collision_lane_change_episodes, collision_out_of_road_episodes, runover_light_episodes, static_episodes, max_steps_episodes))
    if dump_results:
        env.logger.log_scalar('test/term_success', success_episodes, model_step)
        env.logger.log_scalar('test/term_obstacle', collision_obs_episodes, model_step)
        env.logger.log_scalar('test/term_out_of_road', collision_out_of_road_episodes, model_step)
        env.logger.log_scalar('test/term_lane_change', collision_lane_change_episodes, model_step)
        env.logger.log_scalar('test/term_runover_light', runover_light_episodes, model_step)
        env.logger.log_scalar('test/term_static', static_episodes, model_step)
        env.logger.log_scalar('test/term_max_steps', max_steps_episodes, model_step)
        env.logger.log_scalar('test/total_reward', total_reward, model_step)

        with open(path + 'test_results.csv','a') as f:
                csvwriter = csv.writer(f, delimiter=',')
                csvwriter.writerow([model_step, success_episodes, total_reward, collision_obs_episodes,
                        collision_out_of_road_episodes, collision_lane_change_episodes, runover_light_episodes, static_episodes, max_steps_episodes])
    return total_reward, success_episodes, results

def get_scratch_dir(base_log_dir):
    return base_log_dir.split(base_log_dir.split("/home")[0])[1].replace("/home", "/home/scratch")

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

    if not os.path.exists(ALTA_LOGS):
        os.makedirs(ALTA_LOGS)

    POLICY_PLOTS = ALTA_LOGS + 'policy_plots/'
    if not os.path.exists(POLICY_PLOTS):
        os.makedirs(POLICY_PLOTS)

    TF_MODELS = ALTA_LOGS+'tf-models/checkpoint/'
    if not os.path.exists(TF_MODELS):
        os.makedirs(TF_MODELS)

    FRAME_SKIP = 1
    MODEL_PATH = os.path.join(ALTA_LOGS, 'models')
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    SAVE_PATH = ALTA_LOGS + 'ppo2_weights'
    TB_LOGS_DIR = ALTA_LOGS+ 'tb/'

    MAX_TRIALS = 5
    
    steps = args.timesteps
    
    def get_latest_model(log_dir=ALTA_LOGS, ext='*.pkl', sep='_'):
        list_of_files = glob.glob(log_dir + ext)
        latest_file = max(list_of_files, key=os.path.getctime)
        latest_file = latest_file.split('{}'.format(ext[1:]))[0]
        ind = int(latest_file.split(sep)[1])
        return ind, latest_file

    def get_completed_episodes(log_dir=ALTA_LOGS, ext='*.pkl', sep1='_step_', sep2='_ind_'):
        list_of_files = glob.glob(log_dir + ext)
        latest_file = max(list_of_files, key=os.path.getctime)
        latest_file = latest_file.split('{}'.format('.' + ext.split('.')[1]))[0]
        ind = int(latest_file.split(sep1)[1].split(sep2)[0])
        return ind, latest_file

    
    for i in range(MAX_TRIALS):
        try:
            # Create the environment
            logger = tf_log.Logger(TB_LOGS_DIR)
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
                    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP, videos=config.config["videos"])

                    config.config['spawn_points_fixed_idx'] = list(spawn_points_fixed_idx[test_idx])

                    # Sending logger as None so as to not affect existing validation plots
                    env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=None, log_dir=ALTA_LOGS)
                    dummy_env = DummyVecEnv([lambda: env])
                    model = PPO.load(args.agent_model_path, dummy_env)

                    with open(ALTA_LOGS + 'test_results_' + config.config["city_name"] +  config.config['scenarios'] +  '_run_' + str(test_idx) + ".txt", "w") as f:
                        total_reward, success_episodes, results = test(model, env)
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
                with open(ALTA_LOGS + 'final_test_results_' + config.config["city_name"]+  config.config['scenarios'] + ".txt", "w") as f:
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

                IMAGES_PATH = SCRATCH_DIR+'images/'
                VIDEO_PATH = SCRATCH_DIR+'videos/'
                vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP, videos=config.config["videos"])

                config.config['spawn_points_fixed_idx'] = list(spawn_points_fixed_idx)

                # Sending logger as None so as to not affect existing validation plots
                env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=None, log_dir=ALTA_LOGS)
                dummy_env = DummyVecEnv([lambda: env])

                rewards = []
                successes = []
                updates = []
                model_files = [os.path.join(ALTA_LOGS, model) for model in os.listdir(ALTA_LOGS) if model.endswith('.pkl')]
                model_files = sorted(model_files, key=os.path.getctime)

                update = 0
                for model_file in model_files[:-1]:
                    model = PPO.load(model_file, dummy_env)
                    total_reward, success_episodes, results = test(model, env)
                    print("Model: {}, Success: {}, Reward: {}".format(model_file, success_episodes, total_reward))
                    rewards.append(total_reward)
                    successes.append(success_episodes)
                    updates.append(update)
                    plot_test_results(successes, rewards, updates, ALTA_LOGS)
                    with open(ALTA_LOGS + 'test_results2.csv','a') as f:
                        csvwriter = csv.writer(f, delimiter=',')
                        csvwriter.writerow([update, success_episodes, total_reward])
                    update += 40000
                env.close()
            else:
                print("Training begins")
                IMAGES_PATH = SCRATCH_DIR+'images/'
                VIDEO_PATH = SCRATCH_DIR+'videos/'
                vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP, videos=config.config["videos"])
                if args.task=="self-driving":
                    print("Creating Carla Env")         
                    from ppo import PPO
                    env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=logger, log_dir=ALTA_LOGS)
                else:
                    print("Creating Mujoco Env")
                    from ppo_mujoco import PPO
                    env = gym.make(args.task)
                dummy_env = DummyVecEnv([lambda: env])
                
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
                if config.config["ent_coef"]==-1:
                    config.config["ent_coef"]==0.005
                model = PPO(policy=policy, env=dummy_env, n_steps=args.n_steps, nminibatches=4, verbose=1, learning_rate=args.lr, 
                        tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=False, ent_coef=config.config["ent_coef"],
                        noptepochs=args.no_epochs, cliprange=args.clip)
                if any(fname.endswith('.pkl') for fname in os.listdir(ALTA_LOGS)):
                    with open(ALTA_LOGS + "seed.txt", "r") as f:
                        seed = int(f.readline())
                    print("Using the pre-initialized seed: {}".format(seed))
                    set_global_seeds(seed)
                    completed_steps, latest_model = get_latest_model(log_dir=ALTA_LOGS, ext='*.pkl', sep='hts')
                    env.total_steps = completed_steps
                    completed_episodes, _ = get_completed_episodes(log_dir=ALTA_LOGS + 'val_episode_info_plots/', ext='*.png', sep1='_TrainEp_', sep2='_step_')
                    env.episode_num = completed_episodes
                    print(env.total_steps, env.episode_num)
                    print("Completed episodes: {}".format(completed_episodes))
                    print("Loading Latest model!!!")
                    model = PPO.load(latest_model, dummy_env)
                    print("Model: {} loaded successfully".format(latest_model))
                    best_model = model.learn(steps, completed_steps, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=False, seed=seed, policy_plots=False)
                else:
                    dt = datetime.now()
                    millis = dt.microsecond
                    print(millis)
                    with open(ALTA_LOGS + "seed.txt", "w") as f:
                        f.write(str(millis))
                    best_model = model.learn(steps, 0, env, tb_log_name="PPO2", save_file=SAVE_PATH, reset_num_timesteps=True, seed=millis, policy_plots=False, custom_logger = logger)
                
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

if __name__ == '__main__':
    #run_ids = [3,1,4,5]
    run_ids = [3]
    base_log_dir = '/home/scratch/vkadi/alta-logs/sac_vs_ppo_dynamic-navigation/'

    config = ConfigManager(algo="PPO")
    config.config["videos"] = True
    config.config["carla_gpu"] = '2'
    config.config["code_gpu"]  = '2'
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(config.config["code_gpu"])
    config.config["testing"] = True
    config.config["test_fixed_spawn_points"] = True
    config.config["city_name"] = "Town02"
    config.config["input_type"] = "wp_obs_info_speed_steer_ldist_goal_light"
    config.config["num_npc"] = 20
    config.config['spawn_points_fixed_idx'] = np.load(base_log_dir+'spawn_pt_order_2.npy')
    config.config["ent_coef"] = -1
    config.config["n_steps"] = 1
    config.config["gradient_steps_per_iteration"] = 1
    config.config["target_update_interval"] = 1
    config.config["task"] = "self-driving"
    config.config["network"] = "2_layer"
    config.config["num_npc_lower_threshold"] = 15
    #config.config["num_npc_upper_threshold"] = 20
    config.config["num_episodes"] = 25

    num_successes = []
    tot_rewards = []


    base_prefix = 'algo_PPO_task_self-driving_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__mb_4__dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000/'
    for run_id in run_ids:
        prefix = 'algo_PPO_task_self-driving_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__mb_4__dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000_runid_run'+str(run_id)+'/'
    
        ALTA_LOGS = base_log_dir + base_prefix + prefix

        IMAGES_PATH = ALTA_LOGS+'images/'
        VIDEO_PATH = ALTA_LOGS+'videos/'
        vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, 1)

        logger = None
        set_global_seeds(5)
        env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, logger=None, log_dir = ALTA_LOGS, base_prefix = None, prefix = None)
        dummy_env = DummyVecEnv([lambda: env])

        MODEL_PATH = ALTA_LOGS+'ppo2_weights3600000.pkl'

        from ppo import PPO
        model = PPO.load(MODEL_PATH, dummy_env)
        tot_reward, success_episodes, _ = test(model, env, False, path = ALTA_LOGS, model_step=0)
        print(success_episodes)
        print(tot_reward)
        num_successes.append(success_episodes)
        tot_rewards.append(tot_reward)
    print(num_successes)
    print(tot_rewards)
    env.close()

