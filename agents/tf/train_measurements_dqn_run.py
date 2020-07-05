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
from custom_dqn_ebu import Custom_DQN_EBU
import csv, os
import matplotlib.pyplot as plt
from environment.carla_9_4.config import DISCRETE_ACTIONS


def compute_discounted_returns(rewards, gamma):
    returns = np.zeros_like(rewards)
    n = np.size(rewards)
    
    returns[-1] = rewards[-1]
    for i in range(n-2, 0, -1):
        returns[i] = rewards[i] + gamma* returns[i+1]

    return returns

def test(model, env, path=None):
    
    
    '''
    This method is used for testing a model.
    Current it includes hard-coded changes for testing with custom actions.
    TODO: Remove this function and reuse function from custom_dqn_new.py
    '''
    
    dummy_env = DummyVecEnv([lambda: env])
    # dummy_env = env

    success_episodes = 0
    e_obs_collision = 0
    e_out_of_road = 0
    e_lane_change = 0
    e_runover_light = 0
    e_static = 0
    e_max_steps = 0
    e_max_steps_obstacle = 0
    e_max_steps_light = 0
    e_unexpected_collision = 0
    e_unknown = 0

    results = {}
    total_reward = 0
    # import pdb;
    # pdb.set_trace()
    for ind in range(25):
        obs = np.zeros((dummy_env.num_envs,) + dummy_env.observation_space.shape)
        # obs[:] = env.reset(unseen=True, index=ind)
        obs[:] = env.reset(unseen=True, index=ind, expert_agent=True)
        done = False
        reward = 0
        
        q_values_matrix = []
        q_values_matrix_normalized = []
        rewards = []
        action_q_values = []
        actions_taken = []
        validation_ep_index = '0'
        t = 0
        last_t = 245
        thres = last_t - ind
        while not done:
            control = env.vehicle_agent.run_step(debug=True)

            # action = control

            steer = control.steer
            brake = control.brake

            if steer < -0.49:
                steer = -0.5
            elif steer >= -0.49 and steer < -0.29:
                steer = -0.3
            elif steer >= -0.29 and steer < -0.05:
                steer = -0.1
            elif steer >= -0.05 and steer < 0.05:
                steer = 0.0
            elif steer >= 0.05 and steer < 0.29:
                steer = 0.1
            elif steer >= 0.29 and steer < 0.49:
                steer = 0.3
            elif steer > 0.49:
                steer = 0.5

            # if steer < -0.4:
            #     steer = -0.5
            # elif steer >= -0.4 and steer < -0.2:
            #     steer = -0.3
            # elif steer >= -0.2 and steer < -0.05:
            #     steer = -0.1
            # elif steer >= -0.05 and steer < 0.05:
            #     steer = 0.0
            # elif steer >= 0.05 and steer < 0.2:
            #     steer = 0.1
            # elif steer >= 0.2 and steer < 0.4:
            #     steer = 0.3
            # elif steer > 0.4:
            #     steer = 0.5

            if brake > 0:
                target_speed = 0
            else:
                target_speed = 20

            for i in range(len(DISCRETE_ACTIONS)):
                if DISCRETE_ACTIONS[i][0] == target_speed and DISCRETE_ACTIONS[i][1] == steer:
                    action = np.array(i)

            # control = env.expert_agent.run_step()
            
            # action, q_values, actions_proba = model.predict(obs, deterministic=True)
            # # action = np.array([4])
            # # if t >= 476:
            # #     action = np.array([1])
            # if t >= thres and t < last_t:
            #     action = np.array([3])
            # q_values_matrix.append(q_values[0])
            # q_values_matrix_normalized.append(actions_proba[0])
            # action_q_values.append(q_values[0][action])
            # actions_taken.append(action)
            
            t = t+1

            info = env.step(action)
            # print(info)
            reward += info[1][0]
            done = info[2]
            obs = np.expand_dims(info[0], axis=0)
            rewards.append(info[1][0])
            
            if done:
                validation_ep_index = info[3]['val_ep_idx']
        
        total_reward += reward
        if info[3]['termination_state'] == 'success':
            success_episodes += 1
            results[ind] = 1
        else:
            results[ind] = 0
            if info[3]['termination_state'] == 'obs_collision':
                e_obs_collision += 1
            elif info[3]['termination_state'] == 'out_of_road':
                e_out_of_road += 1
            elif info[3]['termination_state'] == 'lane_invasion':
                e_lane_change += 1
            elif info[3]['termination_state'] == 'runover_light':
                e_runover_light += 1
            elif info[3]['termination_state'] == 'static':
                e_static += 1
            elif info[3]['termination_state'] == 'max_steps':
                e_max_steps += 1
            elif info[3]['termination_state'] == 'max_steps_obstacle':
                e_max_steps_obstacle += 1
            elif info[3]['termination_state'] == 'max_steps_light':
                e_max_steps_light += 1
            elif info[3]['termination_state'] == 'unexpected_collision':
                e_unexpected_collision += 1
            else:
                e_unknown += 1
        
        with open(path + 'my_test_results.csv','a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([ind, success_episodes, total_reward[0],
                e_obs_collision,  e_out_of_road, e_lane_change,
                e_runover_light, e_static, e_max_steps, e_max_steps_obstacle, e_max_steps_light])

        # action_q_values = np.array(action_q_values)
        # actions_taken = np.array(actions_taken)
        # returns = compute_discounted_returns(np.array(rewards), gamma=0.975) 

        # plot q_values for this validation episode
        # plot_q_values(np.array(q_values_matrix), np.array(q_values_matrix_normalized),
        # validation_ep_index, returns, action_q_values, actions_taken, path)

    # Reset env after testing
    # env.reset()
    # print("Results of train scenarios")
    # print(results)
    # print("Step: {0} Total Success Episodes: {1}".format(model_step, success_episodes))

    # with open(path + 'test_results.csv','a') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow([model_step, success_episodes, total_reward,
    #         e_obs_collision,  e_out_of_road, e_lane_change,
    #         e_runover_light, e_static, e_max_steps])

    return total_reward, success_episodes

def plot_q_values(q_values_matrix, q_values_matrix_normalized, validation_ep_index,
                returns, action_q_values, actions_taken, path):
    # import pdb; pdb.set_trace()
    path = os.path.join(path, 'Test_qvalue_plots')
    if not os.path.exists(path):
        os.makedirs(path)
    # Transpose the matrix
    q_values_matrix = q_values_matrix.T
    q_values_matrix_normalized = q_values_matrix_normalized.T

    figure_name = os.path.join(path, validation_ep_index + '_qvalues.png')
    # fig, ax = plt.subplots()
    fig, (ax1, ax2, ax3, ax4, ax5)  = plt.subplots(5, 1, figsize=(12, 12))
    fig.suptitle('Q Values {}'.format(validation_ep_index))

    ax1.matshow(q_values_matrix, cmap=plt.cm.Blues, aspect='auto')
    ax2.matshow(q_values_matrix_normalized, cmap=plt.cm.Blues, aspect='auto')
    ax5.hist(q_values_matrix.reshape(-1), bins=20, density=True)
    # row, col = np.shape(q_values_matrix)
    # for i in range(row):
    #     for j in range(col):
    #         c = q_values_matrix[i,j]
    #         ax.text(j, i, str(c), va='center', ha='center')

    returns = returns.reshape(-1)
    action_q_values = action_q_values.reshape(-1)
    n = np.size(returns)
    ax3.plot(np.arange(n), np.array(returns),'b')
    ax3.plot(np.arange(n), np.array(action_q_values),'g')
    ax3.legend('r', 'q', loc='best')
    ax3.set_xlim(left=0, right=n)

    actions_taken = actions_taken.reshape(-1)
    ax4.plot(np.arange(n), np.array(actions_taken),'b')
    ax4.set_xlim(left=0, right=n)

    ax3.set_ylabel('returns(b)')
    ax4.set_ylabel('actions')

    ax1.set_ylabel('Actions')
    
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Actions (Normalized))')
    
    plt.savefig(figure_name)
    plt.close()

    # also saving arrays
    fpath = os.path.join(path, validation_ep_index + '_qvalues.txt')
    with open(fpath, 'a') as f:
        f.write("\nq_values_matrix\n")
        f.write(str(q_values_matrix))
        f.write("\n q_values_matrix_normalized\n")
        f.write(str(q_values_matrix_normalized))
        f.write("\n returns\n")
        f.write(str(returns))
        f.write("\n action_q_values\n")
        f.write(str(action_q_values))
        f.write("\n actions_taken\n")
        f.write(str(actions_taken))
    
    fpath_array = os.path.join(path, validation_ep_index + '_qvalues_array')
    np.savez(fpath_array, q_values_matrix= q_values_matrix, q_values_matrix_normalized=q_values_matrix_normalized,
             returns=returns, action_q_values=action_q_values, actions_taken=actions_taken)
    # fpath2 = os.path.join(path, validation_ep_index + '_qvalues.txt')
    # with open(path + 'test_results.csv','a') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow([model_step, success_episodes, total_reward,
    #         e_obs_collision,  e_out_of_road, e_lane_change,
    #         e_runover_light, e_static, e_max_steps])



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

    # Adding another folder to avoid over-writing exiting files while training from buffer
    # ALTA_LOGS += 'buffer_training/'
    
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

    MAX_TRIALS = 5
    steps = args.timesteps
    
    env = None
    model = None

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
                    vis_wrapper_vae = None

                    config.config['spawn_points_fixed_idx'] = list(spawn_points_fixed_idx[test_idx])
                    config.config["verbose"] = True
                    
                    # Sending logger as None so as to not affect existing validation plots
                    
                    RETRIES_ON_ERROR = 5
                    serverStartRetries = 0
                    serverStarted = False
                    
                    env = None
                    while ((not serverStarted) and serverStartRetries < RETRIES_ON_ERROR):
                        try:

                            env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, vis_wrapper_vae=vis_wrapper_vae, logger=None, log_dir=ALTA_LOGS)
                            serverStarted = True
                        
                        except Exception as identifier:
                            print(prefix, identifier, serverStartRetries)
                            traceback.print_exc()
                            if env is not None:
                                env.close()
                                serverStartRetries += 1
                                time.sleep(20)
                    
                    dummy_env = DummyVecEnv([lambda: env])

                    model = Custom_DQN.load(args.agent_model_path, dummy_env)
                    with open(ALTA_LOGS + 'test_results1_' + config.config["city_name"] +  config.config['scenarios'] +  '_run_' + str(test_idx) + ".txt", "w") as f:
                        #TODO: Add test() method
                        total_reward, success_episodes= test(model, env, path=ALTA_LOGS)
                        results = 0
                        # total_reward, success_episodes, results = 0, 0, 0
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
            
            elif args.train_buffer:

                TB_LOGS_DIR = ALTA_LOGS+'tb_buffer/'
                logger = tf_log.Logger(TB_LOGS_DIR)
                IMAGES_PATH = SCRATCH_DIR+'images_buffer/'
                VIDEO_PATH = SCRATCH_DIR+'videos_buffer/'
                IMAGES_PATH_VAE = SCRATCH_DIR+'vae_images_buffer/'
                VIDEO_PATH_VAE = SCRATCH_DIR+'vae_videos_buffer/'
                
                vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP, videos=config.config["videos"])
                vis_wrapper_vae = None

                RETRIES_ON_ERROR = 5
                serverStartRetries = 0
                serverStarted = False
                env = None
                # import pdb
                # pdb.set_trace()
                while ((not serverStarted) and serverStartRetries < RETRIES_ON_ERROR):
                    try:

                        env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, vis_wrapper_vae=vis_wrapper_vae, logger=logger, log_dir=ALTA_LOGS)
                        serverStarted = True
                    
                    except Exception as identifier:
                        print(prefix, identifier, serverStartRetries)
                        traceback.print_exc()
                        if env is not None:
                            env.close()
                            serverStartRetries += 1
                            time.sleep(20)
                
                dummy_env = DummyVecEnv([lambda: env])
            
                model = Custom_DQN.load(args.agent_model_path, env)
                SAVE_PATH = ALTA_LOGS + 'dqn_measurements_weights_buffer'
                model.learn_from_buffer(env, 50000, tb_log_name="DQN", save_file=SAVE_PATH)

            else:
                print("Training begins, Retry run iteration: {0}", i)
                IMAGES_PATH = SCRATCH_DIR+'images/'
                VIDEO_PATH = SCRATCH_DIR+'videos/'
                IMAGES_PATH_VAE = SCRATCH_DIR+'vae_images/'
                VIDEO_PATH_VAE = SCRATCH_DIR+'vae_videos/'
                
                vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, FRAME_SKIP, videos=config.config["videos"])
                vis_wrapper_vae = None

                RETRIES_ON_ERROR = 5
                serverStartRetries = 0
                serverStarted = False
                # env = None
                if env is not None:
                    completed_steps = env.total_steps
                    completed_episodes = env.episode_num
                else:
                    completed_steps = 0
                    completed_episodes = 0

                while ((not serverStarted) and serverStartRetries < RETRIES_ON_ERROR):
                    try:

                        env = CarlaEnv(config=config.config, vis_wrapper=vis_wrapper, vis_wrapper_vae=vis_wrapper_vae, logger=logger, log_dir=ALTA_LOGS)
                        serverStarted = True
                    
                    except Exception as identifier:
                        print(prefix, identifier, serverStartRetries)
                        traceback.print_exc()
                        if env is not None:
                            env.close()
                            serverStartRetries += 1
                            time.sleep(20)
                
                env.episode_num = completed_episodes
                env.total_steps = completed_steps
                
                policy = MlpPolicy

                # TODO: Need to add resume training logic if replay buffer gets saved.
                
                dt = datetime.now()
                millis = dt.microsecond
                print(millis)
                with open(ALTA_LOGS + "seed.txt", "w") as f:
                    f.write(str(millis))
                
                reset_num_timesteps = True

                latest_model_path = os.path.join(ALTA_LOGS, "dqn_measurements_weights_buffer_latest.zip")
                

                if model is not None:
                    dummy_env = DummyVecEnv([lambda: env])
                    model.env = dummy_env
                    reset_num_timesteps = False
                
                elif os.path.exists(latest_model_path):
                    
                    # Currently named it as _best_model, should have been _latest model
                    training_info_file = os.path.join(ALTA_LOGS, "dqn_measurements_weights_best_model.txt")

                    if os.path.exists(training_info_file):
                        with open(training_info_file, "r") as f:
                            line = f.readline()
                            line_info = line.split(',')
                            completed_episodes = int(line_info[0])
                            completed_steps = int(line_info[1])
                        print("Loaded training info: Completed Episodes {0}, Completed steps {1}".format(completed_episodes, completed_steps))
                    else:
                        print("Warning: Found model file, but not training info file!")
                        completed_episodes = 0
                        completed_steps = 0

                    # update environment episode number and total_steps
                    env.episode_num = completed_episodes
                    env.total_steps = completed_steps
                    dummy_env = DummyVecEnv([lambda: env])

                    # continue training from saved path
                    model = Custom_DQN.load(latest_model_path, dummy_env)
                    reset_num_timesteps = False
                    print("Loading last saved agent from: {}".format(latest_model_path))

                elif args.agent_model_path is not None:
                    # Train using agent model path provided

                    if args.train_from_scratch:
                        # Train from scratch using trained model and buffer
                        # reset optimizer and training

                        dummy_env = DummyVecEnv([lambda: env])

                        kwargs = {}
                        kwargs["skip_optimizer_state_load"] = True
                        model = Custom_DQN.load(args.agent_model_path, dummy_env, **kwargs)
                        model.num_timesteps = 0
                        model.exploration = None
                        reset_num_timesteps = True
                        model.exploration_final_eps=args.exp_final_eps
                        model.target_network_update_freq=args.target_freq
                        model.exploration_fraction=0.1
                        model.learning_starts=25000
                        model.batch_size=512
                        
                        print("Loading pretrained agent from: {}".format(args.agent_model_path))
                    else:

                        # continue training with agent model path with saved optimizer
                        dummy_env = DummyVecEnv([lambda: env])
                        model = Custom_DQN.load(args.agent_model_path, dummy_env)
                        reset_num_timesteps = False
                        print("Loading pretrained agent from: {}".format(args.agent_model_path))

                else:
                    # Create a new model and train


                    # model = DQN(policy=policy, env=dummy_env, learning_rate=args.lr, buffer_size=args.buffer_size, exploration_fraction=0.1,
                    #             exploration_final_eps=0.02, batch_size=32, prioritized_replay=False, param_noise=False,
                    #             tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=False)
                    # model = Custom_DQN(policy=policy, env=dummy_env, learning_rate=args.lr, buffer_size=args.buffer_size,
                    #             exploration_fraction=0.050,learning_starts=25000,exploration_final_eps=0.05, gamma=0.99,
                    #             batch_size=512, target_network_update_freq=2000,
                    #             prioritized_replay=args.prioritized_replay, param_noise=args.param_noise,
                    #             tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=args.full_tensorboard_log)
                    dummy_env = DummyVecEnv([lambda: env])
                    if args.ebu:
                        model = Custom_DQN_EBU(policy=policy, env=dummy_env, learning_rate=args.lr, buffer_size=args.buffer_size,
                                    exploration_fraction=0.1,learning_starts=25000,exploration_final_eps=args.exp_final_eps, gamma=0.99,
                                    batch_size=512, target_network_update_freq=args.target_freq,
                                    prioritized_replay=args.prioritized_replay, param_noise=args.param_noise,
                                    tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=args.full_tensorboard_log, ebu_beta=0.5)

                    else:
                        model = Custom_DQN(policy=policy, env=dummy_env, learning_rate=args.lr, buffer_size=args.buffer_size,
                                    exploration_fraction=0.1,learning_starts=25000,exploration_final_eps=args.exp_final_eps, gamma=0.99,
                                    batch_size=512, target_network_update_freq=args.target_freq,
                                    prioritized_replay=args.prioritized_replay, param_noise=args.param_noise,
                                    tensorboard_log=TB_LOGS_DIR, full_tensorboard_log=args.full_tensorboard_log, n_step=args.dqn_n_step)

                

                if args.expert_buffer_path is not None and args.expert_data_sample_percent > 0:
                    
                    expert_model = Custom_DQN.load(args.expert_buffer_path)
                    expert_replay_buffer = expert_model.replay_buffer
                    expert_data_sample_percent = args.expert_data_sample_percent
                    input_type = args.input_type
                else:
                    expert_replay_buffer = None
                    expert_data_sample_percent = 0.0
                    input_type = None


                # Call appropriate learn method
                if args.gen_expert_data:
                    best_model = model.generate_expert_data_nstep(env, steps, tb_log_name="DQN", save_file=SAVE_PATH, reset_num_timesteps=reset_num_timesteps)

                elif args.ebu:
                    best_model = model.learn_new_EBU(env, steps, tb_log_name="DQN", save_file=SAVE_PATH, num_opt_epochs=args.opt_epochs, reset_num_timesteps=reset_num_timesteps, val_trials=args.val_trials)

                elif args.special_sample:
                    # best_model = model.learn_new_buffer(env, steps, tb_log_name="DQN", save_file=SAVE_PATH, num_opt_epochs=args.opt_epochs)
                    best_model = model.learn_new_buffer_nstep(env, steps, tb_log_name="DQN", save_file=SAVE_PATH, num_opt_epochs=args.opt_epochs, reset_num_timesteps=reset_num_timesteps, val_trials=args.val_trials)
                else:
                    best_model = model.learn_new_nstep(env, steps, tb_log_name="DQN", save_file=SAVE_PATH, num_opt_epochs=args.opt_epochs, reset_num_timesteps=reset_num_timesteps,
                        val_trials=args.val_trials, expert_replay_buffer=expert_replay_buffer, expert_data_sample_percent=expert_data_sample_percent, input_type=input_type)
                
                best_model.save(SAVE_PATH)
            break
        except Exception as e:
            with open(ALTA_LOGS + "error.txt", "w") as f:
                print("********** Code ERROR for prefix: {} **********".format(prefix))
                print(e)
                print(traceback.format_exc())
                f.write(str(e))
                f.write(traceback.format_exc())
                # if env is not None:
                #     env.close()
        finally:
            env.close()
            time.sleep(120)