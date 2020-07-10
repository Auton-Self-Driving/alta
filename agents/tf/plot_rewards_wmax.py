import os
import math
import numpy as np
from scipy.interpolate import spline, interp1d
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
import statistics


font = {'size' : 36}
matplotlib.rc('font', **font)

log_path = "/home/scratch/hiteshar/research/alta-logs/dqn_lstjunc_steer7_speed_0_20_ac12_improv_fs3_uniform_nn128_nostpenalty1/lstjunc_steer7_throttle_0_20_ac12/"
run_path = "algo_DQN_input_wp_obs_info_speed_steer_ldist_light_nw_1_layer_lr_3e-05_exp_0.1_long_straight_npc_50_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_3_pid_fs_dqn_n_3_r_norm_24_suc_r_100_const_r_1"

# log_path = "/home/scratch/hiteshar/research/alta-logs/dqn_replayRL1_spl_sample_train2/longst_steer_throttle_1/"
# run_path = "algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_0.0001_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_5_use_pid_fs__special_sample_"

# log_path = "/home/scratch/hiteshar/research/alta-logs/dqn_replayRL1/longst_steer_throttle_3/"
# run_path = "algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_5_use_pid_fs_"

# /zfsauton2/home/hiteshar/local_scratch2/gpu3/research/alta-logs/dqn_replayRL1/longst_steer_throttle_3/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_5_use_pid_fs_/
#buffer_training/test_results_learn_with_buffer.csv"
# /home/scratch/hiteshar/research/alta-logs/dqn_replayRL1/longst_steer_throttle_3/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_5_use_pid_fs_/algo_DQN_input_wp_obs_info_speed_steer_ldist_goal_light_network_1_layer_lr_1e-05_long_straight_npc_50_buffer_1000000_brake_col_3.0_col_sp_3.0_light_3.0_light_sp_3.0_fs_5_use_pid_fs__runid_1/buffer_training/test_results_learn_with_buffer.csv
# log_path = "/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light"
# run_path = "algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__dynamic_navigation_pretrained_agent__npc_110_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000"
indexes = [1,2,3]

def get_data_from_file(indexes):
    successes = []
    rewards = []
    timesteps = []
    completed_timestep = []

    new_rewards = {}
    new_success = {}
    for j in indexes:
        try:
            file_name = os.path.join(log_path, run_path, "{}_runid_{}".format(run_path, j), "test_results.csv")
            # file_name = os.path.join(log_path, "{}_runid_{}".format(run_path, j), "buffer_training/test_results_learn_with_buffer.csv")
            # import pdb
            # pdb.set_trace()
            with open(file_name, "r") as f:
                print(file_name)
                lines = [line for line in f.readlines()]
                arr1 = np.array([line.strip().split(',') for line in lines])

                # reward = np.array([float(x[1:-1]) for x in arr1[:,2]])
                reward = np.array([float(x) for x in arr1[:,2]])
                # data = genfromtxt(file_name, delimiter=',')

                # timestep = data[:, 0]
                timestep = np.array([int(x) for x in arr1[:,0]])
                completed_timestep.append(timestep.shape[0])
                # success = data[:, 1]
                success = np.array([int(x) for x in arr1[:,1]])
                static = np.array([int(x) for x in arr1[:,7]])
                max_steps = np.array([int(x) for x in arr1[:,8]])
                success = success + static + max_steps
                # reward = data[:, 2]
                # import pdb
                # pdb.set_trace()
                for idx in range(timestep.shape[0]):
                    new_rewards.setdefault(timestep[idx], []).append(reward[idx])
                    new_success.setdefault(timestep[idx], []).append(success[idx])

                successes.append(success)
                rewards.append(reward)
                timesteps.append(timestep)
        except Exception as e:
            print("********** File Not Found: {} **********".format(file_name))
    
    return new_rewards, new_success

def compute_datapoints(new_rewards, new_success):
    mean_reward = []
    min_reward = []
    max_reward = []

    mean_success = []
    min_success = []
    max_success = []
    timesteps = []
    for key in sorted(new_rewards):
        timesteps.append(key / 1000000)
        mean_reward.append(statistics.mean(new_rewards[key]))
        min_reward.append(min(new_rewards[key]))
        max_reward.append(max(new_rewards[key]))
        
        mean_success.append(statistics.mean(new_success[key]))
        min_success.append(min(new_success[key]))
        max_success.append(max(new_success[key]))
        
    return mean_reward, min_reward, max_reward, mean_success, min_success, max_success, timesteps

def plot_success(timesteps, mean_success, min_success, max_success, figname="mean_success_wmax.png"):
    plt.figure(figsize=(22, 14))
    plt.title("St Road with dynamic actors (Single scenario) W Max")
    plt.xlabel('Timesteps (in M)', fontdict={'size' : 36})
    plt.ylabel('Total Success Episodes', fontdict={'size' : 36})
    plt.xticks(list(np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)), ('{}'.format(str(x)) for x in np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)))
    plt.plot(timesteps, mean_success, label='WRL',  color='royalblue')
    plt.fill_between(timesteps, min_success, max_success, color='lavender')
    plt.savefig(os.path.join(log_path, run_path, figname), dpi=200)
    
def plot_reward(timesteps, mean_reward, min_reward, max_reward, figname="mean_reward_wmax.png"):
    plt.figure(figsize=(22, 14))
    plt.plot(timesteps, mean_reward, label='WRL+',  color='orangered')
    plt.fill_between(timesteps, min_reward, max_reward, color='mistyrose')

    axes = plt.gca()
    # axes.set_ylim(bottom=-5000)
    # plt.legend(loc='lower right', prop={'size' : 36})
    plt.title("St Road with dynamic actors (Single scenario) W Max")
    plt.xlabel('Timesteps (in M)', fontdict={'size' : 36})
    plt.ylabel('Total Cumulative Reward', fontdict={'size' : 36})
    plt.xticks(list(np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)), ('{}'.format(str(x)) for x in np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)))
    plt.savefig(os.path.join(log_path, run_path, figname), dpi=200)


new_rewards, new_success = get_data_from_file(indexes)
mean_reward, min_reward, max_reward, mean_success, min_success, max_success, timesteps = compute_datapoints(new_rewards, new_success)
plot_success(timesteps, mean_success, min_success, max_success, figname="mean_success_wmax.png")
plot_reward(timesteps, mean_reward, min_reward, max_reward, figname="mean_reward_wmax.png")

for i in range(len(indexes)):
    new_rewards, new_success = get_data_from_file(indexes[i: i+1])
    mean_reward, min_reward, max_reward, mean_success, min_success, max_success, timesteps = compute_datapoints(new_rewards, new_success)
    plot_success(timesteps, mean_success, min_success, max_success, figname="mean_success_wmax{}.png".format(indexes[i]))
    plot_reward(timesteps, mean_reward, min_reward, max_reward, figname="mean_reward_wmax{}.png".format(indexes[i]))


    