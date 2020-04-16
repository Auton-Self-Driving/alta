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

log_path = "/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation/wp_obs_info_speed_steer_ldist_goal_light"
run_path = "algo_PPO_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__dynamic_navigation_pretrained_agent__npc_110_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_1000"
indexes = [5,6,7]

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

            data = genfromtxt(file_name, delimiter=',')

            timestep = data[:, 0]
            completed_timestep.append(timestep.shape[0])
            success = data[:, 1]
            reward = data[:, 2]
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

def plot_success(timesteps, mean_success, min_success, max_success, figname="mean_success.png"):
    plt.figure(figsize=(22, 14))
    plt.title("Navigation with dynamic obstacles")
    plt.xlabel('Timesteps (in M)', fontdict={'size' : 36})
    plt.ylabel('Total Success Episodes', fontdict={'size' : 36})
    plt.xticks(list(np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)), ('{}'.format(str(x)) for x in np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)))
    plt.plot(timesteps, mean_success, label='WRL',  color='royalblue')
    plt.fill_between(timesteps, min_success, max_success, color='lavender')
    plt.savefig(os.path.join(log_path, run_path, figname), dpi=200)
    
def plot_reward(timesteps, mean_reward, min_reward, max_reward, figname="mean_reward.png"):
    plt.figure(figsize=(22, 14))
    plt.plot(timesteps, mean_reward, label='WRL+',  color='orangered')
    plt.fill_between(timesteps, min_reward, max_reward, color='mistyrose')

    axes = plt.gca()
    axes.set_ylim(bottom=-5000)
    # plt.legend(loc='lower right', prop={'size' : 36})
    plt.title("Navigation with dynamic obstacles")
    plt.xlabel('Timesteps (in M)', fontdict={'size' : 36})
    plt.ylabel('Total Cumulative Reward', fontdict={'size' : 36})
    plt.xticks(list(np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)), ('{}'.format(str(x)) for x in np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)))
    plt.savefig(os.path.join(log_path, run_path, figname), dpi=200)


new_rewards, new_success = get_data_from_file(indexes)
mean_reward, min_reward, max_reward, mean_success, min_success, max_success, timesteps = compute_datapoints(new_rewards, new_success)
plot_success(timesteps, mean_success, min_success, max_success, figname="mean_success.png")
plot_reward(timesteps, mean_reward, min_reward, max_reward, figname="mean_reward.png")

for i in range(len(indexes)):
    new_rewards, new_success = get_data_from_file(indexes[i: i+1])
    mean_reward, min_reward, max_reward, mean_success, min_success, max_success, timesteps = compute_datapoints(new_rewards, new_success)
    plot_success(timesteps, mean_success, min_success, max_success, figname="mean_success{}.png".format(indexes[i]))
    plot_reward(timesteps, mean_reward, min_reward, max_reward, figname="mean_reward{}.png".format(indexes[i]))


    
