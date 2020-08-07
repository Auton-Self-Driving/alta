import os
import math
import numpy as np
from scipy.interpolate import spline, interp1d
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
import statistics
import argparse
from shutil import copy

font = {'size' : 36}
matplotlib.rc('font', **font)

def get_data_from_file(log_path, run_path, indexes, log_path_auton=None):
    # import pdb
    # pdb.set_trace()

    successes = []
    rewards = []
    timesteps = []
    completed_timestep = []

    new_rewards = {}
    new_success = {}
    for j in indexes:
        try:


            run_1_array = None
            run_2_array = None
            run_3_array = None

            run_arrays = [None, None, None, None]

            for run_index in [1, 2, 3]:
                for bucket_index in [0, 1, 2]:

                    file_name = os.path.join(log_path, run_path, "{}_runid_{}".format(run_path, j), "test_results_{}_{}.csv".format(run_index, bucket_index))

                    data = genfromtxt(file_name, delimiter=',')

                    if run_arrays[run_index] is None:
                        run_arrays[run_index] = np.array(data)
                    else:
                        run_arrays[run_index] = np.vstack((run_arrays[run_index], np.array(data)))
            
                # sort the arrays by timestep
                a = run_arrays[run_index]
                run_arrays[run_index] = a[a[:,0].argsort()]

            
            # success += 0
            # reward += 0
            # max_steps_obs += 0
            # for run_index in [1, 2, 3]:
            #     data = run_arrays[run_index]
            #     timestep = data[:, 0]    
                
            #     success += data[:, 1]
            #     reward += data[:, 2]
            #     max_steps_obs += data[:, 9]




            # file_name = os.path.join(log_path, run_path, "{}_runid_{}".format(run_path, j), "test_results.csv")

            # # copy file to zfsauton directory
            # if log_path_auton is not None:
            #     auton_dir = os.path.join(log_path_auton, run_path, "{}_runid_{}".format(run_path, j))
            #     if not os.path.exists(auton_dir):
            #         os.makedirs(auton_dir)

            #     file_name_auton = os.path.join(auton_dir, "test_results.csv")
            #     copy(file_name, file_name_auton)

            # data = genfromtxt(file_name, delimiter=',')

            for run_index in [1, 2, 3]:
                data = run_arrays[run_index]
                timestep = data[:, 0]    
                
                success = data[:, 1]
                reward = data[:, 2]
                max_steps_obs = data[:, 9]

            # # Adding max steps due to obstacle
            # success = success + max_steps_obs
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

def plot_success(log_path, run_path, timesteps, mean_success, min_success, max_success, figname="mean_success_val.png", title="Navigation with dynamic obstacles", log_path_auton=None):
    plt.figure(figsize=(22, 14))
    axes = plt.gca()
    axes.set_ylim(top=25)
    axes.set_ylim(bottom=0)
    plt.title(title)
    plt.xlabel('Timesteps (in M)', fontdict={'size' : 36})
    plt.ylabel('Total Success Episodes', fontdict={'size' : 36})
    plt.xticks(list(np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)), ('{}'.format(str(x)) for x in np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)))
    plt.plot(timesteps, mean_success, label='WRL',  color='royalblue')
    plt.fill_between(timesteps, min_success, max_success, color='lavender')
    plt.savefig(os.path.join(log_path, run_path, figname), dpi=200)

    if log_path_auton is not None:
        plt.savefig(os.path.join(log_path_auton, run_path, figname), dpi=200)
    
def plot_reward(log_path, run_path, timesteps, mean_reward, min_reward, max_reward, figname="mean_reward_val.png", title="Navigation with dynamic obstacles", log_path_auton=None):
    plt.figure(figsize=(22, 14))
    
    # axes = plt.gca()
    # axes.set_ylim(bottom=-5000)
    # plt.legend(loc='lower right', prop={'size' : 36})
    plt.title(title)
    plt.xlabel('Timesteps (in M)', fontdict={'size' : 36})
    plt.ylabel('Total Cumulative Reward', fontdict={'size' : 36})
    plt.xticks(list(np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)), ('{}'.format(str(x)) for x in np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)))
    plt.plot(timesteps, mean_reward, label='WRL+',  color='orangered')
    plt.fill_between(timesteps, min_reward, max_reward, color='mistyrose')
    plt.savefig(os.path.join(log_path, run_path, figname), dpi=200)

    if log_path_auton is not None:
        plt.savefig(os.path.join(log_path_auton, run_path, figname), dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser to plot reward curves for RL algos.')
    parser.add_argument('--log-path',dest='log_path',type=str,required=True,help='Log path.')
    parser.add_argument('--run-path',dest='run_path',type=str,required=True,help='Run path.')
    parser.add_argument('--title',dest='title',type=str, help='Title of plot / Scenario',default="Dynamic Navigation")
    parser.add_argument('--inds', nargs="+", type=int)
    parser.add_argument('--save-auton', dest='save_auton', action='store_true', help='Save to auton')

    args = parser.parse_args()

    # log_path = "/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae"
    # run_path = "algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.0005_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_"
    # indexes = [1, 2, 3]
    log_path = args.log_path
    run_path = args.run_path
    indexes = args.inds
    title = args.title

    if args.save_auton:
        log_path_auton = os.path.join("/zfsauton2/home/hiteshar/research/alta-logs/", log_path.split("/alta-logs/")[1])

        if not os.path.exists(log_path_auton):
            os.makedirs(log_path_auton)
    else:
        log_path_auton = None

    new_rewards, new_success = get_data_from_file(log_path, run_path, indexes, log_path_auton)
    mean_reward, min_reward, max_reward, mean_success, min_success, max_success, timesteps = compute_datapoints(new_rewards, new_success)

    # save arrays
    array_path = os.path.join(log_path, run_path, "results_arrays_val")
    np.savez(array_path, mean_reward=mean_reward, min_reward=min_reward, 
        max_reward=max_reward, mean_success=mean_success, min_success=min_success,
        max_success=max_success, timesteps=timesteps)
    
    if log_path_auton is not None:
        array_path_auton = os.path.join(log_path_auton, run_path, "results_arrays_val")
        np.savez(array_path_auton, mean_reward=mean_reward, min_reward=min_reward, 
            max_reward=max_reward, mean_success=mean_success, min_success=min_success,
            max_success=max_success, timesteps=timesteps)

    plot_success(log_path, run_path, timesteps, mean_success, min_success, max_success, figname="mean_success_val.png", title=title, log_path_auton=log_path_auton)
    plot_reward(log_path, run_path, timesteps, mean_reward, min_reward, max_reward, figname="mean_reward_val.png",title=title, log_path_auton=log_path_auton)

    for i in range(len(indexes)):
        new_rewards, new_success = get_data_from_file(log_path, run_path, indexes[i: i+1])
        mean_reward, min_reward, max_reward, mean_success, min_success, max_success, timesteps = compute_datapoints(new_rewards, new_success)
        plot_success(log_path, run_path, timesteps, mean_success, min_success, max_success, figname="mean_success{}_val.png".format(indexes[i]), title=title, log_path_auton=log_path_auton)
        plot_reward(log_path, run_path, timesteps, mean_reward, min_reward, max_reward, figname="mean_reward{}_val.png".format(indexes[i]), title=title, log_path_auton=log_path_auton)


    