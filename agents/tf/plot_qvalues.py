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

# font = {'size' : 36}
# matplotlib.rc('font', **font)

def get_data_from_file(log_path, run_path, indexes, log_path_auton=None):
    successes = []
    rewards = []
    timesteps = []
    completed_timestep = []

    new_rewards = {}
    new_success = {}
    for j in indexes:
        try:
            file_name = os.path.join(log_path, run_path, "{}_runid_{}".format(run_path, j), "test_results.csv")

            # copy file to zfsauton directory
            if log_path_auton is not None:
                auton_dir = os.path.join(log_path_auton, run_path, "{}_runid_{}".format(run_path, j))
                if not os.path.exists(auton_dir):
                    os.makedirs(auton_dir)

                file_name_auton = os.path.join(auton_dir, "test_results.csv")
                copy(file_name, file_name_auton)

            data = genfromtxt(file_name, delimiter=',')

            timestep = data[:, 0]
            completed_timestep.append(timestep.shape[0])
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

def plot_success(log_path, run_path, timesteps, mean_success, min_success, max_success, figname="mean_success.png", title="Navigation with dynamic obstacles", log_path_auton=None):
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
    
def plot_reward(log_path, run_path, timesteps, mean_reward, min_reward, max_reward, figname="mean_reward.png", title="Navigation with dynamic obstacles", log_path_auton=None):
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

def plot_success_CARLA(log_path, timesteps, mean_success, min_success, max_success, test_results=True, with_std=True, figname="mean_success.png", title=None):
    plt.figure(figsize=(11, 7))

    

    if not test_results:
        label_names = ['Without EPS', 'EPS: K = 1', 'EPS: K = 3', 'EPS: K = 5']
        mean_colors = ['orangered', 'lightseagreen', 'goldenrod', 'darkorchid']
        fill_colors = ['mistyrose', 'paleturquoise', 'khaki', 'mediumpurple']
        alphas = [0.5, 0.3, 0.3, 0.2]
    else:
        label_names = ['Uniform', 'Backward', 'PER']
#         label_names = ['Navigation task']
        mean_colors = ['orangered', 'lightseagreen', 'goldenrod']
        fill_colors = ['mistyrose', 'paleturquoise', 'khaki']
        alphas = [0.5, 0.3, 0.2]
    
    for ind in range(len(timesteps)):
        tsteps = [s * 1000000 for s in timesteps[ind]]
        # plot percentage out of 25
        mean_success_i = mean_success[ind] * 4
        min_success_i = min_success[ind] * 4
        max_success_i = max_success[ind] * 4
        plt.plot(tsteps, mean_success_i , label=label_names[ind], color=mean_colors[ind])
        if with_std:
            plt.fill_between(tsteps, min_success_i, max_success_i, color=fill_colors[ind], alpha=alphas[ind])

    axes = plt.gca()
    axes.set_ylim(top=100)
    # axes.set_ylim(bottom=0)
    
    if title is not None:
        plt.title(title, fontdict={'size' : 18})
    else:
        if not test_results:
            plt.title("Success Rate Vs Timesteps", fontdict={'size' : 18})
        else:
            plt.title("Success Rate Vs Timesteps", fontdict={'size' : 18})
#         plt.title("Cumulative Success Metric on Navigation task", fontdict={'size' : 18})
    plt.xlabel('Simulator Timesteps', fontdict={'size' : 18})
    plt.ylabel('Success Rate', fontdict={'size' : 18})
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    # plt.legend()
    plt.legend(loc='lower right', prop={'size' : 18})
#     plt.xticks(list(np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)), ('{}'.format(str(x)) for x in np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)))
    plt.savefig(os.path.join(log_path, figname), dpi=200)
    plt.clf()
    plt.close()

def plot_reward_CARLA(log_path, timesteps, mean_reward, min_reward, max_reward, test_results=True, with_std=True, figname="mean_reward.png", title=None):
    plt.figure(figsize=(11, 7))

    if not test_results:
        label_names = ['Without EPS', 'EPS: K = 1', 'EPS: K = 3', 'EPS: K = 5']
        mean_colors = ['orangered', 'lightseagreen', 'goldenrod', 'darkorchid']
        fill_colors = ['mistyrose', 'paleturquoise', 'khaki', 'mediumpurple']
        alphas = [0.5, 0.3, 0.3, 0.2]
    else:
#         label_names = ['with state A', 'with state A+I', 'with state I']
        # label_names = ['Navigation task']
        label_names = ['Uniform', 'Backward', 'PER']
        mean_colors = ['orangered', 'lightseagreen', 'goldenrod']
        fill_colors = ['mistyrose', 'paleturquoise', 'khaki']
        alphas = [0.5, 0.3, 0.2]
    
    for ind in range(len(timesteps)):
        tsteps = [s * 1000000 for s in timesteps[ind]]
        plt.plot(tsteps, mean_reward[ind], label=label_names[ind], color=mean_colors[ind])
        if with_std:
            plt.fill_between(tsteps, min_reward[ind], max_reward[ind], color=fill_colors[ind], alpha=alphas[ind])

    axes = plt.gca()
    axes.set_ylim()
    if title is not None:
        plt.title(title, fontdict={'size' : 18})
    else:
        if not test_results:
            axes.set_ylim(top=140000, bottom=-5000)
            plt.title("CARLA Reward", fontdict={'size' : 18})
        else:
            # axes.set_ylim(top=140000, bottom=-150000)
    #         plt.title("Cumulative Reward with Dynamic Actors", fontdict={'size' : 18})
            plt.title("Cumulative Reward", fontdict={'size' : 18})
        
    
    plt.xlabel('Simulator Timesteps', fontdict={'size' : 18})
    plt.ylabel('Cumulative Reward', fontdict={'size' : 18})
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.legend(loc='lower right', prop={'size' : 18})
    # plt.legend()
#     plt.xticks(list(np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)), ('{}'.format(str(x)) for x in np.arange(0, (math.ceil(timesteps[-1] / 0.5) + 1) * 0.5, 0.5)))
    plt.savefig(os.path.join(log_path, figname), dpi=200)
    plt.clf()
    plt.close()

def plot_q_values(returns, action_q_values, log_path, figname, title):
    
    plt.figure(figsize=(11, 7))
    
    # path = os.path.join(path, 'Validation_qvalue_plots_buffer')
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # # Transpose the matrix
    # q_values_matrix = q_values_matrix.T
    # q_values_matrix_normalized = q_values_matrix_normalized.T

    # figure_name = os.path.join(path, validation_ep_index + '_qvalues.png')
    # # fig, ax = plt.subplots()
    # fig, (ax1, ax2, ax3, ax4, ax5)  = plt.subplots(5, 1, figsize=(12, 12))
    # fig.suptitle('Q Values {}'.format(validation_ep_index))

    # ax1.matshow(q_values_matrix, cmap=plt.cm.Blues, aspect='auto')
    # ax2.matshow(q_values_matrix_normalized, cmap=plt.cm.Blues, aspect='auto')
    # ax5.hist(q_values_matrix.reshape(-1), bins=20, density=True)
    # row, col = np.shape(q_values_matrix)
    # for i in range(row):
    #     for j in range(col):
    #         c = q_values_matrix[i,j]
    #         ax.text(j, i, str(c), va='center', ha='center')

    returns = returns.reshape(-1)
    action_q_values = action_q_values.reshape(-1)
    n = np.size(returns)
    plt.plot(np.arange(n), np.array(returns),label="Discounted Return", color='b')
    plt.plot(np.arange(n), np.array(action_q_values), label="Q-Value",color='g')
    plt.legend(loc='best', prop={'size' : 18})
    
    axes = plt.gca()
    axes.set_xlim(left=0, right=n)

    plt.title(title, fontdict={'size' : 18})
    # ax3.legend('r', 'q', loc='best')
    # plt.set_xlim(left=0, right=n)

    # actions_taken = actions_taken.reshape(-1)
    # ax4.plot(np.arange(n), np.array(actions_taken),'b')
    # ax4.set_xlim(left=0, right=n)

    # ax3.set_ylabel('returns(blue)')
    # ax4.set_ylabel('actions')
    # ax5.set_ylabel('Q histogram')

    # ax1.set_ylabel('Actions')
    
    # ax2.set_xlabel('Timesteps')
    # ax2.set_ylabel('Actions (Normalized))')
    plt.xlabel('Simulator Timesteps', fontdict={'size' : 18})
    plt.ylabel('Value', fontdict={'size' : 18})

    plt.savefig(os.path.join(log_path, figname), dpi=200)
    plt.clf()
    plt.close()

    # plt.savefig(figure_name)
    # plt.close()

    # saving arrays    
    # fpath_array = os.path.join(path, validation_ep_index + '_qvalues_array')
    # np.savez(fpath_array, q_values_matrix= q_values_matrix, q_values_matrix_normalized=q_values_matrix_normalized,
    #          returns=returns, action_q_values=action_q_values, actions_taken=actions_taken)

def plot_q_value_and_return(returns, action_q_values, path):
    
    plt.figure(figsize=(11, 7))

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

    ax3.set_ylabel('returns(blue)')
    ax4.set_ylabel('actions')
    ax5.set_ylabel('Q histogram')

    ax1.set_ylabel('Actions')
    
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Actions (Normalized))')
    
    plt.savefig(figure_name)
    plt.close()

    # saving arrays    
    fpath_array = os.path.join(path, validation_ep_index + '_qvalues_array')
    np.savez(fpath_array, q_values_matrix= q_values_matrix, q_values_matrix_normalized=q_values_matrix_normalized,
             returns=returns, action_q_values=action_q_values, actions_taken=actions_taken)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser to plot reward curves for RL algos.')
    parser.add_argument('--log-path',dest='log_path',type=str,required=True,help='Log path.')
    parser.add_argument('--run-path',dest='run_path',type=str,required=True,help='Run path.')
    parser.add_argument('--success-title',dest='success_title',type=str, help='Title of plot / Scenario',default="Success Rate Vs Timesteps")
    parser.add_argument('--reward-title',dest='reward_title',type=str, help='Title of plot / Scenario',default="Cumulative Reward")
    parser.add_argument('--inds', nargs="+", type=int)
    parser.add_argument('--save-auton', dest='save_auton', action='store_true', help='Save to auton')
    parser.add_argument('--fs',dest='fs',type=int,default=1, help='frame-skip to multiply to time-steps.')

    args = parser.parse_args()

    # log_path = "/zfsauton2/home/tanmaya/projects/alta-logs/new_env/ppo_runs/dynamic_navigation_images/pretrained_ae"
    # run_path = "algo_PPO_input_wp_vae_obs_info_speed_steer_ldist_goal_light_network_CustomPolicy2_lr_0.0002_ae_lr_0.0005_dynamic_navigation_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_fstack_3_n_10000_finetune_vae_epochs_10__clip_0.2__mb_10_"
    # indexes = [1, 2, 3]
    log_path = args.log_path
    run_path = args.run_path
    indexes = args.inds
    success_title = args.success_title
    reward_title = args.reward_title
    fs = args.fs

    mean_rewards = []
    min_rewards = []
    max_rewards = []
    mean_success = []
    min_success = []
    max_success = []
    steps = []
    file_name = 'E_1080_t_390000_i_0_v_40_qvalues_array.npz'
    file_path = os.path.join(log_path, file_name)
    data = np.load(file_path)
    action_q_values = data['action_q_values']
    returns = data['returns']

    plot_q_values(returns, action_q_values,log_path, figname="Qvalue.png", title="Q-Values Vs Discounted Returns")

    # for file_name in file_names:
        
        
    #     rmean, rmin, rmax, smean, smin, smax, timesteps = data['mean_reward'], data['min_reward'], data['max_reward'], \
    #         data['mean_success'], data['min_success'], data['max_success'], data['timesteps']

    #     mean_rewards.append(rmean)
    #     min_rewards.append(rmin)
    #     max_rewards.append(rmax)
    #     mean_success.append(smean)
    #     min_success.append(smin)
    #     max_success.append(smax)
    #     # multiple by fs to have simulator timesteps
    #     steps.append(timesteps * fs)

    # plot_reward_CARLA(log_path, steps, mean_rewards, min_rewards, max_rewards, test_results=True, with_std=True, figname="mean_reward.png", title=reward_title)
    # plot_success_CARLA(log_path, steps, mean_success, min_success, max_success, test_results=True, with_std=True, figname="mean_success.png", title=success_title)

    # if args.save_auton:
    #     log_path_auton = os.path.join("/zfsauton2/home/hiteshar/research/alta-logs/", log_path.split("/alta-logs/")[1])

    #     if not os.path.exists(log_path_auton):
    #         os.makedirs(log_path_auton)
    # else:
    #     log_path_auton = None

    # new_rewards, new_success = get_data_from_file(log_path, run_path, indexes, log_path_auton)
    # mean_reward, min_reward, max_reward, mean_success, min_success, max_success, timesteps = compute_datapoints(new_rewards, new_success)

    # # save arrays
    # array_path = os.path.join(log_path, run_path, "results_arrays")
    # np.savez(array_path, mean_reward=mean_reward, min_reward=min_reward, 
    #     max_reward=max_reward, mean_success=mean_success, min_success=min_success,
    #     max_success=max_success, timesteps=timesteps)
    
    # if log_path_auton is not None:
    #     array_path_auton = os.path.join(log_path_auton, run_path, "results_arrays")
    #     np.savez(array_path_auton, mean_reward=mean_reward, min_reward=min_reward, 
    #         max_reward=max_reward, mean_success=mean_success, min_success=min_success,
    #         max_success=max_success, timesteps=timesteps)

    # plot_success(log_path, run_path, timesteps, mean_success, min_success, max_success, figname="mean_success.png", title=title, log_path_auton=log_path_auton)
    # plot_reward(log_path, run_path, timesteps, mean_reward, min_reward, max_reward, figname="mean_reward.png",title=title, log_path_auton=log_path_auton)

    # for i in range(len(indexes)):
    #     new_rewards, new_success = get_data_from_file(log_path, run_path, indexes[i: i+1])
    #     mean_reward, min_reward, max_reward, mean_success, min_success, max_success, timesteps = compute_datapoints(new_rewards, new_success)
    #     plot_success(log_path, run_path, timesteps, mean_success, min_success, max_success, figname="mean_success{}.png".format(indexes[i]), title=title, log_path_auton=log_path_auton)
    #     plot_reward(log_path, run_path, timesteps, mean_reward, min_reward, max_reward, figname="mean_reward{}.png".format(indexes[i]), title=title, log_path_auton=log_path_auton)


    