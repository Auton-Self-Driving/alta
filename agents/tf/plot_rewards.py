import numpy as np
from scipy.interpolate import spline, interp1d
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
import csv
import os

import matplotlib.colors as mc
import colorsys

def lighten_color(color, amount=0.5):
	try:
		c = mc.cnames[color]
	except:
		c = color
	c = colorsys.rgb_to_hls(*mc.to_rgb(c))
	return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

font = {'size' : 36}

matplotlib.rc('font', **font)

plt.figure(figsize=(22, 14))
colors = ['b', 'g', 'r', 'c', 'm']

#exp_name = 'algo_SAC_input_wp_network_2_layer_lr_0.0004_buffer_1000000_batchsz_2048_gradupd-per-iter_1_cp-0.0-0.0_navigation'
def plot_runs(idx, exp_name, color, min_steps= 100):
	dir_name = os.path.join(exp_folder, exp_name)
	#print(dir_name)
	rewards = []
	for run_folder in os.listdir(dir_name):
		if(not os.path.isdir(os.path.join(dir_name, run_folder))):
			continue
		file_name = os.path.join(dir_name, run_folder, 'test_results.csv')
		my_data = []
		with open(file_name) as f:
			reader = csv.reader(f, delimiter = ',')
			for row in reader:
				reward = np.asarray(eval(row[2]))
				if reward.ndim>1:
					reward = reward[0][0]
				my_data.append([int(row[0]), int(row[1]), reward])
		#my_data = genfromtxt(file_name, delimiter=',')
		my_data = np.asarray(my_data)
		#print(my_data.shape)
		timesteps = my_data[:,0]
		reward = my_data[:,2]
		rewards.append(reward)
		if(len(timesteps)<=min_steps):
			min_steps = len(timesteps)

	for i in range(len(rewards)):
		rewards[i] = rewards[i][:min_steps]
	timesteps = timesteps[:min_steps]
	#print("Rewards shape", np.asarray(rewards).shape)

	#rewards = np.array(rewards)
	mean_reward = np.mean(rewards, axis=0)
	std_reward = np.std(rewards, axis=0)
	mean_neg_std_reward = np.amin(rewards, axis=0)
	mean_pos_std_reward = np.amax(rewards, axis=0)

	# mean_neg_std_reward = mean_reward - std_reward
	# mean_pos_std_reward = mean_reward + std_reward

	steps = timesteps
	plt.xlabel('Timesteps', fontdict={'size' : 36})
	# plt.ylabel('Total Success Episodes', fontdict={'size' : 36})
	plt.ylabel('Total Reward', fontdict={'size' : 36})
	plt.xticks(list(range(0, 1000001, 100000)), ('0', '100k', '200k', '300k', '400k', '500k', '600k', '700k', '800k', '900k', '1000k'))
	#plt.xticks(list(range(0, 500001, 100000)), ('0', '100k', '200k', '300k', '400k', '500k'))
	plt.yticks(list(range(0, 120001, 20000)), ('0', '20k', '40k', '60k', '80k', '100k', '120k'))
	plt.plot(steps, mean_reward, label='exp_'+str(idx),  color=color, linewidth=4)

	# plt.plot(steps, mean_neg_std_reward)
	# plt.plot(steps, mean_pos_std_reward)
	#color2 = lighten_color(color, 0.3)
	plt.fill_between(steps, mean_neg_std_reward, mean_pos_std_reward, color=color, alpha=0.5)
	# plt.show()

def plot_ppo_rewards(color):
	ppo_reward_info = np.load('ppo_test_rewards.npy')

	min_steps = 10000000
	for i in range(len(ppo_reward_info)):
		if(len(ppo_reward_info[i])<min_steps):
			min_steps = len(ppo_reward_info[i])

	rewards = []
	for i in range(len(ppo_reward_info)):
		#print(ppo_reward_info[i][:min_steps, -1].shape)
		rewards.append(ppo_reward_info[i][:min_steps, -1])
	rewards = np.asarray(rewards)
	#print(rewards.shape)

	mean_reward = np.mean(rewards, axis=0)
	std_reward = np.std(rewards, axis=0)
	mean_neg_std_reward = np.amin(rewards, axis=0)
	mean_pos_std_reward = np.amax(rewards, axis=0)

	steps = np.arange(0, min_steps*10000, 10000)
	plt.xlabel('Timesteps', fontdict={'size' : 36})
	plt.ylabel('Total Reward', fontdict={'size' : 36})
	plt.xticks(list(range(0, 1000001, 100000)), ('0', '100k', '200k', '300k', '400k', '500k', '600k', '700k', '800k', '900k', '1000k'))
	plt.yticks(list(range(0, 120001, 20000)), ('0', '20k', '40k', '60k', '80k', '100k', '120k'))
	plt.plot(steps, mean_reward, label='ppo',  color=color, linewidth=4)

	# plt.plot(steps, mean_neg_std_reward)
	# plt.plot(steps, mean_pos_std_reward)
	#color2 = lighten_color(color, 0.3)
	plt.fill_between(steps, mean_neg_std_reward, mean_pos_std_reward, color=color, alpha=0.5)
	return min_steps

if __name__=='__main__':
	#
	global exp_folder
	
	colors_list = ['darkorange', 'purple', 'green', 'yellow', 'maroon', 'goldenrod']
	ctr = 0

	exp_list = []
	mode = "Neurips_wrl_plus"

	if mode == "Neurips_wrl_plus":
		exp_folder = '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo'
		min_steps = plot_ppo_rewards('black')
		#exp_list.append('algo_PPO_input_wp_network_2_layer_lr_0.0002_navigation')
		#exp_list.append('algo_SAC_input_wp_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_gradupd-per-iter_1_cp-0.0-0.0_navigation')
		'''exp_list.append('algo_SAC_input_wp_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_gradupd-per-iter_4_cp-0.0-0.0_navigation')
		exp_list.append('algo_SAC_input_wp_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_gradupd-per-iter_8_cp-0.0-0.0_navigation')
		exp_list.append('algo_SAC_input_wp_network_2_layer_lr_0.0004_buffer_1000000_batchsz_2048_gradupd-per-iter_1_cp-0.0-0.0_navigation')
		exp_list.append('algo_SAC_input_wp_network_2_layer_lr_0.0004_buffer_50000_batchsz_2048_gradupd-per-iter_1_cp-0.0-0.0_navigation')
		exp_list.append('algo_SAC_input_wp_network_2_layer_lr_0.0004_buffer_10000_batchsz_2048_gradupd-per-iter_1_cp-0.0-0.0_navigation')'''
		exp_list.append('algo_SAC_task_self-driving_input_wp_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_gradupd-per-iter_1_tgt-upd-int_1_ent-coef_0.05_cp-0.0-0.0_navigation')

	elif mode == "Distance2goal_reward":
		min_steps = 100
		exp_folder = '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_reward-dist2goal'
		exp_list.append('algo_SAC_task_self-driving_input_wp_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_gradupd-per-iter_1_tgt-upd-int_1_ent-coef_0.005_cp-0.0-0.0_navigation')
		exp_list.append('algo_PPO_task_self-driving_input_wp_network_2_layer_lr_0.0002_navigation')

	elif mode == "Half-Cheetah":
		min_steps = 100
		exp_folder = '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_mujoco'
		exp_list.append('algo_SAC_task_HalfCheetah-v2_input_wp_network_2_layer64_lr_0.0004_buffer_1000000_batchsz_512_gradupd-per-iter_1_tgt-upd-int_1_ent-coef_0.005_cp-0.0-0.0_navigation')
		exp_list.append('algo_PPO_task_HalfCheetah-v2_input_wp_network_2_layer64_lr_0.0002_navigation')

	elif mode == "Ant":
		min_steps = 100
		exp_folder = '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_mujoco'
		exp_list.append('algo_SAC_task_Ant-v2_input_wp_network_2_layer64_lr_0.0003_buffer_1000000_batchsz_256_gradupd-per-iter_1_tgt-upd-int_1_ent-coef_-1_cp-0.0-0.0_navigation')
		exp_list.append('algo_PPO_task_Ant-v2_input_wp_network_2_layer64_lr_0.0002_navigation')
	else:
		print("Invalid mode")

	for exp in exp_list:
		if(exp.split('_')[0]!='algo'):
			continue
		#try:
		plot_runs(ctr+1, exp, colors_list[ctr], min_steps)
		print(ctr+1, ':', exp)
		#except:
		#	continue
		ctr+=1

	axes = plt.gca()
	axes.set_ylim(bottom=0)
	plt.legend(loc='lower right', prop={'size' : 36})
	print('Saving plot to ', os.path.join(exp_folder, 'actor_rewards_new.png'))
	plt.savefig(os.path.join(exp_folder, 'actor_rewards_new.png'), dpi=200)
