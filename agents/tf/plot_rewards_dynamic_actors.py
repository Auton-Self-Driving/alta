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
def plot_runs(idx, exp_folder, exp_name, color, min_steps= 100, label = None, algo='ppo'):
	dir_name = os.path.join(exp_folder, exp_name)
	#print(dir_name)
	rewards = []
	for run_folder in sorted(os.listdir(dir_name))[:3]:
		if(not os.path.isdir(os.path.join(dir_name, run_folder))):
			continue
		file_name = os.path.join(dir_name, run_folder, algo+'_weightstest_results.csv')
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
	xticks_list = [str(i*100)+'k' for i in range(len(range(0, 8000001, 100000)))]
	plt.xticks(list(range(0, 8000001, 100000)), xticks_list, rotation=60, size = 20)
	#plt.xticks(list(range(0, 500001, 100000)), ('0', '100k', '200k', '300k', '400k', '500k'))
	#plt.yticks(list(range(0, 120001, 20000)), ('0', '20k', '40k', '60k', '80k', '100k', '120k'))

	if label is None:
		label = 'exp_'+str(idx) 

	plt.plot(steps, mean_reward, label=label,  color=color, linewidth=4)

	# plt.plot(steps, mean_neg_std_reward)
	# plt.plot(steps, mean_pos_std_reward)
	#color2 = lighten_color(color, 0.3)
	plt.fill_between(steps, mean_neg_std_reward, mean_pos_std_reward, color=color, alpha=0.5)
	# plt.show()

if __name__=='__main__':
	#
	#global exp_folder
	
	#colors_list = ['black', 'darkorange', 'green', 'purple', 'maroon', 'goldenrod', 'yellow']
	colors_list = ['black', 'sienna', 'darkred', 'yellowgreen', 'darkslategray', 'darkgreen']
	ctr = 0
	plot_another = 0

	exp_list = []
	mode = "dynamic-navigation"

	if mode == "dynamic-navigation":
		#labels_list = ['SAC_n25_nb1', 'SAC_n25_nb10', 'SAC_n25_nb30', 'SAC_n25_nb100', 'SAC_n25_nb30_4layer']
		labels_list = ['SAC_n25_nb100_col10', 'SAC_n25_nb100_ent5e-1', 'SAC_n25_nb100_ent5e-1_bootstrap']
		algo_list = ['sac', 'sac', 'sac', 'sac', 'sac']

		exp_folder = '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_dynamic-navigation_corrNstep'
		min_steps = 100

		'''exp_list.append('algo_SAC_task_self-driving_input_8dim_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_nSteps_25_gdUpdFreq_1_tgtUpdInt_1_ent_0.005_dynamic_navigation_npc_70_cp-250.0-250.0_lp-250.0-250.0')
		exp_list.append('algo_SAC_task_self-driving_input_8dim_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_nSteps_25_gdUpdFreq_10_tgtUpdInt_1_ent_0.005_dynamic_navigation_npc_70_cp-250.0-250.0_lp-250.0-250.0')
		exp_list.append('algo_SAC_task_self-driving_input_8dim_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_nSteps_25_gdUpdFreq_30_tgtUpdInt_1_ent_0.005_dynamic_navigation_npc_70_cp-250.0-250.0_lp-250.0-250.0')
		exp_list.append('algo_SAC_task_self-driving_input_8dim_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_nSteps_25_gdUpdFreq_100_tgtUpdInt_1_ent_0.005_dynamic_navigation_npc_70_cp-250.0-250.0_lp-250.0-250.0')
		exp_list.append('algo_SAC_task_self-driving_input_8dim_network_4_layer_lr_0.0004_buffer_1000000_batchsz_512_nSteps_25_gdUpdFreq_30_tgtUpdInt_1_ent_0.005_dynamic_navigation_npc_70_cp-250.0-250.0_lp-250.0-250.0')'''

		exp_list.append('algo_SAC_task_self-driving_input_8dim_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_nSteps_25_gdUpdFreq_100_tgtUpdInt_1_ent_0.005_dynamic_navigation_npc_70_cp-10.0-10.0_lp-10.0-10.0')
		exp_list.append('algo_SAC_task_self-driving_input_8dim_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_nSteps_25_gdUpdFreq_100_tgtUpdInt_1_ent_0.5_dynamic_navigation_npc_70_cp-250.0-250.0_lp-250.0-250.0')
		exp_list.append('algo_SAC_task_self-driving_input_8dim_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_nSteps_25_gdUpdFreq_100_tgtUpdInt_1_ent_0.5_dynamic_navigation_npc_70_cp-250.0-250.0_lp-250.0-250.0_bootstrap')


	if mode == "long-straight":
		labels_list = ['PPO', 'SAC_n1_ent-1']
		algo_list = ['ppo2', 'sac']

		exp_folder = '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_long-straight'
		min_steps = 100

		# Lane colision on
		exp_list.append('algo_PPO_task_self-driving_input_wp_obs_info_speed_steer_ldist_goal_light_network_2_layer_lr_0.0002_epochs_10__clip_0.1__mb_4__long_straight_npc_70_col_250.0_col_sp_250.0_light_250.0_light_sp_250.0_n_10000')

		exp_list.append('algo_SAC_task_self-driving_input_8dim_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_nSteps_1_gdUpdFreq_1_tgtUpdInt_1_ent_-1.0_long_straight_npc_70_cp-250.0-250.0_lp-250.0-250.0')


	for exp in exp_list:
		if(exp.split('_')[0]!='algo'):
			continue
		#try:
		plot_runs(ctr+1, exp_folder, exp, colors_list[ctr], min_steps, labels_list[ctr], algo_list[ctr])
		print(ctr+1, ':', exp)
		#except:
		#	continue
		ctr+=1

		axes = plt.gca()
		axes.set_ylim(bottom=0, top = 120000)
		plt.legend(loc='lower right', prop={'size' : 36})
		print('Saving plot to ', os.path.join(exp_folder, 'actor_rewards_new.png'))
		plt.savefig(os.path.join(exp_folder, 'actor_rewards_'+str(ctr)+'.png'), dpi=200)
