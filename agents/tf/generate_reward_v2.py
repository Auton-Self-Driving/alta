import numpy as np
from scipy.interpolate import spline, interp1d
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
import csv
import os

font = {'size' : 36}

matplotlib.rc('font', **font)

rewards = []
plt.figure(figsize=(22, 14))
colors = ['b', 'g', 'r', 'c', 'm']

exp_folder = '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_corr-term'
exp_name = 'algo_SAC_input_wp_network_2_layer_lr_0.0004_buffer_1000000_batchsz_2048_gradupd-per-iter_1_cp-0.0-0.0_navigation'

dir_name = os.path.join(exp_folder, exp_name)

for run_folder in os.listdir(dir_name):
    if(not os.path.isdir(os.path.join(dir_name, run_folder))):
        continue
    file_name = os.path.join(dir_name, run_folder, 'test_results.csv')
    my_data = []
    with open(file_name) as f:
        reader = csv.reader(f, delimiter = ',')
        for row in reader:
            my_data.append([int(row[0]), int(row[1]), eval(row[2])[0][0]])
    #my_data = genfromtxt(file_name, delimiter=',')
    my_data = np.asarray(my_data)
    print(np.shape(my_data))
    timesteps = my_data[:,0]
    reward = my_data[:,2]
    rewards.append(reward)

rewards = np.array(rewards)
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
plt.plot(steps, mean_reward, label='WRL',  color='royalblue')

# plt.plot(steps, mean_neg_std_reward)
# plt.plot(steps, mean_pos_std_reward)
plt.fill_between(steps, mean_neg_std_reward, mean_pos_std_reward, color='lavender')
# plt.show()

axes = plt.gca()
axes.set_ylim(bottom=0)
plt.legend(loc='lower right', prop={'size' : 36})
# plt.title("Navigation with dynamic obstacles")
# plt.savefig('actor_rewards.png')
# plt.savefig('actor_rewards_600.png', dpi=600)
print('Saving plot to ', os.path.join(dir_name, 'actor_rewards_new.png'))
plt.savefig(os.path.join(dir_name, 'actor_rewards_new.png'), dpi=200)
# plt.savefig('actor_rewards.eps', format='eps')
