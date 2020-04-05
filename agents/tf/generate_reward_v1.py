import numpy as np
from scipy.interpolate import spline, interp1d
import matplotlib
import matplotlib.pyplot as plt

font = {'size' : 20}

matplotlib.rc('font', **font)

rewards = []
plt.figure(figsize=(15, 8))
colors = ['b', 'g', 'r', 'c', 'm']
total_models = 22
start_model_step = (26 - total_models) * 40000
steps = list(range(start_model_step, 1000001, 40000))
# for j in range(1, 6):
# for j in range(2, 6):
for j in [1, 2, 4]:
    prefix = 'ppo_{}/'.format(j)

    ALTA_LOGS = '/zfsauton2/home/tanmaya/projects/neurips/ppo_pid_wp_scenarios_navigation/' + prefix
    
    with open(ALTA_LOGS + "best_model.txt", "r") as f:
        # with open(ALTA_LOGS + "random_model.txt", "r") as fo:
        #     lines = [line for line in fo.readlines()]
        #     random_reward = float(lines[1].split(" [[")[1].split("]]")[0])
        #     random_success = float(lines[5].split(": ")[1])

        lines = [line for line in f.readlines()]
        nos = lines[4].split("training: ")[1][1:-1].split(', ')
        reward = []
        
        # reward.append(random_reward)
        for no in nos:
            reward.append(float(no.split('[[')[1].split(']]')[0]))
        print(reward)
        reward = np.array(reward)
        reward = reward[26 - total_models:]
        
        success = []
        nos = lines[5].split("episodes: ")[1][1:-2].split(', ')
        # success.append(random_success)
        for no in nos:
            success.append(float(no))
        print(success)
        success = np.array(success)
        success = success[26 - total_models:]
        
        max_rewards = np.zeros(total_models)
        max_success = np.zeros(total_models)
        
        max_rewards[0] = reward[0]
        max_success[0] = success[0]
        
        for k in range(1, total_models):
            max_success[k] = success[k]
            max_success[k] = np.amax(max_success[:k+1])
            max_inds = np.array([i for i, l in enumerate(max_success[:k+1]) if l == max_success[k]])
            max_rewards[k] = reward[k]
            max_rewards[k] = np.amax(max_rewards[:k+1][max_inds])
        print(max_rewards)
        print(max_success)
        rewards.append(max_rewards)
        # plt.plot(list(range(start_model_step, 1000001, 40000)), max_rewards, color=colors[j-1], linestyle='-', linewidth=1, label='run{}'.format(j))
        plt.xlabel('Steps', fontdict={'size' : 24})
        plt.ylabel('Best Running Model Reward', fontdict={'size' : 24})
        plt.xticks(list(range(0, 1000001, 200000)), ('0', '200k', '400k', '600k', '800k', '1000k'))
        # plt.legend()
    
rewards = np.array(rewards)
mean_reward = np.mean(rewards, axis=0)
std_reward = np.std(rewards, axis=0)
mean_neg_std_reward = np.amin(rewards, axis=0)
mean_pos_std_reward = np.amax(rewards, axis=0)

# mean_neg_std_reward = mean_reward - std_reward
# mean_pos_std_reward = mean_reward + std_reward

xnew = np.linspace(start_model_step, 1000000, 10000) #300 represents number of points to make between T.min and T.max

# power_smooth = spline(steps, mean_reward, xnew)
# plt.plot(xnew, power_smooth, color='#bd83ce', linestyle='-', linewidth=2, markersize=8, label='mean')
f = interp1d(steps, mean_reward, kind='cubic')
plt.plot(xnew, f(xnew), color='#bd83ce', linestyle='-', linewidth=2, markersize=8, label='Navigation')

# plt.plot(steps, mean_reward, color='#bd83ce', linestyle='-', linewidth=2, markersize=8, label='One Turn')

# neg_smooth = spline(steps, mean_neg_std_reward, xnew)
# plt.plot(xnew, neg_smooth, color='#f1c6e7', linestyle='-', linewidth=1, label='mean-std')
f_neg = interp1d(steps, mean_neg_std_reward, kind='cubic')
plt.plot(xnew, f_neg(xnew), color='#f1c6e7', linestyle='-', linewidth=1)

# plt.plot(steps, mean_neg_std_reward, color='#f1c6e7', linestyle='-', linewidth=1)

# pos_smooth = spline(steps, mean_pos_std_reward, xnew)
# plt.plot(xnew, pos_smooth, color='#f1c6e7', linestyle='-', linewidth=1, label='mean+std')
f_pos = interp1d(steps, mean_pos_std_reward, kind='cubic')
plt.plot(xnew, f_pos(xnew), color='#f1c6e7', linestyle='-', linewidth=1)

# plt.plot(steps, mean_pos_std_reward, color='#f1c6e7', linestyle='-', linewidth=1)

axes = plt.gca()
# axes.set_ylim(bottom=20000)
axes.set_xlim(left=start_model_step, right=1000000)


# plt.fill_between(xnew, neg_smooth, pos_smooth, color='#f1c6e7')
plt.fill_between(xnew, f_neg(xnew), f_pos(xnew), color='#f1c6e7')

# plt.fill_between(steps, mean_neg_std_reward, mean_pos_std_reward, color='#f1c6e7')
plt.legend(loc='lower right', prop={'size' : 20})
plt.savefig(ALTA_LOGS + '../zoomed_reward.png')
        
    