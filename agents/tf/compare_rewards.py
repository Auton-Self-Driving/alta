import tensorflow as tf
import numpy as np
import os
import pdb
import time
#log_dir = '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac/'
log_dir = '/zfsauton2/home/vkadi/projects/alta/alta-logs/'

#exp_name = 'algo_SAC_input_wp_network_2_layer_lr_0.0003_buffer_100000_batchsz_512_gradupd-per-iter_2_navigation'
exp_name = 'algo_PPO_lr_0.0002_layers_2_layer_steer_coeff_0_ent_coef_0.005'

runs = ['1','2','3', '4','5']
#events_file = 'events.out.tfevents.1582853702.gpu9.int.autonlab.org'
events_file = 'events.out.tfevents.1571733751.gpu10.int.autonlab.org'

#filepath = os.path.join(log_dir, exp_name, exp_name+'_runid_run'+str(run_id), 'tb/SAC_1', events_file)

reward_logs = []
for run_id in runs:
	filepath = os.path.join(log_dir, exp_name, exp_name+'_runid_'+run_id, 'tb/', events_file)
	print(filepath)

	data = []
	st = time.time()
	for e in tf.train.summary_iterator(filepath):
		for v in e.summary.value:
			if v.tag=='test/total_reward':
				data.append([e.step, v.simple_value])
	data = np.asarray(data)
	reward_logs.append(data)

reward_logs = np.asarray(reward_logs)
np.save('ppo_test_rewards.npy', reward_logs)
print(reward_logs.shape, time.time()-st)
