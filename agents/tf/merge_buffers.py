import numpy as np
import os

folders = []

#data_path = '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_dynamic-navigation_corrNstep_envSem3'
data_path = '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_dynamic-navigation_corrNstep_envSem3/algo_SAC_task_self-driving_input_8dim_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_nSteps_25_trainFreq_512_gdUpdFreq_100_tgtUpdInt_1_ent_0.05_dynamic_navigation_npc_70_cp-250.0-250.0_lp-250.0-250.0_corr/algo_SAC_task_self-driving_input_8dim_network_2_layer_lr_0.0004_buffer_1000000_batchsz_512_nSteps_25_trainFreq_512_gdUpdFreq_100_tgtUpdInt_1_ent_0.05_dynamic_navigation_npc_70_cp-250.0-250.0_lp-250.0-250.0_corr_runid_run1/'
folders.append('replay_buffer.npy')

mega_buff = []
'''for folder in folders:
	curr_path = os.path.join(data_path, folder)
	for curr_run in os.listdir(curr_path):
		print(curr_run)
		curr_buff = np.load(os.path.join(curr_path, curr_run, 'replay_buffer.npy'))
		mega_buff = mega_buff+list(curr_buff)'''

'''for folder in folders:
	curr_path = os.path.join(data_path, folder)
	for curr_run in os.listdir(curr_path):
		print(curr_run)
		curr_buff = np.load(os.path.join(curr_path, curr_run))
		mega_buff = mega_buff+list(curr_buff)'''
for folder in folders:
	curr_path = os.path.join(data_path, folder)
	curr_buff = np.load(curr_path)
	mega_buff = mega_buff+list(curr_buff)

mega_buff = np.asarray(mega_buff)
np.save('mega_buffer1.npy', mega_buff)

