import numpy as np
import os

folders = []

data_path = '/zfsauton2/home/vkadi/projects/alta/alta-logs/sac_vs_ppo_dynamic-navigation_corrNstep'

folders.append('buffers/buff_set1')

mega_buff = []
'''for folder in folders:
	curr_path = os.path.join(data_path, folder)
	for curr_run in os.listdir(curr_path):
		print(curr_run)
		curr_buff = np.load(os.path.join(curr_path, curr_run, 'replay_buffer.npy'))
		mega_buff = mega_buff+list(curr_buff)'''

for folder in folders:
	curr_path = os.path.join(data_path, folder)
	for curr_run in os.listdir(curr_path):
		print(curr_run)
		curr_buff = np.load(os.path.join(curr_path, curr_run))
		mega_buff = mega_buff+list(curr_buff)

mega_buff = np.asarray(mega_buff)
np.save('mega_buffer1.npy', mega_buff)

