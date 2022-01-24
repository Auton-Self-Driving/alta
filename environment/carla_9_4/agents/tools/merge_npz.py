import argparse
import torch
import numpy as np
import glob
from pathlib import Path
# import matplotlib.pyplot as plt
# import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target', required=True, type=str,
    help='target folder containing npz files')

parser.add_argument('-x', '--truncate-min', default=150, type=int,
    help='truncate episodes that has a length of traj less than x')

parser.add_argument('-X', '--truncate-max', default=5000, type=int,
    help='truncate episodes that has a length of traj greater than X')

parser.add_argument('-l', '--length', default=1200000, type=int,
    help='num of timesteps target (round-up)')

args = parser.parse_args()

target_folder = Path(args.target)
npz_list = glob.glob('{}/*.npz'.format(target_folder))

npz_keys = list(np.load(npz_list[0]).keys())
summary_dict = {key: [] for key in npz_keys}
total_sz = max_traj = 0
episode_len_list = []

for npz_file in npz_list:
    data = np.load(npz_file)
    is_included = False
    for k, v in data.items():
        if v.size == 1 or len(v) <= args.truncate_min:
            print('ignore short traj', k, v.shape)
            continue
        if len(v) > args.truncate_max:
            print('ignore long traj', k, v.shape)
            continue
        summary_dict[k].append(v)
        is_included = True

    # for creating histogram
    if data[npz_keys[0]].size != 1:
        episode_len_list.append(len(data[npz_keys[0]]))

    if is_included:
        total_sz += len(data[npz_keys[0]])
        max_traj = max(max_traj, len(data[npz_keys[0]]))
    if total_sz >= args.length:
        break

for key in summary_dict:
    summary_dict[key] = np.concatenate(summary_dict[key])

save_fname = target_folder.parent / (target_folder.name + \
    '_total{}_maxtraj{}.npz'.format(total_sz, max_traj))
with open(save_fname, 'wb') as f:
    np.savez(f, **summary_dict)

# with open('episode_len_list.pkl', 'wb') as f:
#     pickle.dump(episode_len_list, f)
# print(episode_len_list)

print('summary npz (total timestep [{}]) has been saved to [{}]'.format(
    len(summary_dict[npz_keys[0]]), save_fname))


