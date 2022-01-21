import argparse
import torch
import numpy as np
import glob
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target', required=True, type=str,
    help='target folder containing npz files')

parser.add_argument('-x', '--truncate', default=15, type=int,
    help='truncate episodes that has a length of traj less than x')

args = parser.parse_args()

target_folder = Path(args.target)
npz_list = glob.glob('{}/*.npz'.format(target_folder))

npz_keys = list(np.load(npz_list[0]).keys())
summary_dict = {key: [] for key in npz_keys}

for npz_file in npz_list:
    data = np.load(npz_file)
    for k, v in data.items():
        if v.size == 1 or len(v) <= args.truncate:
            print(k, v.shape)
            continue
        summary_dict[k].append(v)

for key in summary_dict:
    summary_dict[key] = np.concatenate(summary_dict[key])

save_fname = target_folder.parent / (target_folder.name + '.npz')
with open(save_fname, 'wb') as f:
    np.savez(f, **summary_dict)

print('summary npz (total timestep [{}]) has been saved to [{}]'.format(
    len(summary_dict[npz_keys[0]]), save_fname))


