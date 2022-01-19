import argparse
import torch
import numpy as np
import glob
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target', required=True, type=str,
    help='target folder containing npz files')

args = parser.parse_args()

target_folder = Path(args.target)
npz_list = glob.glob('{}/*.npz'.format(target_folder))

summary_dict = {key: [] for key in np.load(npz_list[0]).keys()}

for npz_file in npz_list:
    data = np.load(npz_file)
    for k, v in data.items():
        summary_dict[k].append(v)

for key in summary_dict:
    summary_dict[key] = np.concatenate(summary_dict[key])

save_fname = target_folder.parent / (target_folder.name + '.npz')
with open(save_fname, 'wb') as f:
    np.savez(f, **summary_dict)

print('summary npz has been saved to [{}]'.format(save_fname))


