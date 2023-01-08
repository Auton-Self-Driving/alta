"""Sensitivity Analysis
"""

import os
import glob
import pickle
import sys
import torch
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import argparse
import copy
import numpy as np

sys.path.append('..')

from network import PPOActorCritic_Continuous


"""

[ppo_agent : test()] action chosen: [[ 0.346822  -0.7617278]] # Steer Throttle
[ppo_agent : test()] state space: {'Wp': 0.35276280477077, 'S': 3.226407327668418e-06, 'Str': 0.1733655333518982, 'D2T': 0.7235249136231705, 'Lgt': 1.0, 
    'D_F_Fr_Br_Bl_Fl': [-2.0, -1.0, -1.0, -1.0, -1.0], 
    'S_F_Fr_Br_Bl_Fl': [20.0, 20.0, 20.0, 20.0, 20.0]} 
    [ 3.52762805e-01 -6.66666667e-02  1.00000000e+00 -2.00000000e-01
  1.00000000e+00 -2.00000000e-01  1.00000000e+00 -2.00000000e-01
  1.00000000e+00 -2.00000000e-01  1.00000000e+00  3.22640733e-07
  1.73365533e-01  7.23524914e-01  1.00000000e+00]

"""

BASE_STATE = np.asarray([[ 3.52762805e-01, -6.66666667e-02,  1.00000000e+00, -2.00000000e-01,
  1.00000000e+00, -2.00000000e-01,  1.00000000e+00, -2.00000000e-01,
  1.00000000e+00, -2.00000000e-01,  1.00000000e+00,  3.22640733e-07,
  1.73365533e-01,  7.23524914e-01,  1.00000000e+00]])

BASE_ACTION = np.asarray([[ 0.346822,  -0.7617278]])

STATE_BOUNDS = {
    0:[-1,1], # Wp orientation
    11:[-1,1], # Speed
    12:[-0.5,0.5], # Steer
    13:[-1,1] # D2T
}

STATE_REP = {
    0: "Orientation",
    11: "Speed",
    12: "Steer",
    13: "D2T",
}

INTERVAL = 0.02

def __create_all_uniq_pairs(array):

    n = len(array)

    pairings = []

    for i in range(n-1):
        for j in range(i+1,n):
            pairings.append((array[i],array[j]))

    return pairings

def compute_single_axis_variations(base_tensor, policy, out_path, args):

    base_target_speed = (BASE_ACTION[0,1] * 1.5) + 1 
    base_target_speed = np.clip(base_target_speed * args.target_speed / 2, 0, args.target_speed)

    for idx in list(STATE_BOUNDS.keys()):
        offset, span = STATE_BOUNDS[idx][0], STATE_BOUNDS[idx][1] - STATE_BOUNDS[idx][0]
        num_entries = int(span / INTERVAL + 1)
        perturbed_obs = base_tensor.repeat(num_entries, 1)
        perturbed_dim = torch.arange(offset,offset+span+1e-6,INTERVAL)

        perturbed_obs[:,idx] = perturbed_dim

        action, logprob = policy.act(perturbed_obs,deterministic=True)

        target_speed = (action[:,1] * 1.5) + 1
        target_speed = np.clip(target_speed * args.target_speed / 2, 0, args.target_speed)

        plt.plot(perturbed_dim, target_speed, label=STATE_REP[idx])
        plt.plot(BASE_STATE[0,idx],base_target_speed,'o', markersize=7, label=STATE_REP[idx]) 
        plt.xlabel("State Space")
        plt.ylabel("Target Speed")
    plt.legend()
    plt.savefig("{}/single.png".format(out_path),dpi=600)
    plt.clf()



def compute_dual_axis_variations(base_tensor, policy, out_path, args):


    state_pairs = __create_all_uniq_pairs(list(STATE_BOUNDS.keys()))

    for idx1, idx2 in state_pairs:
        offset1, span1 = STATE_BOUNDS[idx1][0], STATE_BOUNDS[idx1][1] - STATE_BOUNDS[idx1][0]
        num_entries1 = int(span1 / INTERVAL + 1)
        offset2, span2 = STATE_BOUNDS[idx2][0], STATE_BOUNDS[idx2][1] - STATE_BOUNDS[idx2][0]
        num_entries2 = int(span2 / INTERVAL + 1)

        perturbed_dim1 = torch.arange(offset1,offset1+span1+1e-6,INTERVAL)
        perturbed_dim2 = torch.arange(offset2,offset2+span2+1e-6,INTERVAL)

        X,Y = np.meshgrid(perturbed_dim1.numpy(),perturbed_dim2.numpy())

        speed_outputs = np.zeros(X.shape)

        for itr in range(num_entries2):
            perturbed_obs = base_tensor.repeat(num_entries1, 1)
            
            perturbed_obs[:,idx1] = perturbed_dim1
            perturbed_obs[:,idx2] = perturbed_dim2[itr]

            action, logprob = policy.act(perturbed_obs,deterministic=True)

            target_speed = (action[:,1] * 1.5) + 1
            target_speed = np.clip(target_speed * args.target_speed / 2, 0, args.target_speed)

            speed_outputs[itr,:] = target_speed

        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, speed_outputs, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel(STATE_REP[idx1])
        ax.set_ylabel(STATE_REP[idx2])
        ax.set_zlabel("Target Speed")
        # ax.legend()
        pickle.dump({"x":X,"y":Y,"z":speed_outputs,"x_lab":STATE_REP[idx1],"y_lab":STATE_REP[idx2]}, open('{}/{}_{}.pickle'.format(out_path,STATE_REP[idx1],STATE_REP[idx2]), 'wb')) 
        plt.savefig("{}/{}_{}.png".format(out_path,STATE_REP[idx1],STATE_REP[idx2]),dpi=600)
        plt.clf()



def probe_policy(args):

    ckpt_base_path = "../checkpoints/{}/".format(args.ckpt)
    ckpt_path = glob.glob(ckpt_base_path+"*_{}_*".format(args.ckpt_iter))[0]

    policy = PPOActorCritic_Continuous(15, 2).to(args.device)
    ckpt = torch.load(ckpt_path, map_location=args.device)
    policy.load_state_dict(ckpt['glb_policy'])
    print("Loaded weights from",args.ckpt)
    
    out_path = '{}/{}/{}'.format(args.out_root,args.ckpt,args.ckpt_iter)
    os.makedirs(out_path ,exist_ok=True)

    base_tensor = torch.from_numpy(BASE_STATE).to(torch.float).to(args.device)
    
    compute_single_axis_variations(base_tensor, policy, out_path, args)
    compute_dual_axis_variations(base_tensor, policy, out_path, args)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Sensitivity Analysis")
    parser.add_argument("--ckpt", default=argparse.SUPPRESS)
    parser.add_argument("--ckpt_iter", default=argparse.SUPPRESS)
    parser.add_argument("--out_root", default='../tests/sensitivity')
    parser.add_argument("--target_speed", default=50)
    parser.add_argument("--device", default='0', type=str)

    args = parser.parse_args()

    probe_policy(args)
    

# python sensitivity_analysis.py --ckpt 15dim_nocrach_dense_no_lane_term_tanh_squashed --ckpt_iter 7216852 --device 'cuda:2'
# python sensitivity_analysis.py --ckpt 7dim_nocrach_dense_no_lane --ckpt_iter 7210465 --device 'cuda:2'

