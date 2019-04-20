import os
import shutil
import numpy as np
import torch

# Reset episode until planner can find a path to target
def reset_env(env):
    while True:
        try:
            obs = env.reset()
        except:
            continue
        else:
             return obs 

# Remove directories
def silent_remove(*directories):
    for d in directories:
        try:
            shutil.rmtree(d)
        except FileNotFoundError:
            pass

# Create directory if not existing
def silent_add(*directories):
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)

def to_numpy(tensor):
    return tensor.cpu().data.numpy()

def from_numpy(array):
    return torch.from_numpy(array.astype(np.float32))

def convert_observation(obs):
    if 'image' in obs:
        obs['image'] = np.expand_dims(np.transpose(obs['image'], (2, 0, 1)), axis=0)
    for k, v in obs.items():
        obs[k] = from_numpy(obs[k])
    
    return obs
