from collections import deque, namedtuple

import numpy as np
import torch

Experience = namedtuple('Experience', field_names=['obs', 'action', 'reward', 'next_obs', 'done'])

class ReplayBuffer:
    """ 
    Adapted from PyTorch Lightning examples:
    https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/rl/common/memory.py
    """
    def __init__(self, capacity=int(1e6)):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, actions, rewards, next_obs, dones = zip(*[self.buffer[idx] for idx in indices])
        return (
            torch.FloatTensor(obs).reshape(batch_size, -1).cuda(),
            torch.FloatTensor(actions).reshape(batch_size, -1).cuda(),
            torch.FloatTensor(rewards).reshape(batch_size, -1).cuda(),
            torch.FloatTensor(next_obs).reshape(batch_size, -1).cuda(),
            torch.FloatTensor(dones).reshape(batch_size, -1).cuda()
        )
