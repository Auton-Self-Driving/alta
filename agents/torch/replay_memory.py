import random
import torch 
from typing import Dict, Tuple


def dict_concat(dicts):
    # Make a single dictionary of concatenated tensors from a list of dictionaries
    # of tensors
    return {k: torch.cat([d[k] for d in dicts], dim=0) for k in dicts[0].keys()}

def concat(items):
    # Concatenate tensors either from a list of dictionaries of tensors
    # or a list of tensors
    if isinstance(items[0], dict):
        return dict_concat(items)
    else:
        return torch.cat(items)


class ReplayMemory(object):
    def __init__(self, capacity):
        # List of transition tuples (obs, action, reward, next_obs, done)
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, 
             transition: Tuple[Dict[str, torch.Tensor], 
                               torch.Tensor,
                               torch.Tensor,
                               Dict[str, torch.Tensor],
                               torch.Tensor]):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        
        # Sample a batch of transitions
        batch = random.sample(self.memory, batch_size)
        # print(batch)
        
        # Avoid batch of size 1 which is problematic for batchnorm
        if len(batch) == 1:
            batch += batch

        return map(concat, zip(*batch))