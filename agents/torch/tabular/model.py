from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import hydra


class DiscreteModel(pl.LightningModule):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

    def training_step(self, batch, batch_idx):
        import ipdb; ipdb.set_trace()
        pass

    def configure_optimizers(self):
        pass
