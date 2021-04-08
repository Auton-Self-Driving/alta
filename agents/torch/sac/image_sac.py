from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import hydra

from .actor import DiagGaussianActor
from .critic import DoubleQCritic, Critic
from .utils import to_np, soft_update_params
from agents.torch.models import make_conv_preprocessor
from sac import SAC

from agents.torch.utils import COLOR, CONVERTER

class ImageSAC(SAC):
    """ CQL for mixed observation spaces """
    def __init__(self, *args, **kwargs):
        self.frame_stack = kwargs.pop('frame_stack', 2)
        self.freeze_conv = kwargs.pop('freeze_conv', False)
        self.conv_arch = kwargs.pop('conv_arch', 'vanilla')

        super().__init__(*args, **kwargs)

        # self.encoder = make_conv_preprocessor(256, arch=self.conv_arch, frame_stack=self.frame_stack, freeze_conv=self.freeze_conv)
        # self.target_encoder = make_conv_preprocessor(256, arch=self.conv_arch, frame_stack=self.frame_stack, freeze_conv=self.freeze_conv)
        # self.target_encoder.load_state_dict(self.encoder.state_dict())
        # if not self.freeze_conv:
        #     self.encoder_optimizer = optim.Adam(self.encoder.parameters())

    def predict(self, x):
        if len(x.shape) > 1:
            x = x.flatten()
        img, mlp_features = x[:-8], x[-8:]

        num_channels = int(img.size / (112 * 112))
        img = img.reshape(112, 112, num_channels)

        img = torch.FloatTensor(img).permute(2,0,1).cuda() / 255.

        mlp_features = torch.FloatTensor(mlp_features).cuda()
        img_features = self.encoder(img[None]).reshape(1,8192)[:,:32]
        state = torch.cat([img_features, mlp_features[None]], dim=1)
        action = self.policy(state).rsample()
        return action.detach().cpu().numpy().reshape(-1, self.action_dim)

    def convert_batch_obs(self, batch):
        (obs, actions, rewards, next_obs, terminals), indices, weights = batch

        img, mlp_features = obs[:,:-8], obs[:,-8:]
        num_channels = int(img.size(1) / (112 * 112))
        img = img.reshape(-1, 112, 112, num_channels)

        img = torch.cuda.FloatTensor(img).permute(0,3,1,2) / 255.
        mlp_features = torch.cuda.FloatTensor(mlp_features)

        next_img, next_mlp_features = next_obs[:,:-8], next_obs[:,-8:]
        next_img = next_img.reshape(-1, 112, 112, num_channels)

        next_img = torch.cuda.FloatTensor(next_img).permute(0,3,1,2) / 255.
        next_mlp_features = torch.cuda.FloatTensor(next_mlp_features)

        img_features = self.encoder(img).reshape(-1,8192)[:,:32]
        next_img_features = self.encoder(next_img).reshape(-1,8192)[:,:32]

        state = torch.cat([img_features, mlp_features], dim=1)
        next_state = torch.cat([next_img_features, next_mlp_features], dim=1)
        return (state, actions, rewards, next_state, terminals), indices, weights

    def training_step(self, batch, batch_idx, optimizer_idx):
        new_batch = self.convert_batch_obs(batch)
        super().training_step(new_batch, batch_idx, optimizer_idx)

    def configure_optimizers(self):
        return super().configure_optimizers()

    def set_encoder(self, encoder):
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False


class DBC(SAC):
    """ Uses DBC to learn representations for control using bisimulation metrics
    https://arxiv.org/abs/2006.10742
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dynamics_model = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.obs_dim + 1)
        )
        self.target_dynamics_model = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.obs_dim + 1)
        )
        self.target_dynamics_model.load_state_dict(self.dynamics_model.state_dict())
        self.dynamics_optimizer = optim.Adam(self.dynamics_model.parameters())

    def training_step(self, batch, batch_idx, optimizer_idx):
        # with torch.no_grad():
        #     new_batch = self.convert_batch_obs(batch)

        new_batch = batch

        # train policy
        # torch.autograd.set_detect_anomaly(True)
        CQL.training_step(self, new_batch, batch_idx, optimizer_idx)

        # train encoder + dynamics
        curr_z, action, reward, next_z, terminal = new_batch
        batch_size = curr_z.size(0)
        perm = np.random.permutation(batch_size)
        curr_z_2 = curr_z[perm]
        reward_2 = reward[perm]

        curr_z_and_action = torch.cat([curr_z, action], dim=1)
        pred_z_and_reward = self.dynamics_model(curr_z_and_action)
        pred_z, pred_reward = pred_z_and_reward[:,:-1], pred_z_and_reward[:,-1:]
        pred_z_2 = pred_z[perm]

        z_dist = F.smooth_l1_loss(curr_z, curr_z_2, reduction='none')
        r_dist = F.smooth_l1_loss(reward, reward_2, reduction='none')
        t_dist = F.smooth_l1_loss(pred_z, pred_z_2, reduction='none')

        b_dist = r_dist + t_dist # bisimulation distance
        z_loss = F.mse_loss(b_dist, z_dist)
        self.log('encoder_loss', z_loss)

        pred_loss = F.mse_loss(pred_z, next_z.detach())
        reward_loss = F.mse_loss(pred_reward, reward)
        self.log('pred_loss', pred_loss)
        self.log('reward_loss', reward_loss)

        total_loss = z_loss + pred_loss + reward_loss
        self.log('total_dbc_loss', total_loss)

        self.encoder_optimizer.zero_grad()
        self.dynamics_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.dynamics_optimizer.step()

        for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_target_tau) + param.data * self.soft_target_tau
            )

        for target_param, param in zip(self.target_dynamics_model.parameters(), self.dynamics_model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_target_tau) + param.data * self.soft_target_tau
            )

    def configure_optimizers(self):
        return super().configure_optimizers() + [self.dynamics_optimizer]