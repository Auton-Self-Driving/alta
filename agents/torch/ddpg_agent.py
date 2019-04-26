import copy
from itertools import chain
from typing import Union

import torch
import torch.nn as nn
from torch.distributions import Distribution
from torchvision import transforms

from .abstract_agent import Agent
from .networks import CNN, VAE, Pretrained, DeterministicPolicy, QValue


class DDPGAgent(Agent):
    def __init__(self,
                 encoder: Union[CNN, VAE, None], 
                 actor: DeterministicPolicy, 
                 critic: QValue, 
                 noise: Distribution,
                 actor_lr,
                 critic_lr,
                 target_lr, 
                 discount, 
                 max_grad_norm):
        super(DDPGAgent, self).__init__()
        
        self.target_lr = target_lr
        self.discount = discount
        self.noise = noise
        self.max_grad_norm = max_grad_norm
        
        # Current networks
        if encoder is not None:
            self.curr_nets["encoder"] = encoder.to(self.device)
        self.curr_nets["actor"] = actor.to(self.device)
        self.curr_nets["critic"] = critic.to(self.device)
        
        # Target networks
        for k, v in self.curr_nets.items():
            self.targ_nets[k] = copy.deepcopy(v)
        
        # Optimizers
        self.actor_params = self.curr_nets["actor"].parameters()
        self.critic_params = self.curr_nets["critic"].parameters()
        
        if isinstance(self.curr_nets["encoder"], CNN):
            self.actor_params = chain(
                self.actor_params, self.curr_nets["encoder"].parameters())
            self.critic_params = chain(
                self.critic_params, self.curr_nets["encoder"].parameters())
        elif isinstance(self.curr_nets["encoder"], VAE):
            self.vae_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
            self.vae_params = self.curr_nets["encoder"].parameters()
            # self.vae_optim = torch.optim.RMSprop(self.vae_params, lr=1e-5, eps=1e-5, momentum=0)
            self.vae_optim = torch.optim.Adam(self.vae_params, lr=1e-4, eps=1e-5)
            
        # self.actor_optim = torch.optim.RMSprop(self.actor_params, lr=actor_lr, 
        #                                        eps=1e-5, momentum=0)
        # self.critic_optim = torch.optim.RMSprop(self.critic_params, lr=critic_lr,
        #                                      eps=1e-5, momentum=0)
        self.actor_optim = torch.optim.Adam(
            self.actor_params, lr=actor_lr, eps=1e-5)
        self.critic_optim = torch.optim.Adam(
            self.critic_params, lr=critic_lr, eps=1e-5)
    
    def get_action(self, obs, eval_mode):
        # Set eval mode for batchnorm and dropout
        self._set_network_states("eval")
        
        # Put inputs on GPU
        obs = self._put_on_device(obs)
        
        with torch.set_grad_enabled(False):
            # Process image input
            if isinstance(self.curr_nets["encoder"], CNN):
                obs["image_features"] = self.curr_nets["encoder"](obs["image"])
            elif isinstance(self.curr_nets["encoder"], VAE):
                image = torch.squeeze(obs["image"]).cpu()
                image = self.vae_transforms(image)
                image = torch.unsqueeze(image, dim=0).cuda()
                obs["vae_features"] = self.curr_nets["encoder"](image)
                obs["image_features"] = self.curr_nets["encoder"].get_encoded_features()
                
            # Compute action
            action = self.curr_nets["actor"](obs).cpu()
                
        # Add noise for exploration
        if not eval_mode:
            action += self.noise.sample()
            action = torch.clamp(action, -1, 1)
        
        return action, {}
   
    def update(self, batch):
        # Set train mode for batchnorm and dropout
        self._set_network_states("train")
        
        # Put tensors on GPU
        obs, action, reward, next_obs, done = [self._put_on_device(x) for x in batch]
        
        # Process image inputs if necessary
        if isinstance(self.curr_nets["encoder"], CNN):
            obs["image_features"] = self.curr_nets["encoder"](obs["image"])
            next_obs["image_features"] = self.targ_nets["encoder"](next_obs["image"])
        elif isinstance(self.curr_nets["encoder"], VAE):
            obs_batch = self.process_batch(obs["image"])
            next_obs_batch = self.process_batch(next_obs["image"])
            obs["vae_features"] = self.curr_nets["encoder"](obs_batch)
            obs["image_features"] = self.curr_nets["encoder"].get_encoded_features()
            next_obs["vae_features"] = self.targ_nets["encoder"](next_obs_batch)
            next_obs["image_features"] = self.targ_nets["encoder"].get_encoded_features()

            # Compute VAE loss and update
            vae_loss1 = self.update_vae(obs["vae_features"], obs_batch)
            vae_loss2 = self.update_vae(next_obs["vae_features"], next_obs_batch)
            vae_loss = (vae_loss1 + vae_loss2) / 2.0

        # Compute current Q estimate
        current_Q = self.curr_nets["critic"](obs, action)
        
        # Compute target Q value
        next_action = self.targ_nets["actor"](next_obs)
        target_Q = self.targ_nets["critic"](next_obs, next_action)
        target_Q = reward + ((1 - done) * self.discount * target_Q).detach()
        
        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize critic (and cnn)
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
        self.critic_optim.step()

        # Compute actor loss (avoiding gradient w.r.t CNN through obs)
        action = self.curr_nets["actor"](obs)
        obs["image_features"] = obs["image_features"].detach()
        actor_loss = -self.curr_nets["critic"](obs, action).mean()

        # Optimize actor (and cnn)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
        self.actor_optim.step()

        if isinstance(self.curr_nets["encoder"], CNN):
            # Record critic optimization statistics
            losses, weights, grads = {}, {}, {}
            losses["critic"] = critic_loss.item()
            self._log_weights_and_grads(
                "critic_update", ["critic", "encoder"], weights, grads)

            # Record actor optimization statistics
            losses["actor"] = actor_loss.item()
            self._log_weights_and_grads(
                "actor_update", ["actor", "encoder"], weights, grads)
        elif isinstance(self.curr_nets["encoder"], VAE):
            # Record VAE loss optimization statistics
            losses, weights, grads = {}, {}, {}
            losses["encoder"] = vae_loss.item()
            self._log_weights_and_grads(
                "encoder_update", ["encoder"], weights, grads)
            
            # Record critic optimization statistics
            losses, weights, grads = {}, {}, {}
            losses["critic"] = critic_loss.item()
            self._log_weights_and_grads(
                "critic_update", ["critic"], weights, grads)

            # Record actor optimization statistics
            losses["actor"] = actor_loss.item()
            self._log_weights_and_grads(
                "actor_update", ["actor"], weights, grads)
        
        # Update frozen target models
        self._soft_update_target()

        return losses, weights, grads

    def update_vae(self, outputs, batch):
        recon_batch, mu, logvar = outputs
        vae_loss = self.curr_nets["encoder"].loss_function(
            recon_batch, batch, mu, logvar)
        # print("VAE loss = {}".format(vae_loss))
        # Optimize
        self.vae_optim.zero_grad()
        vae_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.vae_params, self.max_grad_norm)
        self.vae_optim.step()
        
        return vae_loss
    
    def process_batch(self, batch):
        processed_batch = []
        for image in batch:
            image = image.cpu()
            image = self.vae_transforms(image)
            processed_batch.append(image)
        processed_batch = torch.stack(processed_batch)
        return processed_batch.cuda()
        
