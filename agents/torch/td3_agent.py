import copy
from itertools import chain
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Distribution

from .abstract_agent import Agent
from .networks import CNN, DeterministicPolicy, QValue


class TD3Agent(Agent):
    def __init__(self,
                 cnn: Optional[CNN], 
                 actor: DeterministicPolicy, 
                 critic1: QValue, 
                 critic2: QValue,
                 exploration_noise: Distribution,
                 target_noise: Distribution,
                 noise_clip,
                 policy_interval,
                 actor_lr, 
                 critic_lr,
                 target_lr, 
                 discount, 
                 max_grad_norm):
        super(TD3Agent, self).__init__()
        
        self.target_lr = target_lr
        self.discount = discount
        self.noise = exploration_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_interval = policy_interval
        self.max_grad_norm = max_grad_norm
        
        # Current networks
        if cnn is not None:
            self.curr_nets["cnn"] = cnn.to(self.device)
        self.curr_nets["actor"] = actor.to(self.device)
        self.curr_nets["critic1"] = critic1.to(self.device)
        self.curr_nets["critic2"] = critic2.to(self.device)
        
        # Target networks
        for k, v in self.curr_nets.items():
            self.targ_nets[k] = copy.deepcopy(v)

        # Optimizers
        self.actor_params = self.curr_nets["actor"].parameters()
        self.critic_params = chain(self.curr_nets["critic1"].parameters(),
                                   self.curr_nets["critic2"].parameters())
        
        if "cnn" in self.curr_nets:
            self.actor_params = chain(
                self.actor_params, self.curr_nets["cnn"].parameters())
            self.critic_params = chain(
                self.critic_params, self.curr_nets["cnn"].parameters())
        
        self.actor_optim = torch.optim.RMSprop(self.actor_params, lr=actor_lr, 
                                               eps=1e-5, momentum=0)
        self.critic_optim = torch.optim.RMSprop(self.critic_params, lr=critic_lr,
                                             eps=1e-5, momentum=0)
    
    def get_action(self, obs, eval_mode):
        # Set eval mode for batchnorm and dropout
        self._set_network_states("eval")
        
        # Put inputs on GPU
        obs = self._put_on_device(obs)
        
        with torch.set_grad_enabled(False):
            # Process image input
            if "cnn" in self.curr_nets:
                obs["image_features"] = self.curr_nets["cnn"](obs["image"])
                
            # Compute action
            action = self.curr_nets["actor"](obs).cpu()
                
        # Add noise for exploration
        if not eval_mode:
            action += self.noise.sample()
            action = torch.clamp(action, -1, 1)
        
        return action, {}
   
    def update(self, batch):
        self.num_update_calls += 1
        
        # Set train mode for batchnorm and dropout
        self._set_network_states("train")
        
        # Put tensors on GPU
        obs, action, reward, next_obs, done = [self._put_on_device(x) for x in batch]
        
        # Process image inputs if necessary
        if "cnn" in self.curr_nets:
            obs["image_features"] = self.curr_nets["cnn"](obs["image"])
            next_obs["image_features"] = self.targ_nets["cnn"](next_obs["image"])

        # Compute current Q estimates
        current_Q1 = self.curr_nets["critic1"](obs, action)
        current_Q2 = self.curr_nets["critic2"](obs, action)
        
        # Select next action according to policy and add clipped noise
        noise = self.target_noise.sample(torch.Size([action.size(0)]))
        noise = noise.clamp(-self.noise_clip, self.noise_clip).to(self.device)
        next_action = (self.targ_nets["actor"](next_obs) + noise).clamp(-1, 1)
        
        # Compute target Q value
        target_Q1 = self.targ_nets["critic1"](next_obs, next_action)
        target_Q2 = self.targ_nets["critic2"](next_obs, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1 - done) * self.discount * target_Q).detach()

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + \
            nn.MSELoss()(current_Q2, target_Q)
        
        # Optimize critic (and cnn)
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
        self.critic_optim.step()
        
        # Record critic optimization statistics
        losses, weights, grads = {}, {}, {}
        losses["critic"] = critic_loss.item()
        self._log_weights_and_grads(
            "critic_update", ["critic1", "critic2", "cnn"], weights, grads)

        if self.num_update_calls % self.policy_interval == 0:
            # Compute actor loss (avoiding gradient w.r.t CNN through obs)
            action = self.curr_nets["actor"](obs)
            obs["image_features"] = obs["image_features"].detach()
            actor_loss = -self.curr_nets["critic1"](obs, action).mean()
            
            # Optimize actor (and cnn)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
            self.actor_optim.step()
            
            # Record actor optimization statistics
            losses["actor"] = actor_loss.item()
            self._log_weights_and_grads(
                "actor_update", ["actor", "cnn"], weights, grads)

            # Update frozen target models
            self._soft_update_target()

        return losses, weights, grads