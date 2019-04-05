import copy
from itertools import chain
from typing import Optional

import torch
import torch.nn as nn

from .abstract_agent import Agent
from .networks import CNN, StochasticPolicy, QValue, StateValue


class SACAgent(Agent):
    def __init__(self,
                 cnn: Optional[CNN], 
                 actor: StochasticPolicy, 
                 qcritic1: QValue, 
                 qcritic2: QValue,
                 vcritic: StateValue,
                 entropy_coeff,
                 policy_l2,
                 policy_freq,
                 actor_lr, 
                 critic_lr,
                 target_lr, 
                 discount):
        super(SACAgent, self).__init__()
        
        self.target_lr = target_lr
        self.discount = discount
        self.entropy_coeff = entropy_coeff
        self.policy_l2 = policy_l2
        self.policy_freq = policy_freq
        
        # Current networks
        if cnn is not None:
            self.curr_nets["cnn"] = cnn.to(self.device)
        self.curr_nets["actor"] = actor.to(self.device)
        self.curr_nets["qcritic1"] = qcritic1.to(self.device)
        self.curr_nets["qcritic2"] = qcritic2.to(self.device)
        self.curr_nets["vcritic"] = vcritic.to(self.device)
        
        # Target networks
        for k, v in self.curr_nets.items():
            self.targ_nets[k] = copy.deepcopy(v)
        
        # Optimizers
        actor_params = self.curr_nets["actor"].parameters()
        critic_params = chain(self.curr_nets["qcritic1"].parameters(),
                              self.curr_nets["qcritic2"].parameters(),
                              self.curr_nets["vcritic"].parameters())
        if "cnn" in self.curr_nets:
            actor_params = chain(actor_params, self.curr_nets["cnn"].parameters())
            critic_params = chain(critic_params, self.curr_nets["cnn"].parameters())
            
        self.actor_optim = torch.optim.Adam(actor_params, lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic_params, lr=critic_lr)
    
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
            if eval_mode:
                # Deterministic TODO
                raise NotImplementedError
            else:
                # Sampled
                distrib = self.curr_nets["actor"](obs)
                action = distrib.sample().cpu()
                
        debug_info = {}
        debug_info["normal_mean"] = distrib.normal_mean
        debug_info["normal_std"] = distrib.normal_std
        debug_info["action"] = action
        
        return action, debug_info
   
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

        # Compute current Q estimate
        current_Q1 = self.curr_nets["qcritic1"](obs, action)
        current_Q2 = self.curr_nets["qcritic2"](obs, action)
        
        # Compute current state value
        current_V = self.curr_nets["vcritic"](obs)
        
        # Compute target Q value
        target_Q = self.targ_nets["vcritic"](next_obs)
        target_Q = (reward + (1 - done) * self.discount * target_Q).detach()
        
        # Compute target state value
        dist = self.targ_nets["actor"](obs)
        new_action = dist.sample() 
        # TODO check computation of log_prob (sum?)
        logprob = dist.log_prob(new_action)
        target_V = torch.min(self.targ_nets["qcritic1"](obs, new_action),
                             self.targ_nets["qcritic2"](obs, new_action))
        target_V = (target_V - self.entropy_coeff * logprob).detach()
        
        # Compute critic losses
        qcritic_loss = nn.MSELoss()(current_Q1, target_Q) + \
            nn.MSELoss()(current_Q2, target_Q)
        vcritic_loss = nn.MSELoss()(current_V, target_V)
        critic_loss = qcritic_loss + vcritic_loss

        # Optimize critics
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()
        
        metrics = {}
        metrics["qcritic_loss"] = qcritic_loss.item()
        metrics["vcritic_loss"] = vcritic_loss.item()

        if self.num_update_calls % self.policy_freq == 0:
            # Compute actor loss
            dist = self.curr_nets["actor"](obs)
            new_action = dist.rsample() # Reparameterized sample
            logprob = dist.log_prob(new_action)
            current_Q = self.curr_nets["qcritic1"](obs, new_action)
            actor_loss = (-current_Q + self.entropy_coeff * logprob).mean()
            
            # Add regularization for policy parameters
            reg_loss = (dist.normal_mean ** 2 + dist.normal_std ** 2).mean()
            actor_loss = actor_loss + self.policy_l2 * reg_loss
            
            # Optimize actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            
            # TODO Debug NaN
            for p in self.curr_nets["actor"].parameters():
                grad = p.grad.data
                if (grad != grad).sum() > 0:
                    print(grad)
                    
            self.actor_optim.step()
            
            metrics["actor_loss"] = actor_loss.item()
        
            # Update frozen target models
            self._soft_update_target()

        return metrics