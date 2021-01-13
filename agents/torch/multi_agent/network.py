"""A2C network for discrete_actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical

from a2c_utils import set_init

class Basic_Discrete(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Basic_Discrete, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).detach()
        m = self.distribution(prob)
        return m.sample().cpu().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class PPOActorCritic_Continuous(nn.Module):
    def __init__(self, state_dim, action_dim, action_std=.5):
        super(PPOActorCritic_Continuous, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
            )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )
        self.action_var = torch.full(
            (action_dim,) , action_std * action_std)

    def forward(self):
        raise NotImplementedError('please use act and eval instead')

    def act(self, state):
        device = next(self.actor.parameters()).device
        action_mean = self.actor(state.to(device))
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        action = action.detach().cpu().numpy()
        action_logprob = action_logprob.detach().cpu().numpy()

        return action, action_logprob

    def evaluate(self, state, action):
        device = next(self.actor.parameters()).device
        action_mean = self.actor(state.to(device))
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def __str__(self):
        device = next(self.actor.parameters()).device
        return 'PPOActorCritic_Continuous:\n ' + \
            'device: {}\n actor: {}\n critic: {}\n'.format(device, 
            self.actor, self.critic)


if __name__ == '__main__':
    glb_net = Basic_Discrete(12, 24).to('cpu')
    print(glb_net)
    policy = PPOActorCritic_Continuous(12, 24)
    print(policy)