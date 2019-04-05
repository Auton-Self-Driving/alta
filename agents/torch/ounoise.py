from torch.distributions import Distribution, Normal
import torch

class OUNoise(object):
    def __init__(self, mu, sigma, theta):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        action_dim = mu.size(0) if (type(mu) == torch.tensor) else 1
        self.normal = Normal(torch.zeros(action_dim), 
                             torch.ones(action_dim))
        self.reset()

    def reset(self):
        self.state = self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * self.normal.sample()
        self.state = x + dx
        return self.state