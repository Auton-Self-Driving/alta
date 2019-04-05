import torch
from torch.distributions import Distribution, Normal


# Code adapted from
# https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/distributions.py
class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :normal_mean: Mean of the normal distribution
        :normal_std: Std of the normal distribution
        :epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def log_prob(self, value):
        """
        :value: some value, x
        """
        pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        logprob = self.normal.log_prob(pre_tanh_value)
        logprob = logprob - torch.log(1 - value ** 2 + self.epsilon)
        return logprob.sum(dim=-1, keepdim=True)

    def sample(self):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()
        return torch.tanh(z)

    def rsample(self):
        """
        Sampling in the reparameterization case.
        """
        
        z = Normal(torch.zeros(self.normal_mean.size()),
                   torch.ones(self.normal_std.size())).sample()
        z = self.normal_mean + self.normal_std * z.to(self.normal_mean.device)
        z.requires_grad_()
        return torch.tanh(z)