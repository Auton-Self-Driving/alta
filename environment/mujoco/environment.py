import gym
import torch
from scipy.misc import imresize


class MujocoEnv(object):
    def __init__(self, env_name, pixel_obs=False):
        self.env = gym.make(env_name)
        self.pixel_obs = pixel_obs
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
    def reset(self):
        obs = self.env.reset()
        return self._get_obs(obs)
    
    def step(self, action):
        action = action.numpy()[0]
        
        obs, reward, done, _ = self.env.step(action)
        obs = self._get_obs(obs)
        
        reward = torch.tensor([[reward]], dtype=torch.float32)
        done = torch.tensor([[done]], dtype=torch.float32)
        
        info = {}
        for idx, elem in enumerate(action):
            info["action" + str(idx)] = elem 
            
        return obs, reward, done, info
    
    def _get_obs(self, obs):
        if self.pixel_obs:
            img = self.env.render(mode="rgb_array")
            img = imresize(img, (84, 84))
            img = img / 255.
            img = torch.tensor([img], dtype=torch.float32)
            obs = {}
            obs["image"] = img.permute(0, 3, 1, 2)
        else:
            obs = torch.tensor([obs], dtype=torch.float32)
        return obs

"""
class MujocoEnv(object):
    def __init__(self, env_name, pixel_obs=False):
        self.env = gym.make(env_name)
        self.pixel_obs = pixel_obs
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
    def reset(self):
        obs = self.env.reset()
        return self._get_obs(obs)
    
    def step(self, action):
        action = action.numpy()[0]
        
        obs, reward, done, _ = self.env.step(action)
        obs = self._get_obs(obs)
        
        reward = torch.tensor([[reward]], dtype=torch.float32)
        done = torch.tensor([[done]], dtype=torch.float32)
        
        info = {}
        for idx, elem in enumerate(action):
            info["action" + str(idx)] = elem 
            
        return obs, reward, done, info
    
    def _get_obs(self, obs):
        if self.pixel_obs:
            
            img = self.env.render(mode="rgb_array")
            img = imresize(img, (28, 28))[:, :, 1]
            img = img / 255.
            img = torch.tensor([img], dtype=torch.float32)
            #obs = img.view(1, 28 * 28)
            obs = {}
            obs["image"] = img.view(1, -1)
            #TODO 
            #img = torch.tensor([obs], dtype=torch.float32)
            #obs = {}
            #obs["image"] = img
        else:
            obs = torch.tensor([obs], dtype=torch.float32)
        return obs
"""