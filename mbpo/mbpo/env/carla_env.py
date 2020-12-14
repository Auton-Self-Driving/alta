import os

from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import DEFAULT_ENV
import agents.tf.vis_module
from agents.tf.ae.controller import AEController

import numpy as np

import gym
from gym import utils
from gym.spaces import Box

from datetime import datetime

class CarlaEnvWrapper(CarlaEnv, utils.EzPickle):
	def __init__(self, **kwargs):
		config = {}
		config['algo'] = 'MBPO'
		config['train_config'] = 'PPO'

		for key, value in kwargs.items():
			config[key] = value

		config['carla_gpu'] = os.environ.get('CUDA_VISIBLE_DEVICES')

		self._gym_disable_underscore_compat = True
		super(CarlaEnvWrapper, self).__init__(config=config)
		utils.EzPickle.__init__(self)

		# if config.get('semantic', False):
		# 	vae = AEController(image_size=(128, 128, 5), frame_stack=3, learning_rate=1e-3)
		# 	vae.load('/home/brian/alta/agents/tf/trained_models/ae_16_32_64_64_fs_3')
		# 	self.set_vae(vae)
		# self.reset()

	def render(self, mode=''):
		pass


class CarlaImageEnvWrapper(CarlaEnvWrapper):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		obs_lower = np.zeros((64 * 64 * 3))
		obs_upper = np.ones((64 * 64 * 3))
		self.observation_space = Box(low=obs_lower, high=obs_upper, dtype=np.float32)

	def reset(self):
		_ = super(CarlaEnvWrapper, self).reset()
		return self.render()

	def step(self, action):
		_, reward, done, info = super(CarlaEnvWrapper, self).step(action)
		return self.render(), reward, done, info

	def render(self, mode=''):
		import ipdb; ipdb.set_trace()
		return self.image.copy()