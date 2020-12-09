import os

from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import DEFAULT_ENV
import agents.tf.vis_module
from agents.tf.ae.controller import AEController

import gym
from gym import utils

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

		if config.get('semantic', False):
			vae = AEController(image_size=(128, 128, 5), frame_stack=3, learning_rate=1e-3)
			vae.load('/home/brian/alta/agents/tf/trained_models/ae_16_32_64_64_fs_3')
			self.set_vae(vae)

		self.reset()

	def render(self, mode=''):
		pass