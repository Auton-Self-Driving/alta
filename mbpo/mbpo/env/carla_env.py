import os

from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import DEFAULT_ENV
import agents.tf.vis_module

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

		date = datetime.now().strftime('%Y-%m-%d')
		log_dir = '{}/{}-seed={}'.format(config['log_dir'], date, config['seed'])
		if not os.path.isdir(log_dir):
			os.mkdir(log_dir)

		IMAGES_PATH = '{}/test_images/'.format(log_dir)
		VIDEO_PATH = '{}/test_videos/'.format(log_dir)
		vis_wrapper = agents.tf.vis_module.vis(IMAGES_PATH, VIDEO_PATH, videos=config['videos'])

		self._gym_disable_underscore_compat = True
		super(CarlaEnvWrapper, self).__init__(config=config, vis_wrapper=vis_wrapper, log_dir=log_dir)
		utils.EzPickle.__init__(self)