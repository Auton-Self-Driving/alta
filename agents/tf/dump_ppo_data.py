import argparse
import os
from collections import defaultdict

from PIL import Image
import deepdish as dd
from ppo import PPO
from stable_baselines.common.vec_env import DummyVecEnv
from environment.carla_9_4.env import CarlaEnv


def make_dir(path):
	if not os.path.isdir(path):
		os.mkdir(path)


class Logger:
	def __init__(self, path):
		make_dir(path)
		make_dir(os.path.join(path, 'topdown'))
		make_dir(os.path.join(path, 'front'))

		self.path = path
		self.save_dict = defaultdict(list)
		self.sample_counter = 0

	def update(self, returns, dump_images=True):
		self.sample_counter += 1
		obs, next_obs, reward, done, info = returns
		
		self.save_dict['obs'].append(obs)
		self.save_dict['next_obs'].append(next_obs)
		self.save_dict['reward'].append(reward)
		self.save_dict['done'].append(done)

		if dump_images:
			topdown, front = info['image'], info['rgb']
			topdown_path = os.path.join(self.path, 'topdown', str(self.sample_counter)+'.png')
			front_path = os.path.join(self.path, 'front', str(self.sample_counter)+'.png')

			Image.fromarray(topdown).save(topdown_path)
			Image.fromarray(front).save(front_path)

		# don't store images in dict
		del info['image']
		del info['rgb']
		self.save_dict['infos'].append(info)

	def store_dict(self):
		dict_dir = os.path.join(self.path, 'data_dict.h5')
		dd.io.save(dict_dir, self.save_dict)
		print('Saved dict to {} | Iteration {}'.format(dict_dir, self.sample_counter))


def main(args):
	logger = Logger(args.save_path)
	print('Logging to {}'.format(args.save_path))

	env = CarlaEnv()
	dummy_env = DummyVecEnv([lambda: env])
	env.observation_space.shape = (1,8) # changed shape for MBPO before, temp fix
	env.config['log_images'] = True
	env.config['semantic'] = True
	model = PPO.load(args.model_path, env=dummy_env)

	obs = env.reset()
	for i in range(args.num_samples):
		action = model.predict(obs.reshape(1,-1), deterministic=True)[0]
		next_obs, reward, done, info = env.step(action)
		logger.update((obs, next_obs, reward, done, info))

		if i % 100 == 99:
			logger.store_dict()

		if done:
			obs = env.reset()
		else:
			obs = next_obs

	logger.store_dict()
	print('Done')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_samples', type=int, default=1000)
	parser.add_argument('--model_path', type=str, default='/home/brian/alta-logs/ppo_expert/ppo2_weights6000000.zip')
	parser.add_argument('--save_path', type=str, default='/home/brian/offline_carla/ppo_data/')
	args = parser.parse_args()
	main(args)