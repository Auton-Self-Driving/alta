import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_data(path):
	obs = np.load('{}/obs.npy'.format(path))
	next_obs = np.load('{}/next_obs.npy'.format(path))
	actions = np.load('{}/actions.npy'.format(path))
	rewards = np.load('{}/rewards.npy'.format(path))
	return obs, actions, next_obs, rewards


def main():
	obs, actions, next_obs, rewards = load_data('path/')

	state_dim = obs.shape[1]
	action_dim = actions.shape[1]
	latent_dim = 