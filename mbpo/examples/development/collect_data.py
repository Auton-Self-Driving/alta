import argparse
from distutils.util import strtobool
import json
import os
import pickle

import tensorflow as tf
import numpy as np

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant, get_uniform_policy
from softlearning.samplers import rollouts, rollout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path',
                        type=str,
                        help='Path to the checkpoint.')
    parser.add_argument('--save_path', type=str, default='data')
    args = parser.parse_args()

    return args


def collect_trajectories(policy, environment, num_samples=50000):
    obs, next_obs, actions, rewards = [], [], [], []
    sample_counter = 0
    while sample_counter < num_samples:
        path = rollout(environment, policy, path_length=1000)
        obs.extend(path['observations'])
        next_obs.extend(path['next_observations'])
        actions.extend(path['actions'])
        rewards.extend(path['rewards'])
        sample_counter += path['rewards'].size
    obs, next_obs, actions, rewards = np.array(obs), np.array(next_obs), np.array(actions), np.array(rewards)
    return obs[:num_samples], next_obs[:num_samples], actions[:num_samples], rewards[:num_samples]


def simulate_policy(args):
    session = tf.keras.backend.get_session()
    checkpoint_path = args.checkpoint_path.rstrip('/')
    experiment_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(experiment_path, 'params.json')
    with open(variant_path, 'r') as f:
        variant = json.load(f)

    with session.as_default():
        pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    picklable['evaluation_environment']._env.env.CarlaServer.close()
    evaluation_environment = picklable['training_environment']

    policy = (
        get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
    policy.set_weights(picklable['policy_weights'])
    with policy.set_deterministic(True):
        expert_obs, expert_actions, expert_next_obs, expert_rewards = collect_trajectories(policy, evaluation_environment)

    policy = get_uniform_policy(evaluation_environment)
    random_obs, random_actions, random_next_obs, random_rewards = collect_trajectories(policy, evaluation_environment)

    obs = np.concatenate([expert_obs, random_obs], axis=0)
    actions = np.concatenate([expert_actions, random_actions], axis=0)
    next_obs = np.concatenate([expert_next_obs, random_next_obs], axis=0)
    rewards = np.concatenate([expert_rewards, random_rewards], axis=0)

    np.save('{}/obs'.format(args.save_path), obs)
    np.save('{}/actions'.format(args.save_path), actions)
    np.save('{}/next_obs'.format(args.save_path), next_obs)
    np.save('{}/rewards'.format(args.save_path), rewards)

    print('Done')


if __name__ == '__main__':
    args = parse_args()
    simulate_policy(args)
