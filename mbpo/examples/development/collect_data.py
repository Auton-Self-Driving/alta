import argparse
from distutils.util import strtobool
import json
import os
import pickle

import tensorflow as tf
import numpy as np
import deepdish as dd
from PIL import Image

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant, get_uniform_policy
from softlearning.samplers import rollouts, rollout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path',
                        type=str,
                        help='Path to the experiment.')
    parser.add_argument('policy_id',
                        type=str)
    parser.add_argument('--save_path', type=str, default='/home/brian/offline_carla/mixed_data/')
    args = parser.parse_args()

    return args


def collect_trajectories(policy, environment, num_samples=500, save_offset=0, image_save_dir=''):
    obs, next_obs, actions, rewards, terminals, infos = [], [], [], [], [], []
    sample_counter = 0
    while sample_counter < num_samples:
        path = rollout(environment, policy, path_length=1000)
        rollout_length = len(path['rewards'])
        for i in range(rollout_length):
            sem, rgb = path['infos'][i]['image'].copy(), path['infos'][i]['rgb'].copy()
            del path['infos'][i]['image']
            del path['infos'][i]['rgb']
            sem_path = os.path.join(image_save_dir, 'sem', str(i+sample_counter+save_offset)+'.png')
            rgb_path = os.path.join(image_save_dir, 'rgb', str(i+sample_counter+save_offset)+'.png')
            Image.fromarray(sem).save(sem_path)
            Image.fromarray(rgb).save(rgb_path)

        obs.extend(path['observations'])
        next_obs.extend(path['next_observations'])
        actions.extend(path['actions'])
        rewards.extend(path['rewards'])
        terminals.extend(path['terminals'])
        infos.extend(path['infos'])
        sample_counter += path['rewards'].size

        print('Samples: {}'.format(sample_counter+save_offset))

    obs, next_obs, actions, rewards, terminals, infos = \
        np.array(obs), np.array(next_obs), np.array(actions), np.array(rewards), np.array(terminals), np.array(infos)
    # return obs[:num_samples], next_obs[:num_samples], actions[:num_samples], rewards[:num_samples], terminals[:num_samples], infos[:num_samples]
    return obs, next_obs, actions, rewards, terminals, infos


def simulate_policy(args):
    session = tf.keras.backend.get_session()
    experiment_path = args.experiment_path.rstrip('/')

    variant_path = os.path.join(experiment_path, 'params.json')
    with open(variant_path, 'r') as f:
        variant = json.load(f)

    with session.as_default():
        pickle_path = os.path.join(experiment_path, 'models/policy_{}0000.pkl'.format(args.policy_id))
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    environment_params = variant['environment_params']
    environment_params['evaluation']['kwargs']['pedestrians'] = 150
    environment_params['evaluation']['kwargs']['log_images'] = True
    environment_params['evaluation']['kwargs']['semantic'] = True
    evaluation_environment = get_environment_from_params(environment_params['evaluation'])

    policy = (
        get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
    policy.set_weights(picklable['policy_weights'])

    with policy.set_deterministic(True):
        expert_trajectories = collect_trajectories(policy, evaluation_environment, num_samples=500000, save_offset=0, image_save_dir=args.save_path)

    policy = get_uniform_policy(evaluation_environment)
    random_trajectories = collect_trajectories(policy, evaluation_environment, num_samples=500000, save_offset=500000, image_save_dir=args.save_path)

    obs = np.concatenate([expert_trajectories[0], random_trajectories[0]], axis=0)
    next_obs = np.concatenate([expert_trajectories[1], random_trajectories[1]], axis=0)
    actions = np.concatenate([expert_trajectories[2], random_trajectories[2]], axis=0)
    rewards = np.concatenate([expert_trajectories[3], random_trajectories[3]], axis=0)
    terminals = np.concatenate([expert_trajectories[4], random_trajectories[4]], axis=0)
    infos = np.concatenate([expert_trajectories[5], random_trajectories[5]], axis=0)

    # num_samples = infos.shape[0]
    # for i in range(num_samples):
    #     semantic = infos[i]['image']
    #     rgb = infos[i]['rgb']

    #     semantic_path = os.path.join(args.save_path, 'semantic', str(i) + '.png')
    #     rgb_path = os.path.join(args.save_path, 'rgb', str(i) + '.png')

    #     Image.fromarray(semantic).save(semantic_path)
    #     Image.fromarray(rgb).save(rgb_path)

    data_dict = {
        'obs': obs,
        'actions': actions,
        'next_obs': next_obs,
        'rewards': rewards,
        'terminals': terminals,
        'infos': infos
    }

    dd.io.save('mixed_data.h5', data_dict)

    # np.save('{}/obs'.format(args.save_path), obs)
    # np.save('{}/actions'.format(args.save_path), actions)
    # np.save('{}/next_obs'.format(args.save_path), next_obs)
    # np.save('{}/rewards'.format(args.save_path), rewards)

    print('Done')


if __name__ == '__main__':
    args = parse_args()
    simulate_policy(args)
