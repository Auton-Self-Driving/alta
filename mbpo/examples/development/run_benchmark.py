import argparse
from distutils.util import strtobool
import json
import os
import pickle

import tensorflow as tf
import numpy as np

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path',
                        type=str,
                        help='Path to the checkpoint.')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=10)
    parser.add_argument('--render-mode', '-r',
                        type=str,
                        default='human',
                        choices=('human', 'rgb_array', None),
                        help="Mode to render the rollouts in.")
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        help="Evaluate policy deterministically.")

    args = parser.parse_args()

    return args


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

    evaluation_environment = picklable['evaluation_environment']
    env = evaluation_environment._env.env
    # env.config['use_scenarios'] = True

    policy = (
        get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
    policy.set_weights(picklable['policy_weights'])

    rewards = []
    successes = []

    with policy.set_deterministic(True):
        for ep_idx in range(25):
            total_reward = 0
            obs = env.reset(index=ep_idx)
            print('==EPISODE {}'.format(ep_idx+1))

            for i in range(10000):
                action = policy.actions_np(np.array([obs]))[0]
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break

            print('EPISODE {} | REWARD: {} | EP LEN: {}'.format(ep_idx+1, total_reward, i+1) )
            rewards.append(total_reward)
            completed = reward > 0.
            successes.append(completed)

    import ipdb; ipdb.set_trace()
    pass

    # with policy.set_deterministic(args.deterministic):
    #     paths = rollouts(args.num_rollouts,
    #                      evaluation_environment,
    #                      policy,
    #                      path_length=args.max_path_length,
    #                      render_mode=args.render_mode)

    # if args.render_mode != 'human':
    #     from pprint import pprint; import pdb; pdb.set_trace()
    #     pass

    # return paths


if __name__ == '__main__':
    args = parse_args()
    simulate_policy(args)
