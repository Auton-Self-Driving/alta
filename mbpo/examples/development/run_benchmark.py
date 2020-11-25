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
    parser.add_argument('experiment_path',
                        type=str,
                        help='Path to the experiment.')
    parser.add_argument('policy_id',
                        type=str)
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
    # checkpoint_path = args.checkpoint_path.rstrip('/')
    # experiment_path = os.path.dirname(checkpoint_path)
    experiment_path = args.experiment_path.rstrip('/')

    variant_path = os.path.join(experiment_path, 'params.json')
    with open(variant_path, 'r') as f:
        variant = json.load(f)

    # with session.as_default():
    #     pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
    #     with open(pickle_path, 'rb') as f:
    #         picklable = pickle.load(f)
    # evaluation_environment = picklable['evaluation_environment']

    with session.as_default():
        pickle_path = os.path.join(experiment_path, 'models/policy_{}0000.pkl'.format(args.policy_id))
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    environment_params = variant['environment_params']
    # environment_params['evaluation']['kwargs']['sample_npc'] = False
    environment_params['evaluation']['kwargs']['city_name'] = 'Town01'
    # environment_params['evaluation']['kwargs']['verbose'] = True
    evaluation_environment = get_environment_from_params(environment_params['evaluation'])
    
    env = evaluation_environment._env.env
    print('Scenarios: {}'.format(env.config['scenarios']))

    policy = (
        get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
    policy.set_weights(picklable['policy_weights'])

    with policy.set_deterministic(True):
        paths = rollouts(25,
                         evaluation_environment,
                         policy,
                         path_length=5000)

    termination_states = []
    for path in paths:
        termination_states.append(path['infos'][-1]['termination_state'])

    print(termination_states)

    import ipdb; ipdb.set_trace()
    pass


if __name__ == '__main__':
    args = parse_args()
    simulate_policy(args)
