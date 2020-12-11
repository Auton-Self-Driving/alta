from ray import tune
import numpy as np
import pdb
from copy import deepcopy

from softlearning.misc.utils import get_git_rev, deep_update

M = 256
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {}

POLICY_PARAMS_BASE = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
})

POLICY_PARAMS_FOR_DOMAIN = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],
})

DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'Pendulum': 200,
    'Carla': 1000,
}

ALGORITHM_PARAMS_ADDITIONAL = {
    'MBPO': {
        'type': 'MBPO',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(5000),
        }
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'target_update_interval': 1,
            'n_initial_exploration_steps': int(1e3),
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker2d': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                    'Pendulum': 1,
                }.get(
                    spec.get('config', spec)
                    ['environment_params']
                    ['training']
                    ['domain'],
                    1.0
                ),
            )),
        }
    },
    'MVE': {
        'type': 'MVE',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(5000),
        }
    },
}

DEFAULT_NUM_EPOCHS = 200

NUM_EPOCHS_PER_DOMAIN = {
    'Hopper': int(1e3),
    'HalfCheetah': int(3e3),
    'Walker2d': int(3e3),
    'Ant': int(3e3),
    'Humanoid': int(1e4),
    'Pendulum': 10,
}

ALGORITHM_PARAMS_PER_DOMAIN = {
    **{
        domain: {
            'kwargs': {
                'n_epochs': NUM_EPOCHS_PER_DOMAIN.get(
                    domain, DEFAULT_NUM_EPOCHS),
                'n_initial_exploration_steps': (
                    MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH
                    ) * 10),
            }
        } for domain in NUM_EPOCHS_PER_DOMAIN
    }
}

ENVIRONMENT_PARAMS = {
    'Carla': {
        'StateDriving-v0': {
            'input_type': 'wp_obs_info_speed_steer_ldist_goal_light',
            'action_type': 'merged_speed_scaled_tanh',
            'scenarios': 'no_crash_dense',
            'use_scenarios': True,
            'train_fixed_spawn_points': True,
            'test_fixed_spawn_points': True,
            'reward_function': 'simple2',
            'sample_npc': False,
            'num_npc': 50,
            'num_pedestrians': 25,
            'city_name': 'Town01',
            'const_collision_penalty': 250,
            'collision_penalty_speed_coeff': 250,
            'const_light_penalty': 250,
            'light_penalty_speed_coeff': 250,
            'vehicle_proximity_threshold': 10,
            'traffic_light_proximity_threshold': 10,
            'videos': False,
            'testing': False,
            'frame_skip': 2,
            'reward_normalize_factor': 16,
            'verbose': False
        },
        'ImageDriving-v0': {
            'input_type': 'wp_obs_info_speed_steer_ldist_goal_light',
            'semantic': True,
            'frame_stack_size': 1,
            'action_type': 'merged_speed_scaled_tanh',
            'scenarios': 'dynamic_navigation',
            'use_scenarios': False,
            'train_fixed_spawn_points': True,
            'test_fixed_spawn_points': True,
            'reward_function': 'simple2',
            'sample_npc': False,
            'num_npc': 50,
            'city_name': 'Town01',
            'const_collision_penalty': 250,
            'collision_penalty_speed_coeff': 250,
            'const_light_penalty': 250,
            'light_penalty_speed_coeff': 250,
            'vehicle_proximity_threshold': 10,
            'traffic_light_proximity_threshold': 10,
            'videos': False,
            'testing': False,
            'frame_skip': 2,
            'reward_normalize_factor': 16,
            'verbose': False,
            'image_shape': (64,64,3),
            'pixel_wrapper_kwargs': {
            }
        }
    }
}

NUM_CHECKPOINTS = 10


def get_variant_spec_base(universe, domain, task, policy, algorithm, env_params):
    algorithm_params = deep_update(
        ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {}),
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
    )
    algorithm_params = deep_update(
        algorithm_params,
        env_params
    )

    variant_spec = {
        # 'git_sha': get_git_rev(),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': (
                    ENVIRONMENT_PARAMS.get(domain, {}).get(task, {})),
            },
            'evaluation': tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['environment_params']
                ['training']
            )),
        },
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': tune.sample_from(lambda spec: (
                    {
                        'SimpleReplayPool': int(1e6),
                        'TrajectoryReplayPool': int(1e4),
                    }.get(
                        spec.get('config', spec)
                        ['replay_pool_params']
                        ['type'],
                        int(1e6))
                )),
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': NUM_EPOCHS_PER_DOMAIN.get(
                domain, DEFAULT_NUM_EPOCHS) // NUM_CHECKPOINTS,
            'checkpoint_replay_pool': True,
        },
    }

    return variant_spec

def is_image_env(universe, domain, task, variant_spec):
    return 'pixel_wrapper_kwargs' in (
        variant_spec['environment_params']['training']['kwargs'])

def get_variant_spec_image(universe,
                           domain,
                           task,
                           policy,
                           algorithm,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, *args, **kwargs)

    if is_image_env(universe, domain, task, variant_spec):
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                'image_shape': (
                    variant_spec
                    ['environment_params']
                    ['training']
                    ['kwargs']
                    ['image_shape']),
                'output_size': M,
                'conv_filters': (4, 4),
                'conv_kernel_sizes': ((3, 3), (3, 3)),
                'pool_type': 'MaxPool2D',
                'pool_sizes': ((2, 2), (2, 2)),
                'pool_strides': (2, 2),
                'dense_hidden_layer_sizes': (),
            },
        }
        variant_spec['policy_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())
        variant_spec['Q_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())

    return variant_spec

def get_variant_spec(args, env_params):
    universe, domain, task = env_params.universe, env_params.domain, env_params.task

    variant_spec = get_variant_spec_image(
        universe, domain, task, args.policy, env_params.type, env_params)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
