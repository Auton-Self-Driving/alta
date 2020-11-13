params = {
    'type': 'MBPO',
    'universe': 'gym',
    'domain': 'Carla',
    'task': 'Driving-v0',

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'logtests',

    'kwargs': {
        'epoch_length': 10000,
        'train_every_n_steps': 1,
        'n_train_repeat': 20,
        'eval_render_mode': None,
        'eval_n_episodes': 25,
        'eval_deterministic': True,
        'n_initial_exploration_steps': int(5000),

        'discount': 0.95,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 250,
        'model_retain_epochs': 5,
        'rollout_batch_size': 100e3,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 0.05,
        'target_entropy': -2,
        'max_model_t': None,
        'rollout_schedule': [20, 150, 1, 1],
    }
}

