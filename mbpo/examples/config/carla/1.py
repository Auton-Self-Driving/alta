params = {
    'type': 'SAC',
    'universe': 'gym',
    'domain': 'Carla',
    'task': 'ImageDriving-v0',

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'sac-test',

    'kwargs': {
        'epoch_length': 10000,
        'train_every_n_steps': 1,
        'n_train_repeat': 20,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'discount': 0.95,
        'tau': 5e-3,
        'reward_scale': 1.0,

        # 'policy_lr': 3e-4,
        # 'Q_lr': 3e-4,
        # 'alpha_lr': 3e-4,
        'lr': 3e-4,
        'target_update_interval': 1,
        'tau': 5e-3,
        'target_entropy': 'auto',
    }
}

