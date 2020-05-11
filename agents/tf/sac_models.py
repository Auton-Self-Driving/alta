from stable_baselines.sac.policies import FeedForwardPolicy

class My_MlpPolicy_1layer(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space=2, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(My_MlpPolicy_1layer, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        layers=[64],
                                        feature_extraction="mlp", **_kwargs)

class My_MlpPolicy_2layer(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space=2, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(My_MlpPolicy_2layer, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        layers=[64, 64],
                                        feature_extraction="mlp", **_kwargs)

class My_MlpPolicy_3layer(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space=2, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(My_MlpPolicy_3layer, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        layers=[32, 16, 8],
                                        feature_extraction="mlp", **_kwargs)
