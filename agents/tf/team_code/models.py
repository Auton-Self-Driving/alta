from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
from abc import abstractmethod
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer
from stable_baselines.common.policies import BasePolicy, register_policy, mlp_extractor
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines.common.distributions import CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution
from stable_baselines.common.input import observation_input
from distributions import make_proba_dist_type


def CoRLModel(inputs, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        activation = tf.nn.tanh
        convs1 = [
            [8, [3, 3], 1],
            [16, [3, 3], 1],
        ]
        pool1 = [
            [[2,2], 2]
        ]
        convs2 = [
            [16, [3, 3], 1],
            [8, [3, 3], 1],
        ]
        pool2 = [
            [[2,2], 2]
        ]
        hidden = 3528
        net = inputs
        out_size, kernel, stride = convs1[0]
        net = slim.conv2d(net, out_size, kernel, stride, scope="conv1/conv3_1")
        out_size, kernel, stride = convs1[1]
        net = slim.conv2d(net, out_size, kernel, stride, scope="conv1/conv3_2")
        kernel, stride = pool1[0]
        net = slim.pool(net, kernel, "MAX", stride=stride, scope="pool1")
        #--------
        out_size, kernel, stride = convs2[0]
        net = slim.conv2d(net, out_size, kernel, stride, scope="conv2/conv3_1")
        out_size, kernel, stride = convs2[1]
        net = slim.conv2d(net, out_size, kernel, stride, scope="conv2/conv3_2")
        kernel, stride = pool2[0]
        net = slim.pool(net, kernel, "MAX", stride=stride, scope="pool2")
        net = tf.squeeze(net)
        net = tf.reshape(net, [-1, 21, 21, 8])
        #Flatten pool layer
        net = slim.flatten(net, scope="flatten3")
        #--------
        net = slim.fully_connected(
            net,
            hidden,
            weights_initializer=xavier_initializer(uniform=False),
            activation_fn=activation,
            scope="fc4")
        net = slim.fully_connected(
            net,
            num_actions,
            weights_initializer=xavier_initializer(uniform=False),
            activation_fn=None,
            scope="y")
    return net

def MeasurementsModel(inputs, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        activation = tf.nn.relu
        net = inputs
        net = slim.fully_connected(
            net,
            128,
            weights_initializer=xavier_initializer(uniform=False),
            activation_fn=activation,
            scope="fc1")
        net = slim.fully_connected(
            net,
            128,
            weights_initializer=xavier_initializer(uniform=False),
            activation_fn=activation,
            scope="fc2")
        net = slim.fully_connected(
            net,
            num_actions,
            weights_initializer=xavier_initializer(uniform=False),
            activation_fn=None,
            scope="y")
    return net

def nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=64, init_scale=np.sqrt(2)))


class ActorCriticPolicy(BasePolicy):
    """
    Policy object that implements actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False):
        super(ActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=scale)
        self._pdtype = make_proba_dist_type(ac_space)
        self._policy = None
        self._proba_distribution = None
        self._value_fn = None
        self._action = None
        self._deterministic_action = None

    def _setup_init(self):
        """
        sets up the distibutions, actions, and value
        """
        with tf.variable_scope("output", reuse=True):
            assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None
            self.logstd = self.proba_distribution.logstd
            self.mean = self.proba_distribution.mode() 
            self._action = self.proba_distribution.sample()
            self._deterministic_action = self.proba_distribution.mode()
            self._neglogp = self.proba_distribution.neglogp(self.action)
            if isinstance(self.proba_distribution, CategoricalProbabilityDistribution):
                self._policy_proba = tf.nn.softmax(self.policy)
            elif isinstance(self.proba_distribution, DiagGaussianProbabilityDistribution):
                self._policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]
            elif isinstance(self.proba_distribution, BernoulliProbabilityDistribution):
                self._policy_proba = tf.nn.sigmoid(self.policy)
            elif isinstance(self.proba_distribution, MultiCategoricalProbabilityDistribution):
                self._policy_proba = [tf.nn.softmax(categorical.flatparam())
                                     for categorical in self.proba_distribution.categoricals]
            else:
                self._policy_proba = []  # it will return nothing, as it is not implemented
            self._value_flat = self.value_fn[:, 0]

    @property
    def pdtype(self):
        """ProbabilityDistributionType: type of the distribution for stochastic actions."""
        return self._pdtype

    @property
    def policy(self):
        """tf.Tensor: policy output, e.g. logits."""
        return self._policy

    @property
    def proba_distribution(self):
        """ProbabilityDistribution: distribution of stochastic actions."""
        return self._proba_distribution

    @property
    def value_fn(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, 1)"""
        return self._value_fn

    @property
    def value_flat(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, )"""
        return self._value_flat

    @property
    def action(self):
        """tf.Tensor: stochastic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action

    @property
    def deterministic_action(self):
        """tf.Tensor: deterministic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._deterministic_action

    @property
    def neglogp(self):
        """tf.Tensor: negative log likelihood of the action sampled by self.action."""
        return self._neglogp

    @property
    def policy_proba(self):
        """tf.Tensor: parameters of the probability distribution. Depends on pdtype."""
        return self._policy_proba

    @abstractmethod
    def step(self, obs, state=None, mask=None, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def value(self, obs, state=None, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError


class FeedForwardPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                activ = tf.nn.tanh

                observation_features = self.processed_obs[:, :, -8:]
                observation_features_flat = tf.layers.flatten(observation_features)

                visual_features = self.processed_obs[:, :, :-8]
                visual_features = tf.reshape(visual_features, [-1, 128, 128, 15])

                vis_pi_latent = vis_vf_latent = cnn_extractor(visual_features, **kwargs)
                vis_pi_latent = tf.reshape(vis_pi_latent, [-1, 1, 512])
                vis_vf_latent = tf.reshape(vis_vf_latent, [-1, 1, 512])

                meas_pi_h = activ(linear(observation_features_flat, "pi_meas_fc", 512, init_scale=np.sqrt(2)))
                meas_pi_latent = tf.reshape(meas_pi_h, [-1, 1, 512])
                features = tf.layers.flatten(tf.concat([vis_pi_latent, meas_pi_latent], axis=2))
                pi_latent = activ(linear(features, "pi_fc", 128, init_scale=np.sqrt(2)))

                meas_vf_h = activ(linear(observation_features_flat, "vf_meas_fc", 512, init_scale=np.sqrt(2)))
                meas_vf_latent = tf.reshape(meas_vf_h, [-1, 1, 512])
                features = tf.layers.flatten(tf.concat([vis_vf_latent, meas_vf_latent], axis=2))
                vf_latent = activ(linear(features, "vf_fc", 128, init_scale=np.sqrt(2)))

                # observation_features = self.processed_obs[:, :, -8:]
                # observation_features_flat = tf.layers.flatten(observation_features)

                # visual_features = self.processed_obs[:, :, :-8]
                # visual_features = tf.reshape(visual_features, [-1, 64, 64, 5])

                # vis_pi_latent = vis_vf_latent = cnn_extractor(visual_features, **kwargs)
                # vis_pi_latent = tf.reshape(vis_pi_latent, [-1, 1, 64])
                # vis_vf_latent = tf.reshape(vis_vf_latent, [-1, 1, 64])

                # meas_pi_h = activ(linear(observation_features_flat, "pi_meas_fc", 64, init_scale=np.sqrt(2)))
                # meas_pi_latent = tf.reshape(meas_pi_h, [-1, 1, 64])
                # features = tf.layers.flatten(tf.concat([vis_pi_latent, meas_pi_latent], axis=2))
                # pi_latent = activ(linear(features, "pi_fc", 64, init_scale=np.sqrt(2)))

                # meas_vf_h = activ(linear(observation_features_flat, "vf_meas_fc", 64, init_scale=np.sqrt(2)))
                # meas_vf_latent = tf.reshape(meas_vf_h, [-1, 1, 64])
                # features = tf.layers.flatten(tf.concat([vis_vf_latent, meas_vf_latent], axis=2))
                # vf_latent = activ(linear(features, "vf_fc", 64, init_scale=np.sqrt(2)))

            else:
                pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(self.processed_obs), net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()
    
    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class Policy_1_layer(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(Policy_1_layer, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[64],
                                                          vf=[64])],
                                           feature_extraction="mlp")
    
    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
            return action, value, self.initial_state, neglogp, None, None
        else:
            action, value, neglogp, logstd, mean = self.sess.run([self.action, self.value_flat, self.neglogp, 
                                                                    self.logstd, self.mean],
                                                   {self.obs_ph: obs})
            return action, value, self.initial_state, neglogp, logstd, mean
        
class Policy_2_layer(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(Policy_2_layer, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[64, 64],
                                                          vf=[64, 64])],
                                           feature_extraction="mlp")
    
    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
            return action, value, self.initial_state, neglogp, None, None
        else:
            action, value, neglogp, logstd, mean = self.sess.run([self.action, self.value_flat, self.neglogp, 
                                                                    self.logstd, self.mean],
                                                   {self.obs_ph: obs})
            return action, value, self.initial_state, neglogp, logstd, mean


class CustomPolicy(ActorCriticPolicy):
    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
            return action, value, self.initial_state, neglogp, None, None
        else:
            action, value, neglogp, logstd, mean = self.sess.run([self.action, self.value_flat, self.neglogp,
                                                                  self.logstd, self.mean],
                                                                 {self.obs_ph: obs})
            return action, value, self.initial_state, neglogp, logstd, mean

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class CustomPolicy1(CustomPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy1, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.tanh
            
            measurement_features = tf.expand_dims(self.processed_obs[:, :, -1], axis=1)
            vae_features = self.processed_obs[:, :, :-1]
            vae_features_flat = tf.layers.flatten(vae_features)
            
            pi_h = activ(linear(vae_features_flat, "pi_vae_fc", 64, init_scale=np.sqrt(2)))
            pi_latent = tf.reshape(pi_h, [-1, 1, 64])
            features = tf.layers.flatten(tf.concat([pi_latent, measurement_features], axis=2))
            pi_latent = activ(linear(features, "pi_fc", 64, init_scale=np.sqrt(2)))


            vf_h = activ(linear(vae_features_flat, "vf_vae_fc", 64, init_scale=np.sqrt(2)))
            vf_latent = tf.reshape(vf_h, [-1, 1, 64])
            features = tf.layers.flatten(tf.concat([vf_latent, measurement_features], axis=2))
            vf_latent = activ(linear(features, "vf_fc", 64, init_scale=np.sqrt(2)))
            
            value_fn = linear(vf_latent, 'vf', 1, init_scale=np.sqrt(2))

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()


class CustomPolicy2(CustomPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy2, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.tanh
            
            # HARD CODED: Taking last 8 observation input
            observation_features = self.processed_obs[:, :, -6:]
            observation_features_flat = tf.layers.flatten(observation_features)
            vae_features = self.processed_obs[:, :, :-6]
            vae_features_flat = tf.layers.flatten(vae_features)

            vae_pi_h = activ(linear(vae_features_flat, "pi_vae_fc", 64, init_scale=np.sqrt(2)))
            vae_pi_latent = tf.reshape(vae_pi_h, [-1, 1, 64])
            meas_pi_h = activ(linear(observation_features_flat, "pi_meas_fc", 64, init_scale=np.sqrt(2)))
            meas_pi_latent = tf.reshape(meas_pi_h, [-1, 1, 64])
            features = tf.layers.flatten(tf.concat([vae_pi_latent, meas_pi_latent], axis=2))
            pi_latent = activ(linear(features, "pi_fc", 64, init_scale=np.sqrt(2)))


            vae_vf_h = activ(linear(vae_features_flat, "vf_vae_fc", 64, init_scale=np.sqrt(2)))
            vae_vf_latent = tf.reshape(vae_vf_h, [-1, 1, 64])
            meas_vf_h = activ(linear(observation_features_flat, "vf_meas_fc", 64, init_scale=np.sqrt(2)))
            meas_vf_latent = tf.reshape(meas_vf_h, [-1, 1, 64])
            features = tf.layers.flatten(tf.concat([vae_vf_latent, meas_vf_latent], axis=2))
            vf_latent = activ(linear(features, "vf_fc", 64, init_scale=np.sqrt(2)))
            
            # value_fn = linear(vf_latent, 'vf', 1, init_scale=np.sqrt(2))
            value_fn = tf.layers.dense(vf_latent, 1, name='vf')

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()


class CustomPolicy3(CustomPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy3, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.tanh

            # HARD CODED: Taking last 8 observation input
            observation_features = self.processed_obs[:, :, -1:]
            observation_features_flat = tf.layers.flatten(observation_features)
            vae_features = self.processed_obs[:, :, :-1]
            vae_features_flat = tf.layers.flatten(vae_features)

            vae_pi_h = activ(linear(vae_features_flat, "pi_vae_fc", 64, init_scale=np.sqrt(2)))
            vae_pi_latent = tf.reshape(vae_pi_h, [-1, 1, 64])
            meas_pi_h = activ(linear(observation_features_flat, "pi_meas_fc", 64, init_scale=np.sqrt(2)))
            meas_pi_latent = tf.reshape(meas_pi_h, [-1, 1, 64])
            features = tf.layers.flatten(tf.concat([vae_pi_latent, meas_pi_latent], axis=2))
            pi_latent = activ(linear(features, "pi_fc", 64, init_scale=np.sqrt(2)))


            vae_vf_h = activ(linear(vae_features_flat, "vf_vae_fc", 64, init_scale=np.sqrt(2)))
            vae_vf_latent = tf.reshape(vae_vf_h, [-1, 1, 64])
            meas_vf_h = activ(linear(observation_features_flat, "vf_meas_fc", 64, init_scale=np.sqrt(2)))
            meas_vf_latent = tf.reshape(meas_vf_h, [-1, 1, 64])
            features = tf.layers.flatten(tf.concat([vae_vf_latent, meas_vf_latent], axis=2))
            vf_latent = activ(linear(features, "vf_fc", 64, init_scale=np.sqrt(2)))

            # value_fn = linear(vf_latent, 'vf', 1, init_scale=np.sqrt(2))
            value_fn = tf.layers.dense(vf_latent, 1, name='vf')

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()


class CustomPolicy4(CustomPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy4, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.tanh

            # HARD CODED: Taking last 6 observation input
            observation_features = self.processed_obs[:, :, -6:]
            observation_features_flat = tf.layers.flatten(observation_features)
            vae_features = self.processed_obs[:, :, :-6]
            vae_features_flat = tf.layers.flatten(vae_features)

            vae_pi_h = activ(linear(vae_features_flat, "pi_vae_fc", 64, init_scale=np.sqrt(2)))
            vae_pi_latent = tf.reshape(vae_pi_h, [-1, 1, 64])
            meas_pi_h = activ(linear(observation_features_flat, "pi_meas_fc", 64, init_scale=np.sqrt(2)))
            meas_pi_latent = tf.reshape(meas_pi_h, [-1, 1, 64])
            features = tf.layers.flatten(tf.concat([vae_pi_latent, meas_pi_latent], axis=2))
            pi_latent = activ(linear(features, "pi_fc", 64, init_scale=np.sqrt(2)))


            vae_vf_h = activ(linear(vae_features_flat, "vf_vae_fc", 64, init_scale=np.sqrt(2)))
            vae_vf_latent = tf.reshape(vae_vf_h, [-1, 1, 64])
            meas_vf_h = activ(linear(observation_features_flat, "vf_meas_fc", 64, init_scale=np.sqrt(2)))
            meas_vf_latent = tf.reshape(meas_vf_h, [-1, 1, 64])
            features = tf.layers.flatten(tf.concat([vae_vf_latent, meas_vf_latent], axis=2))
            vf_latent = activ(linear(features, "vf_fc", 64, init_scale=np.sqrt(2)))

            # value_fn = linear(vf_latent, 'vf', 1, init_scale=np.sqrt(2))
            value_fn = tf.layers.dense(vf_latent, 1, name='vf')

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

class CustomWPPolicy(CustomPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomWPPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.tanh
            
            measurement_features = tf.expand_dims(self.processed_obs[:, -1], axis=1)
            measurement_features_flat = tf.layers.flatten(measurement_features)
            
            pi_h = activ(linear(measurement_features_flat, "pi_vae_fc", 64, init_scale=np.sqrt(2)))
            pi_latent = activ(linear(pi_h, "pi_fc", 64, init_scale=np.sqrt(2)))
            
            vf_h = activ(linear(measurement_features_flat, "vf_vae_fc", 64, init_scale=np.sqrt(2)))
            vf_latent = activ(linear(pi_h, "vf_fc", 64, init_scale=np.sqrt(2)))
            
            value_fn = linear(vf_latent, 'vf', 1, init_scale=np.sqrt(2))

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()


class CnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", **_kwargs)

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
            return action, value, self.initial_state, neglogp, None, None
        else:
            action, value, neglogp, logstd, mean = self.sess.run([self.action, self.value_flat, self.neglogp,
                                                                    self.logstd, self.mean],
                                                   {self.obs_ph: obs})
            return action, value, self.initial_state, neglogp, logstd, mean
