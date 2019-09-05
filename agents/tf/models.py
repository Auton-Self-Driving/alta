from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer
from stable_baselines.common.policies import ActorCriticPolicy, FeedForwardPolicy, register_policy, nature_cnn


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

class MlpPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(MlpPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[64, 64],
                                                          vf=[64, 64])],
                                           feature_extraction="mlp")

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu
            
            measurement_features = tf.expand_dims(self.processed_obs[:, :, -1], axis=1)
            vae_features = self.processed_obs[:, :, :-1]
            vae_features_flat = tf.layers.flatten(vae_features)
            
            pi_h = activ(tf.layers.dense(vae_features_flat, 1, name='pi_vae_fc'))
            pi_latent = tf.reshape(pi_h, [-1, 1, 1])
            features = tf.layers.flatten(tf.concat([pi_latent, measurement_features], axis=2))
            pi_latent = activ(tf.layers.dense(features, 64, name='pi_fc'))
            
            vf_h = activ(tf.layers.dense(vae_features_flat, 1, name='vf_vae_fc'))
            vf_latent = tf.reshape(vf_h, [-1, 1, 1])
            features = tf.layers.flatten(tf.concat([vf_latent, measurement_features], axis=2))
            vf_latent = activ(tf.layers.dense(features, 64, name='vf_fc'))
            
            value_fn = tf.layers.dense(vf_latent, 1, name='vf')

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
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

