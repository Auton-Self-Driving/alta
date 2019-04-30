from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer

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
