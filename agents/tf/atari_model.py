from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer

def AtariModel(inputs, num_actions, scope, reuse=False):
    print('inputs.shape', inputs)
    with tf.variable_scope(scope, reuse=reuse):
        activation = tf.nn.relu
        convs1 = [
            [16, [3, 3], 1],
        ]
        convs2 = [
            [32, [3, 3], 1],
        ]
        net = inputs
        out_size, kernel, stride = convs1[0]
        net = slim.conv2d(net, out_size, kernel, stride, scope="conv1/conv3_1")
        #--------
        out_size, kernel, stride = convs2[0]
        net = slim.conv2d(net, out_size, kernel, stride, scope="conv2/conv3_1")
        net = tf.squeeze(net)
        net = tf.reshape(net, [-1, 84, 84, 32])
        net = slim.flatten(net, scope="flatten3")
        #--------
        net = tf.layers.dense(inputs=net, 
        units= 256, 
        activation=activation)
        net = tf.layers.dense(inputs=net,
        units=num_actions)
    return net