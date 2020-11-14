# Copyright (c) 2018 Roma Sokolkov
# Copyright (c) 2018 hardmaru
# MIT License

'''
AE models.
'''

import time
import numpy as np
import os
import tensorflow as tf
import json


def normalize(data):
    return data / 255.0


def denormalize(data):
    return data * 255.0


class ConvPrImitator(object):
    def __init__(self, z_size=512, image_size= (128, 128, 3), gt_size = 2, batch_size=100, learning_rate=0.0001, is_training=True,
                 reuse=False, frame_stack=1, gpu_mode=True):
        self.z_size = z_size # Unused
        self.gt_size = gt_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.reuse = reuse
        self.image_size = image_size
        self.frame_stack = frame_stack
        with tf.variable_scope('conv_ae', reuse=self.reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using cpu.')
                    self._build_graph()
            else:
                tf.logging.info('Model using gpu.')
                self._build_graph()
        self._init_session()

    def compute_outshape(self, inp_size, kernel_size, padding, stride):
        return (inp_size-kernel_size+2*padding)//stride+1

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            h,w,c = self.image_size 
            out_w = w
            out_h = h
            self.x = tf.placeholder(tf.float32, shape=[None, h, w, c* self.frame_stack])
            self.gt = tf.placeholder(tf.float32, shape=[None, self.gt_size])

            # Encoder
            h = tf.layers.conv2d(self.x, 16, 5, strides=2, activation=tf.nn.relu, \
                                kernel_initializer=tf.initializers.variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform'), \
                                bias_initializer=tf.initializers.zeros(), name="enc_conv1")
            out_w = self.compute_outshape(out_w, 5, 0, 2)
            out_h = self.compute_outshape(out_h, 5, 0, 2)

            h = tf.layers.conv2d(h, 32, 5, strides=2, activation=tf.nn.relu, \
                                kernel_initializer=tf.initializers.variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform'), \
                                bias_initializer=tf.initializers.zeros(), name="enc_conv2")
            out_w = self.compute_outshape(out_w, 5, 0, 2)
            out_h = self.compute_outshape(out_h, 5, 0, 2)

            h = tf.layers.conv2d(h, 64, 5, strides=2, activation=tf.nn.relu, \
                                kernel_initializer=tf.initializers.variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform'), \
                                bias_initializer=tf.initializers.zeros(), name="enc_conv3")
            out_w = self.compute_outshape(out_w, 5, 0, 2)
            out_h = self.compute_outshape(out_h, 5, 0, 2)
            # Model used for Learning to drive using Waypoints (last layer dim = 16)
            # h = tf.layers.conv2d(h, 16, 5, strides=2, activation=tf.nn.relu, name="enc_conv4")
            # self.encoded = tf.reshape(h, [-1, 5 * 5 * 16])

            # Model used for Learning to Drive with Dynamic Actors (last layer dim = 64)
            h = tf.layers.conv2d(h, 64, 5, strides=2, activation=tf.nn.relu, \
                                kernel_initializer=tf.initializers.variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform'), \
                                bias_initializer=tf.initializers.zeros(), name="enc_conv4")
            out_w = self.compute_outshape(out_w, 5, 0, 2)
            out_h = self.compute_outshape(out_h, 5, 0, 2)

            self.encoded = tf.reshape(h, [-1, out_h * out_w * 64])

            self.z = tf.placeholder(tf.float32, shape = [None, self.z_size])

            mod_h = tf.concat([self.encoded, self.z], 1)
            self.y = tf.layers.dense(mod_h, self.gt_size, activation = tf.nn.tanh)

            # train ops
            if self.is_training:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

                eps = 1e-6  # avoid taking log of zero

                # Commented code for weighting classes
                # class_weights = tf.constant([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 500.0, 1.0, 1.0]])
                # weights = tf.reduce_sum(class_weights * labels, axis=1)
                # weighted_losses = entropy_loss * weights
                # self.entropy_loss = tf.reduce_mean(weighted_losses)
                

                labels = tf.reshape(self.gt, (-1, self.gt_size))
                preds = tf.reshape(self.y, (-1,self.gt_size))

                regression_loss = tf.nn.l2_loss(labels-preds)
                self.regression_loss = tf.reduce_mean(regression_loss)
                self.loss = self.regression_loss
                
                self.output_preds = preds
                
                # training
                self.lr = tf.Variable(self.learning_rate, trainable=False)
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                grads = self.optimizer.compute_gradients(self.loss)  # can potentially clip gradients here.

                self.train_op = self.optimizer.apply_gradients(
                    grads, global_step=self.global_step, name='train_step')

            # initialize vars
            self.init = tf.global_variables_initializer()
            self.init_l = tf.local_variables_initializer()

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.g, config=config)
        self.sess.run(self.init)
        self.sess.run(self.init_l)

    def encode(self, x, z):
        return self.sess.run(self.encoded, feed_dict={self.x: x, self.z:z})

    def predict(self, x, z):
        return self.sess.run(self.y, feed_dict={self.x: x, self.z:z})

    def get_model_params(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                param_name = var.name
                p = self.sess.run(var)
                model_names.append(param_name)
                params = np.round(p * 10000).astype(np.int).tolist()
                model_params.append(params)
                model_shapes.append(p.shape)
        return model_params, model_shapes, model_names

    def get_random_model_params(self, stdev=0.5):
        # get random params.
        _, mshape, _ = self.get_model_params()
        rparam = []
        for s in mshape:
            # rparam.append(np.random.randn(*s)*stdev)
            rparam.append(np.random.standard_cauchy(s) * stdev)  # spice things up!
        return rparam

    def set_model_params(self, params):
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            print("No of trainable variables: {}".format(len(t_vars)))
            assign_ops = []
            for var in t_vars:
                time_start = time.time()
                # pshape = self.sess.run(var).shape
                p = np.array(params[idx])
                # assert pshape == p.shape, "inconsistent shape"
                assign_op = var.assign(p.astype(np.float) / 10000.)
                assign_ops.append(assign_op)
                idx += 1
                print("Time to set AE target model param: {}, shape: {}, time: {}".format(idx, p.shape, time.time() - time_start))
            time_start = time.time()
            assign_ops = tf.group(*assign_ops)
            self.sess.run(assign_ops)
            print("Time to assign AE target model params: {}".format(time.time() - time_start))

    def load_json(self, jsonfile='ae.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)

    def set_random_params(self, stdev=0.5):
        rparam = self.get_random_model_params(stdev)
        self.set_model_params(rparam)

    def load_checkpoint(self, checkpoint_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print('loading model', ckpt.model_checkpoint_path)
        tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)