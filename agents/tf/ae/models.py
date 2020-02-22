# Copyright (c) 2018 Roma Sokolkov
# Copyright (c) 2018 hardmaru
# MIT License

'''
AE models.
'''

import numpy as np
import os
import tensorflow as tf
import json


def normalize(data):
    return data / 255.0


def denormalize(data):
    return data * 255.0


class ConvAutoEncoder(object):
    def __init__(self, z_size=512, batch_size=100, learning_rate=0.0001, is_training=True,
                 reuse=False, num_classes=5, frame_stack=1, gpu_mode=True):
        self.z_size = z_size # Unused
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.reuse = reuse
        self.num_classes = num_classes
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

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, 128, 128, self.num_classes * self.frame_stack])

            # Encoder
            h = tf.layers.conv2d(self.x, 16, 5, strides=2, activation=tf.nn.relu, name="enc_conv1")
            h = tf.layers.conv2d(h, 32, 5, strides=2, activation=tf.nn.relu, name="enc_conv2")
            h = tf.layers.conv2d(h, 64, 5, strides=2, activation=tf.nn.relu, name="enc_conv3")
            h = tf.layers.conv2d(h, 16, 5, strides=2, activation=tf.nn.relu, name="enc_conv4")
            self.encoded = tf.reshape(h, [-1, 5 * 5 * 16])

            # Decoder
            h = tf.reshape(self.encoded, [-1, 5, 5, 16])
            h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name="dec_deconv1")
            h = tf.layers.conv2d_transpose(h, 32, 5, strides=2, activation=tf.nn.relu, name="dec_deconv2")
            h = tf.layers.conv2d_transpose(h, 16, 6, strides=2, activation=tf.nn.relu, name="dec_deconv3")
            self.y = tf.layers.conv2d_transpose(h, self.num_classes * self.frame_stack, 6, strides=2, activation=None, name="dec_deconv4")

            # train ops
            if self.is_training:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

                eps = 1e-6  # avoid taking log of zero

                # # cross-entropy pixel wise loss
                self.decoded = tf.nn.softmax(self.y, name='decoded')
                
                labels = tf.reshape(self.x, (-1, self.frame_stack, self.num_classes))
                logits = tf.reshape(self.y, (-1, self.frame_stack, self.num_classes))

                entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
                self.entropy_loss = tf.reduce_mean(entropy_loss)
                self.loss = self.entropy_loss
                
                probs = tf.reshape(self.decoded, (-1, self.frame_stack, self.num_classes))
                input_labels = tf.argmax(labels, axis=-1)
                self.output_preds = tf.argmax(probs, axis=-1)
                
                input_labels_flattened = tf.reshape(input_labels, [-1])
                output_preds_flattened = tf.reshape(self.output_preds,[-1])
                confusion_matrix = np.zeros((self.num_classes, self.num_classes))
                
                my_accuracy = tf.reduce_mean(tf.cast(tf.equal(input_labels_flattened, output_preds_flattened), tf.float32))
                self.my_accuracy = my_accuracy
                
                self.accuracy, self.accuracy_op = tf.metrics.accuracy(input_labels, self.output_preds)

                self.confusion_matrix = tf.confusion_matrix(input_labels_flattened, output_preds_flattened, num_classes=self.num_classes)

                self.my_confusion_matrix = self.confusion_matrix
                self.my_confusion_matrix_normalized = self.confusion_matrix / tf.reshape(tf.reduce_sum(self.confusion_matrix, axis=1), [-1,1])

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

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def encode(self, x):
        return self.sess.run(self.encoded, feed_dict={self.x: x})

    def decode(self, z):
        return self.sess.run(self.y, feed_dict={self.encoded: z})

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
            for var in t_vars:
                pshape = self.sess.run(var).shape
                p = np.array(params[idx])
                assert pshape == p.shape, "inconsistent shape"
                assign_op = var.assign(p.astype(np.float) / 10000.)
                self.sess.run(assign_op)
                idx += 1

    def load_json(self, jsonfile='ae.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)

    def save_json(self, jsonfile='ae.json'):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

    def set_random_params(self, stdev=0.5):
        rparam = self.get_random_model_params(stdev)
        self.set_model_params(rparam)

    def save_model(self, model_save_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = os.path.join(model_save_path, 'ae')
        tf.logging.info('saving model %s.', checkpoint_path)
        saver.save(sess, checkpoint_path, 0)  # just keep one

    def load_checkpoint(self, checkpoint_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print('loading model', ckpt.model_checkpoint_path)
        tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)