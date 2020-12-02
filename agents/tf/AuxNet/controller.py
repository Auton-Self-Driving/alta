# Copyright (c) 2018 Roma Sokolkov
# MIT License

'''
AE controller for runtime optimization.
'''

import time
import numpy as np
import sys

from .models import ConvPrImitator
import ipdb
st = ipdb.set_trace

class AuxNetController:
    def __init__(self, z_size=512, image_size=(160, 80, 5),
                 frame_stack=1,
                 learning_rate=0.001, kl_tolerance=0.5,
                 epoch_per_optimization=10, batch_size=64,
                 buffer_size=500, multi_task = None, gt_size = None, save_every_epoch = False, model_filepath=None):
        # AE input and output shapes
        self.z_size = z_size # Unused
        self.frame_stack = frame_stack
        self.image_size = (image_size[0], image_size[1], image_size[2] * self.frame_stack)
        self.num_classes = int(self.image_size[2] / self.frame_stack)

        # AE params
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance # Unused

        # Training params
        self.epoch_per_optimization = epoch_per_optimization
        self.batch_size = batch_size

        # Buffer
        self.buffer_size = buffer_size
        self.buffer_pos = -1
        self.buffer_full = False
        self.buffer_reset(gt_size) 
        self.multi_task = multi_task   
        self.gt_size = gt_size   

        self.priv_imitator = ConvPrImitator(z_size=self.z_size,
                           gt_size = self.gt_size,
                           batch_size=self.batch_size,
                           learning_rate=self.learning_rate,
                           is_training=True,
                           reuse=False,
                           num_classes=self.num_classes,
                           frame_stack=self.frame_stack,
                           multi_task = self.multi_task,
                           gpu_mode=True)
        self.save_every_epoch = False
        self.model_filepath = model_filepath

    def buffer_append(self, arr):
        # print(arr.shape, self.image_size)
        assert arr.shape == self.image_size
        self.buffer_pos += 1
        if self.buffer_pos > self.buffer_size - 1:
            #self.buffer_pos = 0
            self.buffer_full = True
        self.buffer_pos = self.buffer_pos%self.buffer_size
        self.buffer[self.buffer_pos] = arr

    def buffer_reset(self, gt_size = None):
        self.buffer_pos = -1
        self.buffer_full = False
        self.buffer_images = np.zeros((self.buffer_size,
                                self.image_size[0],
                                self.image_size[1],
                                self.image_size[2]),
                               dtype=np.uint8)
        if gt_size is not None:
            self.buffer_manual_states = np.zeros((self.buffer_size, self.z_size))
            self.buffer_gt = np.zeros((self.buffer_size, gt_size))
            self.buffer = list(zip(self.buffer_images, self.buffer_manual_states, self.buffer_gt))
        else:
            self.buffer = list(zip(self.buffer_images))

    def buffer_get_copy(self):
        if self.buffer_full:
            return self.buffer.copy()
        return self.buffer[:self.buffer_pos]

    def encode(self, arr):
        # print(arr.shape, self.image_size)
        assert arr.shape == self.image_size
        # # Normalize
        # arr = arr.astype(np.float) / 255.0
        # Reshape
        arr = arr.reshape(1,
                          self.image_size[0],
                          self.image_size[1],
                          self.image_size[2])
        return self.priv_imitator.encode(arr)

    def optimize(self):
        ds = self.buffer_get_copy()
        # TODO: may be do buffer reset.
        # self.buffer_reset()

        num_batches = int(np.floor(len(ds) / self.batch_size))
        train_step = 0

        train_loss_hist = []

        for epoch in range(self.epoch_per_optimization):
            time_start = time.time()
            np.random.shuffle(ds)

            train_loss_array = []
            accuracy_array = []
            confusion_matrix_final = 0
            my_accuracy_array = []
            my_confusion_matrix_final = 0

            for idx in range(num_batches):
                batch = ds[idx * self.batch_size:(idx + 1) * self.batch_size]
                # obs = batch.astype(np.float) / 255.0
                images = np.asarray([tup[0] for tup in batch]).astype(np.float)
                manual_state = np.asarray([tup[1] for tup in batch]).astype(np.float)
                gt = np.asarray([tup[2] for tup in batch]).astype(np.float)

                if self.multi_task=='state_pred':
                    state_supervision = np.asarray([tup[-1] for tup in batch]).astype(np.float)
                    feed = {self.priv_imitator.x: images, self.priv_imitator.z: manual_state, self.priv_imitator.gt: gt, self.priv_imitator.state_gt: state_supervision}
                else:
                    feed = {self.priv_imitator.x: images, self.priv_imitator.z: manual_state, self.priv_imitator.gt: gt}

                (train_loss, train_step, _) = self.priv_imitator.sess.run([
                    self.priv_imitator.loss,
                    self.priv_imitator.global_step,
                    self.priv_imitator.train_op
                ], feed) 
                if ((train_step + 1) % 100 == 0):
                    sys.stdout.write("Privileged agent Imitator: optimization step {%.2f}, Loss: {%.2f}\r"%(train_step + 1, train_loss))

                train_loss_array.append(train_loss)
            train_loss_hist.append(np.mean(np.array(train_loss_array)))
            print("\nTime to run Imitator epoch {}: {}".format(epoch, time.time() - time_start))
            if self.save_every_epoch:
                self.save(self.model_filepath)
                print(train_loss_hist)

        time_start = time.time()
        
        # my_confusion_matrix_normalized_final = my_confusion_matrix_final / np.sum(my_confusion_matrix_final, axis=1).reshape((-1,1))
        return train_loss_hist, train_step

    def save(self, path):
        self.priv_imitator.save_json(path)

    def load(self, path):
        self.priv_imitator.load_json(path)