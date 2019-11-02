# Copyright (c) 2018 Roma Sokolkov
# MIT License

'''
VAE controller for runtime optimization.
'''

import numpy as np

from .model import ConvVAE


class VAEController:
    def __init__(self, z_size=512, image_size=(80, 160, 3),
                 learning_rate=0.0001, kl_tolerance=0.5,
                 epoch_per_optimization=10, batch_size=64,
                 buffer_size=500):
        # VAE input and output shapes
        self.z_size = z_size
        self.image_size = image_size

        # VAE params
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance

        # Training params
        self.epoch_per_optimization = epoch_per_optimization
        self.batch_size = batch_size

        # Buffer
        self.buffer_size = buffer_size
        self.buffer_pos = -1
        self.buffer_full = False
        self.buffer_reset()

        self.vae = ConvVAE(z_size=self.z_size,
                           batch_size=self.batch_size,
                           learning_rate=self.learning_rate,
                           kl_tolerance=self.kl_tolerance,
                           is_training=True,
                           reuse=False,
                           gpu_mode=True)

        self.target_vae = ConvVAE(z_size=self.z_size,
                                  batch_size=1,
                                  is_training=False,
                                  reuse=False,
                                  gpu_mode=True)

    def buffer_append(self, arr):
        # print(arr.shape, self.image_size)
        assert arr.shape == self.image_size
        self.buffer_pos += 1
        if self.buffer_pos > self.buffer_size - 1:
            self.buffer_pos = 0
            self.buffer_full = True
        self.buffer[self.buffer_pos] = arr

    def buffer_reset(self):
        self.buffer_pos = -1
        self.buffer_full = False
        self.buffer = np.zeros((self.buffer_size,
                                self.image_size[0],
                                self.image_size[1],
                                self.image_size[2]),
                               dtype=np.uint8)

    def buffer_get_copy(self):
        if self.buffer_full:
            return self.buffer.copy()
        return self.buffer[:self.buffer_pos]

    def encode(self, arr):
        # print(arr.shape, self.image_size)
        assert arr.shape == self.image_size
        # Normalize
        # arr = arr.astype(np.float) / 255.0
        # Reshape
        arr = arr.reshape(1,
                          self.image_size[0],
                          self.image_size[1],
                          self.image_size[2])
        return self.target_vae.encode(arr)

    def decode(self, arr):
        assert arr.shape == (1, self.z_size)
        # Decode
        arr = self.target_vae.decode(arr)
        # Denormalize
        # arr = arr * 255.0
        return arr

    def optimize(self):
        ds = self.buffer_get_copy()
        # TODO: may be do buffer reset.
        # self.buffer_reset()

        num_batches = int(np.floor(len(ds) / self.batch_size))

        train_step = 0

        for epoch in range(self.epoch_per_optimization):
            np.random.shuffle(ds)

            train_loss_array = []
            entropy_loss_array = []
            kl_loss_array = []
            accuracy_array = []
            confusion_matrix_final = 0
            my_accuracy_array = []

            for idx in range(num_batches):
                batch = ds[idx * self.batch_size:(idx + 1) * self.batch_size]
                # obs = batch.astype(np.float) / 255.0
                obs = batch.astype(np.float)
                feed = {self.vae.x: obs, }
                (train_loss, entropy_loss, kl_loss, train_step, _, confusion_matrix, my_confusion_matrix_normalized, accuracy, accuracy_op, my_accuracy) = self.vae.sess.run([
                    self.vae.loss,
                    self.vae.entropy_loss,
                    self.vae.kl_loss,
                    self.vae.global_step,
                    self.vae.train_op,
                    self.vae.confusion_matrix,
                    self.vae.my_confusion_matrix_normalized,
                    self.vae.accuracy,
                    self.vae.accuracy_op,
                    self.vae.my_accuracy
                ], feed)
                if ((train_step + 1) % 50 == 0):
                    print("VAE: optimization step",
                          (train_step + 1), train_loss, entropy_loss, kl_loss)

                train_loss_array.append(train_loss)
                entropy_loss_array.append(entropy_loss)
                kl_loss_array.append(kl_loss)

                accuracy_array.append(accuracy)
                confusion_matrix_final += confusion_matrix
                my_accuracy_array.append(my_accuracy)

        self.set_target_params()

        # Average values in last epoch
        train_loss_avg = np.mean(np.array(train_loss_array))
        entropy_loss_avg = np.mean(np.array(entropy_loss_array))
        kl_loss_avg = np.mean(np.array(kl_loss_array))

        accuracy_avg = np.mean(np.array(accuracy_array))
        confusion_matrix_final = np.array(confusion_matrix_final)

        my_accuracy_avg = np.mean(np.array(my_accuracy_array))

        return train_loss_avg, entropy_loss_avg, kl_loss_avg, accuracy_avg, my_accuracy_avg, confusion_matrix_final, confusion_matrix_final, train_step

    def save(self, path):
        self.target_vae.save_json(path)

    def load(self, path):
        self.target_vae.load_json(path)

    def set_target_params(self):
        params, _, _ = self.vae.get_model_params()
        self.target_vae.set_model_params(params)
