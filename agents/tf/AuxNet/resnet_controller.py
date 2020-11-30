# Copyright (c) 2018 Roma Sokolkov
# MIT License

'''
AE controller for runtime optimization.
'''

import time
import numpy as np
import sys

from .resnet import ResNet34
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

import ipdb
from .dataset import CarlaDatasetGenerator

st = ipdb.set_trace

class ResnetController:
    def __init__(self, zdim=3, image_size=(224, 224, 3),
                epoch_per_optimization=50,
                batch_size=300,
                frozen=True,
                checkpoint_path='/home/scratch/mayankgu/model_weights.ckpt',
                ):
        self.epoch_per_optimization = epoch_per_optimization
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path

        # Buffer
        self.buffer = []
        
        self.model = ResNet34((224,224,3), weights='imagenet', include_top=False, zdim=zdim)
        self.model.compile(loss='mean_squared_error', optimizer="adam")
        
        self.checkpoint_path="/home/scratch/mayankgu/cp.ckpt"

        if frozen:        
            for l in self.model.layers:
                if l.name not in ['pool1', 'concatenate_1', 'fc_1', 'fc_2', 'fc_3', 'relu2', 'relu3']:
                    l.trainable = False

    def optimize(self, iter):
        self.iter=iter
        self.checkpoint_path="/home/scratch/mayankgu/DAGGER_iter_"+str(iter)+".ckpt"
        cp_callback = ModelCheckpoint(filepath=self.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        train_images = np.stack([i[0][0] for i in self.buffer])
        train_manual = np.stack([i[0][1] for i in self.buffer])
        train_labels = np.stack([i[1] for i in self.buffer])
        carla_gen = CarlaDatasetGenerator(train_images, train_manual, train_labels, self.batch_size)
        self.model.fit_generator(generator=carla_gen,
          steps_per_epoch=len(carla_gen),
          epochs=self.epoch_per_optimization,
          callbacks=[cp_callback])
    
    def predict(self, img, manual_states):
        self.model(img, manual_states)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path, unfrozen=False):
        self.model.load_weights(path)
        if unfrozen:
            for l in model.layers:
                if l.name not in ['data', 'manual_states_data']:
                    l.trainable = True