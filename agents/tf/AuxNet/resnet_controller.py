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
from keras.callbacks import ModelCheckpoint, EarlyStopping

import ipdb
from .dataset import CarlaDatasetGenerator

st = ipdb.set_trace

class ResnetController:
    def __init__(self, zdim=3, image_size=(224, 224, 3),
                epoch_per_optimization=30,
                batch_size=300,
                frozen=False,
                checkpoint_path='/home/scratch/vkadi/model_weights.ckpt',
                val_data=None,
                return_feat = False):
        self.epoch_per_optimization = epoch_per_optimization
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path

        # Buffer
        self.buffer = []
        self.val_data=val_data

        self.feat_model = None
        if return_feat:
            self.model, self.feat_model = ResNet34((224,224,3), weights='imagenet', include_top=False, zdim=zdim, return_feat = True)            
        else:
            self.model = ResNet34((224,224,3), weights='imagenet', include_top=False, zdim=zdim)
        self.model.compile(loss=['mean_squared_error', 'binary_crossentropy'], optimizer="adam")
        
        self.checkpoint_path="/home/scratch/vkadi/cp.ckpt"

        if frozen:        
            for l in self.model.layers:
                if l.name not in ['pool1', 'concatenate_1', 'fc_1', 'fc_2', 'fc_3', 'relu2', 'relu3']:
                    l.trainable = False
    
    def unfreeze(self):
        for l in self.model.layers:
            if l.name not in ['data', 'manual_states_data']:
                l.trainable = True

    def optimize(self, iter, epochs=None, patience=6):
        self.iter=iter
        self.checkpoint_path="/home/scratch/vkadi/resnet_DAGGER_iter_"+str(iter)+".ckpt"
        cp_callback = ModelCheckpoint(filepath=self.checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 mode='min',
                                                 monitor='val_loss',
                                                 verbose=1)
        train_images = np.stack([i[0][0] for i in self.buffer])
        train_manual = np.stack([i[0][1] for i in self.buffer])
        train_labels = np.stack([i[1] for i in self.buffer])
        train_helper_states = np.stack([i[2] for i in self.buffer])
        
        val_images = np.stack([i[0][0] for i in self.val_data])
        val_manual = np.stack([i[0][1] for i in self.val_data])
        val_labels = np.stack([i[1] for i in self.val_data])
        val_helper_states = np.stack([i[2] for i in self.val_data])
        
        carla_gen = CarlaDatasetGenerator(train_images, train_manual, train_labels, train_helper_states, self.batch_size)
        if epochs is None:
            max_ep = self.epoch_per_optimization
        else:
            max_ep = epochs
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience)
        history = self.model.fit_generator(generator=carla_gen,
          steps_per_epoch=len(carla_gen),
          validation_data=([val_images, val_manual], [val_labels, val_helper_states]),
          epochs=max_ep,
          callbacks=[early_stopping_callback, cp_callback])
    
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