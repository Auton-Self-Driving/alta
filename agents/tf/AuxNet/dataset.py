import numpy as np
import keras

class CarlaDatasetGenerator(keras.utils.Sequence):
    def __init__(self, train_images, train_manual, train_labels, train_helper_states, batch_size) :
        self.train_images = train_images
        self.train_manual = train_manual
        self.train_labels = train_labels
        self.train_helper_states = train_helper_states
        self.batch_size = batch_size
    
    def __len__(self) :
        return (np.ceil(len(self.train_labels) / float(self.batch_size))).astype(np.int)
  
  
    def __getitem__(self, idx) :
        batch_images = self.train_images[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_manual = self.train_manual[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_labels = self.train_labels[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_helper_states = self.train_helper_states[idx * self.batch_size : (idx+1) * self.batch_size]        
        
        input_data, labels = [np.array(batch_images), np.array(batch_manual)], [np.array(batch_labels), np.array(batch_helper_states)]
        return input_data, labels