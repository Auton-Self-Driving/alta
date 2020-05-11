from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import io
import itertools
# from packaging import version
from six.moves import range

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn                                                    
import pandas as pd
import os
# import sklearn.metrics

# print("TensorFlow version: ", tf.__version__)
# assert version.parse(tf.__version__).release[0] >= 2, \
#     "This notebook requires TensorFlow 2.0 or above."


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  buf_value = buf.getvalue()
  image = tf.image.decode_png(buf_value, channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return buf_value, image

# def image_grid():
#   """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
#   # Create a figure to contain the plot.
#   figure = plt.figure(figsize=(10,10))
#   for i in range(25):
#     # Start next subplot.
#     plt.subplot(5, 5, i + 1, title=class_names[train_labels[i]])
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
  
#   return figure

def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Normalize the confusion matrix.
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

def log_confusion_matrix(cm, file_writer_cm, class_names, epoch):
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names=class_names)
  _, cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)

REDUCED_SEMANTIC_COLOR_MAP = {
    0	: ["Everything Else", ( 0, 0, 0)],
    1	: ["Pedestrian",	(220, 20, 60)],
    2	: ["Road line",	(157, 234, 50)],
    3	: ["Road",	(128, 64, 128)],
    4	: ["Car",	( 0, 0, 142)],
    5	: ["Total",	( 0, 0, 142)]
}

def get_cm_image(cm):
    df_cm = pd.DataFrame(cm, index = [REDUCED_SEMANTIC_COLOR_MAP[i][0] for i in range(5)], columns = [REDUCED_SEMANTIC_COLOR_MAP[i][0] for i in range(5)])
    figure = plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    cm_image_png, cm_image = plot_to_image(figure)

    return cm_image_png, cm_image

def save_cm(cm, path, ind):
    if not os.path.exists(path):
        os.makedirs(path)
    df_cm = pd.DataFrame(cm, index = [REDUCED_SEMANTIC_COLOR_MAP[i][0] for i in range(5)], columns = [REDUCED_SEMANTIC_COLOR_MAP[i][0] for i in range(5)])
    figure = plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(path + 'cm_{}.png'.format(ind))
    plt.close()




