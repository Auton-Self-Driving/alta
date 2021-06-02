import glob
import argparse

import tensorflow as tf
import skimage as sk
import skimage.io as skio
import skimage.transform as sktf
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models as sm
from segmentation_models.losses import CategoricalCELoss


class CARLASegmentationData(tf.keras.utils.Sequence):
    def __init__(self, path):
        self.pathsX = sorted(glob.glob('{}/route_*/rgb/*.png'.format(path)))
        self.pathsY = sorted(glob.glob('{}/route_*/seg/*.png'.format(path)))
        assert len(self.pathsX) == len(self.pathsY), 'Mismatch in RGB and segmentation map samples'

    def __getitem__(self, idx):
        pathX, pathY = self.pathsX[idx], self.pathsY[idx]
        X, Y = skio.imread(pathX), skio.imread(pathY)
        return process_input(X, Y)

    def __len__(self):
        return len(self.pathsX)


def process_input(img, seg):
    img = sktf.resize(img, (224,224))
    seg = sktf.resize(seg, (224,224), preserve_range=True)
    seg = tf.keras.utils.to_categorical(seg, num_classes=23)
    return img[None,:,:,:], seg[None,:,:,:]


def convert_to_color(seg):
    height, width, channels = seg.shape
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[seg[:,:,0] == 1] = np.array([0,0,0])
    img[seg[:,:,1] == 1] = np.array([70,70,70])
    img[seg[:,:,2] == 1] = np.array([100,40,40])
    img[seg[:,:,3] == 1] = np.array([55,90,80])
    img[seg[:,:,4] == 1] = np.array([220,20,60])
    img[seg[:,:,5] == 1] = np.array([153,153,153])
    img[seg[:,:,6] == 1] = np.array([157,234,50])
    img[seg[:,:,7] == 1] = np.array([128,64,128])
    img[seg[:,:,8] == 1] = np.array([244,35,232])
    img[seg[:,:,9] == 1] = np.array([107,142,35])
    img[seg[:,:,10] == 1] = np.array([0,0,142])
    img[seg[:,:,11] == 1] = np.array([102,102,156])
    img[seg[:,:,12] == 1] = np.array([220,220,0])
    img[seg[:,:,13] == 1] = np.array([70,130,180])
    img[seg[:,:,14] == 1] = np.array([81,0,81])
    img[seg[:,:,15] == 1] = np.array([150,100,100])
    img[seg[:,:,16] == 1] = np.array([230,150,140])
    img[seg[:,:,17] == 1] = np.array([180,165,180])
    img[seg[:,:,18] == 1] = np.array([250,170,30])
    img[seg[:,:,19] == 1] = np.array([110,190,160])
    img[seg[:,:,20] == 1] = np.array([170,120,50])
    img[seg[:,:,21] == 1] = np.array([45,60,150])
    img[seg[:,:,22] == 1] = np.array([145,170,100])
    return img


def main(args):
    train_dataset = CARLASegmentationData(args.train_path)
    val_dataset = CARLASegmentationData(args.val_path)

    model = sm.Unet('resnet34', encoder_weights='imagenet', classes=23, encoder_freeze=True, activation='softmax')
    model.compile('Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    callbacks = [tf.keras.callbacks.ModelCheckpoint(args.save_path + '/model.h5', save_weights_only=True, save_best_only=True)]

    # training
    model.fit(
        x=train_dataset,
        batch_size=32,
        epochs=3,
        callbacks=callbacks,
        validation_data=val_dataset)

    # visualize output
    img, seg = val_dataset[0]
    pred = model.predict(img)[0]
    pred = tf.keras.utils.to_categorical(pred.argmax(axis=2), num_classes=23)

    seg = convert_to_color(seg[0])
    pred = convert_to_color(pred)

    fig, axs = plt.subplots(3)
    axs[0].imshow(img[0])
    axs[1].imshow(seg)
    axs[2].imshow(pred)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='sample_data/train/')
    parser.add_argument('--val_path', type=str, default='sample_data/val/')
    parser.add_argument('--save_path', type=str, default='sample_data/')
    args = parser.parse_args()
    main(args)