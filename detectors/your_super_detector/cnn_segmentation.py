# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:28:14 2021

@author: Kert PC
"""

import os

import numpy as np
import tensorflow as tf
from scipy import ndimage, signal


from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
import tensorflow_addons as tfa
from scipy import ndimage
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, DepthwiseConv2D, Dropout, UpSampling2D, concatenate
import pandas as pd
from focal_loss import SparseCategoricalFocalLoss
from skimage.color import rgb2hsv, hsv2rgb
import scipy

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Add
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense, LeakyReLU, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
import cv2

import PIL
from PIL import ImageOps, Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import math


class EarImages(keras.utils.Sequence):
    
    def image_histogram_equalization(self, channel, number_bins=256):
        image_histogram, bins = np.histogram(channel.flatten(), number_bins, density=True)
        cdf = image_histogram.cumsum() # cumulative distribution function
        cdf = 255 * cdf / cdf[-1] # normalize
    
        # use linear interpolation of cdf to find new pixel values
        channel_equalized = np.interp(channel.flatten(), bins[:-1], cdf)
    
        return channel_equalized.reshape(channel.shape), cdf
    
    def histogram_equlization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)

        return img
    
    def sobel_edge_calc(self, img) :
        img = np.expand_dims((img[:, :, 0] + img[:, :, 1] + img[:, :, 2]), axis=-1) / 3
        
        # Get x-gradient in "sx"
        sx = ndimage.sobel(img,axis=0,mode='constant')
        # Get y-gradient in "sy"
        sy = ndimage.sobel(img,axis=1,mode='constant')
        # Get square root of sum of squares
        sobel = np.hypot(sx, sy)
        
        roi = 3
        size = 2 * roi + 1
        sobel = ndimage.maximum_filter(sobel, size=size, mode='constant')
        
        sobel = (sobel / np.max(sobel)) * 255
        
        return sobel
    
    def edge_enhance(self, img) :
        filt = [[-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]]
        
        #img = signal.convolve2d(img, filt, mode='same', boundary='wrap')
        
        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
        intensity_img[:,:,2] = signal.convolve2d(intensity_img[:,:,2], filt, mode='same', boundary='wrap')
        img = cv2.cvtColor(intensity_img, cv2.COLOR_HSV2BGR )
        
        
        return img


    def __init__(self, images, masks, batch_size=8, img_size=(512,608)):
        self.images = images
        self.masks = masks
        
        self.batch_size = batch_size
        self.img_size = img_size
        

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        
        if i >= len(self.images) :
            i = i % len(self.images)
            print('Taking pictures from the beginning')
            
        
        batch_images = self.images[i : i + self.batch_size]
        batch_masks = self.masks[i : i + self.batch_size]
        
        return self.getData(batch_images, batch_masks)
    
    def __len__(self):
        ret = len(self.images) // self.batch_size
        
        return ret
    
    def getData(self, batch_images, batch_masks, aug_round=0) :
        
        x = []
        y = []
        
        for j, path in enumerate(batch_images):
            img = np.array(load_img(path, color_mode="rgb"))
            
            """
            img_eq = np.zeros(img.shape)
            for i in range(img.shape[2]):
                channel = img[:, :, i]
                img_eq[:, :, i] = self.image_histogram_equalization(channel)[0]
            img = img_eq
            """
            #img = self.edge_enhance(img)
            
            img = self.histogram_equlization_rgb(img)
            
            #img = np.append(img, self.sobel_edge_calc(img), axis=-1)
            
            img = img - np.min(img)
            img = ((img / np.max(img)) * 2) - 1

            """
            
            img = img - 128
            img = img / 128
            """

            #print(np.shape(img))
             
            x.append(img)
            

        for j, path in enumerate(batch_masks):
            img = load_img(path, color_mode="grayscale")
            
            img = np.expand_dims(img, axis=-1) / 255
            
            y.append(img)
        return np.array(x), np.array(y)
    


class Segmentor:  
    
    
    def __init__(self) :
        self.input_dir = './data/ears/'
        self.target_dir = './data/ears/annotations/segmentation/'
        self.model_dir = './models/regular/'


    def segment(self, img) :
        result = self.model.predict(img)
        return result


    def segment_list(self, img_list) :
        val_images = sorted(

            [   os.path.join(self.input_dir, fname.split('/')[-2], fname.split('/')[-1])
                for fname in img_list ] )
        
        val_masks = sorted(
            [   os.path.join(self.target_dir, fname.split('/')[-2], fname.split('/')[-1])
                for fname in img_list] )
        
        keras.backend.clear_session()
    
        val_gen = EarImages(val_images, val_masks, batch_size=1)
        
        return self.model.predict(val_gen)

    
    def train_model(self, num_classes = 2, batch_size = 8, num_epochs = 50, block_number = 2, filter_number = 16) :

        train_images = sorted(
            [   os.path.join(self.input_dir + 'train/', fname)
                for fname in os.listdir(self.input_dir + 'train/')
                if fname.endswith(".png") ] )

        train_masks = sorted(
            [   os.path.join(self.target_dir + 'train/', fname)
                for fname in os.listdir(self.target_dir + 'train/')
                if fname.endswith(".png") ] )
        
        val_masks = train_masks[-80:]
        val_images = train_images[-80:]
        
        train_images = train_images[:-80]
        train_masks = train_masks[:-80]
        
        
        keras.backend.clear_session()
        
        train_gen = EarImages(train_images, train_masks, batch_size=batch_size)
        val_gen = EarImages(val_images, val_masks, batch_size=batch_size)
        
        

        #inputs, outputs, self.model = self.unet_model_blocks(block_number=block_number, filter_number=filter_number)
    
        self.model = self.get_model()
              
        self.model.compile(optimizer="adam", loss=SparseCategoricalFocalLoss(gamma=2.0), metrics=["sparse_categorical_accuracy"])

            
        self.model.summary()

        epochs = num_epochs
    
        callbacks = [
            keras.callbacks.ModelCheckpoint("ear_segmentation", save_best_only=True)
        ]
            
        history = self.model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

        
        model_names = [ mod_name for mod_name in os.listdir(self.model_dir) ]
        
        model_type_num = str(len(model_names) + 1)
        model_path = self.model_dir + 'model' + '_' + model_type_num
        
        self.model.save(model_path)
        
        hist_df = pd.DataFrame(history.history) 
        
        hist_json_file = self.model_dir + 'history_' + model_type_num + '.json' 
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)
    
    
    def load_model(self, model_name) :
        self.model = keras.models.load_model(self.model_dir + model_name)
        
    def get_model(self):
         #model = get_efficient_unet_b0((None, None, 3),
         #    pretrained=True, block_type='transpose', concat_input=True, out_channels=2)
         #model = EfficientNetB0(input_shape=(None, None, 3), classes=2, include_top=False, pooling='max')
         model = MobileNetV2(input_shape=(None, None, 3), alpha=0.5, weights=None, include_top=False, pooling='max', classes=2)

         return model
    """
    def unet_model_blocks(self, inputs=None, num_classes=2, block_number=3, filter_number=16):
        if inputs is None:
            num_of_channels = 3
            
            inputs = layers.Input((None, None) + (num_of_channels, ))
            
            
        filter_num = filter_number
        x = inputs
        block_features = []
        for i in range(block_number):
            fn_cur = filter_num*(2**(i))
            conv1 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(x)
            conv1 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(conv1)
            conv1 = Dropout(0.15)(conv1)
            block_features.append(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            x = pool1

        fn_cur = filter_num*(2**(block_number))
        conv3 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(x)
        conv3 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(conv3)
        conv3 = BatchNormalization()(conv3)
        drop3 = Dropout(0.15)(conv3)
        x = drop3
        for i in range(block_number):
            fn_cur = filter_num*(2**(block_number - i - 1))
            up8 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(
                UpSampling2D(size=(2, 2))(x))
            merge8 = concatenate([block_features.pop(), up8], axis=3)
            
            conv8 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(merge8)
            conv8 = Conv2D(fn_cur, (3, 3), activation='relu', padding='same')(conv8)
            conv8 = Dropout(0.15)(conv8)
            x = conv8

        conv10 = Conv2D(num_classes, (3,3), activation='softmax', padding="same")(x)
        
        model = keras.Model(inputs, conv10)

        return inputs, conv10, model
    """
    
    def unet_model_blocks(self, inputs=None, num_classes=2, block_number=3, filter_number=16):
        if inputs is None:
            num_of_channels = 3
            
            inputs = layers.Input((None, None) + (num_of_channels, ))
            
            
        filter_num = filter_number
        x = inputs
        block_features = []
        for i in range(block_number):
            fn_cur = filter_num*(2**(i))
            conv1 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(x)
            conv1 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(conv1)
            conv1 = self.inverted_residual_block(conv1, fn_cur*2, fn_cur)
            block_features.append(conv1)
            conv1 = Dropout(0.15)(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            x = pool1

        fn_cur = filter_num*(2**(block_number))
        conv3 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(x)
        conv3 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(conv3)
        conv3 = BatchNormalization()(conv3)
        drop3 = Dropout(0.15)(conv3)
        x = drop3
        for i in range(block_number):
            fn_cur = filter_num*(2**(block_number - i - 1))
            up8 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(
                UpSampling2D(size=(2, 2))(x))
            merge8 = concatenate([block_features.pop(), up8], axis=3)
            
            conv8 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(merge8)
            conv8 = Conv2D(fn_cur, (3, 3), activation='relu', padding='same')(conv8)
            conv8 = self.inverted_residual_block(conv8, fn_cur*2, fn_cur)
            conv8 = Dropout(0.15)(conv8)
            x = conv8

        conv10 = Conv2D(num_classes, (3,3), activation='softmax', padding="same")(x)
        
        model = keras.Model(inputs, conv10)

        return inputs, conv10, model
    
    
    def inverted_residual_block(self, x, expand=64, squeeze=16):
          m = Conv2D(expand, (1,1), activation='relu', padding="same")(x)
          m = DepthwiseConv2D((3,3), activation='relu', padding="same")(m)
          m = Conv2D(squeeze, (1,1), activation='relu', padding="same")(m)
          return Add()([m, x])
    
    
if __name__ == '__main__':
    segmentor = Segmentor()
    segmentor.train_model(batch_size=4, num_epochs=40, block_number=3, filter_number=12)

    
    
    print('blah')