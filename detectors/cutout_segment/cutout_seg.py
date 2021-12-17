# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 15:16:32 2021

@author: Kert PC
"""

import os

from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy import ndimage, signal


from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
import tensorflow_addons as tfa
from scipy import ndimage
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
import pandas as pd
from focal_loss import SparseCategoricalFocalLoss
from skimage.color import rgb2hsv, hsv2rgb
import scipy

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
    
    def edge_enhance(self, img) :
        filt = [[-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]]
        
        #img = signal.convolve2d(img, filt, mode='same', boundary='wrap')
        
        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
        intensity_img[:,:,2] = signal.convolve2d(intensity_img[:,:,2], filt, mode='same', boundary='wrap')
        img = cv2.cvtColor(intensity_img, cv2.COLOR_HSV2BGR )
        
        
        return img
    
    def get_annotations(self, annot_name):
            with open(annot_name) as f:
                lines = f.readlines()
                annot = []
                for line in lines:
                    l_arr = line.split(" ")[1:5]
                    l_arr = [int(i) for i in l_arr]
                    annot.append(l_arr)
            return annot
    
    
    def convert2mask(self, mt, shape):
        # Converts coordinates of bounding-boxes into blank matrix with values set where bounding-boxes are.

        t = np.zeros([shape, shape])
        for m in mt:
            x, y, w, h = m
            cv2.rectangle(t, (x,y), (x+w, y+h), 1, -1)
        return t

    def prepare_for_detection(self, prediction, ground_truth):
            # For the detection task, convert Bounding-boxes to masked matrices (0 for background, 1 for the target). If you run segmentation, do not run this function

            if len(prediction) == 0:
                return [], []

            # Large enough size for base mask matrices:
            shape = 2*max(np.max(prediction), np.max(ground_truth)) 
            
            p = self.convert2mask(prediction, shape)
            gt = self.convert2mask(ground_truth, shape)

            return p, gt

    def cutout_img(self, img, zone) :
        x, y, w, h = zone
        
        x = x + w // 2
        y = y + h // 2
        
        fixed_size = 64
        start_y = abs(min(y-fixed_size, 0))
        start_x = abs(min(x-fixed_size, 0))

        cutout = np.array(img[y-fixed_size+start_y:y+fixed_size, x-fixed_size+start_x:x+fixed_size])
        shape = np.shape(cutout)

                
        ret_img = np.zeros((fixed_size * 2, fixed_size * 2, shape[2]))
        ret_img[start_y:shape[0]+start_y, start_x:shape[1]+start_x] = cutout
        
        """
        if np.max(ret_img) == 0 :
            print((start_y, start_x))
            print(np.shape(cutout))
            print(shape[0]+start_y - start_y)
            print(shape[1]+start_x - start_x)
            
            if shape[2] > 1 :
                plt.imshow(ret_img)
            else :
                plt.imshow(ret_img[:, :, 0], cmap='gray', vmin=0, vmax=255)
            plt.show()
        """
    
        return ret_img


    def __init__(self, images, masks, batch_size=8, img_size=(512,608), yolo_dir='', yolo_list=[]):
        self.images = images
        self.masks = masks
        
        self.batch_size = batch_size
        self.img_size = img_size
        
        self.yolo_dir = yolo_dir
        self.yolo_list = yolo_list
        

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
            img = self.histogram_equlization_rgb(img)
        
            if self.yolo_dir != '' :
                annot_name = os.path.join(self.yolo_dir, Path(os.path.basename(path)).stem) + '.txt'
                annot_list = self.get_annotations(annot_name)
                #annot_list.append((0, 0, 128, 128))
                #annot_list.append((128 , 0, 128, 128))
            else :
                annot_list = self.yolo_list
                
                
            for annot in annot_list :
                
                subimg = self.cutout_img(img, annot)
                
                minimg = np.min(subimg)
                maximg = np.max(subimg)
                
                if minimg < maximg:
                    subimg = subimg - minimg
                    subimg = ((subimg / np.max(subimg)) * 2) - 1
                
                x.append(subimg)
            

        for j, path in enumerate(batch_masks):
            
            img = np.expand_dims(load_img(path, color_mode="grayscale"), axis=-1) 
            
            if self.yolo_dir != '' :
                annot_name = os.path.join(self.yolo_dir, Path(os.path.basename(path)).stem) + '.txt'
                annot_list = self.get_annotations(annot_name)
                #annot_list.append((0, 0, 128, 128))
                #annot_list.append((128 , 0, 128, 128))
            else :
                annot_list = self.yolo_list
            
            for annot in annot_list :      
                subimg = self.cutout_img(img, annot)
                subimg = subimg / 255
                
                y.append(subimg)
                
            
        x_arr = np.array(x, dtype=np.float)
        y_arr = np.array(y, dtype=np.float)
          
        return x_arr, y_arr
    


class Segmentor:  
    
    
    def __init__(self) :
        self.input_dir = 'C:/Users/Kert PC/Desktop/SB/data/ears/'
        self.target_dir = 'C:/Users/Kert PC/Desktop/SB/data/ears/annotations/segmentation/'
        self.model_dir = 'C:/Users/Kert PC/Desktop/SB/models/cutout/'
        self.yolo_dir = 'C:/Users/Kert PC/Desktop/SB/data/ears/annotations/detection/train_YOLO_format'
        self.yolo_dir_test = 'C:/Users/Kert PC/Desktop/SB/data/ears/annotations/detection/test_YOLO_format'

    def segment(self, img) :
        result = self.model.predict(img)
        return result


    def segment_list(self, img_list) :
        val_images = sorted(
            [   os.path.join(self.input_dir, fname)
                for fname in img_list ] )
        
        val_masks = sorted(
            [   os.path.join(self.target_dir, fname)
                for fname in img_list ] )
        
        keras.backend.clear_session()
    
        val_gen = EarImages(val_images, val_masks, batch_size=1, yolo_dir=self.yolo_dir_test)
        
        return self.model.predict(val_gen)


    def segment_list_w_bb(self, img_list, yolo_list) :
        val_images = sorted(
            [   os.path.join(self.input_dir, fname)
                for fname in img_list ] )
        
        val_masks = sorted(
            [   os.path.join(self.target_dir, fname)
                for fname in img_list ] )
        
        keras.backend.clear_session()
    
        val_gen = EarImages(val_images, val_masks, batch_size=1, yolo_list=yolo_list)
        
        return self.model.predict(val_gen)

    
    def train_model(self, num_classes = 2, batch_size = 8, num_epochs = 50, block_number = 2, filter_number = 16) :

        train_images = sorted(
            [   os.path.join(self.input_dir + 'train/', fname)
                for fname in os.listdir(self.input_dir + 'train/')
                if fname.endswith(".png") ] )
        """
        val_images = sorted(
            [   os.path.join(self.input_dir + 'test/', fname)
                for fname in os.listdir(self.input_dir + 'test/')
                if fname.endswith(".png") ] )
        """
        train_masks = sorted(
            [   os.path.join(self.target_dir + 'train/', fname)
                for fname in os.listdir(self.target_dir + 'train/')
                if fname.endswith(".png") ] )
        """
        val_masks = sorted(
            [   os.path.join(self.target_dir + 'test/', fname)
                for fname in os.listdir(self.target_dir + 'test/')
                if fname.endswith(".png") ] )
        """
    
        
        val_masks = train_masks[-80:]
        val_images = train_images[-80:]
        
        train_images = train_images[:-80]
        train_masks = train_masks[:-80]
        
        
        keras.backend.clear_session()
        
        train_gen = EarImages(train_images, train_masks, batch_size=batch_size, yolo_dir=self.yolo_dir)
        val_gen = EarImages(val_images, val_masks, batch_size=batch_size, yolo_dir=self.yolo_dir)
        
        
        #train_gen = EarImages(train_images, train_masks, batch_size=batch_size, yolo_list=yolo_list)
        #val_gen = EarImages(val_images, val_masks, batch_size=batch_size, yolo_list=yolo_list)
        
        inputs, outputs, self.model = self.unet_model_blocks(block_number=block_number, filter_number=filter_number)
        
        self.model.compile(optimizer="adam", loss=SparseCategoricalFocalLoss(gamma=2), metrics=["sparse_categorical_accuracy"])
            
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


    def unet_model_blocks(self, inputs=None, num_classes=2, block_number=3, filter_number=16):
        if inputs is None:
            num_of_channels = 3
            
            inputs = layers.Input((None, None) + (num_of_channels, ))
            
        drop_rate = 0.1
        filter_num = filter_number
        x = inputs
        block_features = []
        for i in range(block_number):
            fn_cur = filter_num*(2**(i))
            conv1 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(x)
            conv1 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(conv1)
            conv1 = Dropout(drop_rate)(conv1)
            block_features.append(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            x = pool1

        fn_cur = filter_num*(2**(block_number))
        conv3 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(x)
        conv3 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(conv3)
        conv3 = BatchNormalization()(conv3)
        drop3 = Dropout(drop_rate)(conv3)
        x = drop3
        for i in range(block_number):
            fn_cur = filter_num*(2**(block_number - i - 1))
            up8 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(
                UpSampling2D(size=(2, 2))(x))
            merge8 = concatenate([block_features.pop(), up8], axis=3)
            
            conv8 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(merge8)
            conv8 = Conv2D(fn_cur, (3, 3), activation='relu', padding='same')(conv8)
            conv8 = Dropout(drop_rate)(conv8)
            x = conv8

        conv10 = Conv2D(num_classes, (3,3), activation='softmax', padding="same")(x)
        
        model = keras.Model(inputs, conv10)

        return inputs, conv10, model
    
    
    

if __name__ == '__main__':
    segmentor = Segmentor()
    segmentor.train_model(batch_size=16, num_epochs=50, block_number=4, filter_number=16)
    
    
    print('blah')