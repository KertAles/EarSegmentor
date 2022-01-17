# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 15:44:01 2022

@author: Kert PC
"""

import os

import numpy as np
import tensorflow as tf
from scipy import ndimage, signal
import csv


from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
import tensorflow_addons as tfa
from scipy import ndimage
from tensorflow.keras.layers import RandomContrast, RandomFlip, RandomRotation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, DepthwiseConv2D, Dropout, UpSampling2D, concatenate
import pandas as pd
from focal_loss import SparseCategoricalFocalLoss
from skimage.color import rgb2hsv, hsv2rgb
import scipy
from scipy.ndimage import rotate

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Add
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense, LeakyReLU, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
import cv2

import PIL
from PIL import ImageOps, Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import math

import random



class EarImages(keras.utils.Sequence):
    
    def histogram_equlization_rgb(self, img):

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        return img
    

    def __init__(self, file_list, input_dir, batch_size=8, img_size=(128, 128), aug_rounds=4, do_aug = True):
        self.file_list = file_list
        self.input_dir = input_dir
        
        self.do_aug = do_aug
        self.aug_rounds = aug_rounds

        self.batch_size = batch_size
        self.img_size = img_size
        

    def __getitem__(self, idx):
        i = idx * self.batch_size
        
        #i = i % len(self.file_list)
        
        #aug_iter = i // len(self.file_list)
        if i >= len(self.file_list) :
            i = i % len(self.file_list)
            #print('Taking pictures from the beginning')
            
        
        batch_files = self.file_list[i : i + self.batch_size]
        
        return self.getData(batch_files, 5)
    
    def __len__(self):
        ret = len(self.file_list) // self.batch_size
        
        ret = ret * self.aug_rounds
        
        return ret
    
    
    def getData(self, batch_files, aug_round=0) :
        
        x = []
        y = []
        
        for j, file_info in enumerate(batch_files):
            img = np.array(load_img(self.input_dir + file_info[0], target_size=(128, 128,3), color_mode="rgb"))
        
            img = self.histogram_equlization_rgb(img)
            
            if self.do_aug :
                flip = random.random()
                
                if flip < 0.5 :
                    img = np.fliplr(img)
                    
                rot_angle = random.random() * 30 - 15
                img = rotate(img, rot_angle, reshape=False)
            
            
            img = img - np.min(img)
            img = img / np.max(img)
            #img = img / 255
            
            x.append(img)
            y.append(int(file_info[1])-1)
        return np.array(x), np.array(y)
    


class Classifier:  
    
    def __init__(self, input_dir='F:/Faks/SB/App/data/ears/',
                       rec_dir = 'F:/Faks/SB/App/data/ears/annotations/recognition/',
                       model_dir = 'F:/Faks/SB/App/models/cnn_extract/') :
        self.input_dir = input_dir
        self.rec_dir = rec_dir
        self.model_dir = model_dir


    def segment(self, img) :
        result = self.model.predict(img)
        return result


    def segment_list(self, file_list) :
       
        keras.backend.clear_session()
    
        val_gen = EarImages(file_list, self.input_dir, batch_size=1, aug_rounds=1, do_aug=False)
        
        return self.model.predict(val_gen)

    
    def train_model(self, num_classes = 10, batch_size = 8, num_epochs = 50, block_number = 4, filter_number = 16) :
        files = []
        classes = []
        with open(self.rec_dir + 'ids.csv', 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader :
                if 'train' in row['file']:
                    #if int(row['class']) <= num_classes :
                    files.append((row['file'], row['class']))
                    
                    #else :
                    #    files.append((row['file'], str(num_classes)))


        random.seed(1337)
        random.shuffle(files)
        
        val_split = round(len(files) * 0.8)
            
        val_files = files[val_split:]
        train_files = files[:val_split]
        #train_files = files
        
        keras.backend.clear_session()
        
        train_gen = EarImages(train_files, self.input_dir, batch_size=batch_size)
        val_gen = EarImages(val_files, self.input_dir, batch_size=batch_size)
        
        
        inputs, outputs, self.model = self.create_model_aleshnet(num_classes=num_classes)
        
        """
        model_pre = tf.keras.applications.ResNet101(
                        include_top=False,
                        weights="imagenet",
                        #pooling='avg',
                        input_shape=(128, 128, 3)
                    )
        model_pre.trainable = False
        flat = Flatten()
        dropout1 = Dropout(0.3)
        dense1 = Dense(512, activation="relu")
        dropout2 = Dropout(0.3)
        dense2 = Dense(256, activation="relu")
        dropout3 = Dropout(0.3)
        dense3 = Dense(128, activation="relu")
        dropout4 = Dropout(0.3)
        output = Dense(num_classes, activation="softmax")

        self.model = tf.keras.Sequential([
                model_pre,
                flat,
                dropout1,
                output
            ])
        #self.model = Model(model_pre.layers[0].output, lay)
        """

        self.model.compile(optimizer='adam',
                           #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           loss=SparseCategoricalFocalLoss(gamma=3.0),
                           metrics=["sparse_categorical_accuracy"])
        
        self.model.summary()

        epochs = num_epochs
        """
        callbacks = [
            keras.callbacks.ModelCheckpoint("ear_segmentation", save_best_only=True)
        ]
        """
            
        history = self.model.fit(train_gen, validation_data=val_gen, epochs=epochs)
        
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                           #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           loss=SparseCategoricalFocalLoss(gamma=4.5),
                           metrics=["sparse_categorical_accuracy"])
        
        history = self.model.fit(train_gen, validation_data=val_gen, epochs=30)

        
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
        
    def pop_last_layer(self, n=1) :
        x = self.model.layers[(-3*n - 1)].output 
        
        model = Model(inputs = self.model.layers[0].output, outputs = x)
        
        model.compile(optimizer='adam',
                           loss=SparseCategoricalFocalLoss(gamma=2.0),
                           metrics=["sparse_categorical_accuracy"])
        self.model = model
        
    def pop_last_layer_2(self, n=1) :
        x = self.model.layers[(-1*n)].output 
        
        model = Model(inputs = self.model.inputs, outputs = x)
        
        model.compile(optimizer='adam',
                           loss=SparseCategoricalFocalLoss(gamma=2.0),
                           metrics=["sparse_categorical_accuracy"])
        self.model = model
    
    def create_model_aleshnet(self, inputs=None, num_classes=100):
        if inputs is None:
            num_of_channels = 3
            
            inputs = layers.Input((128, 128) + (num_of_channels, ))
            
            
        x = inputs
    
        conv1 = Conv2D(192, (11, 11), strides=(2, 2), activation='relu', padding="same")(x)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  
        conv2 = Conv2D(224, (5, 5), activation='relu', padding="same")(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(256, (3, 3), activation='relu', padding="same")(pool2)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding="same")(conv3)
        conv3 = Conv2D(224, (3, 3), activation='relu', padding="same")(conv3)
        conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        flat = Flatten()(conv3)
       
        dense1 = Dense(1024, activation='relu')(flat)
        dense1 = Dropout(0.2)(dense1)
        dense1 = BatchNormalization()(dense1)
    
        dense2 = Dense(512, activation='relu')(dense1)
        dense2 = Dropout(0.2)(dense2)
        dense2 = BatchNormalization()(dense2)
        
        dense3 = Dense(256, activation='relu')(dense2)
        dense3 = Dropout(0.2)(dense3)
        dense3 = BatchNormalization()(dense3)
  
        dense5 = Dense(num_classes, activation='softmax')(dense3)
        
        model = keras.Model(inputs, dense5)

        return inputs, dense5, model
    
    def create_model(self, inputs=None, num_classes=100, block_number=4, filter_number=4):
        if inputs is None:
            num_of_channels = 3
            
            inputs = layers.Input((128, 128) + (num_of_channels, ))
            
            
        filter_num = filter_number
        x = inputs
        for i in range(block_number):
            fn_cur = filter_num*(i + 1)
            conv1 = Conv2D(fn_cur, (3, 3), activation='relu', padding="same")(x)
            conv1 = Conv2D(fn_cur, (3, 3), activation='relu', padding="same")(conv1)
            conv1 = Conv2D(fn_cur, (3, 3), activation='relu', padding="same")(conv1)
            conv1 = Dropout(0.05)(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            x = pool1

        fn_cur = filter_num*(block_number+1)
        conv3 = Conv2D(fn_cur, (3, 3), activation='relu', padding="same")(x)
        conv3 = Conv2D(fn_cur, (3, 3), activation='relu', padding="same")(conv3)
        conv3 = Conv2D(fn_cur, (3, 3), activation='relu', padding="same")(conv3)
        #conv3 = BatchNormalization()(conv3)

        flat = Flatten()(conv3)
        
        dense1 = Dense(600, activation='relu')(flat)
        dense1 = Dropout(0.1)(dense1)
    
        dense2 = Dense(200, activation='relu')(dense1)
        dense2 = Dropout(0.1)(dense2)
        
        dense3 = Dense(150, activation='relu')(dense2)
        dense3 = Dropout(0.1)(dense3)

        dense4 = Dense(100, activation='softmax')(dense3)
        
        model = keras.Model(inputs, dense4)

        return inputs, dense4, model
    

if __name__ == '__main__':
    segmentor = Classifier()
    segmentor.train_model(batch_size=4, num_epochs=100, block_number=4, filter_number=64, num_classes=100)
    