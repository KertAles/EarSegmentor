# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:14:45 2022

@author: Kert PC
"""

import cv2, sys
from skimage import feature
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import os
import matplotlib.pyplot as plt
from sklearn import svm

class HOG_SVM:
    def __init__(self, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2,2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.svm_model = None
        
    def extract(self, img):
        resized_img = np.array(load_img(img, target_size=(128, 128, 3)))

        fd = hog(resized_img,
                            orientations=self.orientations,
                            pixels_per_cell=self.pixels_per_cell,
                            cells_per_block=self.cells_per_block,
                            multichannel=True)
        
        fd = (fd - np.mean(fd)) / (np.std(fd) + 1e-7)
        
        return fd
    
    
    def extract_list(self, path, file_list) :
        features = []
        
        for file in file_list :
            features.append(self.extract(path + file[0]))
            
        return features
    
    
    def train_svm(self, path, file_list) :
        features_train = self.extract_list(path, file_list)
        
        gt = []
        for file in file_list :
            gt.append(file[1])
        
        self.svm_model = svm.SVC(decision_function_shape='ovr', probability=True)
        
        self.svm_model.fit(features_train, gt)
        
    def predict_list(self, path, file_list) :
        predictions = []
        
        if self.svm_model != None :
            features_pred = self.extract_list(path, file_list)
            
            predictions = self.svm_model.predict_proba(features_pred)
            
        return predictions
        

if __name__ == '__main__':
    img = 'C:/Users/Kert PC/Desktop/par.jpeg'
    extractor = HOG_SVM()
    features = extractor.extract(img)