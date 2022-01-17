# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 18:00:26 2022

@author: Kert PC
"""

import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist 
from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation
import feature_extractors.pix2pix.extractor as p2p_ext
import feature_extractors.lbp.extractor as lbp_ext
import csv
from tensorflow.keras.models import Model

from feature_extractors.your_super_extractor.cnn_extractor import Classifier
from run_recognition_evaluation import EvaluateAll
from tensorflow.keras import backend as K


K.clear_session()
extractor = Classifier()

extractor.load_model('model_65')
#extractor.model.summary()

#extractor.pop_last_layer(1)
extractor.pop_last_layer_2(3)

extractor.model.summary()

annotations_path = 'F:/Faks/SB/App/data/ears/annotations/recognition/'
data_set = 'test'
files = []
num_classes = 100


with open(annotations_path + 'ids.csv', 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
            
    for row in reader :
        if data_set in row['file']:
            if int(row['class']) <= num_classes :
                files.append((row['file'], row['class']))
            #elif int(row['class']) < 25 :
            #    files.append((row['file'], str(21)))
        
files_base = []
with open(annotations_path + 'ids.csv', 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
            
    for row in reader :
        if 'train' in row['file'] :
            if int(row['class']) <= num_classes:
                files_base.append((row['file'], row['class']))
            #else :
            #    files_base.append((row['file'], str(21)))


base_features = extractor.segment_list(files_base)

cmp_features = extractor.segment_list(files)

distances_unfil = []

for idx1, feat in enumerate(cmp_features) :
    dist_arr = []
            
    for idx, b_feat in enumerate(base_features) :
        dist_arr.append((np.linalg.norm(feat - b_feat), files_base[idx][1]))
    
    dist_arr = sorted(dist_arr, key=lambda x: x[0])
    distances_unfil.append(dist_arr)

distances = []

for dist_vect in distances_unfil :
    nu_dist = []
    added_classes = []
    
    for dist in dist_vect :
        if not dist[1] in added_classes :
            nu_dist.append(dist)
            added_classes.append(dist[1])
            
    distances.append(nu_dist)

gt = []
for file in files :
    gt.append(file[1])
    
ev = Evaluation()

ev.compute_display_cmc(distances, gt)
    
rank1 = ev.compute_rank1(distances, gt)
rank5 = ev.compute_rankn(distances, gt, 5)
rank10 = ev.compute_rankn(distances, gt, 10)
rank20 = ev.compute_rankn(distances, gt, 20)
rank100 = ev.compute_rankn(distances, gt, 100)
