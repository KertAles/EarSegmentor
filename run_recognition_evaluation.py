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
from tensorflow.keras.preprocessing.image import load_img
import csv
from feature_extractors.your_super_extractor.cnn_extractor import Classifier
from feature_extractors.hog_svm.hog_extractor import HOG_SVM

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config_recognition.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']
        
        self.lbp = lbp_ext.LBP()

    def clean_file_name(self, fname):
        return fname.split('/')[1].split(' ')[0]

    def get_annotations(self, annot_f):
        d = {}
        with open(annot_f) as f:
            lines = f.readlines()
            for line in lines:
                (key, val) = line.split(',')
                # keynum = int(self.clean_file_name(key))
                d[key] = int(val)
        return d
    
    def get_lbp_vectors(self, data_set='train', filename='ids.csv') :
        
        files = []
        with open(self.annotations_path + filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader :
                if data_set in row['file']:
                    files.append((row['file'], row['class']))
        
        vectors = []
        
        for file in files :
            lbp_vect = self.lbp.extract(np.array(load_img(self.images_path + file[0],
                                                 target_size=(128, 128, 3),
                                                 color_mode="rgb")))
            
            vectors.append(lbp_vect)
        
        return vectors
    
    def get_files(self, data_set='train', filename='ids.csv') :
        files = []
        with open(self.annotations_path + filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
                    
            for row in reader :
                if data_set in row['file']:
                    files.append((row['file'], row['class']))
                    
        return files
                        
    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        preprocess = Preprocess()
        eval = Evaluation()
        hog_ext = HOG_SVM()
        classifier_cnn = Classifier()
        
        pred_type = 'hog'
        filename = 'ids_p.csv'

        train_files = self.get_files('train', filename)
        test_files = self.get_files('test', filename)
        
        if pred_type == 'lbp' :
            train_vect = self.get_lbp_vectors('train', filename)
            test_vect = self.get_lbp_vectors('test', filename)
            
        elif pred_type == 'cnn1' : 
            classifier_cnn.load_model('model_65')
            classifier_cnn.pop_last_layer_2(3)
            
            train_vect = classifier_cnn.segment_list(train_files)
            test_vect = classifier_cnn.segment_list(test_files)
        elif pred_type == 'cnn2' :     
            classifier_cnn.load_model('model_61')
            classifier_cnn.pop_last_layer_2(11)
            
            train_vect = classifier_cnn.segment_list(train_files)
            test_vect = classifier_cnn.segment_list(test_files)
        elif pred_type == 'hog' :
            train_vect = hog_ext.extract_list(self.images_path, train_files)
            test_vect = hog_ext.extract_list(self.images_path, test_files)
        
        if pred_type != 'hog_svm' :
            distances_unfil = []
            
            for idx1, feat in enumerate(test_vect) :
                dist_arr = []
                        
                for idx, b_feat in enumerate(train_vect) :
                    dist_arr.append((np.linalg.norm(feat - b_feat), train_files[idx][1]))
                
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
        else :
            hog_ext.train_svm(self.images_path, train_files)
            
            test_vect = hog_ext.predict_list(self.images_path, test_files)
            
            distances = []
            
            for vect in test_vect :
                dist_arr = []
                
                for idx, prob in enumerate(vect) :
                    dist_arr.append((prob, str(idx+1)))
                
                dist_arr = sorted(dist_arr, key=lambda x: x[0], reverse=True)
                distances.append(dist_arr)
        
        gt = []
        for file in test_files :
            gt.append(file[1])
            
      
        ev = Evaluation()

        ev.compute_display_cmc(distances, gt)
            
        rank1 = ev.compute_rank1(distances, gt)
        rank5 = ev.compute_rankn(distances, gt, 5)
        rank10 = ev.compute_rankn(distances, gt, 10)
        
        print(rank1)
        print(rank5)
        print(rank10)
        

if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()