# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 13:32:05 2022

@author: Kert PC
"""

import csv
import os


rec_dir = './data/ears/annotations/recognition/'
cutout_dir = './data/ears/'

with open(rec_dir + 'ids_old.csv', 'r', newline='') as csvfile:
     reader = csv.DictReader(csvfile)
     
     
     with open(rec_dir + 'ids_p.csv', 'w', newline='') as csvfile_w:
         fieldnames = ['file', 'class']
         writer = csv.DictWriter(csvfile_w, fieldnames=fieldnames)
         
         writer.writeheader()
         
         for row in reader:
             file = row['file']
             file_set = file.split('/')[0] + '_p'
             file_id = file.split('/')[-1].split('.')[0]
             
             images = sorted(
                     [   os.path.join(file_set, fname)
                          for fname in os.listdir(cutout_dir + file_set)
                          if file_id in fname ])
                     
             for img in images :
                 writer.writerow({'file': img, 'class': row['class']})
            
