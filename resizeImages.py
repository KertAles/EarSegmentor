# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 15:51:36 2022

@author: Kert PC
"""

from PIL import Image
import os

cutout_dir = './data/ears/train_c/'

image_list = os.listdir(cutout_dir)

newsize = (224, 224)

for image in image_list :
    im = Image.open(cutout_dir + image)
    width, height = im.size
     
    im = im.resize(newsize)
    
    im.save(cutout_dir + image)
                        
