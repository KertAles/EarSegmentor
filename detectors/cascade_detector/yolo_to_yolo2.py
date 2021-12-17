# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 15:29:12 2021

@author: Kert PC
"""

import os

rootdir = 'C:/Users/Kert PC/Desktop/SB/YOLO/yolov3/data/ears/yoloTest/labels/'


for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + file

        if filepath.endswith(".txt"):
            f = open(filepath, "r")
            content = f.read()
            content = content.strip('\n').split('\n')
            # content = [list(map(any, line.split(' '))) for line in content if line != '']
            
            annot = []
            for line in content:
                if line != '' :
                    l_arr = line.split(" ")[0:5]
                    #l_arr_re = l_arr[1:5]
                    
                    l_arr_re = [int(i) for i in l_arr]
                    l_arr_re[0] = str(l_arr_re[0])
                    l_arr_re[1] = str((l_arr_re[1] + (l_arr_re[3] // 2)) / 480)
                    l_arr_re[2] = str((l_arr_re[2] + (l_arr_re[4] // 2)) / 360)
                    l_arr_re[3] = str(l_arr_re[3] / 480)
                    l_arr_re[4] = str(l_arr_re[4] / 360)
                    
                    annot.append(' '.join(l_arr_re))
                    
            f.close()
            
            fw = open(filepath, "w")
            fw.write('\n'.join(annot))
            fw.close()
            