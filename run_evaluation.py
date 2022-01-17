import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation

from tensorflow.keras.preprocessing.image import load_img, array_to_img
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage.measurements import label, center_of_mass
from skimage import morphology
from skimage.measure import regionprops
from mpl_toolkits.axes_grid1 import ImageGrid

import PIL
from PIL import ImageOps, Image

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']
        self.zones_path = config['zones_path']

    def get_annotations(self, annot_name):
            with open(annot_name) as f:
                lines = f.readlines()
                annot = []
                for line in lines:
                    l_arr = line.split(" ")[1:5]
                    l_arr = [int(i) for i in l_arr]
                    annot.append(l_arr)
            return annot
        
    def get_mask(self, prediction):
            mask = np.argmax(prediction, axis=-1)
            mask = np.expand_dims(mask, axis=-1) 
            img = np.expand_dims(np.array(PIL.ImageOps.autocontrast(array_to_img(mask))), axis=-1) / 255
            #plt.imshow(PIL.ImageOps.autocontrast(array_to_img(mask)))
            #plt.show()

            
            return mask
        
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
    
        return ret_img


    def run_evaluation(self):  

        #im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        im_list = sorted(
            [   os.path.join(self.images_path, fname)
                for fname in os.listdir(self.images_path) ] )
        iou_arr = []
        pr_arr = []
        re_arr = []
        f1_arr = []
        acc_arr = []
        preprocess = Preprocess()
        eval = Evaluation()
      
        self.cutout_img_dir = './data/ears/train_c/'
        

        # Change the following detector and/or add your detectors below
        import detectors.your_super_detector.cnn_segmentation as cnn_segmentation
        import detectors.cutout_segment.cutout_seg as cutout_seg
        # import detectors.your_super_detector.detector as super_detector
        
        
        segmentor_cnn = cnn_segmentation.Segmentor()
        segmentor = cutout_seg.Segmentor()

        segmentor_cnn.load_model('model_38')
        segmentor.load_model('model_47')
        
        draw = False
                
        #whole cnn + centroid pred
        #predictions = segmentor_cnn.segment_list(im_list)
        
        
        for idx, im_name in enumerate(im_list) :
            #print(im_name)
            predictions = segmentor_cnn.segment_list([im_name])

            annot_file = im_name.split('/')[-1]
            annot_name = os.path.join(self.annotations_path, annot_file)
            annot = load_img(annot_name, color_mode="grayscale")
            annot = np.expand_dims(annot, axis=-1) / 255
            
            img_file = im_name.split('/')[-1]
            img_name = os.path.join(self.images_path, img_file)
            org_img = np.array(load_img(img_name))
        

            p = np.zeros((360, 480, 1))
            gt = annot

            kernel = np.ones((5, 5, 1), 'uint8')
            eroded_mask = morphology.dilation(self.get_mask(predictions[0]), selem=kernel)
            
            if np.max(eroded_mask) > 0 :
                labels = label(eroded_mask)
                centers = center_of_mass(eroded_mask, labels=labels[0], index=list(range(1,labels[1]+1)))
 
                yolo_list = []
                
                for center in centers :
                    yolo_list.append((round(center[1]) , round(center[0]), 0, 0))                    
                
                #with open('C:/Users/Kert PC/Desktop/SB/data/ears/annotations/cutout/train/' + annot_file.split('.')[0] + '.txt', 'w') as f:
                #    for yolo in yolo_list :
                #        f.write('0 ' + str(yolo[0]) + ' ' + str(yolo[1]) + ' ' + str(yolo[2])  + ' ' + str(yolo[3]) + '\n')
                
                        
                
                cutout_predictions = segmentor.segment_list_w_bb([im_name], yolo_list)
                
                for j, pred in enumerate(cutout_predictions):
                    x, y, w, h = yolo_list[j]
                    
                    x = x + w // 2
                    y = y + h // 2

                    fixed_size = 64
                    offset_y = abs(min(y-fixed_size, 0))
                    offset_x = abs(min(x-fixed_size, 0))
                    
                    start_y = y-fixed_size+offset_y
                    start_x = x-fixed_size+offset_x
                    
                    shape = np.shape(p[start_y:y+fixed_size, start_x:x+fixed_size])
                    
                    c_img = org_img[start_y:start_y+shape[0], start_x:start_x+shape[1]]
                    mask = self.get_mask(pred)[offset_y:offset_y + shape[0],offset_x:offset_x + shape[1], 0]

                    result = c_img.copy()
                    result[mask==0] = (0,0,0)
                    #plt.imshow(result);
                    
                    results_idxs = np.nonzero(result[:, :, 0])
                    if len(results_idxs[0] > 0) :
                        i_leftmost = np.min(results_idxs[0])
                        i_rightmost = np.max(results_idxs[0])
                        
                        i_topmost = np.min(results_idxs[1])
                        i_bottommost = np.max(results_idxs[1])
                        
                        nu_img = Image.fromarray(result[i_leftmost:i_rightmost, i_topmost:i_bottommost, :], 'RGB')
                        
                        #plt.imshow();
                        
                        nu_img = nu_img.save(os.path.join(self.cutout_img_dir, im_name.split('/')[-1].split('.')[0] + '_' + str(j+1) + '.png'))
                        
                        
                        
                    p[start_y:start_y+shape[0],
                      start_x:start_x+shape[1]] = np.logical_or(
                                                      p[start_y:start_y+shape[0],
                                                      start_x:start_x+shape[1]],
                                                      self.get_mask(pred)[offset_y:offset_y + shape[0],
                                                                          offset_x:offset_x + shape[1]]) 
                          
                          
                
        
            iou = eval.iou_compute(p, gt)
            pr, re, f1, acc = eval.prref1_compute(p, gt)
            iou_arr.append(iou)
            pr_arr.append(pr)
            re_arr.append(re)
            f1_arr.append(f1)
            acc_arr.append(acc)

            
            if draw :
                fig = plt.figure(figsize=(9, 9))
                grid = ImageGrid(fig, 111, 
                             nrows_ncols=(1, 4),
                             axes_pad=0.1, 
                             )
                
                for ax, j in zip(grid, range(0, 4)) :

                    if j == 0:
                        ax.imshow(cv2.imread(im_name))
                    elif j == 1:
                        ax.imshow(PIL.ImageOps.autocontrast(array_to_img(gt)))
                    elif j == 2:
                        ax.imshow(PIL.ImageOps.autocontrast(array_to_img(self.get_mask(predictions[0]))))
                    elif j == 3:
                        ax.imshow(PIL.ImageOps.autocontrast(array_to_img(p)))
                        
                plt.show()
            
        
        
        
        """
        # Change the following detector and/or add your detectors below
        import detectors.cascade_detector.detector as cascade_detector
        cascade_detector = cascade_detector.Detector()
        
        
        #cascade + cutout segm
        for im_name in im_list:
            img = cv2.imread(im_name)
            

            annot_file = im_name.split('/')[-1]
            annot_name = os.path.join(self.annotations_path, annot_file)
            annot = load_img(annot_name, color_mode="grayscale")
            annot = np.expand_dims(annot, axis=-1) / 255

            p = np.zeros((360, 480, 1))
            gt = annot
            
            prediction_list = cascade_detector.detect(img)
            
            if len(prediction_list) > 0 :
                cutout_predictions = segmentor.segment_list_w_bb([im_name], prediction_list)
                    
                for j, pred in enumerate(cutout_predictions):
                    x, y, w, h = prediction_list[j]
                            
                    x = x + w // 2
                    y = y + h // 2
    
    
                    fixed_size = 64
                    offset_y = abs(min(y-fixed_size, 0))
                    offset_x = abs(min(x-fixed_size, 0))
                        
                    start_y = y-fixed_size+offset_y
                    start_x = x-fixed_size+offset_x
        
                    shape = np.shape(p[start_y:y+fixed_size, start_x:x+fixed_size])
                        
                    p[start_y:start_y+shape[0], start_x:start_x+shape[1]] = np.logical_or(p[start_y:start_y+shape[0], start_x:start_x+shape[1]], self.get_mask(pred)[offset_y:offset_y + shape[0], offset_x:offset_x + shape[1]]) 
   


            iou = eval.iou_compute(p, gt)
            pr, re, f1, acc = eval.prref1_compute(p, gt)
            iou_arr.append(iou)
            pr_arr.append(pr)
            re_arr.append(re)
            f1_arr.append(f1)
            acc_arr.append(acc)
            
            if draw :
                fig = plt.figure(figsize=(8, 8))
                grid = ImageGrid(fig, 111, 
                             nrows_ncols=(1, 3),
                             axes_pad=0.1, 
                             )
                
                for ax, j in zip(grid, range(0, 3)) :
                    if j == 0:
                        ax.imshow(cv2.imread(im_name))
                    elif j == 1:
                        ax.imshow(PIL.ImageOps.autocontrast(array_to_img(gt)))
                    elif j == 2:
                        ax.imshow(PIL.ImageOps.autocontrast(array_to_img(p)))
                        
                plt.show()

        
        """
        """
        
        #whole cnn
        predictions = segmentor_cnn.segment_list(im_list)
        
        
        
        for idx, im_name in enumerate(im_list) :
            annot_file = im_name.split('/')[-1]
            annot_name = os.path.join(self.annotations_path, annot_file)
            annot = load_img(annot_name, color_mode="grayscale")
            annot = np.expand_dims(annot, axis=-1) / 255
            
            #print(annot_name)


            p = self.get_mask(predictions[idx])
            gt = annot
            

            iou = eval.iou_compute(p, gt)
            pr, re, f1, acc = eval.prref1_compute(p, gt)
            iou_arr.append(iou)
            pr_arr.append(pr)
            re_arr.append(re)
            f1_arr.append(f1)
            acc_arr.append(acc)
            
            if draw :
                fig = plt.figure(figsize=(8, 8))
                grid = ImageGrid(fig, 111, 
                             nrows_ncols=(1, 3),
                             axes_pad=0.1, 
                             )
                
                for ax, j in zip(grid, range(0, 3)) :
                    if j == 0:
                        ax.imshow(cv2.imread(im_name))
                    elif j == 1:
                        ax.imshow(PIL.ImageOps.autocontrast(array_to_img(gt)))
                    elif j == 2:
                        ax.imshow(PIL.ImageOps.autocontrast(array_to_img(p)))
                        
                plt.show()
                
        
        """

        """
        #centered cutout
        for im_name in im_list:
            
            annot_file = im_name.split('/')[-1]
            annot_name = os.path.join(self.annotations_path, annot_file)
            annot = load_img(annot_name, color_mode="grayscale")
            annot = np.expand_dims(annot, axis=-1) / 255
            
            p = np.zeros((360, 480, 1))
            gt = annot
            

            yolo_name = os.path.join(self.zones_path, Path(os.path.basename(im_name)).stem) + '.txt'
            yolo_list = self.get_annotations(yolo_name)
            
            predictions = segmentor.segment_list_w_bb([im_name], yolo_list)
            
            for idx, pred in enumerate(predictions):
                x, y, w, h = yolo_list[idx]
                    
                x = x + w // 2
                y = y + h // 2
                
                fixed_size = 64
                offset_y = abs(min(y-fixed_size, 0))
                offset_x = abs(min(x-fixed_size, 0))
                
                start_y = y-fixed_size+offset_y
                start_x = x-fixed_size+offset_x

                shape = np.shape(p[start_y:y+fixed_size, start_x:x+fixed_size])
                
                p[start_y:start_y+shape[0], start_x:start_x+shape[1]] = self.get_mask(pred)[offset_y:offset_y + shape[0], offset_x:offset_x + shape[1]]

            
            #plt.imshow(PIL.ImageOps.autocontrast(array_to_img(p)))
            #plt.show()

            iou = eval.iou_compute(p, gt)
            pr, re, f1, acc = eval.prref1_compute(p, gt)
            iou_arr.append(iou)
            pr_arr.append(pr)
            re_arr.append(re)
            f1_arr.append(f1)
            acc_arr.append(acc)
            
            if draw :
                fig = plt.figure(figsize=(8, 8))
                grid = ImageGrid(fig, 111, 
                             nrows_ncols=(1, 3),
                             axes_pad=0.1, 
                             )
                
                for ax, j in zip(grid, range(0, 3)) :
                    if j == 0:
                        ax.imshow(cv2.imread(im_name))
                    elif j == 1:
                        ax.imshow(PIL.ImageOps.autocontrast(array_to_img(gt)))
                    elif j == 2:
                        ax.imshow(PIL.ImageOps.autocontrast(array_to_img(p)))
                        
                plt.show()

        """
        
        miou = np.average(iou_arr)  
        mpr = np.average(pr_arr) * 100 
        mre = np.average(re_arr) * 100
        mf1 = np.average(f1_arr) * 100
        macc = np.average(acc_arr) * 100
        
        siou = np.std(iou_arr) 
        spr = np.std(pr_arr)  * 100
        sre = np.std(re_arr)  * 100
        sf1 = np.std(f1_arr) * 100
        sacc = np.std(acc_arr) * 100
        print("\n")
        print("Average IOU:", f"{miou:.2%}", "+-" , f"{siou:.2%}")
        print("Average Accuracy:", "{:.2f}".format(macc), "+-" , "{:.2f}".format(sacc))
        print("Average Precision:", "{:.2f}".format(mpr), "+-" , "{:.2f}".format(spr))
        print("Average Recall:", "{:.2f}".format(mre), "+-" , "{:.2f}".format(sre))
        print("Average F1:", "{:.2f}".format(mf1), "+-" , "{:.2f}".format(sf1))
        print("\n")


if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()