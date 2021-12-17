import cv2
import numpy as np

class Evaluation:

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

    def iou_compute_bb(self, p, gt):
            # Computes Intersection Over Union (IOU)
            if len(p) == 0:
                return 0

            intersection = np.logical_and(p, gt)
            union = np.logical_or(p, gt)

            iou = np.sum(intersection) / np.sum(union)

            return iou
        
        
    def iou_compute(self, p, gt):
            # Computes Intersection Over Union (IOU)
            if len(p) == 0:
                return 0

            intersection = np.logical_and(p, gt)
            union = np.logical_or(p, gt)

            iou = np.sum(intersection) / np.sum(union)

            return iou

    def prref1_compute(self, p, gt):
        
            TP = np.sum(np.logical_and(p, gt))
            FP = np.sum(p) - TP
            FN = np.sum(np.logical_and(np.logical_not(p), gt))
            TN = np.sum(np.logical_not(p)) - FN
            
            if TP > 0 :
                acc = (TP + TN) / (TP + FP + FN + TN)
                pr = TP / (TP + FP)
                re = TP / (TP + FN)
                f1 = 2 / (1/pr + 1/re)
            else : 
                if TN > 0 :
                    acc = TN / (FP + FN + TN)
                else :
                    acc = 0.0
                pr = 0.0
                re = 0.0
                f1 = 0.0
            
            return (pr, re, f1, acc)  
        
    def prref1_compute_2(self, p, gt):
            if len(p) == 0:
                return 0

            TP = np.sum(np.logical_and(p, gt))
            FP = np.sum(p) - TP
            FN = np.sum(np.logical_and(np.logical_not(p), gt))
            TN = np.sum(np.logical_not(p)) - FN
            
            if TP > 0 :
                acc = (TP + TN) / (TP + FP + FN + TN)
                pr = TP / (TP + FP)
                re = TP / (TP + FN)
                f1 = 2 / (1/pr + 1/re)
            else : 
                if TN > 0 :
                    acc = TN / (FP + FN + TN)
                else :
                    acc = 0.0
                pr = 0.0
                re = 0.0
                f1 = 0.0
            
            return (pr, re, f1, acc)  
    # Add your own metrics here, such as mAP, class-weighted accuracy, ...