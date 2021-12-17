import cv2, sys, os
import numpy as np

class Detector:
	# This example of a detector detects faces. However, you have annotations for ears!

	#cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'lbpcascade_frontalface.xml'))
    
    def __init__(self) :
        self.cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'lbpcascade_frontalface.xml'))
        self.cascade1 = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_leftear.xml'))
        self.cascade2 = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_rightear.xml'))

    def detect(self, img):

        det_list1 = self.cascade1.detectMultiScale(img, 1.05, 1)
        det_list2 = self.cascade2.detectMultiScale(img, 1.05, 1)
        
        ret_list = []
        if len(det_list1) > 0:
            ret_list.append(det_list1[0])
        if len(det_list2) > 0:
            ret_list.append(det_list2[0])
    
        return ret_list

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	detector = Detector()
	detected_loc = detector.detect(img)
	for x, y, w, h in detected_loc:
		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
	cv2.imwrite(fname + '.detected.jpg', img)