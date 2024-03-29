'''
SDRE 203 CV and Deep Learning Mini Project
face_detection.py
Script by Gary Chew
21 Jan 2024

Script implements the Haar Cascade, YoloFace, RetinaFace detection algorithms
Haar Cascade: Traditional CV method
YoloFace: CNN-based SSD (Small network of 1.726M parameters)
RetinaFace: CNN-based SSD (Large network of 29.5M parameters) 

Algorithms are tested on Face Detection Data Set and Benchmark (FDDB), specifically 
a. FDDB/FDDB-folds/FDDB-fold-01.txt (291 images)
Results are saved in a JSON file with the necessary parameters for future evaluation (in GT_eval_face_detection.py)
'''

#Import libraries
import sys, os
import time
import cv2
import numpy as np
from PIL import Image
import json
import torch
sys.path.append("yoloface/")
from yoloface.face_detector import YoloDetector
from retinaface import RetinaFace


#To download FDDB if have not done so
path_masterfile = 'FDDB/FDDB-folds/FDDB-fold-01.txt'

class CascadeDectector:
    #FDDB-fold-01: 53.5s,53.2s,52.8s

    def __init__(self, masterfile):
        self.start = time.time()    
        self.path_masterfile = masterfile #Master file with all the image ID
        self.path_cascade_detect_result = 'detect_face_cascade.json' #Results file
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #Load pre-trained cascade classifiers

    def detect_face_cascade(self, path_img):

        face_img = cv2.imread(path_img)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) #Convert to greyscale
    
        #Returns an rectangle around each detected face (x,y,width,height)
        face_coordinates = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize = (10,10))

        return face_coordinates

    def run_save_results(self):

        # Initialize COCO result structure
        coco_results = []

        with open(self.path_masterfile) as f:
            lines = [line.rstrip('\n') for line in f]

        f = open(self.path_cascade_detect_result,'w')

        for line in lines:
            img_path = 'FDDB/images/' + line + '.jpg'
            face_coordinates = self.detect_face_cascade(img_path)
            
            img = cv2.imread(img_path)

            #each coordinate is in the form: (x,y,width,height)
            try:
                for (x,y,w,h) in face_coordinates:
                    
                    #Sanity check;Uncomment if showing image
                    # cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),3)

                    result_entry = {
                    "image_id": line.replace('/', '-'),  # Extract image ID
                    "category_id": 100,  # ID: 100 is face
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "score": 1.0  #Haar cascade has no confidence intervals
                    }
                
                    coco_results.append(result_entry)

            except TypeError as e:
                print(e)
        
            #Sanity check;Uncomment if showing image
            # cv2.imshow('test',img)
            # cv2.waitKey(3000)

        # Save COCO formatted results to a JSON file
        with open(self.path_cascade_detect_result, 'w') as outfile:
            json.dump(coco_results, outfile, indent=2)
        
        f.close()

        self.end = time.time()
        print("Completed face detection. Time taken: {}".format(self.end-self.start))

        return True

class YoloFaceDetector:
    #FDDB-fold-01: 23.4s,22.8s,23.4s

    def __init__(self,masterfile):
        self.start = time.time()
        self.path_masterfile = masterfile #Master file with all the image ID
        self.path_yoloface_detect_result = 'detect_face_yolo.json' #Results file

    def run_save_results(self):

        # Initialize COCO result structure
        coco_results = []
        
        with open(self.path_masterfile) as f:
            lines = [line.rstrip('\n') for line in f]
        
        # model = YoloDetector(weights_name='yolov5n-face.pt',
        #                      target_size=None, 
        #                      device="cpu", min_face=48)
        
        if not torch.cuda.is_available():
            #model = torch.load('yoloface/weights/yolov5n-face.pt', map_location=torch.device('cpu'))
            #torch.save(model.state_dict(),'yoloface/weights/yolov5n-face.pt')
            model = YoloDetector(weights_name='yolov5n-face.pt',config_name='yolov5n.yaml',target_size=720, min_face=10, device="cpu")
        else:
            model = torch.load('yoloface/weights/yolov5n-face.pt')

        
        for line in lines:
            img_path = 'FDDB/images/' + line + '.jpg'
            img = np.array(Image.open(img_path))
            img_plot = cv2.imread(img_path)

            try:
                bboxes,points = model.predict(img, conf_thres = 0.5)
                bboxes = bboxes[0]

                for bbox in bboxes:
                    xmin = bbox[0]
                    ymin = bbox[1]
                    xmax = bbox[2]
                    ymax = bbox[3]
                    
                    #Sanity check; Uncomment if showing image
                    #cv2.rectangle(img_plot, (xmin,ymin), (xmax,ymax),(0,255,0),3)


                    result_entry = {
                        "image_id": line.replace('/', '-'),  # Extract image ID
                        "category_id": 100,  # ID: 100 is face
                        "bbox": [float(xmin), float(ymin), float(xmax-xmin), float(ymax-ymin)],
                        "score": 1  #Confidence level
                    }

                    coco_results.append(result_entry)

            except ValueError as e:
                print(e)
            
            #Sanity check; Uncomment if showing image
            # cv2.imshow('test',img_plot)
            # cv2.waitKey(0)
        
        # Save COCO formatted results to a JSON file
        with open(self.path_yoloface_detect_result, 'w') as outfile:
            json.dump(coco_results, outfile, indent=2)

        f.close()

        self.end = time.time()
        print("Completed face detection. Time taken: {}".format(self.end-self.start))


class RetinaFaceDetector:
    #FDDB-fold-01: 507.2s, 511.3s, 503.5s

    def __init__(self,masterfile):
        self.start = time.time()
        self.path_masterfile = masterfile #Master file with all the image ID
        self.path_retinafae_detect_result = 'detect_face_retina.json' #Results file

    def run_save_results(self):

        # Initialize COCO result structure
        coco_results = []
        
        with open(self.path_masterfile) as f:
            lines = [line.rstrip('\n') for line in f]

        for line in lines:
            img_path = 'FDDB/images/' + line + '.jpg'
            face_coordinates = RetinaFace.detect_faces(img_path, threshold = 0.5)

            img = cv2.imread(img_path)

            try:
                for key in face_coordinates.keys():

                    face = face_coordinates[key]
                    x = face['facial_area'][0]
                    y = face['facial_area'][1]
                    w = face['facial_area'][2] - face['facial_area'][0]
                    h = face['facial_area'][3] - face['facial_area'][1]
                    score = face['score']

                    #Sanity check;Uncomment if showing image
                    # cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),3)


                    result_entry = {
                        "image_id": line.replace('/', '-'),  # Extract image ID
                        "category_id": 100,  # ID: 100 is face
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": 1  #Confidence level
                    }

                    coco_results.append(result_entry)

            except AttributeError as e:
                print(e)
            
            #Sanity check;Uncomment if showing image
            # cv2.imshow('test',img)
            # cv2.waitKey(0)
        
        # Save COCO formatted results to a JSON file
        with open(self.path_retinafae_detect_result, 'w') as outfile:
            json.dump(coco_results, outfile, indent=2)

        f.close()
        self.end = time.time()
        print("Completed face detection. Time taken: {}".format(self.end-self.start))

if __name__ == '__main__':
        
    cascade = CascadeDectector(path_masterfile)
    cascade.run_save_results()

    # yoloface = YoloFaceDetector(path_masterfile)
    # yoloface.run_save_results()

    # retinaface = RetinaFaceDetector(path_masterfile)
    # retinaface.run_save_results()

