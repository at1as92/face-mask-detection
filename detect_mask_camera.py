'''
SDRE 203 CV and Deep Learning Mini Project
detect_mask_camera.py
Script by Gary Chew
21 Jan 2024

This script implements the selected face detection and mask-no mask classifier model to detect face mask in videos
Script is adapted from materials in the SDRE 203 CV and Deep Learning course. 
'''
#Import libraries
import sys
import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model

import torch
sys.path.append("yoloface/")
from yoloface.face_detector import YoloDetector

#Resize and reshape the extracted face image
def preprocess_image(img):
	
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 96, 96, 3)
	# prepare pixel data
	img = img.astype('float32')
    
	img = img / 255.0
	return img

#Process the incoming image to draw bounding boxes on faces, with and without masks
def detect_face_mask(img, detector_model, classifier_model, width, height):
    CLASSES = ['without_mask', 'with_mask']

    face_img = img.copy()  # Color image
    
    #detector_start = time.time()
    bboxes,points = detector_model.predict(img, conf_thres = 0.5) # Predict the faces
    #detector_end = time.time()
    #print("Detector:", str(detector_end-detector_start))
    
    bboxes = bboxes[0]

    #If no faces detected
    if len(bboxes) == 0:
        return face_img
    else:
        #For each face found, extract and feed into the classifier
        #classifier_start = time.time()
        for bbox in bboxes:
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[2]
            y_max = bbox[3]

            roi_color = face_img[y_min:y_max, x_min:x_max] 
            image = cv2.resize(roi_color, (96, 96))
            keras_image = preprocess_image(image)
            result = classifier_model.predict(keras_image)

            #Discriminate between 2 classes using threshold
            if result[0][0] >= 0.5:
                label = 'Mask' + str(result[0][0])
                cv2.rectangle(face_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                cv2.putText(face_img, label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                label = 'No Mask' + str(1-result[0][0])
                cv2.rectangle(face_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                cv2.putText(face_img, label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        #classifier_end = time.time()
        #print("Classifier:", str(classifier_end-classifier_start))
        return face_img

#Load model
if not torch.cuda.is_available():
    #model = torch.load('yoloface/weights/yolov5n-face.pt', map_location=torch.device('cpu'))
    #torch.save(model.state_dict(),'yoloface/weights/yolov5n-face.pt')
    detector_model = YoloDetector(weights_name='yolov5n-face.pt',config_name='yolov5n.yaml',target_size=720, min_face=32, device="cpu")
else:
    detector_model = torch.load('yoloface/weights/yolov5n-face.pt')

classifier_model = load_model('results/ResNet50V2_TL_large-loss-0.073.h5')

#Loading video from file or webcam
#cap = cv2.VideoCapture('test/COVID-19-shortclip-2.mp4') #load video
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("width=",width,"height=",height)

#Declare the video writer object
writer = cv2.VideoWriter('long_clip.mp4',cv2.VideoWriter_fourcc(*'mp4v'),30,(width,height))

#Capture frame by frame
while True:
    ret, frame = cap.read() #.read returns 2 variables: bool, 
    if not ret:
        break
    
    result = detect_face_mask(frame, detector_model, classifier_model, width, height)
    cv2.imshow("detected face and mask", result)
    #writer.write(result)

    if cv2.waitKey(1) & 0xFF == ord('q'): #When you press q, the video will exit
        break

cap.release()
cv2.destroyAllWindows()