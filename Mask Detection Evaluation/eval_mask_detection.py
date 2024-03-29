'''
SDRE 203 CV and Deep Learning Mini Project
eval_mask_detection.py
Script by Gary Chew
21 Jan 2024

Script feeds in 100 unseen labelled images from: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection to test the trained models
Face detection is first done with YOLO face and the detected face is inputted into the trained classifier model.
The results will be saved in a JSON and used to evaluated against a ground truth dataset (in GT_eval_mask_detection.py).

'''
#Import libraries
import sys
import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model

import torch
sys.path.append("yoloface/")
from yoloface.face_detector import YoloDetector

def preprocess_image(img):
	
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 96, 96, 3)
	# prepare pixel data
	img = img.astype('float32')
    
	img = img / 255.0
	return img

def detect_face_mask():
    # Initialize COCO result structure
    coco_results = []

    CLASSES = ['without_mask', 'with_mask']
    
    if not torch.cuda.is_available():
        #model = torch.load('yoloface/weights/yolov5n-face.pt', map_location=torch.device('cpu'))
        #torch.save(model.state_dict(),'yoloface/weights/yolov5n-face.pt')
        detector_model = YoloDetector(weights_name='yolov5n-face.pt',config_name='yolov5n.yaml',target_size=720, min_face=10, device="cpu")
    else:
        detector_model = torch.load('yoloface/weights/yolov5n-face.pt')
    
    #Choose either A or B
    #A: Sequential Train: 44sec for 100 images
    # classifier_model = load_model('results/Sequential_Train_large-loss-0.161.h5')
    # path_detect_mask_result = 'detect_mask_Sequential_Train_large-loss-0.161.json'

    #B: RestNet50V2 TL: 59 sec for 100 images
    classifier_model = load_model('results/ResNet50V2_TL_large-loss-0.073.h5')
    path_detect_mask_result = 'detect_mask_ResNet50V2_TL_large-loss-0.073.json'
    
    for i in np.arange(101):
        filename = f'face-mask-dataset-test/images/maksssksksss{i}.png'
        image_id = f'maksssksksss{i}.png'
        print(image_id)

        img = cv2.imread(filename)
        face_img = img.copy()  # Color image
        height, width, channels = face_img.shape

        try:
            bboxes,points = detector_model.predict(img, conf_thres = 0.5)
            bboxes = bboxes[0]

            for bbox in bboxes:
                x_min = bbox[0]
                y_min = bbox[1]
                x_max = bbox[2]
                y_max = bbox[3]

                roi_color = face_img[y_min:y_max, x_min:x_max] 

                image = cv2.resize(roi_color, (96, 96))
                keras_image = preprocess_image(image)
                result = classifier_model.predict(keras_image)

                if result[0][0] >= 0.5:
                    category_id = 101
                    score = result[0][0]
                    label = 'with mask'
                    cv2.rectangle(face_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                    cv2.putText(face_img, label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    category_id = 102
                    score = 1 - result[0][0]
                    label = 'without mask'
                    cv2.rectangle(face_img, (x_min, y_min), (x_max, y_max), (0, 0,255), 3)
                    cv2.putText(face_img, label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                result_entry = {
                            "image_id": image_id,  # Extract image ID
                            "category_id": category_id,  # ID: 100 is face
                            "bbox": [int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)],
                            "score": float(score)  #Confidence level
                        }
                
                coco_results.append(result_entry)

        except ValueError as e:
            print(e)
        
        #Sanity check
        # cv2.imshow("detected face and mask", face_img)
        # cv2.waitKey(0)

    # Save COCO formatted results to a JSON file
    with open(path_detect_mask_result, 'w') as outfile:
        json.dump(coco_results, outfile, indent=2)

if __name__ == '__main__':
    detect_face_mask()