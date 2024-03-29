'''
SDRE 203 CV and Deep Learning Mini Project
GT_eval_mask_detection.py
Script by Gary Chew
21 Jan 2024

Script processes the ground truth image file path and bounding box details into format ingestable by COCO
Runs COCO evaluation to evaluate between 2 mask detection algorithms: Sequential and ResNet50V2 transfer learning  

'''

#Import libraries
import json
import re
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class ProcessGroundTruth:

    def __init__(self):
        self.category_w_mask = {"id": 101, "name": "with_mask"} #1 category: with_mask
        self.category_wo_mask = {"id": 102, "name": "without_mask"} #1 category: with_mask

        #Initialize COCO annotation structure
        self.coco_data = {"images": [], "annotations": [], "categories": [self.category_w_mask, self.category_wo_mask]}


    # Function to add annotations to COCO format
    def add_annotation(self, image_id, x_min, y_min, x_max, y_max, category_id, annotation_id):

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            "area": (x_max - x_min) * (y_max - y_min),
            "iscrowd": 0,
            "segmentation": []
        }

        return annotation

    # Reading and processing annotations from the file
    def process(self):

        annotation_id = 1

        for i in np.arange(101):
            filename = f'face-mask-dataset-test/annotations_json/maksssksksss{i}.json'
            print(filename)
            
            with open(filename, 'r') as f:
                data = json.load(f)

            f.close()

            image_id = data['annotation']['filename']

            image_entry = {
                "id": image_id,
                "file_name": 'face-mask-dataset-test/images/maksssksksss' + str(i) + '.png'
            }
            
            self.coco_data['images'].append(image_entry)
            
            try:
                for object in data['annotation']['object']:
                    if object['name'] == 'with_mask':
                        category_id = 101
                    elif object['name'] == 'without_mask':
                        category_id = 102
                
                    x_min = object['bndbox']['xmin']
                    y_min = object['bndbox']['ymin']
                    x_max = object['bndbox']['xmax']
                    y_max = object['bndbox']['ymax']

                    # Add annotation
                    annotation = self.add_annotation(image_id, int(x_min), int(y_min), int(x_max), int(y_max), category_id, annotation_id)
                    self.coco_data['annotations'].append(annotation)

                    annotation_id += 1
            
            #Single object only
            except TypeError:

                if data['annotation']['object']['name'] == 'with_mask':
                    category_id = 101
                elif data['annotation']['object']['name'] == 'without_mask':
                    category_id = 102

                x_min = data['annotation']['object']['bndbox']['xmin']
                y_min = data['annotation']['object']['bndbox']['ymin']
                x_max = data['annotation']['object']['bndbox']['xmax']
                y_max = data['annotation']['object']['bndbox']['ymax']

                # Add annotation
                annotation = self.add_annotation(image_id, int(x_min), int(y_min), int(x_max), int(y_max), category_id, annotation_id)
                self.coco_data['annotations'].append(annotation)

                annotation_id += 1


        # Save COCO formatted annotations to a JSON file
        with open('face_mask_coco_annotations_gt.json', 'w') as outfile:
            json.dump(self.coco_data, outfile, indent=2)

class COCOEvaluate:

    def __init__(self, ground_truth_file, detector_results_file):
        self.ground_truth_file = ground_truth_file
        self.detector_results_file = detector_results_file

        self.evaluate()
    
    def evaluate(self):
        coco_gt = COCO(self.ground_truth_file)
        coco_dt = coco_gt.loadRes(self.detector_results_file)
        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

if __name__ == '__main__':

    face_GT = ProcessGroundTruth()
    face_GT.process()

    #AP for IoU=0.5 is 0.189
    coco_eval_mask_detector_1 = COCOEvaluate('face_mask_coco_annotations_gt.json','detect_mask_Sequential_Train_large-loss-0.161.json')

    #AP for IoU=0.5 is 0.472
    coco_eval_mask_detector_2 = COCOEvaluate('face_mask_coco_annotations_gt.json','detect_mask_ResNet50V2_TL_large-loss-0.073.json')