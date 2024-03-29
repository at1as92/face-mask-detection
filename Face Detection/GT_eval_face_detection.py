'''
203 CV and Deep Learning Mini Project
GT_process_eval.py
Script by Gary Chew
21 Jan 24

Script processes the ground truth image file path and bounding box details into format ingestable by COCO
Runs COCO evaluation to evaluate between 3 detection algorithms: Haar Cascade, YoloFace, RetinaFace  
'''

#Import libraries
import json
import re
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

#Original ground truth file
ground_truth_file = 'FDDB/FDDB-folds/FDDB-fold-01-rectList.txt'

class ProcessGroundTruth:

    def __init__(self, ground_truth_file):
        self.category = {"id": 100, "name": "face"} #1 category: face

        #Initialize COCO annotation structure
        self.coco_data = {"images": [], "annotations": [], "categories": [self.category]}

        self.ground_truth_file = ground_truth_file

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

        with open(self.ground_truth_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            image_path, x_min, y_min, x_max, y_max = line.strip().split(',')
        
            # Extracting image ID from the image name; need to align with the results files
            image_id = re.split(r'/|\.', image_path)[1:-1]
            image_id = '-'.join(image_id)

            # Define image entry to be added to coco_data
            image_entry = {
                "id": image_id,
                "file_name": image_path
            }

            self.coco_data['images'].append(image_entry)

            # Add annotation
            annotation = self.add_annotation(image_id, int(x_min), int(y_min), int(x_max), int(y_max), self.category['id'], annotation_id)
            self.coco_data['annotations'].append(annotation)

            annotation_id += 1

        # Save COCO formatted annotations to a JSON file
        with open('fddb_coco_annotations_gt.json', 'w') as outfile:
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

if __name__ == "__main__":

    FDDB_GT = ProcessGroundTruth(ground_truth_file)
    FDDB_GT.process()

    #Evaluate detected results using COCO metrics
    
    #mAP for IoU=0.5 is 0.534
    coco_eval_cascade = COCOEvaluate('fddb_coco_annotations_gt.json','detect_face_cascade.json')

    #mAP for IoU=0.5 is 0.812
    coco_eval_yoloface = COCOEvaluate('fddb_coco_annotations_gt.json','detect_face_yolo.json')

    #mAP for IoU=0.5 is 0.937
    coco_eval_retinaface = COCOEvaluate('fddb_coco_annotations_gt.json','detect_face_retina.json')
    