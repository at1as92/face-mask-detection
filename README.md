# Face Mask Detector

## Introduction
This repository contains the code used for NP's Specialist Diploma in Robotics Engineering Computer Vision and Deep Learning mini project.

This mini project requires a CNN-based model to be built to detect if humans are wearing face masks in a real-time video stream. Green and red bounding boxes, along with the corresponding labels and confidence scores shall be drawn to indicate with and without mask respectively.

## Face Detection
Experimented with 3 different models to evaluate and select an optimal face detector model. Evaluation was performed against Fold #1 (291 images) of the Face Detection Data Set and Benchmark ([FDDB](https://vis-www.cs.umass.edu/fddb/)) using the COCO metrics. Ground truth labels were converted from elliptical to rectangular bounding boxes to facilitate evaluation. Time taken to process images was taken as proxy of computation requirements for each face detector. 

YOLOv5 Face outperformed the other models with the smallest computational demands and was selected out of the 3 models tested.

| Model | AP @ IoU=0.50:0.95 | AR @ IoU=0.50:0.95 |  Average Time (s) |
|:---------:|:-------------------:|:-------------------:|:-----------:|
| Haar Cascade Classifier | 0.161 | 0.255 | 53.2  |
| YOLOv5 Face             | 0.499 | 0.625 | 23.2  |
| RetinaFace              | 0.348 | 0.485 | 507.3 |


All code for this section can be found in [Face Detection](https://github.com/at1as92/face-mask-detection/tree/main/Face%20Detection).

## Training of Mask-No Mask Classifier
Two approaches were explored when training the mask-no mask classifier. Both models were trained for 50 epochs with callbacks implemented to prevent overfitting. 

1. Train a Sequential CNN model from scratch: Modified the output layer of standard Tensorflow Keras Image Classification model to one neuron with sigmoid activation function
2. Transfer Learning: Pre-trained ResNet50V2 ImageNet weights from [Keras Applications](https://keras.io/api/applications/) with fully connected and output layers modified

Training was performed with the dataset provided for this assignment (small subset uploaded) and augmented with additional images from [Chandrika Deb](https://github.com/chandrikadeb7/Face-Mask-Detection). Breakdown for training, validation and testing is approximately 60%, 20% and 20% respectively.

|     | Images With Mask | Images Without Mask |
|:---:|:----------------:|:-------------------:|
| Training Set | 270 | 225 |
| Validation Set | 90 | 75 |
| Test Set | 85 | 72 |

Classification report generated to analyse classifer performance against unseen test set showed ResNetV50V2 is marginally more accurate.

| Sequential | Precision  | Recall | F1-Score |
|:----------:|:----------:|:------:|:--------:|
|Without Mask| 0.944 | 0.944 | 0.944 |
|With Mask   | 0.953 | 0.953 | 0.953 |
|Accuracy    | -     | -     | 0.949 |

| ResNet50V2 | Precision  | Recall | F1-Score |
|:----------:|:----------:|:------:|:--------:|
|Without Mask| 0.986 | 0.986 | 0.986 |
|With Mask   | 0.988 | 0.988 | 0.988 |
|Accuracy    | -     | -     | 0.987 |


All code for this section can be found in [Mask Detection Training](https://github.com/at1as92/face-mask-detection/tree/main/Mask%20Detection%20Training)

## Evaluation of Face Detector & Mask-No Mask Classifier


https://github.com/at1as92/face-mask-detection/assets/62200772/28a4c8cc-27e5-4f97-865d-7e3582b83af2

