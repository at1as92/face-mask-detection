'''
SDRE 203 CV and Deep Learning Mini Project
train_mask_detector_2.py
Script by Gary Chew
21 Jan 2024

2nd Model: Explore using transfer learning on ResNet50V2 to classify with mask and without mask images
Script is adapted from transfer learning flower classifers (SDRE 203 CV and Deep Learning course) and 
https://github.com/chandrikadeb7/Face-Mask-Detection

To access tensorboard, type into cmd: tensorboard --logdir logs/MobileNetV2_TL_{dataset_size} --host 127.0.0.1
'''

#Import libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Flatten, Dropout, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import os
import pathlib
import numpy as np
import pandas as pd

#Set parameters
batch_size = 32
epochs = 50
initial_LR = 0.0001 #Learning rate
IMAGE_SHAPE = (96, 96, 3) # size of each image
CLASS_NAMES = ['without_mask','with_mask'] #0: without_mask; #1: with_mask
dataset_size = 'large' #Options: small, medium, large

def load_data(dataset_size, IMAGE_SHAPE):

    #Path to training, validation and test dataset
    data_dir_train = pathlib.Path(fr'C:\SDRE-CV\Mini_Project\face-mask-dataset\{dataset_size}\train')

    #Train: 60%; Val: 20%; Test: 20%
    #Augmentation performed for training images 
    train_image_generator = ImageDataGenerator(rescale=1./255, 
                                               rotation_range=15, 
                                               zoom_range=0.15,
                                               width_shift_range=0.2, 
                                               height_shift_range=0.2, 
                                               shear_range=0.15, 
                                               horizontal_flip=True, 
                                               fill_mode="nearest",
                                               validation_split = 0.25)
    
    #No augmentation performed for validation images
    val_image_generator = ImageDataGenerator(rescale=1./255, validation_split = 0.25)
    
    train_data_gen = train_image_generator.flow_from_directory(directory=str(data_dir_train), 
                                                               batch_size=batch_size,
                                                               classes=list(CLASS_NAMES), 
                                                               class_mode='binary',
                                                               target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                                                               shuffle=True,
                                                               subset='training')

    val_data_gen = val_image_generator.flow_from_directory(directory=str(data_dir_train), 
                                                            batch_size=batch_size, 
                                                            classes=list(CLASS_NAMES), 
                                                            class_mode='binary',
                                                            target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                                                            shuffle=True,
                                                            subset='validation')
    
    return train_data_gen, val_data_gen

#Create a transfer learning model to train the mask vs no mask dataset
def build_model(input_shape, initial_LR, epochs):

    #Load MobileNetV2 with imagenet weights without the fully connected layers
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)

    #Freeze weights in CNN layers
    for layer in base_model.layers:
        layer.trainable = False

    #Construct fully connected layers for classification
    output = base_model.output
    output = GlobalAveragePooling2D()(output)
    output = Flatten()(output)
    output = Dense(128, activation="relu")(output)
    output = Dropout(0.5)(output)
    output = Dense(1, activation="sigmoid")(output)     

    model = Model(inputs=base_model.inputs, outputs=output)

    #Print summary of the model architecture
    model.summary()

    #Train the fully connected layers
    opt = Adam(learning_rate=initial_LR, decay=initial_LR / epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    return model

if __name__ == "__main__":

    #Load data generators
    train_generator, validation_generator = load_data(dataset_size, IMAGE_SHAPE)

    #Construct model
    model = build_model(IMAGE_SHAPE, initial_LR, epochs)

    #Set model name
    model_name = f"ResNet50V2_TL_{dataset_size}"

    #Set up the callbacks
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name)) #Create a subfolder in logs

    checkpoint = ModelCheckpoint(os.path.join("results", f"{model_name}" + "-loss-{val_loss:.3f}.h5"),
                                save_best_only=True,
                                verbose=1)
    
    earlystop = EarlyStopping(monitor='val_loss', patience = 3)

    #Make sure results folder (saves the weights) exist
    if not os.path.isdir("results"):
        os.mkdir("results")

    #Count number of steps per epoch
    training_steps_per_epoch = np.ceil(train_generator.samples / batch_size)
    validation_steps_per_epoch = np.ceil(validation_generator.samples / batch_size)

    #Train the model and save the history for plotting
    history = model.fit(train_generator, 
                        steps_per_epoch=training_steps_per_epoch,
                        validation_data=validation_generator, 
                        validation_steps=validation_steps_per_epoch,
                        epochs=epochs, 
                        verbose=1, 
                        callbacks=[tensorboard, checkpoint, earlystop])
    
    history_df = pd.DataFrame(history.history) 
    with open(model_name + ".json", 'w') as f:
        history_df.to_json(f)