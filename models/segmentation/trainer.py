"""
Trainer class that creates a segmentation model and trains it
Using quvbel segmentation models and tensorflow
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import segmentation_models as sm
import os
from pathlib import Path


class trainer:
    def __init__(self,
                 model,
                 backbone="resnet101",
                 im_size=224,
                 num_classes=2,
                 batch_size=4,
                 epochs=10):
        self.model = str(model).lower()
        self.bacbone = backbone
        self.img_size = im_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps = None
        self.hist = None

        # model parameters
        self.train_gen = None
        self.val_gen = None
        self.callbacks = None
        self.metrics = []
        self.input_shape = (self.img_size, self.img_size, 3)
        self.num_classes = num_classes
        if self.num_classes == 2:
            self.activation = "sigmoind"
        else:
            self.activation = "softmax"

    def model_builder(self):
        if self.model is "unet":
            mod = sm.Unet(self.bacbone, encoder_weights="imagenet", activation=self.activation,
                          input_shape=self.input_shape, classes=self.num_classes)
        elif self.model is "fpn":
            mod = sm.FPN(self.bacbone, encoder_weights="imagenet", activation=self.activation,
                         input_shape=self.input_shape, classes=self.num_classes)

        elif self.model is "pspnet":
            mod = sm.PSPNet(self.bacbone, encoder_weights="imagenet", activation=self.activation,
                            input_shape=self.input_shape, classes=self.num_classes)
        else:
            raise ValueError(f"Model name not supported, got {self.model}")

        return mod

    def initmodel(self, train_data, val_data=None):
        """
        This function takes the data an initializes the model
        :param train_data: training data to use, in a dataframe format or a list of dirs
        :param val_data: validation data to use, if none, taken from splitting the train into 80/20
        :return: None, it initalizes the model
        """
        print("Initializing creation of model")

        self.steps = train_data.shape[0] // self.batch_size
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        datagen.fit(train_data)
        self.train_gen = datagen.flow_from_dataframe(train_data)

        if val_data is not None:
            val_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True
            )
            val_gen.fit(val_data)
            self.val_gen = datagen.flow_from_dataframe(val_data)

        callbacks = []
        results = os.getcwd()
        results = os.path.join(results, "results/")
        results = str(results)
        Path(results).mkdir(exist_ok=True, parents=True)
        callbacks.append(tf.keras.callbacks.CSVLogger(results + "/logs.csv"))
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(results + "/final_weights.h5"))
        self.callbacks = callbacks

        self.metrics = [sm.metrics.IOUScore(name="iou_score"), sm.metrics.FScore(), sm.metrics.Precision(),
                        sm.metrics.Recall()]

        self.model = self.model_builder()
        self.model.compile("Adam", loss=sm.losses.bce_dice_loss, metrics=self.metrics)

    def train_model(self):
        print("\nInitializing training of model")
        self.hist = self.model.fit_generator(self.train_gen, steps_per_epoc=self.steps, epochs=self.epochs,
                                             verbose=1, callbacks=self.callbacks, validation_data=self.val_gen)

        print(self.hist)
        plt.plot(self.hist)
        print("end")
