"""
Trainer class that creates a segmentation model and trains it
Using quvbel segmentation models and tensorflow
"""
import sys
sys.path.append("C:/Users/Arturo/PycharmProjects/Segm/")
import tensorflow as tf
import matplotlib.pyplot as plt
import segmentation_models as sm
import os
from pathlib import Path
from private_models.models.utils.generators import SegmDataGenerator
from private_models.models.segmentation import pspunet


class trainer:
    def __init__(self,
                 model,
                 backbone="resnet101",
                 im_size=224,
                 num_classes=2,
                 batch_size=4,
                 epochs=10):
        self.model_name = str(model).lower()
        self.backbone = backbone
        self.img_size = im_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps = None
        self.hist = None

        # model parameters
        self.train_gen = None
        self.val_gen = None
        self.callbacks = None
        self.model = None
        self.metrics = []
        self.input_shape = (None, None, 3)
        self.num_classes = num_classes
        if self.num_classes == 2:
            self.activation = "sigmoind"
        else:
            self.activation = "softmax"

    def model_builder(self):
        if self.model_name == "unet":
            if self.img_size % 32 != 0:
                # unet requires an input multiple of 32
                self.img_size = self.img_size - (self.img_size % 32)
                print(f"Unet requires an input multiple of 32, image size set to {self.img_size}")
            mod = sm.Unet(self.backbone, encoder_weights="imagenet", activation=self.activation,
                          classes=self.num_classes)

        elif self.model_name == "fpn":
            mod = sm.FPN(self.backbone, encoder_weights="imagenet", activation=self.activation,
                         classes=self.num_classes)

        elif self.model_name == "pspnet":
            if self.img_size % 48 != 0:
                # unet requires an input multiple of 32
                self.img_size = self.img_size - (self.img_size % 48)
                print(f"PSPnet requires an input multiple of 48, image size set to {self.img_size}")
            mod = sm.PSPNet(self.backbone, encoder_weights="imagenet", activation=self.activation,
                            classes=self.num_classes)
        elif self.model_name == "pspunet":
            if self.img_size % 96 != 0:
                # unet requires an input multiple of 32
                self.img_size = self.img_size - (self.img_size % 96)
                print(f"PSP-Unet requires an input multiple of 48, image size set to {self.img_size}")
            mod = pspunet.build_pspunet(self.backbone, classes=self.num_classes, encoder_weights="imagenet",
                                        encoder_freeze=True, in_shape=(self.img_size, self.img_size, 3))
        else:
            raise ValueError(f"Model name not supported, got {self.model_name}")

        return mod

    def initmodel(self, train_data, val_data=None):
        """
        This function takes the data an initializes the model
        :param train_data: training data to use, in a dataframe format or a list of dirs
        :param val_data: validation data to use, if none, taken from splitting the train into 80/20
        :return: None, it initalizes the model
        """
        print("Initializing creation of model")
        self.model = self.model_builder()
        self.steps = train_data.shape[0] // self.batch_size

        self.train_gen = SegmDataGenerator(train_data, batch_size=self.batch_size, img_size=self.img_size,
                                           n_classes=self.num_classes)

        if val_data is not None:
            self.val_gen = SegmDataGenerator(val_data, batch_size=self.batch_size, img_size=self.img_size,
                                             n_classes=self.num_classes)

        callbacks = []
        results = os.getcwd()
        results = os.path.join(results, "results/")
        results = str(results)
        Path(results).mkdir(exist_ok=True, parents=True)
        callbacks.append(tf.keras.callbacks.CSVLogger(results + "/logs.csv", separator=";"))
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(results + f"/{self.model_name}_{self.backbone}_final_weights.h5"))
        self.callbacks = callbacks

        self.metrics = [sm.metrics.IOUScore(name="iou_score"), sm.metrics.FScore(), sm.metrics.Precision(),
                        sm.metrics.Recall()]

        self.model.compile("Adam", loss=sm.losses.categorical_crossentropy, metrics=self.metrics)
        print("Model built")

    def model_sum(self):
        print(self.model.summary())

    def load_prev_weights(self):
        wpath = os.getcwd()
        wpath = os.path.join(wpath, "results", f"{self.model_name}_{self.backbone}_final_weights.h5")
        if Path(wpath).is_file():
            self.model.load_weights(wpath)
            print("Previously trained weights loaded")

    def train_model(self):
        print("\nInitializing training of model")
        self.load_prev_weights()
        self.hist = self.model.fit_generator(self.train_gen, steps_per_epoch=self.steps, epochs=self.epochs,
                                             verbose=1, callbacks=self.callbacks, validation_data=self.val_gen)

    def model_hist(self):
        plt.figure(figsize=(30, 5))
        plt.subplot(121)
        plt.plot(self.hist.history['iou_score'])
        plt.plot(self.hist.history['val_iou_score'])
        plt.title('Model iou_score')
        plt.ylabel('iou_score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(122)
        plt.plot(self.hist.history['loss'])
        plt.plot(self.hist.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
