import tensorflow as tf
import numpy as np
from PIL import Image


class SegmDataGenerator(tf.keras.utils.Sequence):
    """
    Generates data for Keras
    Reads the data from a dataframe and yields a pair of image/mask
    """

    def __init__(self, dataf, labels=None, batch_size=8, img_size=40, channels=3,
                 n_classes=10, shuffle=True, augmentations=None):
        """Initialization"""
        self.datf = dataf
        self.images = self.datf["image"].tolist()
        self.masks = self.datf["label"].tolist()
        self.numpics = len(dataf)
        self.img_size = img_size
        self.channels = channels
        self.input_size = (self.img_size, self.img_size, self.channels)
        self.batch_size = batch_size
        self.labels = labels
        self.n_classes = n_classes
        if n_classes == 2:
            self.outmask = 1  # 1 for binary
            self.binmask = True
        else:
            self.outmask = n_classes  # last dimmension of the mask labels
            self.binmask = False  # to speed up process we create a bool
        self.shuffle = shuffle
        self.indexes = None
        self.augmentations = augmentations
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.numpics / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of images and mask

        imgs = [self.images[i] for i in indexes]
        mask = [self.masks[i] for i in indexes]

        # Generate data
        image, mask = self.image_pair_mask(imgs, mask)

        return image, mask

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.numpics)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def apply_augmentations(self, image, mask):
        nwimage, nmask = self.augmentations(image, mask)
        return nwimage, nmask

    def image_pair_mask(self, images, masks):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        pictures = np.empty((self.batch_size, self.img_size, self.img_size, self.channels), dtype=np.float32)
        labels = np.empty((self.batch_size, self.img_size, self.img_size, self.outmask), dtype=np.float32)
        # Generate data
        for idx, (im, mk) in enumerate(zip(images, masks)):
            # Store sample
            pic = Image.open(im)
            pic = np.array(pic.resize((self.img_size, self.img_size)))
            lab = Image.open(mk)
            lab = np.array(lab.resize((self.img_size, self.img_size)))
            if self.augmentations is not None:
                pic, lab = self.apply_augmentations(pic, lab)
            if not self.binmask:
                lab = tf.keras.utils.to_categorical(lab, num_classes=self.n_classes)

            pictures[idx, :, :, :] = pic
            labels[idx, :, :, :] = lab

        return pictures, labels
