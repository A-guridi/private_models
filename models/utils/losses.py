import numpy as np
import segmentation_models as sm
from tensorflow.keras import backend as K


def slicer(x, indexes):
    """
    Slice a tensor among the given indexes for a channel last
    :param x: tensor
    :param indexes: index of the class to be sliced
    :return:
    """
    x = K.permute_dimensions(x, (3, 0, 1, 2))
    x = K.gather(x, indexes)
    x = K.permute_dimensions(x, (1, 2, 3, 0))
    return x


def CatCrosEntr(weights=1, class_index=None):
    """
    Categorical Cross Entropy Loss
    :param gt: y_true
    :param pred: y_pred
    :param weights: if given, multiplied by the loss, bust be of the same shape as y_true
    :param class_index: the class to be calculated
    :return:
    """
    def lossfunc(gt, pred):
        gt = slicer(gt, indexes=class_index)
        pred = slicer(pred, indexes=class_index)

        pred /= K.sum(pred, axis=3, keepdims=True)

        loss = gt * K.log(pred) * weights
        return K.mean(loss)

    return lossfunc
