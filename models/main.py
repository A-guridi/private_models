from .segmentation.trainer import trainer
import matplotlib.pyplot as plt
from .utils import infer_num_classes

# data dir
dir = ""

# model
model = "Unet"
backbone = "resnet18"
img_size = 40
num_classes = infer_num_classes(dir)
batch_size = 2
epochs = 2

# create the dataset


mod = trainer(model, backbone=backbone, im_size=img_size,
              num_classes=num_classes, batch_size=batch_size, epochs=epochs)
