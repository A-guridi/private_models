import tensorflow as tf
from tensorflow.keras import layers
from segmentation_models import PSPNet
from segmentation_models.backbones.backbones_factory import Backbones
from segmentation_models.models._utils import freeze_model, get_submodules_from_kwargs
from segmentation_models.models.pspnet import SpatialContextBlock
from private_models.models.utils.custom_layers import convblock, atrous_convblock

# copied from qubvel, weirdly not downloaded in package
def filter_keras_submodules(kwargs):
    """Selects only arguments that define keras_application submodules. """
    submodule_keys = kwargs.keys() & {'backend', 'layers', 'models', 'utils'}
    return {key: kwargs[key] for key in submodule_keys}


def build_PSP(x, conv_filters=512, pooling_type="avg", use_batchnorm=True):
    # old function to manually implement the PSP net
    x1 = SpatialContextBlock(1, conv_filters, pooling_type, use_batchnorm)(x)
    x2 = SpatialContextBlock(2, conv_filters, pooling_type, use_batchnorm)(x)
    x3 = SpatialContextBlock(3, conv_filters, pooling_type, use_batchnorm)(x)
    x6 = SpatialContextBlock(6, conv_filters, pooling_type, use_batchnorm)(x)

    x = layers.Concatenate(axis=3, name="psp_concat")([x1, x2, x3, x6])
    x = layers.Conv1D(filters=conv_filters, kernel_size=1, activation="relu")(x)

    return x


def decode_up(x, skip, nfilters):
    # upsamples x2 the layer and adds the skip connection with the next layer
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate(axis=3)([x, skip])
    x = convblock(x, nfilters)
    return x


def decode_atrous(x, skip, nfilters):
    # upsamples x2 the layer and adds the skip connection with the next layer
    # adds the atrous convolution
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate(axis=3)([x, skip])
    x = atrous_convblock(x, nfilters)
    return x


def build_pspunet(backbone_name='resnet50',
                  in_shape=(96, 96, 3),
                  classes=1,
                  encoder_weights='imagenet',
                  encoder_freeze=False,
                  encoder_features='default',
                  decoder_block_type='upsampling',
                  decoder_filters=(256, 128, 64, 32, 16),
                  decoder_use_batchnorm=True,
                  **kwargs
                  ):
    """ Custom Net from combining Unet and a PSP convolutional block, together with more blocks
    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        in_shape: shape of input data/image ``(H, W, C)``, in general
            case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
            able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
            (e.g. ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
            Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
            layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
        decoder_block_type: one of blocks with following layers structure:
            - `upsampling`:  ``UpSampling2D`` -> ``Conv2D`` -> ``Conv2D``
            - `transpose`:   ``Transpose2D`` -> ``Conv2D``
        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
    Returns:
        ``keras.models.Model``: **PSPUnet**
    """
    # added lines from the quvbel package
    if classes == 2:
        activation = "sigmoid"
        outchannels = 1
    else:
        outchannels = classes
        activation = "softmax"

    model = PSPNet(backbone_name=backbone_name,
                   input_shape=in_shape,
                   classes=classes,
                   activation=activation,
                   encoder_weights='imagenet',
                   encoder_freeze=encoder_freeze,
                   downsample_factor=16,
                   psp_conv_filters=512,
                   psp_pooling_type='avg',
                   psp_use_batchnorm=True,
                   psp_dropout=None)

    # extract the skip connections layers
    fet_layers = Backbones.get_feature_layers(backbone_name)
    skips = [0] * len(fet_layers)
    for idx, l in enumerate(fet_layers):
        if isinstance(l, str):
            skips[idx] = model.get_layer(name=l).output
        else:
            skips[idx] = model.get_layer(index=l).output

    # start the upsampling
    if "vgg" in backbone_name:
        # in the vgg case, we need to add an additional layer
        skip_0 = skips[4]
    else:
        skip_0 = convblock(model.input, filters=32, names="encoder-384")

    # we start building the net from bottom to top, being each layer bigger and applying 2 convolutions after
    # each concatenation

    pspout = convblock(skips[0], filters=512)

    dec_24 = layers.UpSampling2D((2, 2), name="up_24")(pspout)
    dec_24 = layers.Concatenate(name="conc_24")([dec_24, skips[1]])
    dec_24 = convblock(dec_24, filters=decoder_filters[0], names="conv_24")

    dec_48 = layers.UpSampling2D((2, 2), name="up_48")(dec_24)
    dec_48 = layers.Concatenate(name="conc_48")([skips[2], dec_48])
    # dec_48 = convblock(dec_48, filters=decoder_filters[1], names="conv_48")
    dec_48 = atrous_convblock(dec_48, filters=decoder_filters[1], names="conv_48")

    # test if dilated conv really offers better results
    dec_96 = layers.UpSampling2D((2, 2), name="up_96")(dec_48)
    dec_96 = layers.Concatenate(name="conc_96")([skips[3], dec_96])
    # dec_96 = convblock(dec_96, filters=decoder_filters[2], names="conv_96")
    dec_96 = atrous_convblock(dec_96, filters=decoder_filters[2], names="conv_96")

    dec_192 = layers.UpSampling2D((2, 2), name="up_192")(dec_96)
    dec_192 = layers.Concatenate(name="conc_192")([skip_0, dec_192])
    dec_192 = convblock(dec_192, filters=decoder_filters[3], names="conv_192")

    # output layer
    out_layer = layers.Conv2D(filters=outchannels, kernel_size=3, padding="same", name="output_conv")(dec_192)
    out_layer = layers.Activation(activation, name="out_act")(out_layer)

    model = tf.keras.Model(inputs=model.input, outputs=out_layer)

    return model
