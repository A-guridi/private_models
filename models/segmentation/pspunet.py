import tensorflow as tf
import segmentation_models as sm
from segmentation_models.backbones.backbones_factory import Backbones
from segmentation_models.models._utils import freeze_model, get_submodules_from_kwargs
from segmentation_models.models.pspnet import SpatialContextBlock
from private_models.models.utils.custom_layers import convblock, atrous_convblock
from classification_models.tfkeras import Classifiers

backend = None
layers = None
models = None
keras_utils = None


# copied from qubvel, weirdly not downloaded in package
def filter_keras_submodules(kwargs):
    """Selects only arguments that define keras_application submodules. """
    submodule_keys = kwargs.keys() & {'backend', 'layers', 'models', 'utils'}
    return {key: kwargs[key] for key in submodule_keys}


def build_PSP(x, conv_filters=512, pooling_type="avg", use_batchnorm=True):
    x1 = SpatialContextBlock(1, conv_filters, pooling_type, use_batchnorm)(x)
    x2 = SpatialContextBlock(2, conv_filters, pooling_type, use_batchnorm)(x)
    x3 = SpatialContextBlock(3, conv_filters, pooling_type, use_batchnorm)(x)
    x6 = SpatialContextBlock(6, conv_filters, pooling_type, use_batchnorm)(x)

    x = layers.Concatenate(axis=3, name="psp_concat")([x1, x2, x3, x6])
    x = layers.Conv1D(filters=conv_filters, kernel_size=1, activation="relu")(x)

    return x


def decode_up(x, skip, nfilters):
    # upsamples x2 the layer and adds the skip connection with the next layer
    x = convblock(x, nfilters)
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate(axis=3)([x, skip])
    x = convblock(x, nfilters)
    x = layers.Dropout(0.3)(x)
    return x


def decode_atrous(x, skip, nfilters):
    # upsamples x2 the layer and adds the skip connection with the next layer
    # adds the atrous convolution
    x = atrous_convblock(x, nfilters)
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate(axis=3)([x, skip])
    x = atrous_convblock(x, nfilters)
    x = layers.Dropout(0.3)(x)
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
    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)
    if classes == 2:
        activation = "sigmoid"
        outchannels = 1
    else:
        outchannels = classes
        activation = "softmax"
    # backbone = Backbones.get_backbone(
    #     backbone_name,
    #     input_shape=in_shape,
    #     weights=encoder_weights,
    #     include_top=False,
    #     **kwargs
    # )
    backbone_model, preproccess_input = Classifiers.get(backbone_name)
    backbone = backbone_model(input_shape=in_shape, weights=encoder_weights, include_top=False)
    #if encoder_freeze:
    #    freeze_model(backbone)
    if encoder_features == "default":
        encoder_features = Backbones.get_feature_layers(backbone_name)

    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = [backbone.get_layer(name=i).output if isinstance(i, str)
             else backbone.get_layer(index=i).output for i in encoder_features]

    x = build_PSP(x)

    x = layers.Dropout(0.3)(x)

    # start the upsampling
    if "vgg" in backbone_name:
        # in the vgg case, we need to add an additional layer
        x = decode_up(x, skips[0], decoder_filters[0])
        skips = skips[1:]
    decoder_filters = decoder_filters[1:]

    for i in range(4):
        x = decode_up(x, skips[i], decoder_filters[i])

    # final layer
    x = layers.Conv2D(filters=outchannels, kernel_size=(3, 3), name="output")(x)
    output = layers.Activation(activation)(x)

    model = tf.keras.Model(input_, output)

    return model
