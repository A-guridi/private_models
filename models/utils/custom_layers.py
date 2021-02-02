import tensorflow as tf


def swish(x):
    # swish activation function
    return tf.compat.v2.nn.swish(x)


def convblock(x, filters, names=None):
    """
    Creates a convolutional 2D block with the previous layer and given filters
    :param x: previous keras layer
    :param filters: list of two filters or int, number of filters for the convolution
    :return: output of the last layer convoluted
    """
    if not isinstance(filters, list):
        filters = [filters, filters]
    elif len(filters) != 2:
        raise ValueError("Please provide only two filters in a list or one filter value")
    if not isinstance(names, list) and names is not None:
        names = [names + "_1", names + "_2"]
    elif names is None:
        names = ["conv_block_1", "conv_block_2"]

    x = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3, name=names[0], padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99)(x)
    x = tf.keras.layers.Activation(tf.compat.v2.nn.swish)(x)

    x = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=3, name=names[1], padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99)(x)
    x = tf.keras.layers.Activation(tf.compat.v2.nn.swish)(x)

    return x


def atrous_convblock(x, filters, dilation=2, names=None):
    """
        Creates an atrous convolutional 2D block with the previous layer and given filters
        :param x: previous keras layer
        :param filters: list of two filters or int, number of filters for the convolution
        :param dilation: dilation for the atrous convolution
        :param names: names of the layers
        :return: output of the last layer convoluted
        """
    if not isinstance(filters, list):
        filters = [filters, filters]
    elif len(filters) != 2:
        raise ValueError("Please provide only two filters in a list or one filter value")
    if not isinstance(names, list) and names is not None:
        names = [names + "_1", names + "_2"]
    elif names is None:
        names = ["conv_block_1", "conv_block_2"]
        raise ValueError("Please provide only two filters in a list or one filter value")
    x = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3, dilation_rate=(dilation, dilation), padding="same", name=names[0])(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99)(x)
    x = tf.keras.layers.Activation(tf.compat.v2.nn.swish)(x)

    x = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=3, dilation_rate=(dilation, dilation), padding="same", name=names[1])(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99)(x)
    x = tf.keras.layers.Activation(tf.compat.v2.nn.swish)(x)

    return x
