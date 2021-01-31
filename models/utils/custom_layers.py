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
    if type(filters) != list:
        filters = [filters, filters]
    elif len(filters) != 2:
        raise ValueError("Please provide only two filters in a list or one filter value")
    x = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3, name=names[0])(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99)(x)
    x = tf.keras.layers.Activation(tf.compat.v2.nn.swish)(x)

    x = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=3, name=names[1])(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99)(x)
    x = tf.keras.layers.Activation(tf.compat.v2.nn.swish)(x)

    return x


def atrous_convblock(x, filters, dilation=2):
    """
        Creates an atrous convolutional 2D block with the previous layer and given filters
        :param x: previous keras layer
        :param filters: list of two filters or int, number of filters for the convolution
        :param dilation: dilation for the atrous convolution
        :return: output of the last layer convoluted
        """
    if type(filters) != list:
        filters = [filters, filters]
    elif len(filters) != 2:
        raise ValueError("Please provide only two filters in a list or one filter value")
    x = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3, dilation_rate=(dilation, dilation))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99)(x)
    x = tf.keras.layers.Activation(tf.compat.v2.nn.swish)(x)

    x = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=3, dilation_rate=(dilation, dilation))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99)(x)
    x = tf.keras.layers.Activation(tf.compat.v2.nn.swish)(x)

    return x
