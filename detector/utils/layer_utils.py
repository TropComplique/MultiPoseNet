import tensorflow.compat.v1 as tf
from detector.constants import DATA_FORMAT


BATCH_NORM_MOMENTUM = 0.95
BATCH_NORM_EPSILON = 1e-3


def batch_norm_relu(x, is_training, use_relu=True, name=None):
    x = tf.layers.batch_normalization(
        inputs=x, axis=1 if DATA_FORMAT == 'channels_first' else 3,
        momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPSILON,
        center=True, scale=True, training=is_training,
        fused=True, name=name
    )
    return x if not use_relu else tf.nn.relu(x)


def conv2d_same(x, num_filters, kernel_size=3, stride=1, name=None):

    assert kernel_size in [1, 3]
    assert stride in [1, 2]

    if kernel_size == 3:

        if DATA_FORMAT == 'channels_first':
            paddings = [[0, 0], [0, 0], [1, 1], [1, 1]]
        else:
            paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]

        x = tf.pad(x, paddings)

    return tf.layers.conv2d(
        inputs=x, filters=num_filters,
        kernel_size=kernel_size, strides=stride,
        padding='valid', use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=DATA_FORMAT, name=name
    )
