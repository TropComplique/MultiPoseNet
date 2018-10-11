import tensorflow as tf
from detector.constants import DATA_FORMAT
from detector.utils import conv2d_same, batch_norm_relu


def resnet(images, is_training):
    """
    This is an implementation of classical ResNet-50.
    It is taken from here:
    https://github.com/tensorflow/models/blob/master/official/resnet/

    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
    Returns:
        a dict with four float tensors.
    """
    features = {}

    with tf.name_scope('standardize_input'):
        means = tf.constants([123.68, 116.78, 103.94], dtype=tf.float32)
        x = (255.0 * images) - means

    with tf.variable_scope('ResNet-50'):

        if DATA_FORMAT == 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])

        x = conv2d_same(x, 64, kernel_size=7, stride=2)
        x = batch_norm_relu(x, is_training)
        x = tf.layers.max_pooling2d(
            inputs=x, pool_size=3, strides=2,
            padding='same', data_format=DATA_FORMAT
        )

        num_units_per_block = [3, 4, 6, 3]
        strides = [1, 2, 2, 2]

        for i, num_units, stride in enumerate(zip(num_units_per_block, strides)):
            num_filters = 64 * (2**i)  # [64, 128, 256, 512]
            x = block(x, is_training, num_filters, num_units, stride)
            features['c' + str(i + 2)] = x
            # `2**(i + 2)` is stride of the feature `x`

    return features


def block(x, is_training, num_filters, num_units, stride):
    x = bottleneck(x, is_training, num_filters, stride, use_projection=True)
    for _ in range(1, num_units):
        x = bottleneck(x, is_training, num_filters)
    return x


def bottleneck(x, is_training, num_filters, stride=1, rate=1, use_projection=False):

    shortcut = x
    num_output_channels = 4 * num_filters

    if use_projection:
        shortcut = conv2d_same(shortcut, num_output_channels, kernel_size=1, stride=stride)
        shortcut = batch_norm_relu(shortcut, is_training, use_relu=False)

    x = conv2d_same(x, num_filters, kernel_size=1)
    x = batch_norm_relu(x, is_training)

    x = conv2d_same(x, num_filters, kernel_size=3, stride=stride, rate=rate)
    x = batch_norm_relu(x, is_training)

    x = conv2d_same(x, num_output_channels, kernel_size=1)
    x = batch_norm_relu(x, is_training, use_relu=False)

    x += shortcut
    x = tf.nn.relu(x)
    return x
