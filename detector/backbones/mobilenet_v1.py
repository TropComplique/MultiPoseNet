import tensorflow.compat.v1 as tf
import tensorflow.contrib as contrib
import tensorflow.contrib.slim as slim
from detector.constants import DATA_FORMAT


BATCH_NORM_MOMENTUM = 0.95
BATCH_NORM_EPSILON = 1e-3


def mobilenet_v1(images, is_training, depth_multiplier=1.0):
    """
    This implementation works with checkpoints from here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md

    Arguments:
        images: a float tensor with shape [b, h, w, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
        depth_multiplier: a float number, multiplier for the number of filters in a layer.
    Returns:
        a dict with four float tensors.
    """

    def depth(x):
        """Reduce the number of filters in a layer."""
        return max(int(x * depth_multiplier), 8)

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=1 if DATA_FORMAT == 'channels_first' else 3,
            center=True, scale=True,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            training=is_training, fused=True,
            name='BatchNorm'
        )
        return x

    with tf.name_scope('standardize_input'):
        x = (2.0 * images) - 1.0

    with tf.variable_scope('MobilenetV1'):
        params = {
            'padding': 'SAME',
            'activation_fn': tf.nn.relu6, 'normalizer_fn': batch_norm,
            'data_format': 'NCHW' if DATA_FORMAT == 'channels_first' else 'NHWC'
        }
        with slim.arg_scope([slim.conv2d, depthwise_conv], **params):
            features = {}

            if DATA_FORMAT == 'channels_first':
                x = tf.transpose(x, [0, 3, 1, 2])

            layer_name = 'Conv2d_0'
            x = slim.conv2d(x, depth(32), (3, 3), stride=2, scope=layer_name)
            features[layer_name] = x

            strides_and_filters = [
                (1, 64),
                (2, 128), (1, 128),
                (2, 256), (1, 256),
                (2, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512),
                (2, 1024), (1, 1024)
            ]
            for i, (stride, num_filters) in enumerate(strides_and_filters, 1):

                layer_name = 'Conv2d_%d_depthwise' % i
                x = depthwise_conv(x, stride=stride, scope=layer_name)
                features[layer_name] = x

                layer_name = 'Conv2d_%d_pointwise' % i
                x = slim.conv2d(x, depth(num_filters), (1, 1), stride=1, scope=layer_name)
                features[layer_name] = x

    return {
        'c2': features['Conv2d_3_pointwise'], 'c3': features['Conv2d_5_pointwise'],
        'c4': features['Conv2d_11_pointwise'], 'c5': features['Conv2d_13_pointwise']
    }


@contrib.framework.add_arg_scope
def depthwise_conv(
        x, kernel=3, stride=1, padding='SAME',
        activation_fn=None, normalizer_fn=None,
        data_format='NHWC', scope='depthwise_conv'):
    with tf.variable_scope(scope):

        if data_format == 'NHWC':
            in_channels = x.shape[3].value
            strides = [1, stride, stride, 1]
        else:
            in_channels = x.shape[1].value
            strides = [1, 1, stride, stride]

        W = tf.get_variable(
            'depthwise_weights',
            [kernel, kernel, in_channels, 1],
            dtype=tf.float32
        )
        x = tf.nn.depthwise_conv2d(x, W, strides, padding, data_format=data_format)
        x = normalizer_fn(x) if normalizer_fn is not None else x  # batch normalization
        x = activation_fn(x) if activation_fn is not None else x  # nonlinearity
        return x
