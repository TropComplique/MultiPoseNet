import tensorflow as tf
from .utils import conv2d_same, batch_norm_relu
from .constants import DATA_FORMAT


def fpn(features, is_training, depth, min_level=3, add_coarse_features=True, scope='fpn'):
    """
    For person detector subnetwork we
    use min_level=3 and add_coarse_features=True
    (like in the original retinanet paper).

    For keypoint detector subnetwork we
    use min_level=2 and add_coarse_features=False
    (like in the original multiposenet paper).

    Arguments:
        features: a dict with four float tensors.
            It must have keys ['c2', 'c3', 'c4', 'c5'].
            Where a number means that a feature has stride `2**number`.
        is_training: a boolean.
        depth: an integer.
        min_level: an integer, minimal feature stride
            that will be used is `2**min_level`.
            Possible values are [2, 3, 4, 5]
        add_coarse_features: a boolean, whether to add
            features with strides 64 and 128.
        scope: a string.
    Returns:
        a dict with float tensors.
    """

    with tf.variable_scope(scope):

        x = conv2d_same(features['c5'], depth, kernel_size=1, name='lateral5')
        p5 = conv2d_same(x, depth, kernel_size=3, name='p5')
        enriched_features = {'p5': p5}

        if add_coarse_features:
            p6 = conv2d_same(features['c5'], depth, kernel_size=3, stride=2, name='p6')
            p7 = conv2d_same(tf.nn.relu(p6), depth, kernel_size=3, stride=2, name='p7')
            enriched_features.update({'p6': p6, 'p7': p7})

        # top-down path
        for i in reversed(range(min_level, 5)):
            i = str(i)
            lateral = conv2d_same(features['c' + i], depth, kernel_size=1, name='lateral' + i)
            x = nearest_neighbor_upsample(x, scope='upsampling' + i) + lateral
            p = conv2d_same(x, depth, kernel_size=3, name='p' + i)
            enriched_features['p' + i] = p

        enriched_features = {
            n: batch_norm_relu(x, is_training, use_relu=False, name=n + '_batch_norm')
            for n, x in enriched_features.items()
        }

    return enriched_features


def nearest_neighbor_upsample(x, rate=2, scope='upsampling'):
    with tf.name_scope(scope):

        shape = tf.shape(x)
        batch_size = x.shape[0].value
        if batch_size is None:
            batch_size = shape[0]

        if DATA_FORMAT == 'channels_first':
            channels = x.shape[1].value
            height, width = shape[2], shape[3]
            x = tf.reshape(x, [batch_size, channels, height, 1, width, 1])
            x = tf.tile(x, [1, 1, 1, rate, 1, rate])
            x = tf.reshape(x, [batch_size, channels, height * rate, width * rate])
        else:
            height, width = shape[1], shape[2]
            channels = x.shape[3].value
            x = tf.reshape(x, [batch_size, height, 1, width, 1, channels])
            x = tf.tile(x, [1, 1, rate, 1, rate, 1])
            x = tf.reshape(x, [batch_size, height * rate, width * rate, channels])

        return x
