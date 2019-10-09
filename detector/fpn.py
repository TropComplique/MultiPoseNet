import tensorflow.compat.v1 as tf
from detector.utils import conv2d_same, batch_norm_relu
from detector.constants import DATA_FORMAT


def feature_pyramid_network(
        features, is_training, depth,
        min_level=3, add_coarse_features=True,
        scope='fpn'):
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
            Where a number in a name means that
            a feature has stride `2 ** number`.
        is_training: a boolean.
        depth: an integer.
        min_level: an integer, minimal feature stride
            that will be used is `2 ** min_level`.
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
            p7 = conv2d_same(batch_norm_relu(p6, name='pre_p7_bn'), depth, kernel_size=3, stride=2, name='p7')
            enriched_features.update({'p6': p6, 'p7': p7})

        # top-down path
        for i in reversed(range(min_level, 5)):
            lateral = conv2d_same(features[f'c{i}'], depth, kernel_size=1, name=f'lateral{i}')
            x = nearest_neighbor_upsample(x) + lateral
            p = conv2d_same(x, depth, kernel_size=3, name=f'p{i}')
            enriched_features['p' + i] = p

    return enriched_features


def nearest_neighbor_upsample(x):
    """
    Arguments:
        x: a float tensor with shape [b, h, w, c].
    Returns:
        a float tensor with shape [b, 2 * h, 2 * w, c].
    """

    if DATA_FORMAT == 'channels_first':
        x = tf.transpose(x, [0, 2, 3, 1])

    shape = tf.shape(x)
    h, w = shape[1], shape[2]
    x = tf.image.resize_nearest_neighbor(x, [2 * h, 2 * w])

    if DATA_FORMAT == 'channels_first':
        x = tf.transpose(x, [0, 3, 1, 2])

    return x
