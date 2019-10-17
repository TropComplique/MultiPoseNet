import tensorflow.compat.v1 as tf
from detector.constants import NUM_KEYPOINTS, DATA_FORMAT
from detector.utils import batch_norm_relu, conv2d_same
from detector.fpn import feature_pyramid_network


DEPTH = 128


class KeypointSubnet:
    def __init__(self, backbone_features, is_training, params):
        """
        Arguments:
            backbone_features: a dict with float tensors.
                It contains keys ['c2', 'c3', 'c4', 'c5'].
            is_training: a boolean.
            params: a dict.
        """

        self.enriched_features = feature_pyramid_network(
            backbone_features, is_training, depth=DEPTH, min_level=2,
            add_coarse_features=False, scope='keypoint_fpn'
        )
        normalized_enriched_features = {
            n: batch_norm_relu(x, is_training, name=f'{n}_batch_norm')
            for n, x in self.enriched_features.items()
        }
        # it is a dict with keys ['p2', 'p3', 'p4', 'p5']

        upsampled_features = []
        for level in range(2, 6):
            with tf.variable_scope(f'phi_subnet_{level}'):
                x = normalized_enriched_features[f'p{level}']
                y = phi_subnet(x, is_training, upsample=2**(level - 2))
                upsampled_features.append(y)

        upsampled_features = tf.concat(upsampled_features, axis=1 if DATA_FORMAT == 'channels_first' else 3)
        x = conv2d_same(upsampled_features, 64, kernel_size=3, name='final_conv3x3')
        x = batch_norm_relu(x, is_training, name='final_bn')

        self.heatmaps = tf.layers.conv2d(
            x, NUM_KEYPOINTS + 1, kernel_size=1, padding='same',
            bias_initializer=tf.constant_initializer(0.0),
            kernel_initializer=tf.random_normal_initializer(stddev=1e-4),
            data_format=DATA_FORMAT, name='heatmaps'
        )

        if DATA_FORMAT == 'channels_first':

            self.heatmaps = tf.transpose(self.heatmaps, [0, 2, 3, 1])
            self.enriched_features = {
                n: tf.transpose(x, [0, 2, 3, 1])
                for n, x in self.enriched_features.items()
            }


def phi_subnet(x, is_training, upsample):
    """
    Arguments:
        x: a float tensor with shape [b, h, w, c].
        is_training: a boolean.
        upsample: an integer.
    Returns:
        a float tensor with shape [b, upsample * h, upsample * w, depth].
    """

    x = conv2d_same(x, DEPTH, kernel_size=3, name='conv1')
    x = batch_norm_relu(x, is_training, name='bn1')
    x = conv2d_same(x, DEPTH, kernel_size=3, name='conv2')
    x = batch_norm_relu(x, is_training, name='bn2')

    if DATA_FORMAT == 'channels_first':
        x = tf.transpose(x, [0, 2, 3, 1])

    shape = tf.shape(x)
    h, w = shape[1], shape[2]
    new_size = [upsample * h, upsample * w]
    x = tf.image.resize_bilinear(x, new_size)

    if DATA_FORMAT == 'channels_first':
        x = tf.transpose(x, [0, 3, 1, 2])

    return x
