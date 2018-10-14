import tensorflow as tf
import math
from .constants import NUM_KEYPOINTS, DATA_FORMAT
from .utils import batch_norm_relu, conv2d_same
from .fpn import fpn


class KeypointSubnet:
    def __init__(self, images, is_training, backbone, params):
        """
        Arguments:
            images: a float tensor with shape [batch_size, height, width, 3],
                a batch of RGB images with pixel values in the range [0, 1].
            is_training: a boolean.
            backbone: it takes a batch of images and returns a dict of features.
            params: a dict.
        """

        # this is a network like resnet or mobilenet
        features = backbone(images, is_training)

        enriched_features = fpn(
            features, is_training, depth=128, min_level=2,
            add_coarse_features=False, scope='keypoint_fpn'
        )
        normalized_enriched_features = {
            n: batch_norm_relu(x, is_training, use_relu=False, name=n + '_batch_norm')
            for n, x in enriched_features.items()
        }
        # it is a dict with keys ['p2', 'p3', 'p4', 'p5']

        upsampled_features = []
        with tf.variable_scope('phi_subnet', reuse=tf.AUTO_REUSE):
            for level in range(2, 6):
                x = normalized_enriched_features['p' + str(level)]
                upsample = 2**(level - 2)
                upsampled_features.append(phi_subnet(x, is_training, upsample, depth=128))

        upsampled_features = tf.concat(upsampled_features, axis=1 if DATA_FORMAT == 'channels_first' else 3)
        x = conv2d_same(upsampled_features, 64, kernel_size=3, name='final_conv3x3')
        x = batch_norm_relu(x, is_training, name='final_bn')

        p = 0.01  # probability of foreground
        # sigmoid(-log((1 - p) / p)) = p

        heatmaps = tf.layers.conv2d(
            x, NUM_KEYPOINTS + 1, kernel_size=(1, 1), padding='same',
            bias_initializer=tf.constant_initializer(-math.log((1.0 - p) / p)),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            data_format=DATA_FORMAT, name='heatmaps'
        )

        if DATA_FORMAT == 'channels_first':
            self.heatmaps = tf.transpose(heatmaps, [0, 2, 3, 1])
            self.enriched_features = {
                n: tf.transpose(x, [0, 2, 3, 1])
                for n, x in enriched_features.items()
            }
        else:
            self.heatmaps = heatmaps
            self.enriched_features = enriched_features

    def get_predictions(self):
        """
        Returns:
            heatmaps: a float tensor with shape [batch_size, h, w, 17].
            segmentation_masks: a float tensor with shape [batch_size, h, w].

            Where (h, w) = (image_height/DOWNSAMPLE, image_width/DOWNSAMPLE).
        """
        with tf.name_scope('postprocessing'):
            heatmaps = self.heatmaps[:, :, :, :18]
            segmentation_masks = self.heatmaps[:, :, :, 18]
        return {'heatmaps': heatmaps, 'segmentation_masks': segmentation_masks}


def phi_subnet(x, is_training, upsample, depth=128):

    x = conv2d_same(x, depth, kernel_size=3, name='conv1')
    x = batch_norm_relu(x, is_training, name='bn1')
    x = conv2d_same(x, depth, kernel_size=3, name='conv2')
    x = batch_norm_relu(x, is_training, name='bn2')

    if DATA_FORMAT == 'channels_first':
        x = tf.transpose(x, [0, 2, 3, 1])

    shape = tf.shape(x)
    height, width = shape[1], shape[2]
    new_size = [upsample * height, upsample * width]
    x = tf.image.resize_bilinear(x, new_size, align_corners=True)

    if DATA_FORMAT == 'channels_first':
        x = tf.transpose(x, [0, 3, 1, 2])

    return x
