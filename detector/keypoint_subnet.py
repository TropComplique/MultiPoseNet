import tensorflow as tf
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

        self.enriched_features = fpn(
            features, is_training, depth, min_level=2,
            add_coarse_features=False, scope='keypoint_fpn'
        )
        # it is a dict with keys ['p2', 'p3', 'p4', 'p5']

        upsampled_features = []
        with tf.variable_scope('phi_subnet', reuse=tf.AUTO_REUSE):
            for level in range(2, 6):
                x = self.enriched_features['p' + str(level)]
                upsample = 2**(2 - level)
                upsampled_features.append(phi_subnet(x, is_training, depth, upsample))

        upsampled_features = tf.concat(upsampled_features, axis=1 if DATA_FORMAT == 'channels_first' else 3)

        x = conv2d_same(x, depth, kernel_size=3, name='final_conv3x3')
        x = batch_norm_relu(x, is_training, name='final_bn')

        p = 0.01  # probability of foreground
        # sigmoid(-log((1 - p) / p)) = p

        self.heatmaps = tf.layers.conv2d(
            x, K + 1, kernel_size=(1, 1), padding='same',
            bias_initializer=tf.constant_initializer(-math.log((1.0 - p) / p)),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            data_format=DATA_FORMAT, name='heatmaps'
        )


    def get_predictions(self, score_threshold=0.05, iou_threshold=0.5, max_detections=20):
        """Postprocess outputs of the network.

        Returns:
            boxes: a float tensor with shape [batch_size, N, 4].
            scores: a float tensor with shape [batch_size, N].
            num_boxes: an int tensor with shape [batch_size], it
                represents the number of detections on an image.

            Where N = max_detections.
        """
        with tf.name_scope('postprocessing'):

            encoded_boxes = self.raw_predictions['encoded_boxes']
            # it has shape [batch_size, num_anchors, 4]

            class_predictions = self.raw_predictions['class_predictions']
            scores = tf.sigmoid(class_predictions)
            # it has shape [batch_size, num_anchors]

        with tf.name_scope('nms'):
            boxes, scores, num_detections = batch_non_max_suppression(
                encoded_boxes, self.anchors, scores, score_threshold=score_threshold,
                iou_threshold=iou_threshold, max_detections=max_detections
            )
        return {'boxes': boxes, 'scores': scores, 'num_boxes': num_detections}

    def loss(self, groundtruth_heatmaps, params):
        """Compute scalar loss tensors with respect to provided groundtruth.

        Arguments:
            groundtruth_heatmaps: a float tensor with shape [batch_size, max_num_boxes, 4].
            params: a dict with parameters.
        Returns:
            two float tensors with shape [].
        """
        regression_targets, matches = self._create_targets(groundtruth)

        with tf.name_scope('losses'):

            # whether an anchor contains something
            is_matched = tf.to_float(tf.greater_equal(matches, 0))

            not_ignore = tf.to_float(tf.greater_equal(matches, -1))
            # if a value is `-2` then we ignore its anchor

            with tf.name_scope('classification_loss'):

                class_predictions = self.raw_predictions['class_predictions']
                # shape [batch_size, num_anchors]

                cls_losses = focal_loss(
                    class_predictions, is_matched, weights=not_ignore,
                    gamma=params['gamma'], alpha=params['alpha']
                )  # shape [batch_size, num_anchors]

                cls_loss = tf.reduce_sum(cls_losses, axis=[0, 1])

            with tf.name_scope('localization_loss'):

                encoded_boxes = self.raw_predictions['encoded_boxes']
                # it has shape [batch_size, num_anchors, 4]

                loc_losses = localization_loss(
                    encoded_boxes, regression_targets,
                    weights=is_matched
                )  # shape [batch_size, num_anchors]

                loc_loss = tf.reduce_sum(loc_losses, axis=[0, 1])

            with tf.name_scope('normalization'):
                matches_per_image = tf.reduce_sum(is_matched, axis=1)  # shape [batch_size]
                num_matches = tf.reduce_sum(matches_per_image)  # shape []
                normalizer = tf.maximum(num_matches, 1.0)

            return {'localization_loss': loc_loss/normalizer, 'classification_loss': cls_loss/normalizer}


def phi_subnet(x, is_training, depth, upsample):

    x = conv2d_same(x, depth, kernel_size=3, name='conv1')
    x = batch_norm_relu(x, is_training, name='bn1')
    x = conv2d_same(x, depth, kernel_size=3, name='conv2')
    x = batch_norm_relu(x, is_training, name='bn2)

    shape = tf.shape(x)
    if DATA_FORMAT == 'channels_first':
        height, width = shape[2], shape[3]
    else:
        height, width = shape[1], shape[2]

    new_size = [upsample * height, upsample * width]
    x = tf.image.resize_bilinear(x, new_size, align_corners=True)
    return x
