import tensorflow as tf
from .constants import PARALLEL_ITERATIONS, POSITIVES_THRESHOLD, NEGATIVES_THRESHOLD
from .utils import batch_non_max_suppression
from .training_target_creation import get_training_targets
from .box_predictor import retinanet_box_predictor
from .anchor_generator import AnchorGenerator
from .fpn import fpn


class RetinaNet:
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
            features, is_training, depth=128, min_level=3,
            add_coarse_features=True, scope='fpn'
        )
        enriched_features = {
            n: batch_norm_relu(x, is_training, use_relu=False, name=n + '_batch_norm')
            for n, x in enriched_features.items()
        }

        # the detector supports images of various sizes
        shape = tf.shape(images)
        image_height, image_width = shape[1], shape[2]

        anchor_generator = AnchorGenerator(
            strides=[8, 16, 32, 64, 128],
            scales=[32, 64, 128, 256, 512],
            scale_multipliers=[1.0, 1.4142],
            aspect_ratios=[1.0, 2.0, 0.5]
        )
        self.anchors = anchor_generator(image_height, image_width)  # shape [num_anchors, 4]
        num_anchors_per_location = anchor_generator.num_anchors_per_location

        self.raw_predictions = retinanet_box_predictor(
            [enriched_features['p' + str(i)] for i in range(3, 8)],
            is_training, num_anchors_per_location=num_anchors_per_location,
            depth=64, min_level=3
        )
        # it returns a dict with two float tensors:
        # `encoded_boxes` has shape [batch_size, num_anchors, 4],
        # `class_predictions` has shape [batch_size, num_anchors]

    def get_predictions(self, score_threshold=0.05, iou_threshold=0.5, max_detections=25):
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

    def loss(self, groundtruth, params):
        """Compute scalar loss tensors with respect to provided groundtruth.

        Arguments:
            groundtruth: a dict with the following keys
                'boxes': a float tensor with shape [batch_size, max_num_boxes, 4].
                'num_boxes': an int tensor with shape [batch_size],
                    where max_num_boxes = max(num_boxes).
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

    def _create_targets(self, groundtruth):
        """
        Arguments:
            groundtruth: a dict with the following keys
                'boxes': a float tensor with shape [batch_size, N, 4].
                'num_boxes': an int tensor with shape [batch_size].
        Returns:
            regression_targets: a float tensor with shape [batch_size, num_anchors, 4].
            matches: an int tensor with shape [batch_size, num_anchors],
                `-1` means that an anchor box is negative (background),
                and `-2` means that we must ignore this anchor box.
        """
        def fn(x):
            boxes, num_boxes = x
            boxes = boxes[:num_boxes]

            regression_targets, matches = get_training_targets(
                self.anchors, boxes,
                positives_threshold=POSITIVES_THRESHOLD,
                negatives_threshold=NEGATIVES_THRESHOLD
            )
            return regression_targets, matches

        with tf.name_scope('target_creation'):
            regression_targets, matches = tf.map_fn(
                fn, [groundtruth['boxes'], groundtruth['num_boxes']],
                dtype=(tf.float32, tf.int32),
                parallel_iterations=PARALLEL_ITERATIONS,
                back_prop=False, swap_memory=False, infer_shape=True
            )
            return regression_targets, matches


def localization_loss(predictions, targets, weights):
    """A usual L1 smooth loss.

    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, 4],
            representing the (encoded) predicted locations of objects.
        targets: a float tensor with shape [batch_size, num_anchors, 4],
            representing the regression targets.
        weights: a float tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    abs_diff = tf.abs(predictions - targets)
    abs_diff_lt_1 = tf.less(abs_diff, 1.0)
    loss = tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5)
    return weights * tf.reduce_sum(loss, axis=2)


def focal_loss(predictions, targets, weights, gamma=2.0, alpha=0.25):
    """
    Here it is assumed that there is only one class.

    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors],
            representing the predicted logits.
        targets: a float tensor with shape [batch_size, num_anchors],
            representing binary classification targets.
        weights: a float tensor with shape [batch_size, num_anchors].
        gamma, alpha: float numbers.
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    positive_label_mask = tf.equal(targets, 1.0)

    negative_log_p_t = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=predictions)
    probabilities = tf.sigmoid(predictions)
    p_t = tf.where(positive_label_mask, probabilities, 1.0 - probabilities)
    # they all have shape [batch_size, num_anchors]

    modulating_factor = tf.pow(1.0 - p_t, gamma)
    weighted_loss = tf.where(
        positive_label_mask,
        alpha * negative_log_p_t,
        (1.0 - alpha) * negative_log_p_t
    )
    focal_loss = modulating_factor * weighted_loss
    # they all have shape [batch_size, num_anchors]

    return weights * focal_loss
