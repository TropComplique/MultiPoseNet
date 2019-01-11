import math
import tensorflow as tf
from .constants import DATA_FORMAT
from .utils import batch_norm_relu, conv2d_same


def retinanet_box_predictor(
        image_features, is_training,
        num_anchors_per_location=6,
        depth=256, min_level=3):
    """
    Adds box predictors to each feature map,
    reshapes, and returns concatenated results.

    Arguments:
        image_features: a list of float tensors where the ith tensor
            has shape [batch_size, channels_i, height_i, width_i].
        is_training: a boolean.
        num_anchors_per_location, depth, min_level: integers.
    Returns:
        encoded_boxes: a float tensor with shape [batch_size, num_anchors, 4].
        class_predictions: a float tensor with shape [batch_size, num_anchors].
    """

    encoded_boxes = []
    class_predictions = []

    """
    The convolution layers in the box net are shared among all levels, but
    each level has its batch normalization to capture the statistical
    difference among different levels. The same for the class net.
    """

    with tf.variable_scope('box_net', reuse=tf.AUTO_REUSE):
        for level, p in enumerate(image_features, min_level):
            encoded_boxes.append(box_net(
                p, is_training, depth, level,
                num_anchors_per_location
            ))

    with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
        for level, p in enumerate(image_features, min_level):
            class_predictions.append(class_net(
                p, is_training, depth, level,
                num_anchors_per_location
            ))

    return reshape_and_concatenate(
        encoded_boxes, class_predictions,
        num_anchors_per_location
    )


def reshape_and_concatenate(
        encoded_boxes, class_predictions,
        num_anchors_per_location):

    # batch size is a static value
    # during training and evaluation
    batch_size = encoded_boxes[0].shape[0].value
    if batch_size is None:
        batch_size = tf.shape(encoded_boxes[0])[0]

    # it is important that reshaping here is the same as when anchors were generated
    with tf.name_scope('reshaping_and_concatenation'):
        for i in range(len(encoded_boxes)):

            # get spatial dimensions
            shape = tf.shape(encoded_boxes[i])
            if DATA_FORMAT == 'channels_first':
                height_i, width_i = shape[2], shape[3]
            else:
                height_i, width_i = shape[1], shape[2]

            # total number of anchors
            num_anchors_on_feature_map = height_i * width_i * num_anchors_per_location

            y = encoded_boxes[i]
            y = tf.transpose(y, perm=[0, 2, 3, 1]) if DATA_FORMAT == 'channels_first' else y
            y = tf.reshape(y, [batch_size, height_i, width_i, num_anchors_per_location, 4])
            encoded_boxes[i] = tf.reshape(y, [batch_size, num_anchors_on_feature_map, 4])

            y = class_predictions[i]
            y = tf.transpose(y, perm=[0, 2, 3, 1]) if DATA_FORMAT == 'channels_first' else y
            y = tf.reshape(y, [batch_size, height_i, width_i, num_anchors_per_location])
            class_predictions[i] = tf.reshape(y, [batch_size, num_anchors_on_feature_map])

        encoded_boxes = tf.concat(encoded_boxes, axis=1)
        class_predictions = tf.concat(class_predictions, axis=1)

    return {'encoded_boxes': encoded_boxes, 'class_predictions': class_predictions}


def class_net(x, is_training, depth, level, num_anchors_per_location):
    """
    Arguments:
        x: a float tensor with shape [batch_size, depth, height, width].
        is_training: a boolean.
        depth, level, num_anchors_per_location: integers.
    Returns:
        a float tensor with shape [batch_size, num_anchors_per_location, height, width].
    """

    for i in range(4):
        x = conv2d_same(x, depth, kernel_size=3, name='conv3x3_%d' % i)
        x = batch_norm_relu(x, is_training, name='batch_norm_%d_for_level_%d' % (i, level))

    p = 0.01  # probability of foreground
    # sigmoid(-log((1 - p) / p)) = p

    logits = tf.layers.conv2d(
        x, num_anchors_per_location,
        kernel_size=(3, 3), padding='same',
        bias_initializer=tf.constant_initializer(-math.log((1.0 - p) / p)),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        data_format=DATA_FORMAT, name='logits'
    )
    return logits


def box_net(x, is_training, depth, level, num_anchors_per_location):
    """
    Arguments:
        x: a float tensor with shape [batch_size, depth, height, width].
        is_training: a boolean.
        depth, level, num_anchors_per_location: integers.
    Returns:
        a float tensor with shape [batch_size, 4 * num_anchors_per_location, height, width].
    """

    for i in range(4):
        x = conv2d_same(x, depth, kernel_size=3, name='conv3x3_%d' % i)
        x = batch_norm_relu(x, is_training, name='batch_norm_%d_for_level_%d' % (i, level))

    encoded_boxes = tf.layers.conv2d(
        x, 4 * num_anchors_per_location,
        kernel_size=(3, 3), padding='same',
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        data_format=DATA_FORMAT, name='encoded_boxes'
    )
    return encoded_boxes
