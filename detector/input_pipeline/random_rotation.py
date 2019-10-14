import tensorflow.compat.v1 as tf
import tensorflow.contrib as contrib
from detector.constants import DOWNSAMPLE, OVERLAP_THRESHOLD
from detector.input_pipeline.random_crop import prune_non_overlapping_boxes


def random_image_rotation(image, masks, boxes, keypoints, max_angle=45, probability=0.9):
    """
    What this function does:
    1. It takes a random box and rotates everything around its center.
    2. Then it rescales the image so that the box not too small or not too big.
    3. Then it translates the image's center to be at the box's center.

    All coordinates are absolute:
    1. Boxes have coordinates in ranges [0, height] and [0, width].
    2. Keypoints have coordinates in ranges [0, height - 1] and [0, width - 1].

    Arguments:
        image: a float tensor with shape [height, width, 3].
        masks: a float tensor with shape [height / DOWNSAMPLE, width / DOWNSAMPLE, 2].
        boxes: a float tensor with shape [num_persons, 4].
        keypoints: an int tensor with shape [num_persons, 17, 3].
        max_angle: an integer.
        probability: a float number.
    Returns:
        image: a float tensor with shape [height, width, 3].
        masks: a float tensor with shape [height / DOWNSAMPLE, width / DOWNSAMPLE, 2].
        boxes: a float tensor with shape [num_remaining_boxes, 4],
            where num_remaining_boxes <= num_persons.
        keypoints: an int tensor with shape [num_remaining_boxes, 17, 3].
    """
    def rotate(image, masks, boxes, keypoints):

        # get the center of the image
        image_shape = tf.to_float(tf.shape(image))
        image_height = image_shape[0]
        image_width = image_shape[1]
        image_center = 0.5 * tf.stack([image_height, image_width])
        image_center = tf.reshape(image_center, [1, 2])

        box_center, box_width = get_random_box_center(boxes, image_height, image_width)
        rotation = get_random_rotation(max_angle, box_center, image_width)
        scaler = get_random_scaling(box_center, box_width, image_width)

        rotation *= scaler
        translation = image_center - tf.matmul(box_center, rotation)

        """
        Assume tensor `points` has shape [n, 2].
        1. points = points - box_center (translate center of the coordinate system to the box center)
        2. points = points * rotation (rotate and scale relative to the new center)
        3. points = points + box_center (translate back)
        4. points = points - center_translation (translate image center to the box center)

        So full transformation is:
        (points - box_center) * rotation + box_center - center_translation =
        = points * rotation + translation, where translation = image_center - rotation * box_center.
        """

        boxes = transform_boxes(boxes, rotation, translation)
        keypoints = transform_keypoints(keypoints, rotation, translation)
        # after this some boxes and keypoints could be out of the image

        boxes, keypoints = correct(boxes, keypoints, image_height, image_width)
        # now all boxes and keypoints are inside the image

        transform = get_inverse_transform(rotation, translation)
        image = contrib.image.transform(image, transform, interpolation='BILINEAR')

        # masks are smaller than the image
        scaler = tf.stack([1, 1, DOWNSAMPLE, 1, 1, DOWNSAMPLE, 1, 1])
        masks_transform = transform / tf.to_float(scaler)

        masks = contrib.image.transform(masks, masks_transform, interpolation='NEAREST')
        # masks are binary so we use the nearest neighbor interpolation

        return image, masks, boxes, keypoints

    do_it = tf.less(tf.random_uniform([]), probability)
    image, masks, boxes, keypoints = tf.cond(
        do_it,
        lambda: rotate(image, masks, boxes, keypoints),
        lambda: (image, masks, boxes, keypoints)
    )
    return image, masks, boxes, keypoints


def get_random_box_center(boxes, image_height, image_width):
    """
    Arguments:
        boxes: a float tensor with shape [num_persons, 4].
        image_height, image_width: float tensors with shape [].
    Returns:
        box_center: a float tensor with shape [1, 2].
        box_width: a float tensor with shape [].
    """

    # get a random bounding box
    box = tf.random_shuffle(boxes)[0]
    # it has shape [4]

    ymin, xmin, ymax, xmax = tf.unstack(box)
    box_height, box_width = ymax - ymin, xmax - xmin

    # get the center of the box
    cy = ymin + 0.5 * box_height
    cx = xmin + 0.5 * box_width

    # we will rotate around the box's center,
    # but the center mustn't be too near to the border of the image
    cy = tf.clip_by_value(cy, 0.25 * image_height, 0.75 * image_height)
    cx = tf.clip_by_value(cx, 0.2 * image_width, 0.8 * image_width)
    box_center = tf.stack([cy, cx])
    box_center = tf.reshape(box_center, [1, 2])

    return box_center, box_width


def get_random_rotation(max_angle, rotation_center, image_width):
    """
    Arguments:
        max_angle: an integer, angle in degrees.
        rotation_center: a float tensor with shape [1, 2].
        image_width: a float tensor with shape [].
    Returns:
        a float tensor with shape [2, 2].
    """

    PI = 3.141592653589793
    max_angle_radians = max_angle * (PI/180.0)

    # x-coordinate of the rotation center
    cx = rotation_center[0, 1]

    # relative distance between centers
    distance_to_image_center = tf.abs(cx - 0.5 * image_width)
    distance_to_image_center /= image_width

    # if the center is too near to the borders then
    # reduce the maximal rotation angle
    decay = (0.6 - 2.0 * distance_to_image_center)/0.6
    decay = tf.maximum(decay, 0.0)
    max_angle_radians *= decay

    # decay is in [0, 1] range,
    # decay = 1 if cx = 0.5 * image_width,
    # decay = 0 if cx = 0.2 * image_width

    # get a random angle
    theta = tf.random_uniform(
        [], minval=-max_angle_radians,
        maxval=max_angle_radians,
        dtype=tf.float32
    )

    rotation = tf.stack([
        tf.cos(theta), tf.sin(theta),
        -tf.sin(theta), tf.cos(theta)
    ], axis=0)
    rotation = tf.reshape(rotation, [2, 2])

    return rotation


def get_random_scaling(rotation_center, box_width, image_width):
    """
    Arguments:
        rotation_center: a float tensor with shape [1, 2].
        box_width: a float tensor with shape [].
        image_width: a float tensor with shape [].
    Returns:
        a float tensor with shape [].
    """

    # x-coordinate of the rotation center
    cx = rotation_center[0, 1]

    # the distance to the nearest border
    distance = tf.minimum(cx, image_width - cx)

    # i believe this minimizes the amount
    # of zero padding after rescaling
    necessary_scale = image_width/(2.0 * distance)
    # it is always bigger or equal to 1

    # with this scaling the distance to the
    # nearest border will be half of image width

    size_ratio = image_width/box_width
    # it is always bigger or equal to 1

    # new box width will be
    # maximum one third of the image width
    max_scale = size_ratio/3.0

    min_scale = tf.maximum(size_ratio/8.0, necessary_scale)
    # this is all very confusing

    min_scale = tf.minimum(min_scale, max_scale - 1e-4)
    # now always min_scale < max_scale

    # get a random image scaler
    scaler = tf.random_uniform(
        [], minval=min_scale,
        maxval=max_scale,
        dtype=tf.float32
    )

    return scaler


def transform_boxes(boxes, rotation, translation):
    """
    Arguments:
        boxes: a float tensor with shape [num_persons, 4].
        rotation: a float tensor with shape [2, 2].
        translation: a float tensor with shape [1, 2].
    Returns:
        a float tensor with shape [num_persons, 4].
    """

    ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
    p1 = tf.stack([ymin, xmin], axis=1)  # top left
    p2 = tf.stack([ymin, xmax], axis=1)  # top right
    p3 = tf.stack([ymax, xmin], axis=1)  # buttom left
    p4 = tf.stack([ymax, xmax], axis=1)  # buttom right
    points = tf.concat([p1, p2, p3, p4], axis=0)
    # it has shape [4 * num_persons, 2]

    points = tf.matmul(points, rotation) + translation
    p1, p2, p3, p4 = tf.split(points, num_or_size_splits=4, axis=0)

    # get boxes that contain the original boxes
    ymin = tf.minimum(p1[:, 0], p2[:, 0])
    ymax = tf.maximum(p3[:, 0], p4[:, 0])
    xmin = tf.minimum(p1[:, 1], p3[:, 1])
    xmax = tf.maximum(p2[:, 1], p4[:, 1])

    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
    return boxes


def transform_keypoints(keypoints, rotation, translation):
    """
    Arguments:
        keypoints: an int tensor with shape [num_persons, 17, 3].
        rotation: a float tensor with shape [2, 2].
        translation: a float tensor with shape [1, 2].
    Returns:
        an int tensor with shape [num_persons, 17, 3].
    """

    points, v = tf.split(keypoints, [2, 1], axis=2)
    # they have shapes [num_persons, 17, 2] and [num_persons, 17, 1]

    points = tf.to_float(points)
    points = tf.reshape(points, [-1, 2])
    points = tf.matmul(points, rotation) + translation
    points = tf.to_int32(tf.round(points))
    points = tf.reshape(points, [-1, 17, 2])
    keypoints = tf.concat([points, v], axis=2)

    return keypoints


def correct(boxes, keypoints, image_height, image_width):
    """
    Remove boxes and keypoints that are outside of the image.

    Arguments:
        boxes: a float tensor with shape [num_persons, 4].
        keypoints: an int tensor with shape [num_persons, 17, 3].
        image_height, image_width: float tensors with shape [].
    Returns:
        boxes: a float tensor with shape [num_remaining_boxes, 4],
            where num_remaining_boxes <= num_persons.
        keypoints: an int tensor with shape [num_remaining_boxes, 17, 3].
    """

    window = tf.stack([0.0, 0.0, image_height, image_width])
    boxes, keep_indices = prune_non_overlapping_boxes(
        boxes, tf.expand_dims(window, 0),
        min_overlap=OVERLAP_THRESHOLD
    )

    ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
    ymin = tf.clip_by_value(ymin, 0.0, image_height)
    xmin = tf.clip_by_value(xmin, 0.0, image_width)
    ymax = tf.clip_by_value(ymax, 0.0, image_height)
    xmax = tf.clip_by_value(xmax, 0.0, image_width)
    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
    
    keypoints = tf.gather(keypoints, keep_indices)
    y, x, v = tf.split(keypoints, 3, axis=2)
    
    image_height = tf.to_int32(image_height)
    image_width = tf.to_int32(image_width)

    coordinate_violations = tf.concat([
        tf.less(y, 0), tf.less(x, 0),
        tf.greater_equal(y, image_height),
        tf.greater_equal(x, image_width)
    ], axis=2)  # shape [num_remaining_boxes, 17, 4]

    valid_indicator = tf.logical_not(tf.reduce_any(coordinate_violations, axis=2))
    valid_indicator = tf.expand_dims(valid_indicator, 2)
    # it has shape [num_remaining_boxes, 17, 1]

    v *= tf.to_int32(valid_indicator)
    keypoints = tf.concat([y, x, v], axis=2)

    return boxes, keypoints


def get_inverse_transform(rotation, translation):
    """
    If y = x * rotation + translation
    then x = (y - translation) * inverse_rotation.

    Or x = y * inverse_rotation + inverse_translation,
    where inverse_translation = - translation * inverse_rotation.

    This function returns transformation in the
    format required by `tf.contrib.image.transform`.

    Arguments:
        rotation: a float tensor with shape [2, 2].
        translation: a float tensor with shape [1, 2].
    Returns:
        a float tensor with shape [8].
    """

    a, b = rotation[0, 0], rotation[0, 1]
    c, d = rotation[1, 0], rotation[1, 1]

    inverse_rotation = tf.stack([d, -b, -c, a]) / (a * d - b * c)
    inverse_rotation = tf.reshape(inverse_rotation, [2, 2])

    inverse_translation = - tf.matmul(translation, inverse_rotation)
    inverse_translation = tf.squeeze(inverse_translation, axis=0)
    # it has shape [2]

    translate_y, translate_x = tf.unstack(inverse_translation, axis=0)
    transform = tf.stack([
        inverse_rotation[0, 0], inverse_rotation[0, 1], translate_x,
        inverse_rotation[1, 0], inverse_rotation[1, 1], translate_y,
        0.0, 0.0
    ])

    return transform
