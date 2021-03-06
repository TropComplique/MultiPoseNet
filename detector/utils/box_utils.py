import tensorflow.compat.v1 as tf
from detector.constants import EPSILON, SCALE_FACTORS


"""
Tools for dealing with bounding boxes.

All boxes are of the format [ymin, xmin, ymax, xmax] if not stated otherwise.
Also the following must be true: ymin < ymax and xmin < xmax.
And box coordinates are normalized to the [0, 1] range.
"""


def iou(boxes1, boxes2):
    """Computes pairwise intersection-over-union between two box collections.
    Arguments:
        boxes1: a float tensor with shape [N, 4].
        boxes2: a float tensor with shape [M, 4].
    Returns:
        a float tensor with shape [N, M] representing pairwise iou scores.
    """
    intersections = intersection(boxes1, boxes2)
    areas1 = area(boxes1)
    areas2 = area(boxes2)
    unions = tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections
    return tf.clip_by_value(tf.divide(intersections, unions + EPSILON), 0.0, 1.0)


def intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.
    Arguments:
        boxes1: a float tensor with shape [N, 4].
        boxes2: a float tensor with shape [M, 4].
    Returns:
        a float tensor with shape [N, M] representing pairwise intersections.
    """
    ymin1, xmin1, ymax1, xmax1 = tf.split(boxes1, num_or_size_splits=4, axis=1)
    ymin2, xmin2, ymax2, xmax2 = tf.split(boxes2, num_or_size_splits=4, axis=1)
    # they all have shapes like [None, 1]

    all_pairs_min_ymax = tf.minimum(ymax1, tf.transpose(ymax2))
    all_pairs_max_ymin = tf.maximum(ymin1, tf.transpose(ymin2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(xmax1, tf.transpose(xmax2))
    all_pairs_max_xmin = tf.maximum(xmin1, tf.transpose(xmin2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    # they all have shape [N, M]

    return intersect_heights * intersect_widths


def area(boxes):
    """Computes area of boxes.
    Arguments:
        boxes: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N] representing box areas.
    """
    ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
    return (ymax - ymin) * (xmax - xmin)


def to_center_coordinates(boxes):
    """Convert bounding boxes of the format
    [ymin, xmin, ymax, xmax] to the format [cy, cx, h, w].

    Arguments:
        boxes: a float tensor with shape [N, 4].
    Returns:
        a list of float tensors with shape [N]
        that represent cy, cx, h, w.
    """
    ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
    h, w = ymax - ymin, xmax - xmin
    cy, cx = ymin + 0.5 * h, xmin + 0.5 * w
    return [cy, cx, h, w]


def encode(boxes, anchors):
    """Encode boxes with respect to anchors (or proposals).

    Arguments:
        boxes: a float tensor with shape [N, 4].
        anchors: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N, 4],
        anchor-encoded boxes of the format [ty, tx, th, tw].
    """

    ycenter_a, xcenter_a, ha, wa = to_center_coordinates(anchors)
    ycenter, xcenter, h, w = to_center_coordinates(boxes)

    # to avoid NaN in division and log below
    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    ty = (ycenter - ycenter_a)/ha
    tx = (xcenter - xcenter_a)/wa
    th = tf.log(h / ha)
    tw = tf.log(w / wa)

    ty *= SCALE_FACTORS[0]
    tx *= SCALE_FACTORS[1]
    th *= SCALE_FACTORS[2]
    tw *= SCALE_FACTORS[3]

    return tf.stack([ty, tx, th, tw], axis=1)


def decode(codes, anchors):
    """Decode relative codes to normal boxes.

    Arguments:
        codes: a float tensor with shape [N, 4],
            anchor-encoded boxes of the format [ty, tx, th, tw].
        anchors: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N, 4],
        bounding boxes of the format [ymin, xmin, ymax, xmax].
    """

    ycenter_a, xcenter_a, ha, wa = to_center_coordinates(anchors)
    ty, tx, th, tw = tf.unstack(codes, axis=1)

    ty /= SCALE_FACTORS[0]
    tx /= SCALE_FACTORS[1]
    th /= SCALE_FACTORS[2]
    tw /= SCALE_FACTORS[3]

    h = tf.exp(th) * ha
    w = tf.exp(tw) * wa
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a

    ymin, xmin = ycenter - 0.5 * h, xcenter - 0.5 * w
    ymax, xmax = ycenter + 0.5 * h, xcenter + 0.5 * w
    return tf.stack([ymin, xmin, ymax, xmax], axis=1)
