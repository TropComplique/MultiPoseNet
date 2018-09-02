import tensorflow as tf
from detector.constants import PARALLEL_ITERATIONS


def batch_non_max_suppression(
        boxes, scores,
        score_threshold=0.1,
        iou_threshold=0.4,
        max_boxes=20):
    """
    Arguments:
        boxes: a float tensor with shape [batch_size, N, 4].
        scores: a float tensor with shape [batch_size, N].
        score_threshold: a float number.
        iou_threshold: a float number, threshold for IoU.
        max_boxes: an integer, maximum number of retained boxes.
    Returns:
        boxes: a float tensor with shape [batch_size, max_boxes, 4].
        scores: a float tensor with shape [batch_size, max_boxes].
        num_detections: an int tensor with shape [batch_size].
    """
    def fn(x):
        boxes, scores = x

        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_boxes, iou_threshold, score_threshold
        )
        boxes = tf.gather(boxes, selected_indices)
        scores = tf.gather(scores, selected_indices)
        num_boxes = tf.to_int32(tf.shape(boxes)[0])

        zero_padding = max_boxes - num_boxes
        boxes = tf.pad(boxes, [[0, zero_padding], [0, 0]])
        scores = tf.pad(scores, [[0, zero_padding]])

        boxes.set_shape([max_boxes, 4])
        scores.set_shape([max_boxes])
        return boxes, scores, num_boxes

    boxes, scores, num_detections = tf.map_fn(
        fn, [boxes, scores],
        dtype=(tf.float32, tf.float32, tf.int32),
        parallel_iterations=PARALLEL_ITERATIONS,
        back_prop=False, swap_memory=False, infer_shape=True
    )
    return boxes, scores, num_detections
