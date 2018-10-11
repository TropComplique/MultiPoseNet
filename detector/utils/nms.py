import tensorflow as tf
from detector.constants import PARALLEL_ITERATIONS
from .box_utils import decode


def batch_non_max_suppression(
        encoded_boxes,
        anchors, scores,
        score_threshold,
        iou_threshold,
        max_detections):
    """
    Arguments:
        encoded_boxes: a float tensor with shape [batch_size, N, 4].
        anchors: a float tensor with shape [N, 4].
        scores: a float tensor with shape [batch_size, N].
        score_threshold: a float number.
        iou_threshold: a float number.
        max_detections: an integer.
    Returns:
        boxes: a float tensor with shape [batch_size, N', 4].
        scores: a float tensor with shape [batch_size, N'].
        num_detections: an int tensor with shape [batch_size].

        Where N' = max_detections.
    """
    def fn(x):
        encoded_boxes, scores = x

        is_confident = scores >= score_threshold  # shape [N]
        encoded_boxes = tf.boolean_mask(encoded_boxes, is_confident)  # shape [num_confident, 4]
        scores = tf.boolean_mask(scores, is_confident)  # shape [num_confident]
        chosen_anchors = tf.boolean_mask(anchors, is_confident)  # shape [num_confident, 4]
        boxes = decode(encoded_boxes, chosen_anchors)  # shape [num_confident, 4]
        boxes = tf.clip_by_value(boxes, 0.0, 1.0)

        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_output_size=max_detections,
            iou_threshold=iou_threshold, score_threshold=score_threshold
        )

        boxes = tf.gather(boxes, selected_indices)
        scores = tf.gather(scores, selected_indices)
        num_boxes = tf.to_int32(tf.size(selected_indices))

        zero_padding = max_detections - num_boxes
        boxes = tf.pad(boxes, [[0, zero_padding], [0, 0]])
        scores = tf.pad(scores, [[0, zero_padding]])

        boxes.set_shape([max_detections, 4])
        scores.set_shape([max_detections])
        return boxes, scores, num_boxes

    boxes, scores, num_detections = tf.map_fn(
        fn, [encoded_boxes, scores],
        dtype=(tf.float32, tf.float32, tf.int32),
        parallel_iterations=PARALLEL_ITERATIONS,
        back_prop=False, swap_memory=False, infer_shape=True
    )
    return boxes, scores, num_detections
