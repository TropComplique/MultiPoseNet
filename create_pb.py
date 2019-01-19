import tensorflow as tf
import os
import shutil
from detector import RetinaNet
from detector import KeypointSubnet
from detector import prn
from detector.backbones import mobilenet_v1
tf.logging.set_verbosity('INFO')


"""
This code creates a .pb frozen inference graph.
"""


GPU_TO_USE = '0'
PB_FILE_PATH = 'inference/model.pb'
BATCH_SIZE = 1  # must be an integer
CROP_SIZE = [56, 36]
OUTPUT_NAMES = [
    'boxes', 'scores', 'num_boxes',
    'keypoint_heatmaps', 'segmentation_masks',
    'keypoint_scores', 'keypoint_positions'
]
PARAMS = {
    'depth_multiplier': 1.0,
    'score_threshold': 0.3, 'iou_threshold': 0.6, 'max_boxes': 25
}


def create_full_graph(images, params):
    """
    Batch size must be a static value.

    Arguments:
        images: a float tensor with shape [b, h, w, 3].
        params: a dict.
    Returns:
        a dict with tensors that contain predictions.
    """
    is_training = False

    backbone_features = mobilenet_v1(
        images, is_training,
        depth_multiplier=params['depth_multiplier']
    )

    with tf.variable_scope('keypoint_subnet'):
        subnet = KeypointSubnet(backbone_features, is_training, params)

    with tf.variable_scope('retinanet'):
        retinanet = RetinaNet(backbone_features, tf.shape(images), is_training, params)

    predictions = subnet.get_predictions()
    predictions.update(retinanet.get_predictions(
        score_threshold=params['score_threshold'],
        iou_threshold=params['iou_threshold'],
        max_detections=params['max_boxes']
    ))

    batch_size = images.shape[0].value
    assert batch_size is not None

    heatmaps = predictions['heatmaps']
    predicted_boxes = predictions['boxes']
    num_boxes = predictions['num_boxes']

    boxes, box_ind = [], []
    for i in range(batch_size):
        n = num_boxes[i]
        boxes.append(predicted_boxes[i][:n])
        box_ind.append(i * tf.ones([n], dtype=tf.int32))

    boxes = tf.concat(boxes, axis=0)
    box_ind = tf.concat(box_ind, axis=0)

    crops = tf.image.crop_and_resize(
        heatmaps, boxes, box_ind,
        crop_size=CROP_SIZE
    )  # shape [num_boxes, 56, 36, 17]

    num_boxes = tf.shape(crops)[0]
    logits = prn(crops, is_training)

    spatial_area = CROP_SIZE[0] * CROP_SIZE[1]
    logits = tf.reshape(logits, [num_boxes, spatial_area, 17])
    probabilities = tf.nn.softmax(logits, axis=1)
    probabilities = tf.reshape(probabilities, [num_boxes] + CROP_SIZE + [17])

    def argmax_2d(tensor):
        """
        Arguments:
            tensor: a tensor with shape [b, h, w, c].
        Returns:
            an int tensor with shape [b, c, 2].
        """
        shape = tf.shape(tensor)
        flat_tensor = tf.reshape(tensor, (shape[0], -1, shape[3]))

        argmax = tf.argmax(flat_tensor, axis=1, output_type=tf.int32)
        argmax_x = argmax // shape[2]
        argmax_y = argmax % shape[2]

        return tf.stack([argmax_y, argmax_x], axis=2)

    keypoint_scores = tf.reduce_max(probabilities, axis=[1, 2])  # shape [num_boxes, 17]
    keypoint_positions = tf.to_float(argmax_2d(probabilities))  # shape [num_boxes, 17, 2]

    scaler = tf.stack(CROP_SIZE, axis=1)
    keypoint_positions /= scaler

    predictions.update({
        'keypoint_scores': keypoint_scores,
        'keypoint_positions': keypoint_positions
    })

    predictions = {
        n: tf.identity(predictions[n], name=n)
        for n in OUTPUT_NAMES
    }
    return predictions


def convert_to_pb():

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = GPU_TO_USE

    with graph.as_default():

        raw_images = tf.placeholder(dtype=tf.uint8, shape=[BATCH_SIZE, None, None, 3], name='images')
        images = tf.to_float(raw_images)
        images = (1.0/255.0) * images
        predictions = create_full_graph(images, PARAMS)

        tf.train.init_from_checkpoint(
            'models/run00/model.ckpt-175000',
            {'MobilenetV1/': 'MobilenetV1/'}
        )
        tf.train.init_from_checkpoint(
            'models/run00/model.ckpt-175000',
            {'/': 'keypoint_subnet/'}
        )
        tf.train.init_from_checkpoint(
            'models/run01/model.ckpt-150000',
            {'/': 'retinanet/'}
        )
        tf.train.init_from_checkpoint(
            'models/run02/model.ckpt-150000',
            {'PRN/': 'PRN/'}
        )
        init = tf.global_variables_initializer()

        with tf.Session(config=config) as sess:
            sess.run(init)

            input_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(),
                output_node_names=OUTPUT_NAMES
            )

            output_graph_def = tf.graph_util.remove_training_nodes(
                input_graph_def, protected_nodes=OUTPUT_NAMES
            )

            with tf.gfile.GFile(PB_FILE_PATH, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('%d ops in the final graph.' % len(output_graph_def.node))


convert_to_pb()
