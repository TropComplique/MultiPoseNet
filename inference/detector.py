import tensorflow as tf
import numpy as np

MODEL_PATH = 'model.pb'
OUTPUT_NAMES = [
    'boxes', 'scores', 'num_boxes',
    'keypoint_heatmaps', 'segmentation_masks',
    'keypoint_scores', 'keypoint_positions'
]

with tf.gfile.GFile(MODEL_PATH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(graph_def, name='import')

input_image = graph.get_tensor_by_name('import/images:0')
output_ops = {n: graph.get_tensor_by_name(f'import/{n}:0') for n in OUTPUT_NAMES}
sess = tf.Session(graph=graph)
outputs = sess.run(output_ops, feed_dict={input_image: np.expand_dims(np.array(image), 0)})
outputs = {n: v[0] for n, v in outputs.items()}
class Detector:
    def __init__(self, model_path, gpu_memory_fraction=0.25, visible_device_list='0'):
        """
        Arguments:
            model_path: a string, path to a pb file.
            gpu_memory_fraction: a float number.
            visible_device_list: a string.
        """
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name='import')

        self.input_image = graph.get_tensor_by_name('import/images:0')
        self.output_ops = [
            graph.get_tensor_by_name('import/keypoint_heatmaps:0'),
            graph.get_tensor_by_name('import/segmentation_masks:0'),
        ]

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction,
            visible_device_list=visible_device_list
        )
        config_proto = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        self.sess = tf.Session(graph=graph, config=config_proto)

    def __call__(self, image, score_threshold=0.05):
        """Detect faces.

        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 4].
            labels: an int numpy array of shape [num_faces].
            scores: a float numpy array of shape [num_faces].

        Note that box coordinates are in the order: ymin, xmin, ymax, xmax!
        """

        feed_dict = {self.input_image: np.expand_dims(image, 0)}
        heatmaps, segmentation_masks = self.sess.run(self.output_ops, feed_dict)
        
#         n = n[0]
#         to_keep = scores[0][:n] > score_threshold
#         boxes = boxes[0][:n][to_keep]
#         labels = labels[0][:n][to_keep]
#         scores = scores[0][:n][to_keep]

        return heatmaps[0], segmentation_masks[0]
