import tensorflow.compat.v1 as tf
import numpy as np


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
        output_names = [
            'boxes', 'scores', 'num_boxes',
            'keypoint_heatmaps', 'segmentation_masks',
            'keypoint_scores', 'keypoint_positions'
        ]
        self.output_ops = {n: graph.get_tensor_by_name(f'import/{n}:0') for n in output_names}

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction,
            visible_device_list=visible_device_list
        )
        config_proto = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        self.sess = tf.Session(graph=graph, config=config_proto)

    def __call__(self, image, score_threshold=0.05):
        """
        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        """

        h, w, _ = image.shape
        assert h % 128 == 0 and w % 128 == 0

        feed_dict = {self.input_image: np.expand_dims(image, 0)}
        outputs = self.sess.run(self.output_ops, feed_dict)
        outputs.update({
            n: v[0] for n, v in outputs.items()
            if n not in ['keypoint_scores', 'keypoint_positions']
        })

        n = outputs['num_boxes']
        to_keep = outputs['scores'][:n] > score_threshold
        outputs['boxes'] = outputs['boxes'][:n][to_keep]
        outputs['scores'] = outputs['scores'][:n][to_keep]
        outputs['keypoint_positions'] = outputs['keypoint_positions'][to_keep]
        outputs['keypoint_scores'] = outputs['keypoint_scores'][to_keep]

        return outputs
