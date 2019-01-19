import tensorflow as tf
import math
from detector.constants import SHUFFLE_BUFFER_SIZE, NUM_PARALLEL_CALLS, RESIZE_METHOD, DOWNSAMPLE
from .heatmap_creation import get_heatmaps


SIGMA = 1.5
CROP_SIZE = [56, 36]  # height and width


class PoseResidualNetworkPipeline:
    def __init__(self, filenames, is_training, batch_size):
        """
        During the evaluation we resize images keeping aspect ratio.

        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            is_training: a boolean.
            batch_size: an integer.
        """
        self.is_training = is_training

        def get_num_samples(filename):
            return sum(1 for _ in tf.python_io.tf_record_iterator(filename))

        num_examples = 0
        for filename in filenames:
            num_examples_in_file = get_num_samples(filename)
            assert num_examples_in_file > 0
            num_examples += num_examples_in_file
        self.num_examples = num_examples
        assert self.num_examples > 0

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        num_shards = len(filenames)

        if is_training:
            dataset = dataset.shuffle(buffer_size=num_shards)

        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.prefetch(buffer_size=batch_size)

        if is_training:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.repeat(None if is_training else 1)
        dataset = dataset.map(self._parse_and_preprocess, num_parallel_calls=NUM_PARALLEL_CALLS)
        dataset = dataset.apply(tf.data.experimental.unbatch())

        if is_training:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=1)

        self.dataset = dataset

    def _parse_and_preprocess(self, example_proto):
        """
        Returns:
            crops: a float tensor with shape [num_persons, 56, 36, 17].
            labels: a float tensor with shape [num_persons, 56, 36, 17].
        """
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'num_persons': tf.FixedLenFeature([], tf.int64),
            'boxes': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'keypoints': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # get size of the image
        shape = tf.image.extract_jpeg_shape(parsed_features['image'])
        height, width = shape[0], shape[1]
        scaler = tf.to_float(tf.stack([height, width, height, width]))

        # get number of people on the image
        num_persons = tf.to_int32(parsed_features['num_persons'])
        # it is assumed that num_persons > 0

        # get groundtruth boxes, they are in absolute coordinates
        boxes = tf.reshape(parsed_features['boxes'], [num_persons, 4])

        # get keypoints, they are in absolute coordinates
        keypoints = tf.to_int32(parsed_features['keypoints'])
        keypoints = tf.reshape(keypoints, [num_persons, 17, 3])

        heatmaps = tf.py_func(
            lambda k, w, h: get_heatmaps(k, SIGMA, w, h, DOWNSAMPLE),
            [tf.to_float(keypoints), width, height],
            tf.float32, stateful=False
        )
        heatmaps.set_shape([None, None, 17])
        box_ind = tf.zeros([num_persons], dtype=tf.int32)
        crops = tf.image.crop_and_resize(
            tf.expand_dims(heatmaps, 0),
            boxes/scaler, box_ind=box_ind,
            crop_size=CROP_SIZE
        )

        def fn(x):
            """
            Arguments:
                keypoints: a float tensor with shape [17, 3].
                box: a float tensor with shape [4].
            Returns:
                a float tensor with shape [56, 36, 17].
            """
            keypoints, box = x
            ymin, xmin, ymax, xmax = tf.unstack(box, axis=0)
            y, x, v = tf.unstack(keypoints, axis=1)
            keypoints = tf.stack([y, x], axis=1)

            part_id = tf.where(v > 0.0)  # shape [num_visible, 1]
            part_id = tf.to_int32(part_id)
            num_visible = tf.shape(part_id)[0]
            keypoints = tf.gather(keypoints, tf.squeeze(part_id, 1))
            # it has shape [num_visible, 2]

            keypoints -= tf.stack([ymin, xmin])
            h, w = ymax - ymin, xmax - xmin
            scaler = tf.to_float(tf.stack([CROP_SIZE[0]/h, CROP_SIZE[1]/w], axis=0))
            keypoints *= scaler
            keypoints = tf.to_int32(tf.floor(keypoints))  # shape [num_visible, 2]
            
            y, x = tf.unstack(keypoints, axis=1)
            y = tf.clip_by_value(y, 0, CROP_SIZE[0] - 1)
            x = tf.clip_by_value(x, 0, CROP_SIZE[1] - 1)
            keypoints = tf.stack([y, x], axis=1)

            indices = tf.to_int64(tf.concat([keypoints, part_id], axis=1))
            values = tf.ones([num_visible], dtype=tf.float32)
            binary_map = tf.sparse.SparseTensor(indices, values, dense_shape=CROP_SIZE + [17])
            binary_map = tf.sparse.to_dense(binary_map, default_value=0, validate_indices=False)
            return binary_map

        labels = tf.map_fn(
            fn, (tf.to_float(keypoints), boxes),
            dtype=tf.float32, back_prop=False,
        )

        if self.is_training:
            crops, labels = random_flip_left_right(crops, labels)

        return crops, labels


def random_flip_left_right(crops, labels):

    def randomly_flip(x):
        """
        Arguments:
            crops, labels: float tensors with shape [56, 36, 17].
        Returns:
            float tensors with shape [56, 36, 17].
        """
        crops, labels = x

        def flip(crops, labels):

            crops = tf.image.flip_left_right(crops)
            labels = tf.image.flip_left_right(labels)

            """
            The keypoint order:
            0: 'nose',
            1: 'left eye', 2: 'right eye',
            3: 'left ear', 4: 'right ear',
            5: 'left shoulder', 6: 'right shoulder',
            7: 'left elbow', 8: 'right elbow',
            9: 'left wrist', 10: 'right wrist',
            11: 'left hip', 12: 'right hip',
            13: 'left knee', 14: 'right knee',
            15: 'left ankle', 16: 'right ankle'
            """

            correct_order = tf.constant([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
            crops = tf.gather(crops, correct_order, axis=2)
            labels = tf.gather(labels, correct_order, axis=2)
            return crops, labels

        do_it = tf.less(tf.random_uniform([]), 0.5)
        crops, labels = tf.cond(do_it, lambda: flip(crops, labels), lambda: (crops, labels))

        return crops, labels

    with tf.name_scope('random_flip_left_right'):
        crops, labels = tf.map_fn(
            randomly_flip, (crops, labels),
            dtype=(tf.float32, tf.float32),
            back_prop=False,
        )
        return crops, labels
