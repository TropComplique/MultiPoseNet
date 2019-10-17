import tensorflow.compat.v1 as tf
from detector.constants import SHUFFLE_BUFFER_SIZE, NUM_PARALLEL_CALLS, DOWNSAMPLE
from detector.input_pipeline.heatmap_creation import get_heatmaps


# height and width
CROP_SIZE = [56, 36]


class PoseResidualNetworkPipeline:
    """
    """
    def __init__(self, filenames, is_training, batch_size, max_keypoints=None):
        """
        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            is_training: a boolean.
            batch_size: an integer.
            max_keypoints: an integer or None.
        """
        self.is_training = is_training
        self.max_keypoints = max_keypoints

        dataset = tf.data.Dataset.from_tensor_slices(filenames)

        if is_training:
            num_shards = len(filenames)
            dataset = dataset.shuffle(buffer_size=num_shards)

        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.prefetch(buffer_size=batch_size)

        if is_training:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

        dataset = dataset.repeat(None if is_training else 1)
        dataset = dataset.map(self.parse_and_preprocess, num_parallel_calls=NUM_PARALLEL_CALLS)
        dataset = dataset.apply(tf.data.experimental.unbatch())

        if is_training:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=1)

        self.dataset = dataset

    def parse_and_preprocess(self, example_proto):
        """
        Returns:
            crops: a float tensor with shape [num_persons, height, width, 17].
            labels: a float tensor with shape [num_persons, height, width, 17].
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
        image_height, image_width = shape[0], shape[1]
        scaler = tf.to_float(tf.stack(2 * [image_height, image_width]))

        # get number of people on the image
        num_persons = tf.to_int32(parsed_features['num_persons'])
        # it is assumed that num_persons > 0

        # get groundtruth boxes, they are in absolute coordinates
        boxes = tf.reshape(parsed_features['boxes'], [num_persons, 4])

        # get keypoints, they are in absolute coordinates
        keypoints = tf.to_int32(parsed_features['keypoints'])
        keypoints = tf.reshape(keypoints, [num_persons, 17, 3])

        if self.max_keypoints is not None:

            # curriculum learning by sorting
            # annotations based on number of keypoints

            is_visible = tf.to_int32(keypoints[:, :, 2] > 0)  # shape [num_persons, 17]
            is_good = tf.less_equal(tf.reduce_sum(is_visible, axis=1), self.max_keypoints)
            # it has shape [num_persons]

            keypoints = tf.boolean_mask(keypoints, is_good)
            boxes = tf.boolean_mask(boxes, is_good)
            num_persons = tf.shape(boxes)[0]

        heatmaps = tf.py_func(
            lambda k, b, w, h: get_heatmaps(k, b, w, h, DOWNSAMPLE),
            [keypoints, boxes, image_width, image_height],
            tf.float32, stateful=False
        )
        heatmaps.set_shape([None, None, 17])

        box_indices = tf.zeros([num_persons], dtype=tf.int32)
        crops = tf.image.crop_and_resize(
            tf.expand_dims(heatmaps, 0),
            boxes/scaler, box_indices,
            crop_size=CROP_SIZE
        )

        def fn(x):
            """
            Arguments:
                keypoints: a float tensor with shape [17, 3].
                box: a float tensor with shape [4].
            Returns:
                a float tensor with shape [height, width, 17].
            """
            keypoints, box = x

            ymin, xmin, ymax, xmax = tf.unstack(box, axis=0)
            y, x, v = tf.unstack(keypoints, axis=1)
            keypoints = tf.stack([y, x], axis=1)

            part_id = tf.where(v > 0.0)  # shape [num_visible, 1]
            part_id = tf.to_int32(part_id)
            num_visible = tf.shape(part_id)[0]
            keypoints = tf.gather(keypoints, tf.squeeze(part_id, 1))
            # it has shape [num_visible, 2], they have absolute coordinates

            # transform keypoints coordinates
            # to be relative to the box
            h, w = ymax - ymin, xmax - xmin
            height, width = CROP_SIZE
            translation = tf.stack([ymin, xmin])
            scaler = tf.to_float(tf.stack([height/h, width/w], axis=0))

            keypoints -= translation
            keypoints *= scaler
            keypoints = tf.to_int32(tf.round(keypoints))
            # it has shape [num_visible, 2]

            y, x = tf.unstack(keypoints, axis=1)
            y = tf.clip_by_value(y, 0, height - 1)
            x = tf.clip_by_value(x, 0, width - 1)
            keypoints = tf.stack([y, x], axis=1)

            indices = tf.to_int64(tf.concat([keypoints, part_id], axis=1))
            values = tf.ones([num_visible], dtype=tf.float32)
            binary_map = tf.sparse.SparseTensor(indices, values, dense_shape=[height, width, 17])
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
            crops, labels: float tensors with shape [height, width, 17].
        Returns:
            float tensors with shape [height, width, 17].
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

    crops, labels = tf.map_fn(
        randomly_flip, (crops, labels),
        dtype=(tf.float32, tf.float32),
        back_prop=False,
    )
    return crops, labels
