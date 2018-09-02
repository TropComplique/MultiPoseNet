import tensorflow as tf

from detector.input_pipeline.random_crop import random_crop
from detector.input_pipeline.color_augmentations import random_color_manipulations, random_pixel_value_scale
from detector.constants import SHUFFLE_BUFFER_SIZE, NUM_THREADS, RESIZE_METHOD


"""Input pipeline for training or evaluating object detectors."""


class DetectorPipeline:
    def __init__(self, filenames, batch_size, is_training, image_size):
        """
        Note: when evaluating set batch_size to 1 and image_size to None.

        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            batch_size: an integer.
            is_training: a boolean.
            image_size: an integer or None.
        """
        self.is_training = is_training
        self.image_size = image_size

        # when evaluating, images aren't resized
        if not is_training:
            assert batch_size == 1
            assert image_size is None

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
        dataset = dataset.map(self._parse_and_preprocess, num_parallel_calls=NUM_THREADS)

        # we need batches with static first dimension
        padded_shapes = ({'images': [self.image_size, self.image_size, 3]}, {'boxes': [None, 4], 'num_boxes': []})
        dataset = dataset.padded_batch(batch_size, padded_shapes, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=1)

        self.dataset = dataset

    def _parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. (optionally) Augments it.

        Returns:
            image: a float tensor with shape [image_height, image_width, 3],
                an RGB image with pixel values in the range [0, 1].
            boxes: a float tensor with shape [num_boxes, 4], they have normalized coordinates.
            num_boxes: an int tensor with shape [].
        """
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'num_persons': tf.FixedLenFeature([], tf.int64),
            'boxes': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # get an image
        image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # now pixel values are scaled to the [0, 1] range

        # get number of people on the image
        num_boxes = tf.to_int32(parsed_features['num_persons'])
        # it is assumed that num_persons > 0

        # get groundtruth boxes, they are in absolute coordinates
        boxes = tf.reshape(parsed_features['boxes'], [num_boxes, 4])

        # to the [0, 1] range
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        scaler = tf.to_float(tf.stack([height, width, height, width]))
        boxes /= scaler

        if self.is_training:
            image, boxes = self.augmentation(image, boxes)
            num_boxes = tf.shape(boxes)[0]  # it could change after augmentations

        features = {'images': image}
        labels = {'boxes': boxes, 'num_boxes': num_boxes}
        return features, labels

    def augmentation(self, image, boxes):
        image, boxes = self.randomly_crop_and_resize(image, boxes, probability=0.9)
        image = random_color_manipulations(image, probability=0.5, grayscale_probability=0.1)
        image = random_pixel_value_scale(image, probability=0.1, minval=0.8, maxval=1.2)
        boxes = random_jitter_boxes(boxes, ratio=0.03)
        image, boxes = random_flip_left_right(image, boxes)
        return image, boxes

    def randomly_crop_and_resize(self, image, boxes, probability=0.9):

        def crop(image, boxes):
            image, boxes, _, _ = random_crop(
                image, boxes,
                min_object_covered=0.75,
                aspect_ratio_range=(0.9, 1.1),
                area_range=(0.25, 1.0),
                overlap_threshold=0.3
            )
            return image, boxes

        do_it = tf.less(tf.random_uniform([]), probability)
        image, boxes = tf.cond(do_it, lambda: crop(image, boxes), lambda: (image, boxes))

        image = tf.image.resize_images(
            image, [self.image_size, self.image_size],
            method=RESIZE_METHOD
        )
        # note that boxes are with normalized coordinates
        return image, boxes


def random_flip_left_right(image, boxes):

    def flip(image, boxes):
        flipped_image = tf.image.flip_left_right(image)
        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        flipped_xmin = tf.subtract(1.0, xmax)
        flipped_xmax = tf.subtract(1.0, xmin)
        flipped_boxes = tf.stack([ymin, flipped_xmin, ymax, flipped_xmax], axis=1)
        return flipped_image, flipped_boxes

    with tf.name_scope('random_flip_left_right'):
        do_it = tf.less(tf.random_uniform([]), 0.5)
        image, boxes = tf.cond(do_it, lambda: flip(image, boxes), lambda: (image, boxes))
        return image, boxes


def random_jitter_boxes(boxes, ratio=0.05):
    """Randomly jitter bounding boxes.

    Arguments:
        boxes: a float tensor with shape [N, 4].
        ratio: a float number.
            The ratio of the box width and height that the corners can jitter.
            For example if the width is 100 pixels and ratio is 0.05,
            the corners can jitter up to 5 pixels in the x direction.
    Returns:
        a float tensor with shape [N, 4].
    """
    def random_jitter_box(box, ratio):
        """Randomly jitter a box.
        Arguments:
            box: a float tensor with shape [4].
            ratio: a float number.
        Returns:
            a float tensor with shape [4].
        """
        ymin, xmin, ymax, xmax = tf.unstack(box, axis=0)
        box_height, box_width = ymax - ymin, xmax - xmin
        hw_coefs = tf.stack([box_height, box_width, box_height, box_width])

        rand_numbers = tf.random_uniform(
            [4], minval=-ratio, maxval=ratio, dtype=tf.float32
        )
        hw_rand_coefs = tf.multiply(hw_coefs, rand_numbers)

        jittered_box = tf.add(box, hw_rand_coefs)
        return jittered_box

    with tf.name_scope('random_jitter_boxes'):
        distorted_boxes = tf.map_fn(
            lambda x: random_jitter_box(x, ratio),
            boxes, dtype=tf.float32, back_prop=False
        )
        distorted_boxes = tf.clip_by_value(distorted_boxes, 0.0, 1.0)
        return distorted_boxes
