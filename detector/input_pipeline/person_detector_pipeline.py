import tensorflow as tf
from detector.constants import SHUFFLE_BUFFER_SIZE, NUM_THREADS, RESIZE_METHOD, DIVISOR
from .random_crop import random_crop
from .color_augmentations import random_color_manipulations, random_pixel_value_scale


"""
Input pipeline for training or evaluating object detectors.
It is assumed that all boxes are of the same class.
"""


class DetectorPipeline:
    def __init__(self, filenames, is_training, params):
        """
        During the evaluation we resize images keeping aspect ratio.

        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            is_training: a boolean.
            params: a dict.
        """
        self.is_training = is_training

        if not is_training:
            batch_size = 1
            self.image_size = [None, None]
            self.min_dimension = params['min_dimension']
        else:
            batch_size = params['batch_size']
            height = params['image_height']
            width = params['image_width']
            assert height % DIVISOR == 0
            assert width % DIVISOR == 0
            self.image_size = [height, width]

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

        padded_shapes = ({'images': self.image_size + [3]}, {'boxes': [None, 4], 'num_boxes': []})
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
            boxes: a float tensor with shape [num_boxes, 4].
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
        # it is assumed that num_boxes > 0

        # get groundtruth boxes, they are in absolute coordinates
        boxes = tf.reshape(parsed_features['boxes'], [num_boxes, 4])

        # to the [0, 1] range
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        scaler = tf.to_float(tf.stack([height, width, height, width]))
        boxes /= scaler

        if self.is_training:
            image, boxes = self.augmentation(image, boxes)
            num_boxes = tf.shape(boxes)[0]  # it could change after augmentations
        else:
            image, box_scaler = resize_keeping_aspect_ratio(image, self.min_dimension, DIVISOR)
            boxes *= box_scaler

        features = {'images': image}
        labels = {'boxes': boxes, 'num_boxes': num_boxes}
        return features, labels

    def augmentation(self, image, boxes):
        image, boxes = randomly_crop_and_resize(image, boxes, self.image_size, probability=0.9)
        image, boxes = randomly_pad(image, boxes, probability=0.1)
        image = random_color_manipulations(image, probability=0.33, grayscale_probability=0.033)
        image = random_pixel_value_scale(image, probability=0.1, minval=0.8, maxval=1.2)
        boxes = random_box_jitter(boxes, ratio=0.01)
        image, boxes = random_flip_left_right(image, boxes)
        return image, boxes


def randomly_crop_and_resize(image, boxes, image_size, probability=0.9):

    def crop(image, boxes):
        image, boxes, _, _ = random_crop(
            image, boxes,
            min_object_covered=0.9,
            aspect_ratio_range=(0.85, 1.15),
            area_range=(0.33, 1.0),
            overlap_threshold=0.4
        )
        return image, boxes

    do_it = tf.less(tf.random_uniform([]), probability)
    image, boxes = tf.cond(
        do_it,
        lambda: crop(image, boxes),
        lambda: (image, boxes)
    )
    image = tf.image.resize_images(image, image_size, method=RESIZE_METHOD)
    return image, boxes


def randomly_pad(image, boxes, probability=0.9):
    """
    This function makes content of the image
    smaller by scaling and padding with zeros.
    """

    def pad(image, boxes):

        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        # randomly reduce image scale
        scale = tf.random_uniform([], 0.5, 0.9)
        target_height = tf.to_int32(scale * tf.to_float(image_height))
        target_width = tf.to_int32(scale * tf.to_float(image_width))

        image = tf.image.resize_images(
            image, [target_height, target_width],
            method=RESIZE_METHOD
        )

        # randomly pad to the initial size
        offset_y = image_height - target_height
        offset_x = image_width - target_width
        offset_y = tf.random_uniform([], 0, offset_y, dtype=tf.int32)
        offset_x = tf.random_uniform([], 0, offset_x, dtype=tf.int32)

        image = tf.image.pad_to_bounding_box(
            image, offset_y, offset_x,
            image_height, image_width
        )

        # transform boxes
        boxes *= scale
        offset_y = tf.to_float(offset_y/image_height)
        offset_x = tf.to_float(offset_x/image_width)
        translation = tf.stack([offset_y, offset_x, offset_y, offset_x])
        boxes += translation

        return image, boxes

    do_it = tf.less(tf.random_uniform([]), probability)
    image, boxes = tf.cond(do_it, lambda: pad(image, boxes), lambda: (image, boxes))
    return image, boxes


def resize_keeping_aspect_ratio(image, min_dimension, divisor):
    """
    This function resizes and possibly pads with zeros.
    When using a usual FPN, divisor must be equal to 128.

    Arguments:
        image: a float tensor with shape [height, width, 3].
        min_dimension: an integer.
        divisor: an integer.
    Returns:
        image: a float tensor with shape [new_height, new_width, 3],
            where `min_dimension = min(new_height, new_width)`,
            `new_height` and `new_width` are divisible by `divisor`.
        box_scaler: a float tensor with shape [4].
    """
    assert min_dimension % divisor == 0

    min_dimension = tf.constant(min_dimension, dtype=tf.int32)
    divisor = tf.constant(divisor, dtype=tf.int32)

    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    original_min_dim = tf.minimum(height, width)
    scale_factor = tf.to_float(min_dimension / original_min_dim)

    def scale(x):
        unpadded_x = tf.to_int32(tf.round(tf.to_float(x) * scale_factor))
        x = tf.to_int32(tf.ceil(unpadded_x / divisor))
        pad = divisor * x - unpadded_x
        return (unpadded_x, pad)

    zero = tf.constant(0, dtype=tf.int32)
    new_height, pad_height, new_width, pad_width = tf.cond(
        tf.greater_equal(height, width),
        lambda: scale(height) + (min_dimension, zero),
        lambda: (min_dimension, zero) + scale(width)
    )

    # resize keeping aspect ratio
    image = tf.image.resize_images(image, [new_height, new_width], method=RESIZE_METHOD)

    image = tf.image.pad_to_bounding_box(
        image, offset_height=0, offset_width=0,
        target_height=new_height + pad_height,
        target_width=new_width + pad_width
    )
    # it pads image at the bottom or at the right

    # we need to rescale bounding box coordinates
    box_scaler = tf.to_float(tf.stack([
        new_height/(new_height + pad_height),
        new_width/(new_width + pad_width),
        new_height/(new_height + pad_height),
        new_width/(new_width + pad_width),
    ]))

    return image, box_scaler


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


def random_box_jitter(boxes, ratio=0.05):
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
    def jitter_box(box, ratio):
        """
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

    with tf.name_scope('random_box_jitter'):
        distorted_boxes = tf.map_fn(
            lambda x: jitter_box(x, ratio),
            boxes, dtype=tf.float32, back_prop=False
        )
        distorted_boxes = tf.clip_by_value(distorted_boxes, 0.0, 1.0)
        return distorted_boxes
