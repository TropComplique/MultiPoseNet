import tensorflow as tf
import math
from detector.constants import SHUFFLE_BUFFER_SIZE, NUM_THREADS, RESIZE_METHOD, DOWNSAMPLE
from .random_crop import random_crop
from .random_rotation import random_rotation
from .color_augmentations import random_color_manipulations, random_pixel_value_scale
from .person_detector_pipeline import resize_keeping_aspect_ratio
from .heatmap_creation import get_heatmaps


"""Input pipeline for training or evaluating networks for heatmap regression."""


class KeypointPipeline:
    def __init__(self, filenames, batch_size, is_training, image_size):
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

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=1)

        self.dataset = dataset

    def _parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. (optionally) Augments it.

        Returns:
            image: a float tensor with shape [image_height, image_width, 3],
                an RGB image with pixel values in the range [0, 1].
            heatmaps: a float tensor with shape [downsampled_height, downsampled_width, 18].
            loss_mask: a float binary tensor with shape [downsampled_height, downsampled_width].

            where `downsampled_height = image_height/downsample`
            and `downsampled_width = image_width/downsample`.
        """
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'num_persons': tf.FixedLenFeature([], tf.int64),
            'boxes': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'keypoints': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'masks': tf.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # get an image
        image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # now pixel values are scaled to the [0, 1] range

        # get number of people on the image
        num_persons = tf.to_int32(parsed_features['num_persons'])
        # it is assumed that num_persons > 0

        # get groundtruth boxes, they are in absolute coordinates
        boxes = tf.reshape(parsed_features['boxes'], [num_persons, 4])
        # they are only used to guide the data augmentation (when doing a random crop)

        # get keypoints, they are in absolute coordinates
        keypoints = tf.to_int32(parsed_features['keypoints'])
        keypoints = tf.reshape(keypoints, [num_persons, 17, 3])

        # get size of masks, they are downsampled
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]
        masks_height = tf.to_int32(tf.ceil(image_height/DOWNSAMPLE))
        masks_width = tf.to_int32(tf.ceil(image_width/DOWNSAMPLE))
        # (we use the 'SAME' padding in the networks)

        # get masks (loss and segmentation masks)
        masks = tf.decode_raw(parsed_features['masks'], tf.uint8)
        # unpack bits (reverse np.packbits)
        b = tf.constant([128, 64, 32, 16, 8, 4, 2, 1], dtype=tf.uint8)
        masks = tf.reshape(tf.bitwise.bitwise_and(masks[:, None], b), [-1])
        masks = masks[:(masks_height * masks_width * 2)]
        masks = tf.cast(masks > 0, tf.uint8)

        # reshape to the initial form
        masks = tf.reshape(masks, [masks_height, masks_width, 2])
        masks = tf.to_float(masks)  # it has binary values only

        if self.is_training:
            image, masks, keypoints = self.augmentation(image, masks, boxes, keypoints)
            image_height, image_width = self.image_size, self.image_size
        else:
            image, _ = resize_keeping_aspect_ratio(image, self.min_dimension, DIVISOR)

        sigma = 1.0
        heatmaps = tf.py_func(
            lambda k, w, h: get_heatmaps(k, DOWNSAMPLE, sigma, w, h),
            [tf.to_float(keypoints), image_width, image_height],
            tf.float32, stateful=False
        )

        if self.is_training:
            heatmaps_height = math.ceil(self.image_size/DOWNSAMPLE)
            heatmaps_width = math.ceil(self.image_size/DOWNSAMPLE)
            heatmaps.set_shape([heatmaps_height, heatmaps_width, 17])
        else:
            heatmaps.set_shape([None, None, 17])

        features = {'images': image}
        labels = {'heatmaps': heatmaps, 'masks': masks}
        return features, labels

    def augmentation(self, image, masks, boxes, keypoints):
        image, masks, boxes, keypoints = random_rotation(image, masks, boxes, keypoints, max_angle=45, probability=0.9)
        image, masks, keypoints = randomly_crop_and_resize(image, masks, boxes, keypoints, self.image_size, probability=0.9)
        image = random_color_manipulations(image, probability=0.5, grayscale_probability=0.1)
        image = random_pixel_value_scale(image, probability=0.1, minval=0.9, maxval=1.1)
        image, masks, keypoints = random_flip_left_right(image, masks, keypoints)
        return image, masks, keypoints


def randomly_crop_and_resize(image, masks, boxes, keypoints, image_size, probability=0.5):
    """
    Arguments:
        image: a float tensor with shape [height, width, 3].
        masks: a float tensor with shape [height/DOWNSAMPLE, width/DOWNSAMPLE].
        boxes: a float tensor with shape [num_boxes, 4].
        keypoints: an int tensor with shape [num_persons, 17, 3]
        image_size: a list with two integers [new_height, new_width].
        probability: a float number.
    Returns:
        image: a float tensor with shape [new_height, new_width, 3].
        masks: a float tensor with shape [new_height/DOWNSAMPLE, new_width/DOWNSAMPLE].
        keypoints: an int tensor with shape [num_persons, 17, 3],
            note that it has the same shape, but some points became not visible.
    """

    height = tf.to_float(tf.shape(image)[0])
    width = tf.to_float(tf.shape(image)[1])
    scaler = tf.stack([height, width, height, width])
    boxes /= scaler  # to the [0, 1] range

    def crop(image, boxes):
        image, _, window, _ = random_crop(
            image, boxes,
            min_object_covered=0.9,
            aspect_ratio_range=(0.95, 1.05),
            area_range=(0.5, 1.0),
            overlap_threshold=0.3
        )
        return image, window

    whole_image_window = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32)

    do_it = tf.less(tf.random_uniform([]), probability)
    image, window = tf.cond(
        do_it, lambda: crop(image, boxes),
        lambda: (image, whole_image_window)
    )
    image = tf.image.resize_images(image, image_size, method=RESIZE_METHOD)

    # resize masks
    masks_height = math.ceil(image_size[0]/DOWNSAMPLE)
    masks_width = math.ceil(image_size[1]/DOWNSAMPLE)
    masks = tf.image.crop_and_resize(
        image=tf.expand_dims(masks, 0),
        boxes=tf.expand_dims(window, 0),
        box_ind=tf.constant([0], dtype=tf.int32),
        crop_size=[masks_height, masks_width],
        method='nearest'
    )
    masks = masks[0]

    # to absolute coordinates
    window *= scaler

    # now remove keypoints that are outside of the window
    ymin, xmin, ymax, xmax = tf.unstack(window)
    points, v = tf.split(keypoints, [2, 1], axis=2)
    points = tf.to_float(points)
    y, x = tf.unstack(points, axis=2)  # they have shape [num_persons, 17]

    coordinate_violations = tf.stack([
        tf.greater_equal(y, ymax), tf.greater_equal(x, xmax),
        tf.less_equal(y, ymin), tf.less_equal(x, xmin)
    ], axis=2)  # shape [num_persons, 17, 4]
    valid_indicator = tf.logical_not(tf.reduce_any(coordinate_violations, axis=2))
    valid_indicator = tf.expand_dims(valid_indicator, 2)  # shape [num_persons, 17, 1]
    v *= tf.to_int32(valid_indicator)

    translation = tf.stack([ymin, xmin])  # translate coordinate system
    scaler = tf.stack([tf.to_float(image_size[0])/(ymax - ymin), tf.to_float(image_size[1])/(xmax - xmin)])

    points = tf.to_int32(tf.round(scaler * (points - translation)))
    y, x = tf.unstack(points, axis=2)
    y = tf.clip_by_value(y, 0, image_size[0] - 1)
    x = tf.clip_by_value(x, 0, image_size[1] - 1)
    keypoints = tf.concat([tf.stack([y, x], axis=2), v], axis=2)

    return image, masks, keypoints


def random_flip_left_right(image, masks, keypoints):

    def flip(image, masks, keypoints):
        flipped_image = tf.image.flip_left_right(image)
        flipped_masks = tf.image.flip_left_right(masks)

        y, x, v = tf.unstack(keypoints, axis=2)
        width = tf.shape(image)[1]
        flipped_x = tf.subtract(width - 1, x)
        flipped_keypoints = tf.stack([y, flipped_x, v], axis=2)

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
        flipped_keypoints = tf.gather(flipped_keypoints, correct_order, axis=1)

        return flipped_image, flipped_masks, flipped_keypoints

    with tf.name_scope('random_flip_left_right'):
        do_it = tf.less(tf.random_uniform([]), 0.5)
        image, masks, keypoints = tf.cond(
            do_it,
            lambda: flip(image, masks, keypoints),
            lambda: (image, masks, keypoints)
        )
        return image, masks, keypoints
