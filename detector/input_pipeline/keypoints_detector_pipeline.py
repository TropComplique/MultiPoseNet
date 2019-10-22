import tensorflow.compat.v1 as tf
from detector.constants import SHUFFLE_BUFFER_SIZE, NUM_PARALLEL_CALLS, RESIZE_METHOD
from detector.constants import DOWNSAMPLE, DIVISOR, OVERLAP_THRESHOLD
from detector.input_pipeline.random_crop import random_image_crop
from detector.input_pipeline.random_rotation import random_image_rotation
from detector.input_pipeline.color_augmentations import random_color_manipulations, random_pixel_value_scale
from detector.input_pipeline.heatmap_creation import get_heatmaps


class KeypointPipeline:
    """
    Input pipeline for training or evaluating
    networks for heatmaps regression.
    """
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
            min_dimension = params['min_dimension']
            assert min_dimension % DIVISOR == 0
            self.min_dimension = min_dimension
        else:
            batch_size = params['batch_size']
            width, height = params['image_size']
            assert height % DIVISOR == 0
            assert width % DIVISOR == 0
            self.image_size = [height, width]

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

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=1)

        self.dataset = dataset

    def parse_and_preprocess(self, example_proto):
        """
        All heatmaps and masks have values in [0, 1] range.

        Returns:
            image: a float tensor with shape [height, width, 3],
                an RGB image with pixel values in the range [0, 1].
            heatmaps: a float tensor with shape [height / DOWNSAMPLE, width / DOWNSAMPLE, 17].
            loss_masks: a float tensor with shape [height / DOWNSAMPLE, width / DOWNSAMPLE].
            segmentation_masks: a float tensor with shape [height / DOWNSAMPLE, width / DOWNSAMPLE].
            num_boxes: an int tensor with shape [].
        """
        image, masks, boxes, keypoints = self.parse(example_proto)

        if self.is_training:
            image, masks, boxes, keypoints = augmentation(
                image, masks, boxes,
                keypoints, self.image_size
            )
        else:
            image, masks, boxes, keypoints = resize_keeping_aspect_ratio(
                image, masks, boxes, keypoints,
                self.min_dimension, DIVISOR
            )

        shape = tf.shape(image)
        image_height, image_width = shape[0], shape[1]

        heatmaps = tf.py_func(
            lambda k, b, w, h: get_heatmaps(k, b, w, h, DOWNSAMPLE),
            [keypoints, boxes, image_width, image_height],
            tf.float32, stateful=False
        )

        if self.is_training:
            from math import ceil
            height, width = self.image_size
            h = ceil(height/DOWNSAMPLE)
            w = ceil(width/DOWNSAMPLE)
            heatmaps.set_shape([h, w, 17])
        else:
            heatmaps.set_shape([None, None, 17])

        # this is needed for normalization
        num_boxes = tf.shape(boxes)[0]

        features = {'images': image}
        labels = {
            'heatmaps': heatmaps,
            'loss_masks': masks[:, :, 0],
            'segmentation_masks': masks[:, :, 1],
            'num_boxes': num_boxes
        }
        return features, labels

    def parse(self, example_proto):
        """
        Returns:
            image: a float tensor with shape [height, width, 3],
                an RGB image with pixel values in the range [0, 1].
            masks: a float tensor with shape [height / DOWNSAMPLE, width / DOWNSAMPLE, 2].
            boxes: a float tensor with shape [num_persons, 4], in absolute coordinates.
            keypoints: an int tensor with shape [num_persons, 17, 3], in absolute coordinates.
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
        # they are used to guide the data augmentation (when doing a random crop)
        # and to choose sigmas for gaussian blobs

        # get keypoints, they are in absolute coordinates
        keypoints = tf.to_int32(parsed_features['keypoints'])
        keypoints = tf.reshape(keypoints, [num_persons, 17, 3])

        # get size of masks, they are downsampled
        shape = tf.shape(image)
        image_height, image_width = shape[0], shape[1]
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

        return image, masks, boxes, keypoints


def augmentation(image, masks, boxes, keypoints, image_size):
    image, masks, boxes, keypoints = random_image_rotation(image, masks, boxes, keypoints, max_angle=45, probability=0.7)
    image, masks, boxes, keypoints = randomly_crop_and_resize(image, masks, boxes, keypoints, image_size, probability=0.9)
    image = random_color_manipulations(image, probability=0.5, grayscale_probability=0.1)
    image = random_pixel_value_scale(image, probability=0.1, minval=0.9, maxval=1.1)
    image, masks, boxes, keypoints = random_flip_left_right(image, masks, boxes, keypoints)
    return image, masks, boxes, keypoints


def resize_keeping_aspect_ratio(image, masks, boxes, keypoints, min_dimension, divisor):
    """
    This function resizes and possibly pads with zeros.
    When using a usual FPN, divisor must be equal to 128.

    Arguments:
        image: a float tensor with shape [height, width, 3].
        masks: a float tensor with shape [height / DOWNSAMPLE, width / DOWNSAMPLE, 2].
        boxes: a float tensor with shape [num_persons, 4].
        keypoints: an int tensor with shape [num_persons, 17, 3].
        min_dimension, divisor: integers.
    Returns:
        image: a float tensor with shape [h, w, 3],
            where `min_dimension = min(h, w)`,
            `h` and `w` are divisible by `DIVISOR`.
        masks: a float tensor with shape [h / DOWNSAMPLE, w / DOWNSAMPLE, 2].
        boxes: a float tensor with shape [num_persons, 4].
        keypoints: an int tensor with shape [num_persons, 17, 3].
    """

    assert min_dimension % divisor == 0
    min_dimension = tf.constant(min_dimension, dtype=tf.int32)
    divisor = tf.constant(divisor, dtype=tf.int32)

    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    original_min_dim = tf.minimum(height, width)
    scale_factor = tf.to_float(min_dimension / original_min_dim)

    # RESIZE AND PAD IMAGE

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

    # final image size
    h = new_height + pad_height
    w = new_width + pad_width

    # resize keeping aspect ratio
    image = tf.image.resize_images(image, [new_height, new_width], method=RESIZE_METHOD)

    # pad image at the bottom or at the right
    image = tf.image.pad_to_bounding_box(image, offset_height=0, offset_width=0, target_height=h, target_width=w)

    # RESIZE AND PAD MASKS

    # new size of masks with padding
    map_height = tf.to_int32(tf.ceil(h / DOWNSAMPLE))
    map_width = tf.to_int32(tf.ceil(w / DOWNSAMPLE))

    # new size of only masks without padding
    map_only_height = tf.to_int32(tf.ceil(new_height / DOWNSAMPLE))
    map_only_width = tf.to_int32(tf.ceil(new_width / DOWNSAMPLE))

    masks = tf.image.resize_images(
        masks, [map_only_height, map_only_width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    masks = tf.image.pad_to_bounding_box(
        masks, offset_height=0, offset_width=0,
        target_height=map_height, target_width=map_width
    )

    # TRANSFORM KEYPOINTS

    keypoint_scaler = tf.stack([new_height/height, new_width/width])
    keypoint_scaler = tf.to_float(keypoint_scaler)

    points, v = tf.split(keypoints, [2, 1], axis=2)
    points = tf.to_int32(tf.round(tf.to_float(points) * keypoint_scaler))
    y, x = tf.split(points, 2, axis=2)
    y = tf.clip_by_value(y, 0, h - 1)
    x = tf.clip_by_value(x, 0, w - 1)
    keypoints = tf.concat([y, x, v], axis=2)

    # TRANSFORM BOXES

    box_scaler = tf.concat(2 * [keypoint_scaler], axis=0)
    boxes *= box_scaler

    return image, masks, boxes, keypoints


def randomly_crop_and_resize(image, masks, boxes, keypoints, image_size, probability=0.5):
    """
    Arguments:
        image: a float tensor with shape [height, width, 3].
        masks: a float tensor with shape [height / DOWNSAMPLE, width / DOWNSAMPLE, 2].
        boxes: a float tensor with shape [num_persons, 4].
        keypoints: an int tensor with shape [num_persons, 17, 3].
        image_size: a tuple of integers (h, w).
        probability: a float number.
    Returns:
        image: a float tensor with shape [h, w, 3].
        masks: a float tensor with shape [h / DOWNSAMPLE, w / DOWNSAMPLE, 2].
        boxes: a float tensor with shape [num_remaining, 4].
        keypoints: an int tensor with shape [num_remaining, 17, 3].
    """

    shape = tf.to_float(tf.shape(image))
    height, width = shape[0], shape[1]
    scaler = tf.stack([height, width, height, width])
    boxes /= scaler  # to the [0, 1] range

    def crop(image, boxes, keypoints):
        """
        Arguments:
            image: a float tensor with shape [height, width, 3].
            boxes: a float tensor with shape [num_persons, 4].
            keypoints: an int tensor with shape [num_persons, 17, 3].
        Returns:
            image: a float tensor with shape [None, None, 3].
            boxes: a float tensor with shape [num_remaining, 4].
            keypoints: an int tensor with shape [num_remaining, 17, 3].
            window: a float tensor with shape [4].
        """

        image, boxes, window, keep_indices = random_image_crop(
            image, boxes, min_object_covered=0.9,
            aspect_ratio_range=(0.95, 1.05),
            area_range=(0.5, 1.0),
            overlap_threshold=OVERLAP_THRESHOLD
        )

        keypoints = tf.gather(keypoints, keep_indices)
        # it has shape [num_remaining, 17, 3]

        ymin, xmin, ymax, xmax = tf.unstack(window * scaler)
        points, v = tf.split(keypoints, [2, 1], axis=2)
        points = tf.to_float(points)  # shape [num_remaining, 17, 2]

        translation = tf.stack([ymin, xmin])
        points = tf.to_int32(tf.round(points - translation))
        keypoints = tf.concat([points, v], axis=2)

        # note that after this some keypoints will be invisible,
        # so we need to modify the `v` vector later

        return image, boxes, keypoints, window

    whole_image_window = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32)
    do_it = tf.less(tf.random_uniform([]), probability)

    image, boxes, keypoints, window = tf.cond(
        do_it, lambda: crop(image, boxes, keypoints),
        lambda: (image, boxes, keypoints, whole_image_window)
    )

    def correct_keypoints(image_shape, keypoints):
        """
        Arguments:
            image_shape: an int tensor with shape [3].
            keypoints: an int tensor with shape [num_persons, 17, 3].
        Returns:
            an int tensor with shape [num_persons, 17, 3].
        """
        y, x, v = tf.split(keypoints, 3, axis=2)

        height = image_shape[0]
        width = image_shape[1]

        coordinate_violations = tf.concat([
            tf.less(y, 0), tf.less(x, 0),
            tf.greater_equal(y, height),
            tf.greater_equal(x, width)
        ], axis=2)  # shape [num_persons, 17, 4]

        valid_indicator = tf.logical_not(tf.reduce_any(coordinate_violations, axis=2))
        valid_indicator = tf.expand_dims(valid_indicator, 2)
        # it has shape [num_persons, 17, 1]

        v *= tf.to_int32(valid_indicator)
        keypoints = tf.concat([y, x, v], axis=2)
        return keypoints

    def rescale(boxes, keypoints, old_shape, new_shape):
        """
        Arguments:
            boxes: a float tensor with shape [num_persons, 4].
            keypoints: an int tensor with shape [num_persons, 17, 3].
            old_shape, new_shape: int tensors with shape [3].
        Returns:
            a float tensor with shape [num_persons, 4].
            an int tensor with shape [num_persons, 17, 3].
        """
        points, v = tf.split(keypoints, [2, 1], axis=2)
        points = tf.to_float(points)

        old_shape = tf.to_float(old_shape)
        new_shape = tf.to_float(new_shape)
        old_height, old_width = old_shape[0], old_shape[1]
        new_height, new_width = new_shape[0], new_shape[1]

        scaler = tf.stack([new_height/old_height, new_width/old_width])
        points *= scaler

        scaler = tf.stack([new_height, new_width])
        scaler = tf.concat(2 * [scaler], axis=0)
        boxes *= scaler

        new_height = tf.to_int32(new_height)
        new_width = tf.to_int32(new_width)

        points = tf.to_int32(tf.round(points))
        y, x = tf.split(points, 2, axis=2)
        y = tf.clip_by_value(y, 0, new_height - 1)
        x = tf.clip_by_value(x, 0, new_width - 1)
        keypoints = tf.concat([y, x, v], axis=2)
        return boxes, keypoints

    old_shape = tf.shape(image)
    keypoints = correct_keypoints(old_shape, keypoints)

    h, w = image_size  # image size that will be used for training
    image = tf.image.resize_images(image, [h, w], method=RESIZE_METHOD)

    masks_height = tf.to_int32(tf.ceil(h / DOWNSAMPLE))
    masks_width = tf.to_int32(tf.ceil(w / DOWNSAMPLE))

    masks = tf.image.crop_and_resize(
        image=tf.expand_dims(masks, 0),
        boxes=tf.expand_dims(window, 0),
        box_indices=tf.constant([0], dtype=tf.int32),
        crop_size=[masks_height, masks_width],
        method='nearest'
    )
    masks = masks[0]

    boxes, keypoints = rescale(boxes, keypoints, old_shape, tf.shape(image))
    return image, masks, boxes, keypoints


def random_flip_left_right(image, masks, boxes, keypoints):

    def flip(image, masks, boxes, keypoints):

        flipped_image = tf.image.flip_left_right(image)
        flipped_masks = tf.image.flip_left_right(masks)

        y, x, v = tf.unstack(keypoints, axis=2)
        width = tf.shape(image)[1]
        flipped_x = width - 1 - x
        flipped_keypoints = tf.stack([y, flipped_x, v], axis=2)

        width = tf.to_float(width)
        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        flipped_boxes = tf.stack([ymin, width - xmax, ymax, width - xmin], axis=1)

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

        return flipped_image, flipped_masks, flipped_boxes, flipped_keypoints

    do_it = tf.less(tf.random_uniform([]), 0.5)
    image, masks, boxes, keypoints = tf.cond(
        do_it,
        lambda: flip(image, masks, boxes, keypoints),
        lambda: (image, masks, boxes, keypoints)
    )
    return image, masks, boxes, keypoints
