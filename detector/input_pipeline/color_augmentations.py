import tensorflow.compat.v1 as tf


"""
`image` is assumed to be a float tensor with shape [height, width, 3],
it is a RGB image with pixel values in the range [0, 1].
"""


def random_color_manipulations(image, probability=0.5, grayscale_probability=0.1):
    """
    This function randomly changes color of the image.
    It is taken from here:
    https://cloud.google.com/tpu/docs/inception-v3-advanced
    """
    def manipulate(image):
        with tf.name_scope('distort_color_fast'):
            br_delta = tf.random_uniform([], -32.0/255.0, 32.0/255.0)
            cb_factor = tf.random_uniform([], -0.1, 0.1)
            cr_factor = tf.random_uniform([], -0.1, 0.1)
            channels = tf.split(axis=2, num_or_size_splits=3, value=image)
            red_offset = 1.402 * cr_factor + br_delta
            green_offset = -0.344136 * cb_factor - 0.714136 * cr_factor + br_delta
            blue_offset = 1.772 * cb_factor + br_delta
            channels[0] += red_offset
            channels[1] += green_offset
            channels[2] += blue_offset
            image = tf.concat(axis=2, values=channels)
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def to_grayscale(image):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
        return image

    with tf.name_scope('random_color_manipulations'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(do_it, lambda: manipulate(image), lambda: image)

    with tf.name_scope('to_grayscale'):
        do_it = tf.less(tf.random_uniform([]), grayscale_probability)
        image = tf.cond(do_it, lambda: to_grayscale(image), lambda: image)

    return image


def random_pixel_value_scale(image, probability=0.5, minval=0.9, maxval=1.1):
    """
    This function scales each pixel
    independently of the other ones.

    Arguments:
        image: a float tensor with shape [height, width, 3],
            an image with pixel values varying between [0, 1].
        probability: a float number.
        minval: a float number, lower ratio of scaling pixel values.
        maxval: a float number, upper ratio of scaling pixel values.
    Returns:
        a float tensor with shape [height, width, 3].
    """
    def random_value_scale(image):
        color_coefficient = tf.random_uniform(
            tf.shape(image), minval=minval,
            maxval=maxval, dtype=tf.float32
        )
        image = tf.multiply(image, color_coefficient)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    with tf.name_scope('random_pixel_value_scale'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(do_it, lambda: random_value_scale(image), lambda: image)
        return image
