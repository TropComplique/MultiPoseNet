import math
import tensorflow.compat.v1 as tf
from detector.constants import DOWNSAMPLE


def random_rotation(image, masks, boxes, keypoints, max_angle=45, probability=0.9):
    """
    This function takes a random box and rotates everything around its center.
    Then it translates the image's center to be at the box's center.

    Arguments:
        image: a float tensor with shape [height, width, 3].
        masks: a float tensor with shape [mask_height, mask_width, 2],
            they are smaller than the image in DOWNSAMPLE times.
        boxes: a float tensor with shape [num_persons, 4].
        keypoints: an int tensor with shape [num_persons, 17, 3].
        max_angle: an integer.
        probability: a float number.
    Returns:
        image: a float tensor with shape [height, width, 3].
        masks: a float tensor with shape [mask_height, mask_width, 2].
        boxes: a float tensor with shape [num_remaining_boxes, 4],
            where num_remaining_boxes <= num_persons.
        keypoints: an int tensor with shape [num_persons, 17, 3],
            note that some keypoints might be out of the image, but
            we will correct that after doing a random crop.
    """
    def rotate(image, masks, boxes, keypoints):
        with tf.name_scope('random_rotation'):

            # find the center of the image
            image_height = tf.to_float(tf.shape(image)[0])
            image_width = tf.to_float(tf.shape(image)[1])
            image_center = tf.reshape(0.5*tf.stack([image_height, image_width]), [1, 2])

            # get a random bounding box
            box = tf.random_shuffle(boxes)[0]
            ymin, xmin, ymax, xmax = tf.unstack(box)
            box_height, box_width = ymax - ymin, xmax - xmin

            # get the center of the box
            cy, cx = ymin + 0.5*box_height, xmin + 0.5*box_width

            # we will rotate around the box's center,
            # but the center mustn't be too near to the border of the image:
            cy = tf.clip_by_value(cy, 0.25*image_height, 0.75*image_height)
            cx = tf.clip_by_value(cx, 0.2*image_width, 0.8*image_width)
            box_center = tf.reshape(tf.stack([cy, cx]), [1, 2])

            # this changes the center of the image
            center_translation = box_center - image_center

            # to radians
            max_angle_radians = max_angle*(math.pi/180.0)

            # if the center is too near the borders then
            # reduce the maximal rotation angle (where 0.6 = 0.3/0.5)
            max_angle_radians *= (0.6 - 2.0*tf.abs(cx - image_width*0.5)/image_width)/0.6

            # get a random angle
            theta = tf.random_uniform(
                [], minval=-max_angle_radians,
                maxval=max_angle_radians, dtype=tf.float32
            )

            # the distance to the nearest border
            gamma = tf.minimum(cx, image_width - cx)

            # this minimizes the amount of zero padding after rescaling, i believe
            necessary_scale = image_width/(2.0*gamma)
            # always necessary_scale >= 1

            size_ratio = image_width/box_width

            # new box width is maximum one third of the image width
            max_scale = size_ratio/3.0

            min_scale = tf.minimum(tf.maximum(size_ratio/8.0, necessary_scale), max_scale - 1e-4)
            # now always min_scale < max_scale

            # get a random image scaler
            scale = tf.random_uniform([], minval=min_scale, maxval=max_scale, dtype=tf.float32)

            rotation = tf.stack([
                tf.cos(theta), tf.sin(theta),
                -tf.sin(theta), tf.cos(theta)
            ], axis=0)
            rotation_matrix = scale * tf.reshape(rotation, [2, 2])
            # not strictly a rotation, but a rotation with scaling

            # rotate boxes
            ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
            p1 = tf.stack([ymin, xmin], axis=1)  # top left
            p2 = tf.stack([ymin, xmax], axis=1)  # top right
            p3 = tf.stack([ymax, xmin], axis=1)  # buttom left
            p4 = tf.stack([ymax, xmax], axis=1)  # buttom right
            points = tf.concat([p1, p2, p3, p4], axis=0)
            points = tf.matmul(points - box_center, rotation_matrix) + box_center - center_translation
            p1, p2, p3, p4 = tf.split(points, num_or_size_splits=4, axis=0)

            # get boxes that contain the original boxes
            ymin = tf.minimum(p1[:, 0], p2[:, 0])
            ymax = tf.maximum(p3[:, 0], p4[:, 0])
            xmin = tf.minimum(p1[:, 1], p3[:, 1])
            xmax = tf.maximum(p2[:, 1], p4[:, 1])
            ymin = tf.clip_by_value(ymin, 0.0, image_height)
            xmin = tf.clip_by_value(xmin, 0.0, image_width)
            ymax = tf.clip_by_value(ymax, 0.0, image_height)
            xmax = tf.clip_by_value(xmax, 0.0, image_width)

            # in the case if some boxes went over the border too much
            area = (xmax - xmin) * (ymax - ymin)
            boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
            boxes = tf.boolean_mask(boxes, area >= 36)

            # rotate keypoints
            points, v = tf.split(keypoints, [2, 1], axis=2)
            # they have shapes [num_persons, 17, 2] and [num_persons, 17, 1]
            num_persons = tf.shape(points)[0]
            points = tf.reshape(tf.to_float(points), [num_persons * 17, 2])
            points = tf.matmul(points - box_center, rotation_matrix) + box_center - center_translation
            points = tf.reshape(tf.to_int32(tf.round(points)), [num_persons, 17, 2])
            keypoints = tf.concat([points, v], axis=2)

            # `tf.contrib.image.transform` needs inverse transform
            inverse_scale = 1.0 / scale
            inverse_rotation = tf.stack([
                tf.cos(theta), -tf.sin(theta),
                tf.sin(theta), tf.cos(theta)
            ], axis=0)
            inverse_rotation_matrix = inverse_scale * tf.reshape(inverse_rotation, [2, 2])

            # rotate the image
            translate = box_center - tf.matmul(box_center - center_translation, inverse_rotation_matrix)
            translate_y, translate_x = tf.unstack(tf.squeeze(translate, axis=0), axis=0)
            transform = tf.stack([
                inverse_scale * tf.cos(theta), -inverse_scale * tf.sin(theta), translate_x,
                inverse_scale * tf.sin(theta), inverse_scale * tf.cos(theta), translate_y,
                0.0, 0.0
            ])
            image = tf.contrib.image.transform(image, transform, interpolation='BILINEAR')

            # masks are smaller than the image
            scaler = tf.to_float(tf.stack([1.0/DOWNSAMPLE, 1.0/DOWNSAMPLE]))
            box_center *= scaler
            center_translation *= scaler

            # rotate masks
            translate = box_center - tf.matmul(box_center - center_translation, inverse_rotation_matrix)
            translate_y, translate_x = tf.unstack(tf.squeeze(translate, axis=0), axis=0)
            transform = tf.stack([
                inverse_scale * tf.cos(theta), -inverse_scale * tf.sin(theta), translate_x,
                inverse_scale * tf.sin(theta), inverse_scale * tf.cos(theta), translate_y,
                0.0, 0.0
            ])
            masks = tf.contrib.image.transform(masks, transform, interpolation='NEAREST')
            # masks are binary so we use the nearest neighbor interpolation

            return image, masks, boxes, keypoints

    do_it = tf.less(tf.random_uniform([]), probability)
    image, masks, boxes, keypoints = tf.cond(
        do_it,
        lambda: rotate(image, masks, boxes, keypoints),
        lambda: (image, masks, boxes, keypoints)
    )
    return image, masks, boxes, keypoints
