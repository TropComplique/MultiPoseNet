import tensorflow as tf
import math


def random_rotation(image, masks, boxes, keypoints, max_angle=45):
    """
    This function takes a random box and rotates everything around its center.
    Then it translates the image's center to be at the box's center.

    Arguments:
        image: a float tensor with shape [height, width, 3].
        masks: a float tensor with shape [mask_height, mask_width, 2].
        boxes: a float tensor with shape [num_persons, 4].
        keypoints: a float tensor with shape [num_persons, 18, 3].
        max_angle: an integer.
    Returns:
        image: a float tensor with shape [height, width, 3].
        masks: a float tensor with shape [mask_height, mask_width, 2].
        boxes: a float tensor with shape [num_remaining_boxes, 4],
            where num_remaining_boxes <= num_persons.
        keypoints: a float tensor with shape [num_persons, 18, 3],
            note that some keypoints might be out of the image, but
            we will correct that after doing a random crop.
    """
    with tf.name_scope('random_rotation'):

        # get a random angle
        max_angle_radians = max_angle*(math.pi/180.0)
        theta = tf.random_uniform(
            [], minval=-max_angle_radians,
            maxval=max_angle_radians, dtype=tf.float32
        )

        # find the center of the image
        height = tf.to_float(tf.shape(image)[0])
        width = tf.to_float(tf.shape(image)[1])
        image_center = tf.reshape(0.5*tf.stack([height, width]), [1, 2])

        # get a random bounding box
        box = tf.random_shuffle(boxes)[0]
        ymin, xmin, ymax, xmax = tf.unstack(box)
        box_height, box_width = ymax - ymin, xmax - xmin

        # get the center of the box
        cy, cx = ymin + 0.5*box_height, xmin + 0.5*box_width

        # we will rotate around the box's center,
        # but the center mustn't be too near to the border of the image:
        cy = tf.clip_by_value(cy, 0.15*height, 0.85*height)
        cx = tf.clip_by_value(cx, 0.15*width, 0.85*width)
        box_center = tf.reshape(tf.stack([cy, cx]), [1, 2])

        # this changes the center of the image
        center_translation = box_center - image_center

        # get a random image scaler
        size_ratio = box_width/width
        scale = tf.random_uniform(
            [], minval=tf.minimum(5.0*size_ratio, 0.5),
            maxval=tf.minimum(15.0*size_ratio, 1.5),
            dtype=tf.float32
        )

        rotation = tf.stack([
            tf.cos(theta), tf.sin(theta),
            -tf.sin(theta), tf.cos(theta)
        ], axis=0)
        rotation_matrix = (1.0/scale) * tf.reshape(rotation, [2, 2])
        # not strictly a rotation, but a rotation with scaling

        inverse_rotation = tf.stack([
            tf.cos(theta), -tf.sin(theta),
            tf.sin(theta), tf.cos(theta)
        ], axis=0)
        inverse_rotation_matrix = scale * tf.reshape(inverse_rotation, [2, 2])

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
        ymin = tf.clip_by_value(ymin, 0.0, height)
        xmin = tf.clip_by_value(xmin, 0.0, width)
        ymax = tf.clip_by_value(ymax, 0.0, height)
        xmax = tf.clip_by_value(xmax, 0.0, width)

        # in the case if some boxes went over the border
        area = (xmax - xmin) * (ymax - ymin)
        valid_boxes = tf.squeeze(tf.where(area >= 64), axis=1)
        boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
        boxes = tf.gather(boxes, valid_boxes)

        # rotate keypoints
        points, v = tf.split(keypoints, [2, 1], axis=2)
        # they have shapes [num_persons, 18, 2] and [num_persons, 18, 1]
        num_persons = tf.shape(points)[0]
        points = tf.reshape(tf.to_float(points), [num_persons * 17, 2])
        points = tf.matmul(points - box_center, rotation_matrix) + box_center - center_translation
        points = tf.reshape(tf.to_int32(tf.round(points)), [num_persons, 17, 2])
        keypoints = tf.concat([points, v], axis=2)

        # rotate the image
        translate = box_center - tf.matmul(box_center - center_translation, inverse_rotation_matrix)
        translate_y, translate_x = tf.unstack(tf.squeeze(translate, axis=0), axis=0)
        transform = tf.stack([
            scale * tf.cos(theta), -scale * tf.sin(theta), translate_x,
            scale * tf.sin(theta), scale * tf.cos(theta), translate_y,
            0.0, 0.0
        ])
        image = tf.contrib.image.transform(image, transform, interpolation='BILINEAR')

        # find the center of rotation for the masks
        masks_height = tf.to_float(tf.shape(masks)[0])
        masks_width = tf.to_float(tf.shape(masks)[1])
        scaler = tf.stack([masks_height/height, masks_width/width])

        # rotate masks
        translate = box_center * scaler - tf.matmul(scaler * (box_center - center_translation), inverse_rotation_matrix)
        translate_y, translate_x = tf.unstack(tf.squeeze(translate, axis=0), axis=0)
        transform = tf.stack([
            scale * tf.cos(theta), -scale * tf.sin(theta), translate_x,
            scale * tf.sin(theta), scale * tf.cos(theta), translate_y,
            0.0, 0.0
        ])
        masks = tf.contrib.image.transform(masks, transform, interpolation='NEAREST')
        # masks are binary so we use the nearest neighbor interpolation

        return image, masks, boxes, keypoints
