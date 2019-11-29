import tensorflow.compat.v1 as tf
import tensorflow.contrib.slim as slim


def prn(x, is_training):
    """
    Arguments:
        x: a float tensor with shape [b, h, w, c].
        is_training: a boolean.
    Returns:
        a float tensor with shape [b, h, w, c].
    """
    with tf.variable_scope('PRN'):

        b = tf.shape(x)[0]
        _, h, w, c = x.shape.as_list()  # must be static
        x = tf.reshape(x, [b, h * w * c])  # flatten

        with slim.arg_scope([slim.fully_connected], weights_initializer=tf.variance_scaling_initializer()):
            y = slim.fully_connected(x, 1024, activation_fn=tf.nn.relu, scope='fc1')
            # y = slim.dropout(y, keep_prob=0.5, is_training=is_training)
            y = slim.fully_connected(y, h * w * c, activation_fn=tf.nn.relu, scope='fc2')

        x += y
        return tf.reshape(x, [b, h, w, c])
