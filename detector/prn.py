import tensorflow as tf
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

        # flatten
        b = tf.shape(x)[0]
        h, w, c = x.shape.as_list()[1:]  # must be static
        x = tf.reshape(x, [b, h * w * c])

        params = {
            'activation_fn': tf.nn.relu,
            'weights_initializer': tf.variance_scaling_initializer()
        }
        with slim.arg_scope([slim.fully_connected], **params):
            y = slim.fully_connected(x, 1024, scope='fc1')
            y = slim.dropout(y, keep_prob=0.5, is_training=is_training)
            y = slim.fully_connected(y, h * w * c, scope='fc2')

        x += y
        logits = tf.reshape(x, [b, h, w, c])
        return logits
