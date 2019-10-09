import tensorflow.compat.v1 as tf
from detector import prn
from detector.constants import MOVING_AVERAGE_DECAY
from keypoints_model import add_weight_decay


def model_fn(features, labels, mode, params):
    """
    This is a function for creating a computational tensorflow graph.
    The function is in format required by tf.estimator.
    """

    assert mode != tf.estimator.ModeKeys.PREDICT
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    heatmaps = features  # shape [b, h, w, c]
    logits = prn(heatmaps, is_training)
    # it has shape [b, h, w, c]

    with tf.name_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()

    b = tf.shape(heatmaps)[0]
    _, h, w, c = heatmaps.shape.as_list()

    labels = tf.reshape(labels, [b, h * w, c])
    logits = tf.reshape(logits, [b, h * w, c])
    probabilities = tf.nn.softmax(logits, axis=1)

    losses = tf.losses.log_loss(
        labels, probabilities,
        loss_collection=None,
        reduction=tf.losses.Reduction.NONE
    )
    # it has shape [b, h * w, c]

    loss = tf.reduce_mean(losses, axis=[0, 1, 2])
    tf.losses.add_loss(loss)
    tf.summary.scalar('log_loss', loss)
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {'eval_loss': tf.metrics.mean(losses)}
        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.cosine_decay(
            params['initial_learning_rate'], global_step,
            decay_steps=params['num_steps'], alpha=1e-3
        )
        tf.summary.scalar('learning_rate', learning_rate)

    with tf.variable_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    with tf.control_dependencies([train_op]), tf.name_scope('ema'):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
        train_op = ema.apply(tf.trainable_variables())

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)
