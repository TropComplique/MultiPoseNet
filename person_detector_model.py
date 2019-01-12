import tensorflow as tf
from detector import RetinaNet
from detector.backbones import mobilenet_v1
from detector.constants import MOVING_AVERAGE_DECAY, DATA_FORMAT
from metrics import Evaluator
from keypoints_model import add_weight_decay


def model_fn(features, labels, mode, params):
    """
    This is a function for creating a computational tensorflow graph.
    The function is in format required by tf.estimator.
    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    def backbone(images, is_training):
        return mobilenet_v1(
            images, is_training=False,
            depth_multiplier=params['depth_multiplier']
        )

    retinanet = RetinaNet(
        features['images'],
        is_training,
        backbone, params
    )

    # add nms to the graph
    if not is_training:
        predictions = retinanet.get_predictions(
            score_threshold=params['score_threshold'],
            iou_threshold=params['iou_threshold'],
            max_detections=params['max_boxes']
        )

    if mode == tf.estimator.ModeKeys.PREDICT:

        box_scaler = features['box_scaler']
        predictions['boxes'] /= box_scaler

        export_outputs = tf.estimator.export.PredictOutput({
            name: tf.identity(tensor, name)
            for name, tensor in predictions.items()
        })
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions,
            export_outputs={'outputs': export_outputs}
        )

    # add l2 regularization
    with tf.name_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()

    # create localization and classification losses
    losses = retinanet.loss(labels, params)
    tf.losses.add_loss(params['localization_loss_weight'] * losses['localization_loss'])
    tf.losses.add_loss(params['classification_loss_weight'] * losses['classification_loss'])
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('localization_loss', losses['localization_loss'])
    tf.summary.scalar('classification_loss', losses['classification_loss'])
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    if mode == tf.estimator.ModeKeys.EVAL:

        batch_size = features['images'].shape[0].value
        assert batch_size == 1

        evaluator = Evaluator()
        eval_metric_ops = evaluator.get_metric_ops(labels, predictions)

        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.cosine_decay(
            params['initial_learning_rate'], global_step,
            decay_steps=params['num_steps']
        )
        tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # backbone network is frozen
        var_list = [v for v in tf.trainable_variables() if 'MobilenetV1' not in v.name]

        grads_and_vars = optimizer.compute_gradients(total_loss, var_list)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    with tf.control_dependencies([train_op]), tf.name_scope('ema'):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
        train_op = ema.apply(tf.trainable_variables())

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)
