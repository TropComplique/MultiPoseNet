import tensorflow as tf
from detector import RetinaNet
from detector.backbones import mobilenet_v1, resnet
from metrics import Evaluator


MOVING_AVERAGE_DECAY = 0.993


def model_fn(features, labels, mode, params):
    """
    This is a function for creating a computational tensorflow graph.
    The function is in format required by tf.estimator.
    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    def backbone(images, is_training):
        return mobilenet_v1(
            images, is_training,
            depth_multiplier=params['depth_multiplier']
        )

    retinanet = RetinaNet(
        features['images'],
        is_training,
        backbone, params
    )

    # use a pretrained backbone network
    if is_training:
        with tf.name_scope('init_from_checkpoint'):
            # checkpoint_scope = 'ShuffleNetV2/'
            checkpoint_scope = 'MobilenetV1/'
            tf.train.init_from_checkpoint(
                params['pretrained_checkpoint'],
                {checkpoint_scope: checkpoint_scope}
            )

    # add nms to the graph
    if not is_training:
        predictions = retinanet.get_predictions(
            score_threshold=params['score_threshold'],
            iou_threshold=params['iou_threshold'],
            max_boxes_per_class=params['max_boxes_per_class']
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

        evaluator = Evaluator(num_classes=params['num_classes'])
        eval_metric_ops = evaluator.get_metric_ops(labels, predictions)

        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        # learning_rate = tf.train.piecewise_constant(global_step, params['lr_boundaries'], params['lr_values'])
        learning_rate = tf.train.cosine_decay(0.005, global_step, decay_steps=150000)
        tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    with tf.control_dependencies([train_op]), tf.name_scope('ema'):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
        train_op = ema.apply(tf.trainable_variables())

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)
