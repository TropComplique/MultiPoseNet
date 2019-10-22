import tensorflow.compat.v1 as tf
from detector import KeypointSubnet
from detector.backbones import mobilenet_v1


def model_fn(features, labels, mode, params):

    assert mode != tf.estimator.ModeKeys.PREDICT
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    images = features['images']
    # it has shape [b, height, width, 3]

    backbone_features = mobilenet_v1(
        images, is_training,
        params['depth_multiplier']
    )
    subnet = KeypointSubnet(
        backbone_features,
        is_training, params
    )

    # add l2 regularization
    if params['weight_decay'] > 0.0:
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()
        tf.summary.scalar('regularization_loss', regularization_loss)

    losses = {}

    heatmaps = labels['heatmaps']
    # it has shape [b, h, w, 17],
    # where (h, w) = (height / 4, width / 4),
    # and `b` is batch size

    batch_size = tf.shape(heatmaps)[0]
    normalizer = tf.to_float(batch_size)

    segmentation_masks = tf.expand_dims(labels['segmentation_masks'], 3)
    loss_masks = tf.expand_dims(labels['loss_masks'], 3)
    # they have shape [b, h, w, 1]

    predicted_heatmaps = subnet.heatmaps[:, :, :, :17]
    predicted_segmentation_masks = subnet.heatmaps[:, :, :, 17]

    focal_loss = focal_loss(
        heatmaps, labels['num_boxes'],
        predicted_heatmaps, alpha=2.0, beta=4.0
    )  # shape [b, h, w]
    focal_loss = tf.reduce_sum(tf.squeeze(loss_masks, 3) * focal_loss, axis=[0, 1, 2])
    losses['focal_loss'] = focal_loss/normalizer

    regression_loss = tf.nn.l2_loss(loss_masks * (predicted_segmentation_masks - segmentation_masks))
    losses['regression_loss'] = (1.0/normalizer) * regression_loss

    # additional supervision
    # with person segmentation
    for level in range(2, 6):

        x = subnet.enriched_features[f'p{level}']
        x = tf.expand_dims(x[:, :, :, 0], 3)
        # it has shape [b, height / stride, width / stride, 1],
        # where stride is equal to level ** 2

        x = tf.nn.l2_loss(loss_masks * (x - segmentation_masks))
        losses[f'segmentation_loss_at_level_{level}'] = (4.0/normalizer) * x

        shape = tf.shape(segmentation_masks)
        height, width = shape[1], shape[2]
        new_size = [height // 2, width // 2]

        segmentation_masks = tf.image.resize_bilinear(segmentation_masks, new_size)
        loss_masks = tf.image.resize_bilinear(loss_masks, new_size)

    for n, v in losses.items():
        tf.losses.add_loss(v)
        tf.summary.scalar(n, v)
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    with tf.name_scope('eval_metrics'):

        shape = tf.shape(heatmaps)
        height, width = shape[1], shape[2]
        area = tf.to_float(height * width)

        loss_masks = tf.expand_dims(labels['loss_masks'], 3)
        per_pixel_reg_loss = tf.nn.l2_loss(loss_masks * (predicted_heatmaps - heatmaps))/(normalizer * area)
        tf.summary.scalar('per_pixel_reg_loss', per_pixel_reg_loss)

    if mode == tf.estimator.ModeKeys.EVAL:

        eval_metric_ops = {
            'eval_regression_loss': tf.metrics.mean(losses['regression_loss']),
            'eval_focal_loss': tf.metrics.mean(losses['focal_loss']),
            'eval_per_pixel_reg_loss': tf.metrics.mean(per_pixel_reg_loss),
            'eval_segmentation_loss_at_level_2': tf.metrics.mean(losses['segmentation_loss_at_level_2']),
            'eval_segmentation_loss_at_level_5': tf.metrics.mean(losses['segmentation_loss_at_level_5'])
        }

        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.cosine_decay(
            params['initial_learning_rate'], global_step,
            decay_steps=params['num_steps'], alpha=1e-4
        )
        tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        grads_and_vars = [(tf.clip_by_value(g, -200, 200), v) for g, v in grads_and_vars]
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def add_weight_decay(weight_decay):

    trainable_vars = tf.trainable_variables()
    kernels = [
        v for v in trainable_vars
        if ('weights' in v.name or 'kernel' in v.name) and 'depthwise_weights' not in v.name
    ]
    for k in kernels:
        x = tf.multiply(weight_decay, tf.nn.l2_loss(k))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, x)


def focal_loss(heatmaps, num_boxes, predictions, alpha=2.0, beta=4.0):
    """
    This is loss function from here:
    "CornerNet: Detecting Objects as Paired Keypoints"
    (https://arxiv.org/abs/1808.01244)

    Arguments:
        heatmaps: a float tensor with shape [b, h, w, c].
        num_boxes: a long tensor with shape [b].
        predictions: a float tensor with shape [b, h, w, c],
            it represents logits.
        alpha, beta: float numbers.
    Returns:
        a float tensor with shape [b, h, w].
    """

    # notation like in the paper
    y = heatmaps
    y_hat = predictions

    is_extreme_point = tf.equal(y, 1.0)
    # binary tensor with shape [b, h, w, c]

    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=y_hat, labels=tf.to_float(is_extreme_point)
    )  # shape [b, h, w, c]

    # to the [0, 1] range
    y_hat = tf.sigmoid(y_hat)

    weights = tf.where(
        is_extreme_point, tf.pow(1.0 - y_hat, alpha),
        tf.pow(1.0 - y, beta) * tf.pow(y_hat, alpha)
    )  # shape [b, h, w, c]

    b = tf.shape(y)[0]  # batch size
    normalizer = tf.to_float(tf.reshape(num_boxes, [b, 1, 1])) + 1.0
    return tf.reduce_sum(weights * losses, 3)/normalizer
