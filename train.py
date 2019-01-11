import os
import tensorflow as tf
import json
from keypoints_model import model_fn, RestoreMovingAverageHook
from detector.input_pipeline import KeypointPipeline as Pipeline
tf.logging.set_verbosity('INFO')


GPU_TO_USE = '0'
PARAMS = {
    'model_dir': 'models/run00/',
    'train_dataset': '/home/dan/datasets/COCO/multiposenet/train/',
    'val_dataset': '/home/dan/datasets/COCO/multiposenet/val/',
    'pretrained_checkpoint': 'pretrained/mobilenet_v1_1.0_224.ckpt',
    # 'pretrained_checkpoint': 'models/run00/model-175000',

    'backbone': 'mobilenet',
    'depth_multiplier': 1.0,
    'weight_decay': 2e-3,
    # 'score_threshold': 0.3, 'iou_threshold': 0.6, 'max_boxes_per_class': 25,
    # 'localization_loss_weight': 1.0, 'classification_loss_weight': 2.0,

    # 'gamma': 2.0,
    # 'alpha': 0.25,

    'num_steps': 175000,
    'initial_learning_rate' 5e-4,

    'min_dimension': 512,
    'batch_size': 8,  # 1 epoch ~ 7500 steps
    'image_height': 512,
    'image_width': 512,
}


def get_input_fn(is_training=True):

    dataset_path = params['train_dataset'] if is_training else params['val_dataset']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    def input_fn():
        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(filenames, is_training, params)
        return pipeline.dataset

    return input_fn


session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = GPU_TO_USE
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=PARAMS['model_dir'], session_config=session_config,
    save_summary_steps=200, save_checkpoints_secs=1800,
    log_step_count_steps=1000
)


train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
warm_start = tf.estimator.WarmStartSettings(
    'pretrained/mobilenet_v1_1.0_224.ckpt',
    ['MobilenetV1/*']
)
estimator = tf.estimator.Estimator(
    model_fn, params=params, config=run_config,
    warm_start_from=warm_start
)


train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=params['num_steps'])
eval_spec = tf.estimator.EvalSpec(
    val_input_fn, steps=None, start_delay_secs=3600 * 2, throttle_secs=3600 * 2,
    hooks=[RestoreMovingAverageHook(params['model_dir'])]
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
