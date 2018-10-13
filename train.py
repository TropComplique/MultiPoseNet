import os
import tensorflow as tf
import json
from keypoints_model import model_fn, RestoreMovingAverageHook
from detector.input_pipeline import KeypointPipeline as Pipeline
tf.logging.set_verbosity('INFO')


GPU_TO_USE = '1'
# CONFIG = 'config.json'
# params = json.load(open(CONFIG))

params = {
    "model_dir": "models/run00/",
    "train_dataset": "/home/dan/datasets/COCO/multiposenet/train/",
    "val_dataset": "/home/dan/datasets/COCO/multiposenet/val/",
    "pretrained_checkpoint": "pretrained/mobilenet_v1_1.0_224.ckpt",

    "backbone": "mobilenet",
    "depth_multiplier": 1.0,
    "weight_decay": 2e-5,
#     "num_classes": 80,

#     "score_threshold": 0.05, "iou_threshold": 0.6, "max_boxes_per_class": 25,
#     "localization_loss_weight": 1.0, "classification_loss_weight": 4.0,

#     "use_focal_loss": true,
#     "gamma": 2.0,
#     "alpha": 0.25,

    "num_steps": 90000,
    "lr_boundaries": [75000, 110000],
    "lr_values": [0.01, 0.001, 0.0001],

    "min_dimension": 512,
    "batch_size": 8,
    "image_height": 512,
    "image_width": 512,
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
    model_dir=params['model_dir'], session_config=session_config,
    save_summary_steps=60, save_checkpoints_secs=1800,
    log_step_count_steps=1000
)


train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(model_fn, params=params, config=run_config)


train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=params['num_steps'])
eval_spec = tf.estimator.EvalSpec(
    val_input_fn, steps=None, start_delay_secs=3600 * 3, throttle_secs=3600 * 3,
    hooks=[RestoreMovingAverageHook(params['model_dir'])]
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
