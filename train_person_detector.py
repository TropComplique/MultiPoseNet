import os
import tensorflow.compat.v1 as tf
from keypoints_model import RestoreMovingAverageHook
from detector.input_pipeline import DetectorPipeline
from person_detector_model import model_fn


PARAMS = {
    'model_dir': 'models/run01/',
    'train_dataset': '/home/dan/datasets/COCO/multiposenet/train/',
    'val_dataset': '/home/dan/datasets/COCO/multiposenet/val/',
    'pretrained_checkpoint': 'models/run00/model.ckpt-200000',

    'backbone': 'mobilenet',
    'depth_multiplier': 1.0,
    'weight_decay': 5e-5,

    'score_threshold': 0.3, 'iou_threshold': 0.6, 'max_boxes': 25,
    'localization_loss_weight': 1.0, 'classification_loss_weight': 2.0,

    'gamma': 2.0,
    'alpha': 0.25,

    'num_steps': 150000,
    'initial_learning_rate': 1e-3,

    'min_dimension': 640,
    'batch_size': 16,
    'image_size': (640, 640)
}


def get_input_fn(is_training=True):

    dataset_path = PARAMS['train_dataset'] if is_training else PARAMS['val_dataset']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    def input_fn():
        pipeline = DetectorPipeline(filenames, is_training, PARAMS)
        return pipeline.dataset

    return input_fn


tf.logging.set_verbosity('INFO')
session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = '0'


run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=PARAMS['model_dir'], session_config=session_config,
    save_summary_steps=200, save_checkpoints_secs=1800,
    log_step_count_steps=1000
)


train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
warm_start = tf.estimator.WarmStartSettings(PARAMS['pretrained_checkpoint'], ['MobilenetV1/*'])
estimator = tf.estimator.Estimator(model_fn, params=PARAMS, config=run_config, warm_start_from=warm_start)


train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=PARAMS['num_steps'])
eval_spec = tf.estimator.EvalSpec(
    val_input_fn, steps=None, start_delay_secs=3600, throttle_secs=3600,
    hooks=[RestoreMovingAverageHook(PARAMS['model_dir'])]
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
