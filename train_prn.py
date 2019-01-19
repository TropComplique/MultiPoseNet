import os
import tensorflow as tf
from keypoints_model import RestoreMovingAverageHook
from prn_model import model_fn
from detector.input_pipeline import PoseResidualNetworkPipeline as Pipeline
tf.logging.set_verbosity('INFO')


GPU_TO_USE = '0'
PARAMS = {
    'model_dir': 'models/run02/',
    'train_dataset': '/home/dan/datasets/COCO/multiposenet/train/',
    'val_dataset': '/home/dan/datasets/COCO/multiposenet/val/',

    'weight_decay': 5e-5,
    'num_steps': 150000,
    'initial_learning_rate': 7e-4,

    'batch_size': 8,
}


def get_input_fn(is_training=True):

    dataset_path = PARAMS['train_dataset'] if is_training else PARAMS['val_dataset']
    batch_size = PARAMS['batch_size']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    def input_fn():
        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(filenames, is_training, batch_size)
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
estimator = tf.estimator.Estimator(
    model_fn, params=PARAMS, config=run_config
)


train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=PARAMS['num_steps'])
eval_spec = tf.estimator.EvalSpec(
    val_input_fn, steps=None, start_delay_secs=3600 * 2, throttle_secs=3600 * 2,
    hooks=[RestoreMovingAverageHook(PARAMS['model_dir'])]
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
