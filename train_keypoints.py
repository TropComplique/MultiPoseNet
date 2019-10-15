import os
import tensorflow.compat.v1 as tf
from keypoints_model import model_fn, RestoreMovingAverageHook
from detector.input_pipeline import KeypointPipeline


PARAMS = {
    'model_dir': 'models/run00/',
    'train_dataset': '/home/dan/datasets/COCO/multiposenet/train/',
    'val_dataset': '/home/dan/datasets/COCO/multiposenet/val/',
    'pretrained_checkpoint': 'pretrained/mobilenet_v1_1.0_224.ckpt',

    'backbone': 'mobilenet',
    'depth_multiplier': 1.0,
    'weight_decay': 2e-3,

    'num_steps': 200000,
    'initial_learning_rate': 5e-4,

    'min_dimension': 512,
    'batch_size': 8,
    'image_size': (512, 512)
}


def get_input_fn(is_training=True):

    dataset_path = PARAMS['train_dataset'] if is_training else PARAMS['val_dataset']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    def input_fn():
        pipeline = KeypointPipeline(filenames, is_training, PARAMS)
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
