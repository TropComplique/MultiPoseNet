import os
import tensorflow.compat.v1 as tf
from keypoints_model import RestoreMovingAverageHook
from detector.input_pipeline import PoseResidualNetworkPipeline
from prn_model import model_fn


NUM_STEPS_PER_KEYPOINT = 10000
PARAMS = {
    'model_dir': 'models/run02/',
    'train_dataset': '/home/dan/datasets/COCO/multiposenet/train/',
    'val_dataset': '/home/dan/datasets/COCO/multiposenet/val/',

    # 'weight_decay': 1e-9,
    'num_steps': NUM_STEPS_PER_KEYPOINT * 17,
    'initial_learning_rate': 1e-3,

    'batch_size': 32
}


def get_input_fn(is_training=True, max_keypoints=None):

    dataset_path = PARAMS['train_dataset'] if is_training else PARAMS['val_dataset']
    batch_size = PARAMS['batch_size']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    def input_fn():
        pipeline = PoseResidualNetworkPipeline(filenames, is_training, batch_size, max_keypoints)
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


val_input_fn = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(model_fn, params=PARAMS, config=run_config)


for max_keypoints in range(17):
    train_input_fn = get_input_fn(is_training=True, max_keypoints=max_keypoints + 4)
    estimator.train(train_input_fn, steps=NUM_STEPS_PER_KEYPOINT)
    estimator.evaluate(val_input_fn, steps=None, hooks=[RestoreMovingAverageHook(PARAMS['model_dir'])])
