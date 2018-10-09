import tensorflow as tf

DIVISOR = 128
BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-5
DATA_FORMAT = 'channels_first'

NUM_KEYPOINTS = 17

# all heatmaps and masks are downsampled
DOWNSAMPLE = 4

# a small value
EPSILON = 1e-8
# this is used when we are doing box encoding/decoding
SCALE_FACTORS = [10.0, 10.0, 5.0, 5.0]

# here are input pipeline settings.
# you need to tweak these numbers for your system,
# it can accelerate training
SHUFFLE_BUFFER_SIZE = 10000
NUM_PARALLEL_CALLS = 12

# images are resized before feeding them to the network
RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR

# threshold for IoU when creating training targets
MATCHING_THRESHOLD = 0.5

# this is used in tf.map_fn when creating training targets or doing NMS
PARALLEL_ITERATIONS = 10

# this can be important
BATCH_NORM_MOMENTUM = 0.9
