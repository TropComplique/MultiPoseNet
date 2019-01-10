import tensorflow as tf


# this is used for polyak averaging
MOVING_AVERAGE_DECAY = 0.993

# all image sizes must be divisible by this value
DIVISOR = 128

# it is important to set this right
BATCH_NORM_MOMENTUM = 0.993
BATCH_NORM_EPSILON = 1e-5

# or 'channels_last'
DATA_FORMAT = 'channels_first'

# number of body landmarks that will be predicted
NUM_KEYPOINTS = 17

# all heatmaps and masks are downsampled
DOWNSAMPLE = 4

# a small value
EPSILON = 1e-8

# this is used when we are doing box encoding/decoding
SCALE_FACTORS = [10.0, 10.0, 5.0, 5.0]

# here are input pipeline settings,
# you need to tweak these numbers for your system,
# it can accelerate training
SHUFFLE_BUFFER_SIZE = 5000
NUM_PARALLEL_CALLS = 12

# images are resized before feeding them to the network
RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR

# thresholds for iou when creating training targets
POSITIVES_THRESHOLD = 0.5
NEGATIVES_THRESHOLD = 0.5

# this is used in tf.map_fn when creating training targets or doing nms
PARALLEL_ITERATIONS = 10
