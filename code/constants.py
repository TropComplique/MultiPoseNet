import tensorflow as tf

# all heatmaps and masks are downsampled
DOWNSAMPLE = 4

# a small value
EPSILON = 1e-8

# this is used when we are doing box encoding/decoding
SCALE_FACTORS = [10.0, 10.0, 5.0, 5.0]

# here are input pipeline settings.
# you need to tweak these numbers for your system,
# it can accelerate training
SHUFFLE_BUFFER_SIZE = 15000
NUM_THREADS = 12

# images are resized before feeding them to the network
RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR

# threshold for IoU when creating training targets
MATCHING_THRESHOLD = 0.5

# this is used in tf.map_fn when creating training targets or doing NMS
PARALLEL_ITERATIONS = 10

# this can be important
BATCH_NORM_MOMENTUM = 0.9
