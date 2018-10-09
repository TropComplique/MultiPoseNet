import os
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import cv2
import random
import math
import io
from PIL import Image

# you can get this from here:
# https://github.com/cocodataset/cocoapi
import sys
sys.path.append('/home/dan/work/cocoapi/PythonAPI/')
from pycocotools.coco import COCO


"""
Just run:
python create_tfrecords.py

And don't forget set the right paths below.
"""


# paths to downloaded data
IMAGES_DIR = '/home/dan/datasets/COCO/images/'
ANNOTATIONS_DIR = '/home/dan/datasets/COCO/annotations/'

# path where converted data will be stored
RESULT_PATH = '/home/dan/datasets/COCO/multiposenet/'

# because dataset is big we will split it into parts
NUM_TRAIN_SHARDS = 300
NUM_VAL_SHARDS = 1

# all masks are reduced in size
DOWNSAMPLE = 4

# we don't use poorly visible persons
MIN_NUM_KEYPOINTS = 2
MIN_BOX_SIDE = 5


def to_tf_example(image_path, annotations, coco):
    """
    Arguments:
        image_path: a string.
        annotations: a list of dicts.
        coco: an instance of COCO.
    Returns:
        an instance of tf.train.Example.
    """

    with tf.gfile.GFile(image_path, 'rb') as f:
        encoded_jpg = f.read()

    # check image format
    image = Image.open(io.BytesIO(encoded_jpg))
    if not image.format == 'JPEG':
        return None

    width, height = image.size
    if image.mode == 'L':  # if grayscale
        rgb_image = np.stack(3*[np.array(image)], axis=2)
        encoded_jpg = to_jpeg_bytes(rgb_image)
        image = Image.open(io.BytesIO(encoded_jpg))
    assert image.mode == 'RGB'
    assert width > 0 and height > 0

    # whether to use a pixel for computing the loss
    loss_mask = np.ones((height, width), dtype='bool')

    # whether there is a person on a pixel
    segmentation_mask = np.zeros((height, width), dtype='bool')

    boxes, keypoints = [], []
    for a in annotations:

        xmin, ymin, w, h = a['bbox']
        xmax, ymax = xmin + w, ymin + h

        # sometimes boxes go over the edges
        ymin = np.clip(ymin, 0.0, height)
        xmin = np.clip(xmin, 0.0, width)
        ymax = np.clip(ymax, 0.0, height)
        xmax = np.clip(xmax, 0.0, width)

        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)

        h, w = ymax - ymin, xmax - xmin

        # do not add barely visible people,
        # do not add small boxes
        is_bad = (a['num_keypoints'] < MIN_NUM_KEYPOINTS) or\
            (h < MIN_BOX_SIDE) or (w < MIN_BOX_SIDE)

        if is_bad:
            unannotated_person_mask = coco.annToMask(a)
            use_this = unannotated_person_mask == 0
            loss_mask = np.logical_and(use_this, loss_mask)
            continue

        person_mask = coco.annToMask(a)
        segmentation_mask = np.logical_or(person_mask == 1, segmentation_mask)

        points = np.array(a['keypoints'], dtype='int64').reshape(17, 3)
        x, y, v = np.split(points, 3, axis=1)
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        points = np.stack([y, x, v], axis=1)  # note the change (x, y) -> (y, x)

        boxes.append((ymin, xmin, ymax, xmax))
        keypoints.append(points)

    # every image must have boxes
    if len(boxes) < 1:
        return None

    num_persons = len(boxes)
    boxes = np.array(boxes, dtype='float32')
    keypoints = np.stack(keypoints, axis=0).astype('int64')

    # downsample and encode masks
    masks_width, masks_height = math.ceil(width/DOWNSAMPLE), math.ceil(height/DOWNSAMPLE)
    masks = np.stack([loss_mask, segmentation_mask], axis=2)
    masks = masks.astype('uint8')
    masks = cv2.resize(masks, (masks_width, masks_height), cv2.INTER_LANCZOS4)
    masks = np.packbits(masks > 0)
    # we use `ceil` because of the 'SAME' padding

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(encoded_jpg),
        'num_persons': _int64_feature(num_persons),
        'boxes': _float_list_feature(boxes.reshape(-1)),
        'keypoints': _int64_list_feature(keypoints.reshape(-1)),
        'masks': _bytes_feature(masks.tostring())
    }))
    return example


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def to_jpeg_bytes(array):
    image = Image.fromarray(array)
    tmp = io.BytesIO()
    image.save(tmp, format='jpeg')
    return tmp.getvalue()


def convert(coco, image_dir, result_path, num_shards):

    # get all images with people
    catIds = coco.getCatIds(catNms=['person'])
    examples_list = coco.getImgIds(catIds=catIds)

    shutil.rmtree(result_path, ignore_errors=True)
    os.mkdir(result_path)

    # randomize image order
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    print('Number of images:', num_examples)

    shard_size = math.ceil(num_examples/num_shards)
    print('Number of images per shard:', shard_size)

    shard_id = 0
    num_examples_written = 0
    num_skipped_images = 0
    for example in tqdm(examples_list):

        if num_examples_written == 0:
            shard_path = os.path.join(result_path, 'shard-%04d.tfrecords' % shard_id)
            if not os.path.exists(shard_path):
                writer = tf.python_io.TFRecordWriter(shard_path)

        image_metadata = coco.loadImgs(example)[0]
        image_path = os.path.join(image_dir, image_metadata['file_name'])
        annIds = coco.getAnnIds(imgIds=image_metadata['id'], catIds=catIds, iscrowd=None)
        annotations = coco.loadAnns(annIds)

        tf_example = to_tf_example(image_path, annotations, coco)
        if tf_example is None:
            num_skipped_images += 1
            continue
        writer.write(tf_example.SerializeToString())
        num_examples_written += 1

        if num_examples_written == shard_size:
            shard_id += 1
            num_examples_written = 0
            writer.close()

    if num_examples_written != 0:
        writer.close()

    print('Number of skipped images:', num_skipped_images)
    print('Number of shards:', shard_id + 1)
    print('Result is here:', result_path, '\n')


shutil.rmtree(RESULT_PATH, ignore_errors=True)
os.mkdir(RESULT_PATH)

coco = COCO(os.path.join(ANNOTATIONS_DIR, 'person_keypoints_train2017.json'))
image_dir = os.path.join(IMAGES_DIR, 'train2017')
result_path = os.path.join(RESULT_PATH, 'train')
convert(coco, image_dir, result_path, NUM_TRAIN_SHARDS)

coco = COCO(os.path.join(ANNOTATIONS_DIR, 'person_keypoints_val2017.json'))
image_dir = os.path.join(IMAGES_DIR, 'val2017')
result_path = os.path.join(RESULT_PATH, 'val')
convert(coco, image_dir, result_path, NUM_VAL_SHARDS)
