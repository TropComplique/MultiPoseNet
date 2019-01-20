import numpy as np
from PIL import ImageDraw


"""
The keypoint order:
0: 'nose',
1: 'left eye', 2: 'right eye',
3: 'left ear', 4: 'right ear',
5: 'left shoulder', 6: 'right shoulder',
7: 'left elbow', 8: 'right elbow',
9: 'left wrist', 10: 'right wrist',
11: 'left hip', 12: 'right hip',
13: 'left knee', 14: 'right knee',
15: 'left ankle', 16: 'right ankle'
"""


EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (3, 5), (4, 6),
    (5, 11), (6, 12)
]


def get_keypoints(heatmaps, box, threshold):
    """
    Arguments:
        heatmaps: a numpy float array with shape [h, w, 17].
        box: a numpy array with shape [4].
        threshold: a float number.
    Returns:
        a numpy int array with shape [17, 3].
    """
    keypoints = np.zeros([17, 3], dtype='int32')

    ymin, xmin, ymax, xmax = box
    height, width = ymax - ymin, xmax - xmin
    h, w, _ = heatmaps.shape

    for j in range(17):
        mask = heatmaps[:, :, j]
        if mask.max() > threshold:
            y, x = np.unravel_index(mask.argmax(), mask.shape)
            y = np.clip(int(y * height/h), 0, height)
            x = np.clip(int(x * width/w), 0, width)
            keypoints[j] = np.array([x, y, 1])

    return keypoints


def draw_pose(draw, keypoints, box):
    """
    Arguments:
        draw: an instance of ImageDraw.Draw.
        keypoints: a numpy int array with shape [17, 3].
        box: a numpy int array with shape [4].
    """
    ymin, xmin, ymax, xmax = box
    keypoints += np.array([xmin, ymin, 0])

    for (p, q) in EDGES:

        x1, y1, v1 = keypoints[p]
        x2, y2, v2 = keypoints[q]

        both_visible = v1 > 0 and v2 > 0
        if both_visible:
            draw.line([(x1, y1), (x2, y2)])

    for j in range(17):
        x, y, v = keypoints[j]
        if v > 0:
            s = 8
            draw.ellipse([
                (x - s, y - s),
                (x + s, y + s)
            ], fill='red')


def draw_everything(image, outputs):

    image_copy = image.copy()
    image_copy.putalpha(255)
    draw = ImageDraw.Draw(image_copy, 'RGBA')
    width, height = image_copy.size
    scaler = np.array([height, width, height, width])

    n = outputs['num_boxes']
    boxes = scaler * outputs['boxes']
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red')

    mask = outputs['keypoint_heatmaps'][:, :, 0]#outputs['segmentation_masks']
    m, M = mask.min(), mask.max()
    mask = (mask - m)/(M - m)
    mask = np.expand_dims(mask, 2)
    color = np.array([255, 255, 255])
    mask = Image.fromarray((mask*color).astype('uint8'))
    mask.putalpha(mask.convert('L'))
    mask = mask.resize((width, height))
    image_copy.alpha_composite(mask)
    return image_copy
