import math
import numpy as np
from scipy import signal


def get_heatmaps(keypoints, boxes, width, height, downsample):
    """
    Arguments:
        keypoints: a numpy int array with shape [num_persons, 17, 3].
            It is in format (y, x, visibility),
            where coordinates `y, x` are in the ranges
            [0, height - 1] and [0, width - 1].
            And a keypoint is visible if `visibility > 0`.
        boxes: a numpy float array with shape [num_persons, 4],
            person bounding boxes in absolute coordinates.
        width, height: integers, size of the original image.
        downsample: an integer.
    Returns:
        a numpy float array with shape [height/downsample, width/downsample, 17].
    """

    min_sigma, max_sigma = 1.0, 4.0
    scaler = np.array([height - 1.0, width - 1.0], dtype=np.float32)
    keypoints = keypoints.astype(np.float32)

    # compute output size
    h = math.ceil(height / downsample)
    w = math.ceil(width / downsample)

    ymin, xmin, ymax, xmax = np.split(boxes, 4, axis=1)
    # they have shape [num_persons, 1]

    scale = np.sqrt((ymax - ymin) * (xmax - xmin))
    sigmas = np.squeeze(scale * 0.007, axis=1)

    kernels = []  # each person has different blob size
    sigmas = np.clip(sigmas, min_sigma, max_sigma)

    for sigma in sigmas:
        kernels.append(get_kernel(sigma))

    heatmaps = []
    for i in range(17):

        is_visible = keypoints[:, i, 2] > 0
        num_visible = is_visible.sum()

        if num_visible == 0:
            empty = np.zeros([h, w], dtype=np.float32)
            heatmaps.append(empty)
            continue

        person_id = np.where(is_visible)[0]
        body_part = keypoints[is_visible, i, :2]
        # it has shape [num_visible, 2]

        # to the [0, 1] range
        body_part /= scaler

        heatmaps_for_part = []
        for i in range(num_visible):

            kernel = kernels[person_id[i]]
            y, x = body_part[i]

            heatmap = create_heatmap(y, x, kernel, w, h)
            heatmaps_for_part.append(heatmap)

        heatmaps.append(np.stack(heatmaps_for_part, axis=2).max(2))

    heatmaps = np.stack(heatmaps, axis=2)
    return heatmaps


def get_kernel(std):
    """Returns a 2D Gaussian kernel array."""

    k = np.ceil(np.sqrt(- 2.0 * std**2 * np.log(0.01)))
    # it is true that exp(- 0.5 * k**2 / std**2) < 0.01

    size = 2 * int(k) + 1
    x = signal.windows.gaussian(size, std=std).reshape([size, 1])
    x = np.outer(x, x).astype(np.float32)
    return x


def create_heatmap(y, x, kernel, width, height):
    """
    Arguments:
        y, x: float numbers, normalized to the [0, 1] range.
        kernel: a numpy float array with shape [2 * k + 1, 2 * k + 1].
        width, height: integers.
    Returns:
        a numpy float array with shape [height, width].
    """

    # half kernel size
    k = (kernel.shape[0] - 1) // 2

    x = x * (width - 1)
    y = y * (height - 1)
    x, y = int(round(x)), int(round(y))
    # they are in ranges [0, width - 1] and [0, height - 1]

    xmin, ymin = x - k, y - k
    xmax, ymax = x + k, y + k

    shape = [height + 2 * k, width + 2 * k]
    heatmap = np.zeros(shape, dtype=np.float32)

    # shift coordinates
    xmin, ymin = xmin + k, ymin + k
    xmax, ymax = xmax + k, ymax + k

    heatmap[ymin:(ymax + 1), xmin:(xmax + 1)] = kernel
    heatmap = heatmap[k:-k, k:-k]

    return heatmap
