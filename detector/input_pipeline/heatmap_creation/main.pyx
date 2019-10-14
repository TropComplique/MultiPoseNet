import cython
import numpy as np
cimport numpy as np


cdef inline float max(float a, float b):
    return a if a >= b else b


cdef inline float min(float a, float b):
    return a if a <= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
def get_heatmaps(
        np.ndarray[float, ndim=3] keypoints,
        np.ndarray[float, ndim=1] sigmas,
        float min_sigma,
        unsigned int width,
        unsigned int height,
        unsigned int downsample):
    """
    Arguments:
        keypoints: a numpy float array with shape [num_persons, 17, 3].
            It is in format (y, x, visibility),
            where coordinates `y, x` are in the ranges
            [0, height - 1] and [0, width - 1].
            And a keypoint is visible if `visibility > 0`.
        sigmas: a numpy float array with shape [num_persons], sizes of the gaussian blobs.
        min_sigma: a float number.
        width, height: integers, size of the original image.
        downsample: an integer.
    Returns:
        a numpy float array with shape [maps_height, maps_width, 17],
        where (maps_height, maps_width) = (height/downsample, width/downsample).
    """

    cdef float w = float(width)
    cdef float h = float(height)
    cdef unsigned int maps_width = int(np.ceil(w/float(downsample)))
    cdef unsigned int maps_height = int(np.ceil(h/float(downsample)))
    cdef np.ndarray[float, ndim=2] body_part
    cdef np.ndarray[float, ndim=1] scaler = np.array([h - 1.0, w - 1.0], dtype='float32')

    heatmaps = []
    for i in range(17):

        # take a particular body part
        body_part = keypoints[keypoints[:, i, 2] > 0, i, :2].astype('float32')  # shape [num_visible, 2]

        if len(body_part) == 0:
            heatmaps.append(np.zeros((maps_height, maps_width), dtype='float32'))
            continue

        # to the [0, 1] range
        body_part /= scaler

        heatmaps_for_part = []
        for i in range(len(body_part)):
            y, x = body_part[i]
            heatmaps_for_part.append(create_heatmap(y, x, sigma, maps_width, maps_height))

        heatmaps.append(np.stack(heatmaps_for_part, axis=2).max(2))

    heatmaps = np.stack(heatmaps, axis=2)
    return heatmaps


@cython.boundscheck(False)
@cython.wraparound(False)
def create_heatmap(
        float center_y,
        float center_x,
        float sigma,
        unsigned int width,
        unsigned int height):

    # coordinates (center_y, center_x) are normalized to the [0, 1] range

    cdef np.ndarray[float, ndim=2] heatmap = np.zeros((height, width), dtype='float32')
    cdef float theta = 4.6052  # -ln(0.01)
    cdef float delta = np.sqrt(theta * 2.0)
    cdef float distance, value
    cdef int y, x, ymin, xmin, ymax, xmax

    center_y = center_y * (float(height) - 1.0)
    center_x = center_x * (float(width) - 1.0)

    ymin = int(max(0.0, center_y - delta * sigma))
    xmin = int(max(0.0, center_x - delta * sigma))
    ymax = int(min(float(height), center_y + delta * sigma))
    xmax = int(min(float(width), center_x + delta * sigma))

    for y in range(ymin, ymax):
        for x in range(xmin, xmax):
            distance = (float(x) - center_x) ** 2 + (float(y) - center_y) ** 2
            value = distance / (2.0*sigma*sigma)
            if value > theta:
                continue
            heatmap[y, x] = min(1.0, np.exp(-value))
    return heatmap
