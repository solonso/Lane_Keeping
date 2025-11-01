import cv2
import numpy as np


class WarpConfig:
    def __init__(self, src=None, dst=None):
        self.src = src
        self.dst = dst


def default_vertices(width, height):
    src = np.float32([
        [width * 0.42, height * 0.62],
        [width * 0.58, height * 0.62],
        [width * 0.90, height - 1],
        [width * 0.10, height - 1],
    ])

    dst = np.float32([
        [width * 0.25, 0],
        [width * 0.75, 0],
        [width * 0.75, height],
        [width * 0.25, height],
    ])
    return src, dst


def compute_warp_matrices(image_shape, config=None):
    height, width = image_shape
    if config and config.src is not None and config.dst is not None:
        src = config.src.astype(np.float32)
        dst = config.dst.astype(np.float32)
    else:
        src, dst = default_vertices(width, height)
    matrix = cv2.getPerspectiveTransform(src, dst)
    inverse = cv2.getPerspectiveTransform(dst, src)
    return matrix, inverse


def warp_image(image, matrix):
    height, width = image.shape[:2]
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)


def unwarp_points(points, inverse):
    ones = np.ones((points.shape[0], 1))
    homogeneous = np.hstack((points, ones))
    transformed = inverse.dot(homogeneous.T).T
    transformed /= transformed[:, 2][:, None]
    return transformed[:, :2]
