from typing import Tuple

import cv2 as cv
import numpy as np
import torch


def baseline_detect_crack(path: str, thresh: float) -> Tuple[int, float]:
    """
    Predicts anomaly in structure based on amount of edges over the threshold.

    :param path: path to an image
    :param thresh: threshold over which image counts as anomaly
    :return: [int, float] 1 if there is anomaly, and probability of that
    """
    _img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    img_canny = cv.Canny(_img, 50, 200, None, 3)

    prob = np.sum(img_canny > 0) / _img.size
    return int(prob > thresh), prob


def enhanced_detect_crack(path: str, thresh: float, kernel: int = 15) -> Tuple[int, float]:
    """
    Predicts anomaly in structure based on amount of edges over the threshold after bilateral filter.

    :param path: path to an image
    :param thresh: threshold over which image counts as anomaly
    :param kernel: int size of the bilateral filter (the bigger -- the slower is the algo)
    :return: [int, float] 1 if there is anomaly, and probability of that
    """
    _img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    _img = cv.bilateralFilter(_img, kernel, 25, 25)

    img_canny = cv.Canny(_img, 50, 200, None, 3)

    prob = np.sum(img_canny > 0) / _img.size
    return int(prob > thresh), prob


def baseline_trans(img: torch.Tensor) -> np.ndarray:
    _img = img.numpy().transpose(1, 2, 0)

    img_canny = cv.Canny(_img, 50, 200, None, 3)
    return img_canny


def train_trans(img: torch.Tensor) -> np.ndarray:
    _img = img.numpy().transpose(1, 2, 0)

    _img = cv.bilateralFilter(_img, 15, 25, 25)

    img_canny = cv.Canny(_img, 50, 200, None, 3)
    return img_canny
