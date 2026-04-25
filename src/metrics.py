from __future__ import annotations

import numpy as np


def accuracy_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    return float((predictions == targets).mean())


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=False):
        matrix[int(target), int(prediction)] += 1
    return matrix
