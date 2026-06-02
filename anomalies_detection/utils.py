"""
File: utils.py

Description: Helpers for preparing visual features and filtering anomaly
classification results.
"""
from enum import Enum
import numpy as np
import torch


SUSPICIOUS_EVENT_LABELS = {
    0: "No anomaly",
    1: "Anomaly",
}


class Embedding(Enum):
    """Supported quantum embedding strategies for anomaly detection models."""

    AMPLITUDE = "amplitude"
    ANGLE_X = "X"
    ANGLE_Y = "Y"
    ANGLE_Z = "Z"
    IMAGE_ZZ = "image_zz"
    IMAGE_REUPLOAD = "image_reupload"


def prepare_visual_features(features, target_size=None, normalize=True) -> torch.Tensor:
    """
    Prepares photo/image embeddings for the Qiskit QSVM.

    The input should already be numeric features extracted from photos with a
    classical vision model. For amplitude embedding, the
    number of features must be a power of 2, so this helper pads vectors with
    zeros when needed.

    Args:
        features: Array-like visual feature vectors.
        target_size: Optional final feature count. Must be a power of 2.
        normalize: Whether to L2-normalize each feature vector.

    Returns:
        A torch tensor ready for QsvmModel.
    """
    X = _to_tensor(features)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if normalize:
        X = normalize_feature_rows(X)

    return pad_features_to_power_of_two(X, target_size=target_size)


def normalize_feature_rows(features) -> torch.Tensor:
    """
    L2-normalizes each row while keeping all-zero rows unchanged.

    Args:
        features: Array-like feature matrix.

    Returns:
        torch.Tensor: Row-normalized feature matrix.
    """
    X = _to_tensor(features)
    norms = torch.linalg.norm(X, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    return X / norms


def pad_features_to_power_of_two(features, target_size=None) -> torch.Tensor:
    """
    Pads feature vectors so their length is compatible with AmplitudeEmbedding.

    Args:
        features: Array-like feature matrix.
        target_size (int or None): Optional final feature count.

    Returns:
        torch.Tensor: Padded feature matrix.

    Raises:
        ValueError: If ``target_size`` is too small or is not a power of two.
    """
    X = _to_tensor(features)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    current_size = X.shape[1]
    final_size = target_size or _next_power_of_two(current_size)
    if final_size < current_size:
        raise ValueError("target_size cannot be smaller than the current feature count.")
    if not _is_power_of_two(final_size):
        raise ValueError("target_size must be a power of 2 for amplitude embedding.")

    if final_size == current_size:
        return X

    padded = torch.zeros((X.shape[0], final_size), dtype=X.dtype, device=X.device)
    padded[:, :current_size] = X
    return padded


def filter_detection_results(results, tolerance=None):
    """
    Filters model prediction dictionaries using a tolerance threshold.

    Args:
        results: Output from QsvmModel.detect_events() or QsvmModel.predict().
        tolerance: Optional threshold override. If omitted, each result's own
            tolerance value is used.

    Returns:
        Only flagged anomaly results.
    """
    alerts = []
    for result in results:
        threshold = result.get("tolerance", 0.6) if tolerance is None else tolerance
        if result.get("confidence", 0.0) >= threshold and result.get("flagged", False):
            alerts.append(result)
    return alerts


def _to_tensor(values) -> torch.Tensor:
    """
    Converts pandas objects, NumPy arrays, and array-like values to tensors.

    Args:
        values: Values to convert.

    Returns:
        torch.Tensor: Converted float64 tensor.
    """
    if isinstance(values, torch.Tensor):
        return values.detach().to(dtype=torch.float64)
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return torch.as_tensor(values, dtype=torch.float64)


def _to_numpy(values):
    """
    Converts torch, pandas, or array-like values to a NumPy array.

    Args:
        values: Values to convert.

    Returns:
        np.ndarray: Converted NumPy array.
    """
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)


def _next_power_of_two(value) -> int:
    """
    Computes the smallest power of two greater than or equal to a value.

    Args:
        value (int): Input value.

    Returns:
        int: Smallest power of two greater than or equal to ``value``.
    """
    power = 1
    while power < value:
        power *= 2
    return power


def _is_power_of_two(value) -> bool:
    """
    Checks whether a value is a positive power of two.

    Args:
        value (int): Input value.

    Returns:
        bool: True if ``value`` is a power of two, otherwise False.
    """
    if value < 1:
        return False

    power = _next_power_of_two(value)
    return power == value
