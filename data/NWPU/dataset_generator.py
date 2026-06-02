"""
File: dataset_generator.py

Description: This module creates balanced NWPU train and test tensors for anomaly
detection experiments, including image loading, class sampling, normalization,
and anomaly/photo-type label generation.
"""

from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent
ANOMALY_CLASSES = frozenset({"airplane", "ship", "harbor", "airport"})
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def _class_names(split_dir: Path) -> list[str]:
    """
    Returns sorted class folder names for a dataset split.

    Args:
        split_dir (Path): Directory containing class subdirectories.

    Returns:
        list[str]: Sorted class names.
    """
    return sorted(path.name for path in split_dir.iterdir() if path.is_dir())


def _image_paths(class_dir: Path) -> list[Path]:
    """
    Returns sorted supported image paths for a class directory.

    Args:
        class_dir (Path): Directory containing image files.

    Returns:
        list[Path]: Sorted paths whose suffix is supported.
    """
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _balanced_counts(
    total: int, class_names: list[str], rng: random.Random
) -> dict[str, int]:
    """
    Splits a sample count as evenly as possible across class names.

    Args:
        total (int): Total number of samples to distribute.
        class_names (list[str]): Class names that should receive samples.
        rng (random.Random): Random generator used to assign any remainder.

    Returns:
        dict[str, int]: Mapping from class name to requested sample count.
    """
    base_count, remainder = divmod(total, len(class_names))
    shuffled_classes = class_names[:]
    rng.shuffle(shuffled_classes)
    extra_classes = set(shuffled_classes[:remainder])

    return {
        class_name: base_count + (1 if class_name in extra_classes else 0)
        for class_name in class_names
    }


def _load_image(path: Path, image_size: int, normalize: bool) -> torch.Tensor:
    """
    Loads one image as an RGB tensor.

    Args:
        path (Path): Path to the image file.
        image_size (int): Square output image size.
        normalize (bool): Whether to apply ImageNet normalization.

    Returns:
        torch.Tensor: Image tensor shaped ``(3, image_size, image_size)``.
    """
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        array = np.array(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array).permute(2, 0, 1)
    if normalize:
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor


def _sample_split(
    split_dir: Path,
    class_to_idx: dict[str, int],
    num_samples: int,
    ratio_anomaly: float,
    image_size: int,
    normalize: bool,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples one NWPU split into image, anomaly-label, and class-label tensors.

    Args:
        split_dir (Path): Directory containing one dataset split.
        class_to_idx (dict[str, int]): Mapping from class name to numeric label.
        num_samples (int): Number of samples to draw from the split.
        ratio_anomaly (float): Fraction of samples drawn from anomaly classes.
        image_size (int): Square output image size.
        normalize (bool): Whether to apply ImageNet normalization.
        rng (random.Random): Random generator used for sampling and shuffling.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Images, anomaly labels,
        and photo-type labels.

    Raises:
        ValueError: If a class does not contain enough images.
    """
    num_anomalies = round(num_samples * ratio_anomaly)
    num_normal = num_samples - num_anomalies

    anomaly_classes = sorted(
        class_name for class_name in class_to_idx if class_name in ANOMALY_CLASSES
    )
    normal_classes = sorted(
        class_name for class_name in class_to_idx if class_name not in ANOMALY_CLASSES
    )
    samples_per_class = {
        **_balanced_counts(num_anomalies, anomaly_classes, rng),
        **_balanced_counts(num_normal, normal_classes, rng),
    }

    selected: list[tuple[Path, str]] = []
    for class_name in class_to_idx:
        requested = samples_per_class[class_name]
        paths = _image_paths(split_dir / class_name)
        if len(paths) < requested:
            raise ValueError(
                f"Not enough images for class {class_name!r} in {split_dir}: "
                f"requested {requested}, found {len(paths)}"
            )
        selected.extend((path, class_name) for path in rng.sample(paths, requested))

    rng.shuffle(selected)

    images = []
    anomaly_labels = []
    photo_type_labels = []
    for path, class_name in selected:
        images.append(_load_image(path, image_size=image_size, normalize=normalize))
        anomaly_labels.append(1 if class_name in ANOMALY_CLASSES else 0)
        photo_type_labels.append(class_to_idx[class_name])

    return (
        torch.stack(images),
        torch.tensor(anomaly_labels, dtype=torch.long),
        torch.tensor(photo_type_labels, dtype=torch.long),
    )


def create_nwpu_tensors(
    data_dir: str | Path = DATA_DIR,
    num_samples: int = 500,
    ratio_anomaly: float = 0.1,
    image_size: int = 32,
    normalize: bool = True,
    seed: int | None = 0,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Returns train/test NWPU tensors with anomaly and photo-type labels.

    Each split returns:
        images: float tensor shaped (num_samples, 3, image_size, image_size)
        anomaly_labels: long tensor, 1 for anomaly and 0 otherwise
        photo_type_labels: long tensor, class index from alphabetical class order

    Anomaly classes are: airplane, ship, harbor, airport.
    ratio_anomaly is the fraction of each split sampled from anomaly classes.
    Samples are balanced as evenly as possible across anomaly classes and across
    non-anomaly classes.

    Args:
        data_dir (str or Path): Root directory containing ``train`` and ``test``.
        num_samples (int): Number of samples to draw for each split.
        ratio_anomaly (float): Fraction of samples drawn from anomaly classes.
        image_size (int): Square output image size.
        normalize (bool): Whether to apply ImageNet normalization.
        seed (int or None): Random seed for deterministic sampling.

    Returns:
        tuple: Train and test dataset tuples.

    Raises:
        ValueError: If arguments are invalid or class folders do not match.
        FileNotFoundError: If the train or test directory is missing.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if not 0 <= ratio_anomaly <= 1:
        raise ValueError("ratio_anomaly must be between 0 and 1")

    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")

    train_classes = _class_names(train_dir)
    test_classes = _class_names(test_dir)
    if train_classes != test_classes:
        raise ValueError(
            f"Train/test class folders do not match: {train_classes} != {test_classes}"
        )

    missing_anomaly_classes = ANOMALY_CLASSES.difference(train_classes)
    if missing_anomaly_classes:
        missing = ", ".join(sorted(missing_anomaly_classes))
        raise ValueError(f"Missing anomaly class folders: {missing}")

        # anomaly classes first: 0–3
    sorted_anomalies = sorted(ANOMALY_CLASSES)

    # then normal classes
    sorted_normals = sorted(set(train_classes) - set(ANOMALY_CLASSES))

    ordered_classes = sorted_anomalies + sorted_normals

    class_to_idx = {class_name: idx for idx, class_name in enumerate(ordered_classes)}
    rng = random.Random(seed)

    train_tensors = _sample_split(
        train_dir,
        class_to_idx=class_to_idx,
        num_samples=num_samples,
        ratio_anomaly=ratio_anomaly,
        image_size=image_size,
        normalize=normalize,
        rng=rng,
    )
    test_tensors = _sample_split(
        test_dir,
        class_to_idx=class_to_idx,
        num_samples=num_samples,
        ratio_anomaly=ratio_anomaly,
        image_size=image_size,
        normalize=normalize,
        rng=rng,
    )
    return train_tensors, test_tensors


def create_anomaly_dataset(
    ratio_anomaly: float = 1,
    num_samples: int = 500,
    image_size: int = 32,
    normalize: bool = True,
    seed: int | None = 0,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Creates an NWPU anomaly dataset using the default data directory.

    Args:
        ratio_anomaly (float): Fraction of samples drawn from anomaly classes.
        num_samples (int): Number of samples to draw for each split.
        image_size (int): Square output image size.
        normalize (bool): Whether to apply ImageNet normalization.
        seed (int or None): Random seed for deterministic sampling.

    Returns:
        tuple: Train and test dataset tuples.
    """
    return create_nwpu_tensors(
        ratio_anomaly=ratio_anomaly,
        num_samples=num_samples,
        image_size=image_size,
        normalize=normalize,
        seed=seed,
    )


if __name__ == "__main__":
    train, test = create_nwpu_tensors()
    train_images, train_anomaly_labels, train_photo_type_labels = train
    test_images, test_anomaly_labels, test_photo_type_labels = test

    print(f"Train images: {tuple(train_images.shape)}")
    print(f"Train anomaly labels: {tuple(train_anomaly_labels.shape)}")
    print(f"Train photo type labels: {tuple(train_photo_type_labels.shape)}")
    print(f"Test images: {tuple(test_images.shape)}")
    print(f"Test anomaly labels: {tuple(test_anomaly_labels.shape)}")
    print(f"Test photo type labels: {tuple(test_photo_type_labels.shape)}")
    print(f"Train anomalies: {int(train_anomaly_labels.sum())}")
    print(f"Test anomalies: {int(test_anomaly_labels.sum())}")
