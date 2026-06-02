"""
Image feature pipeline for the QSVM classifier.

This module converts images to numeric feature vectors, builds labeled datasets
from class folders, then reduces and scales image vectors to angle values for
angle-based quantum embeddings.
"""

from pathlib import Path
import importlib.util

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float64)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float64)


def find_images(image_dir, recursive=False):
    """Returns sorted supported image paths for a directory."""
    directory = Path(image_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Expected an image directory, got: {directory}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def extract_image_features(image_path, image_size=(32, 32), histogram_bins=8):
    """
    Converts one image to a numeric visual feature vector.

    The features are intentionally lightweight and local: color statistics,
    color histograms, simple edge/texture strength, and a tiny grayscale
    thumbnail. This avoids external model downloads while giving the QSVM
    structured image information.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size, Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    return extract_rgb_features(rgb, histogram_bins=histogram_bins)


def extract_tensor_image_features(image_tensor, histogram_bins=8):
    """
    Converts one image tensor shaped (3, H, W) or (H, W, 3) to features.

    The tensor is expected to contain unnormalized RGB values in [0, 1].
    """
    if isinstance(image_tensor, torch.Tensor):
        image = image_tensor.detach().cpu().numpy()
    else:
        image = np.asarray(image_tensor)

    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image tensor, got shape {image.shape}")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image tensor, got shape {image.shape}")

    rgb = image.astype(np.float64)
    if rgb.min() < 0.0 or rgb.max() > 1.0:
        rgb = (rgb * IMAGENET_STD) + IMAGENET_MEAN
    rgb = np.clip(rgb, 0.0, 1.0)
    return extract_rgb_features(rgb, histogram_bins=histogram_bins)


def extract_rgb_features(rgb, histogram_bins=8):
    """Converts an RGB array in [0, 1] to the shared feature vector."""
    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    color_mean = rgb.mean(axis=(0, 1))
    color_std = rgb.std(axis=(0, 1))
    gray_stats = np.array([
        gray.mean(),
        gray.std(),
        np.quantile(gray, 0.25),
        np.quantile(gray, 0.50),
        np.quantile(gray, 0.75),
    ])

    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(
            rgb[:, :, channel],
            bins=histogram_bins,
            range=(0.0, 1.0),
            density=True,
        )
        histograms.append(hist / max(hist.sum(), 1e-12))

    gray_hist, _ = np.histogram(
        gray,
        bins=histogram_bins,
        range=(0.0, 1.0),
        density=True,
    )
    gray_hist = gray_hist / max(gray_hist.sum(), 1e-12)

    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    texture = np.array([
        np.abs(dx).mean(),
        np.abs(dy).mean(),
        dx.std(),
        dy.std(),
    ])

    gray_thumbnail = np.asarray(
        Image.fromarray((gray * 255).astype(np.uint8)).resize(
            (8, 8),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float64,
    ).reshape(-1) / 255.0
    color_thumbnail = np.asarray(
        Image.fromarray((rgb * 255).astype(np.uint8)).resize(
            (8, 8),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float64,
    ).reshape(-1) / 255.0

    return np.concatenate([
        color_mean,
        color_std,
        gray_stats,
        *histograms,
        gray_hist,
        texture,
        gray_thumbnail,
        color_thumbnail,
    ])


def build_image_class_dataset(
    class_dirs,
    image_size=(32, 32),
    histogram_bins=8,
    max_images_per_class=None,
):
    """
    Builds X/y arrays from class directories where each image is one sample.

    Args:
        class_dirs: Mapping of class directory path to numeric label.
        max_images_per_class: Optional cap to keep QSVM kernel computation small.

    Returns:
        (X, y, image_paths) where X is a torch tensor of image feature vectors.
    """
    vectors = []
    labels = []
    image_paths = []

    for class_dir, label in class_dirs.items():
        paths = find_images(class_dir)
        if max_images_per_class is not None:
            paths = paths[:max_images_per_class]
        if not paths:
            raise ValueError(f"No supported images found in class directory: {class_dir}")

        for image_path in paths:
            vectors.append(extract_image_features(
                image_path,
                image_size=image_size,
                histogram_bins=histogram_bins,
            ))
            labels.append(int(label))
            image_paths.append(str(image_path))

    return (
        torch.as_tensor(np.vstack(vectors), dtype=torch.float64),
        torch.as_tensor(labels, dtype=torch.int64),
        image_paths,
    )


def build_target_detection_dataset(
    data_split_root,
    target_classes,
    image_size=(32, 32),
    histogram_bins=8,
    max_positive_per_class=None,
    max_negative_total=None,
    random_state=42,
):
    """
    Builds a binary dataset: 1 if image belongs to a target class, else 0.

    Args:
        data_split_root: Directory containing class folders, e.g. data/train/train.
        target_classes: Class folder names considered positive.
        max_positive_per_class: Optional cap per target class.
        max_negative_total: Optional cap across all non-target classes.

    Returns:
        (X, y, image_paths) where y is 1 for target and 0 for other.
    """
    root = Path(data_split_root)
    if not root.exists():
        raise FileNotFoundError(f"Data split directory not found: {root}")

    target_classes = set(target_classes)
    positive_paths = []
    negative_paths = []

    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        paths = find_images(class_dir)
        if class_dir.name in target_classes:
            if max_positive_per_class is not None:
                paths = paths[:max_positive_per_class]
            positive_paths.extend(paths)
        else:
            negative_paths.extend(paths)

    if not positive_paths:
        raise ValueError(f"No positive images found for target classes: {sorted(target_classes)}")
    if not negative_paths:
        raise ValueError("No negative images found outside target classes.")

    if max_negative_total is not None and len(negative_paths) > max_negative_total:
        rng = np.random.default_rng(random_state)
        selected = rng.choice(len(negative_paths), size=max_negative_total, replace=False)
        negative_paths = [negative_paths[index] for index in sorted(selected)]

    paths_and_labels = [(path, 1) for path in positive_paths]
    paths_and_labels.extend((path, 0) for path in negative_paths)

    vectors = []
    labels = []
    image_paths = []
    for image_path, label in paths_and_labels:
        vectors.append(extract_image_features(
            image_path,
            image_size=image_size,
            histogram_bins=histogram_bins,
        ))
        labels.append(label)
        image_paths.append(str(image_path))

    return (
        torch.as_tensor(np.vstack(vectors), dtype=torch.float64),
        torch.as_tensor(labels, dtype=torch.int64),
        image_paths,
    )


def build_nwpu_detection_datasets(
    data_dir,
    histogram_bins=8,
):
    """
    Builds train/test binary feature datasets from data/NWPU/dataset_generator.py.

    The NWPU generator returns image tensors and anomaly labels. This function
    converts those tensors to the same handcrafted image features used by the
    rest of the QSVM pipeline.
    """
    generator = _load_nwpu_generator(data_dir)
    train, test = generator.create_anomaly_dataset()

    X_train, y_train = _features_from_nwpu_split(train, histogram_bins=histogram_bins)
    X_test, y_test = _features_from_nwpu_split(test, histogram_bins=histogram_bins)
    train_names = [f"train_{index}" for index in range(len(y_train))]
    test_names = [f"test_{index}" for index in range(len(y_test))]
    return X_train, y_train, train_names, X_test, y_test, test_names


def _load_nwpu_generator(data_dir):
    generator_path = Path(data_dir) / "dataset_generator.py"
    if not generator_path.exists():
        raise FileNotFoundError(f"Missing NWPU dataset generator: {generator_path}")

    spec = importlib.util.spec_from_file_location("nwpu_dataset_generator", generator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _features_from_nwpu_split(split, histogram_bins=8):
    images, anomaly_labels, _photo_type_labels = split
    vectors = [
        extract_tensor_image_features(image, histogram_bins=histogram_bins)
        for image in images
    ]
    return (
        torch.as_tensor(np.vstack(vectors), dtype=torch.float64),
        anomaly_labels.detach().to(dtype=torch.int64),
    )


class ImageAngleTransformer:
    """
    Reduces image vectors and scales them to angles in [0, pi].

    Use this before QsvmModel with embedding="angle_y". The reduced dimension is
    the number of qubits used by the angle embedding.
    """

    def __init__(self, output_dim=16):
        if output_dim < 1:
            raise ValueError("output_dim must be at least 1.")

        self.output_dim = output_dim
        self.standardizer = StandardScaler()
        self.pca = None
        self.angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi))

    def fit(self, X):
        X_numpy = _to_numpy(X)
        X_scaled = self.standardizer.fit_transform(X_numpy)
        n_components = min(self.output_dim, X_scaled.shape[0], X_scaled.shape[1])
        if n_components < self.output_dim:
            raise ValueError(
                "Not enough images/features to fit the requested output_dim. "
                f"Requested {self.output_dim}, but only {n_components} components are available."
            )

        self.pca = PCA(n_components=self.output_dim, random_state=42)
        reduced = self.pca.fit_transform(X_scaled)
        self.angle_scaler.fit(reduced)
        return self

    def transform(self, X):
        if self.pca is None:
            raise ValueError("ImageAngleTransformer must be fitted before transform().")

        X_numpy = _to_numpy(X)
        X_scaled = self.standardizer.transform(X_numpy)
        reduced = self.pca.transform(X_scaled)
        angles = self.angle_scaler.transform(reduced)
        return torch.as_tensor(angles, dtype=torch.float64)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def _to_numpy(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)
