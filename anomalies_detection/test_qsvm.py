"""
File: test_qsvm.py

Description: This script runs a small smoke test for the Qiskit QSVM anomaly
classifier using toy feature vectors and binary anomaly labels.
"""

import torch

from QSVM import QsvmModel
from utils import normalize_feature_rows


def main():
    """Runs the QSVM smoke test and prints toy predictions."""
    # Fake photo features. Each row represents one photo embedding.
    # Class 0 means no anomaly. Class 1 means anomaly.
    raw_train_features = torch.tensor([
        [1.0, 0.0],
        [0.95, 0.05],
        [0.9, 0.1],
        [0.85, 0.15],
        [0.0, 1.0],
        [0.05, 0.95],
        [0.1, 0.9],
        [0.15, 0.85],
    ])
    train_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    raw_test_features = torch.tensor([
        [0.95, 0.05],
        [0.05, 0.95],
    ])
    test_labels = torch.tensor([0, 1])

    # Angle embeddings expect rotation angles. Scaling normalized features by pi
    # keeps similar photos close while making the two toy classes separable.
    train_angles = normalize_feature_rows(raw_train_features) * torch.pi
    test_angles = normalize_feature_rows(raw_test_features) * torch.pi

    model = QsvmModel({
        "embedding": "angle_y",
        "tolerance": 0.5,
        "verbose": False,
    })

    train_accuracy, test_accuracy = model.train(
        train_angles,
        train_labels,
        test_angles,
        test_labels,
    )

    print(f"Train accuracy: {train_accuracy:.2f}")
    print(f"Test accuracy: {test_accuracy:.2f}")

    results = model.detect_events(test_angles)
    print("\nPredictions:")
    for index, result in enumerate(results, start=1):
        status = "ALERTE" if result["flagged"] else "OK"
        label = result["predicted_label"]
        confidence = result["confidence"]
        print(f"Photo {index}: {status} | {label} | confiance={confidence:.2f}")


if __name__ == "__main__":
    main()
