"""
Train a binary QSVM detector for target images.

Default positive classes:
    airplane, airport, harbor, ship

Run from this directory:
    python3 train_target_detector.py
"""

import argparse

import numpy as np
from sklearn.svm import SVC

from QSVM import QsvmModel
from image_pipeline import ImageAngleTransformer
from image_pipeline import build_nwpu_detection_datasets
LABEL_NAMES = {
    0: "autre",
    1: "cible",
}


def main():
    args = _parse_args()

    X_train, y_train, _, X_test, y_test, test_names = build_nwpu_detection_datasets(
        args.data_root,
        histogram_bins=args.histogram_bins,
    )

    transformer = ImageAngleTransformer(output_dim=args.output_dim)
    train_angles = transformer.fit_transform(X_train)
    test_angles = transformer.transform(X_test)

    model = QsvmModel({
        "embedding": "angle_y",
        "label_names": LABEL_NAMES,
        "positive_label": "cible",
        "svm": SVC(
            kernel="precomputed",
            probability=True,
            class_weight=_class_weight(args.class_weight),
        ),
        "tolerance": args.tolerance,
        "verbose": args.verbose,
    })
    train_accuracy, test_accuracy = model.train(
        train_angles,
        y_train,
        test_angles,
        y_test,
    )

    train_results = model.predict(train_angles, tolerance=0.0)
    threshold = (
        _best_threshold(y_train.detach().cpu().numpy(), train_results)
        if args.tolerance == "auto"
        else float(args.tolerance)
    )
    results = model.predict(test_angles, tolerance=0.0)
    predictions = np.array([
        1 if result["scores"]["cible"] >= threshold else 0
        for result in results
    ])
    expected = y_test.detach().cpu().numpy()

    print("Target classes: airplane, airport, harbor, ship")
    print(f"Dataset: {args.data_root}")
    print(f"Train samples: {len(X_train)} ({int(y_train.sum())} cible, {len(y_train) - int(y_train.sum())} autre)")
    print(f"Test samples: {len(X_test)} ({int(y_test.sum())} cible, {len(y_test) - int(y_test.sum())} autre)")
    print(f"Raw image feature dim: {X_train.shape[1]}")
    print(f"QSVM angle dim / qubits: {train_angles.shape[1]}")
    print(f"Class weight: {args.class_weight}")
    print(f"Detection threshold: {threshold:.2f}")
    print(f"Train accuracy: {train_accuracy:.2f}")
    print(f"Test accuracy, SVM hard prediction: {test_accuracy:.2f}")
    print()
    _print_detection_summary(expected, predictions)

    print("\nSample predictions:")
    for name, result in zip(test_names[:args.show_predictions], results[:args.show_predictions]):
        anomaly_score = result["scores"]["cible"]
        status = "ANOMALIE DETECTEE" if anomaly_score >= threshold else "RIEN"
        print(
            f"{name}: {status} | score_anomalie={anomaly_score:.2f}"
        )


def _print_detection_summary(expected, predicted):
    true_positive = int(((expected == 1) & (predicted == 1)).sum())
    false_positive = int(((expected == 0) & (predicted == 1)).sum())
    true_negative = int(((expected == 0) & (predicted == 0)).sum())
    false_negative = int(((expected == 1) & (predicted == 0)).sum())

    total_anomalies = true_positive + false_negative
    total_normal = true_negative + false_positive
    anomaly_detection_rate = true_positive / max(total_anomalies, 1)
    normal_ignore_rate = true_negative / max(total_normal, 1)

    print("Resultats de detection:")
    print(
        f"  Anomalies detectees : {true_positive}/{total_anomalies} "
        f"({anomaly_detection_rate * 100:.0f}%)"
    )
    print(
        f"  Rien detecte        : {true_negative}/{total_normal} "
        f"({normal_ignore_rate * 100:.0f}%)"
    )
    print(f"  Anomalies manquees  : {false_negative}")
    print(f"  Fausses alertes     : {false_positive}")


def _best_threshold(expected, train_results):
    scores = np.array([result["scores"]["cible"] for result in train_results])
    candidates = np.unique(np.round(scores, 4))
    best_threshold = 0.5
    best_score = -1.0

    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        true_positive = ((expected == 1) & (predicted == 1)).sum()
        false_positive = ((expected == 0) & (predicted == 1)).sum()
        true_negative = ((expected == 0) & (predicted == 0)).sum()
        false_negative = ((expected == 1) & (predicted == 0)).sum()
        true_positive_rate = true_positive / max(true_positive + false_negative, 1)
        true_negative_rate = true_negative / max(true_negative + false_positive, 1)
        score = (true_positive_rate + true_negative_rate) / 2

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def _class_weight(value):
    if value == "none":
        return None
    return value


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../data/NWPU")
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--histogram-bins", type=int, default=8)
    parser.add_argument("--tolerance", default="auto")
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--show-predictions", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
