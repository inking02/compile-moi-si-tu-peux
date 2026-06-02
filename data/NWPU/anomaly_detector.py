import torch
import torchvision.models as models
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from dataset_generator import create_anomaly_dataset


def extract_image_features(x: torch.Tensor) -> torch.Tensor:
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Remove classification layer
    resnet.fc = torch.nn.Identity()

    resnet.eval()

    with torch.no_grad():
        return resnet(x).cpu().numpy()


def feature_reductor(x: np.ndarray, num_features: int = 10) -> np.ndarray:
    pca = PCA(n_components=num_features)
    scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
    x_pca = pca.fit_transform(x)
    return scaler.fit_transform(x_pca)


def create_quantum_kernels(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    feature_map = ZFeatureMap(feature_dimension=10, reps=2)
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    print("Building training kernel matrix...")
    K_train = kernel.evaluate(x_train, x_train)
    print("Building test kernel matrix...")
    K_test = kernel.evaluate(x_test, x_train)

    return K_train, K_test


def create_and_train_classifier(K_train: np.ndarray) -> OneClassSVM:
    ocsvm = OneClassSVM(kernel="precomputed", nu=0.1)
    ocsvm.fit(K_train)
    return ocsvm


def detect_anomaly(
    ocsvm: OneClassSVM, K_test: np.ndarray, decision_percentile: float = 90
) -> tuple[list[float], list[int]]:
    decision_scores = (-1) * ocsvm.decision_function(K_test)
    threshold = np.percentile(decision_scores, decision_percentile)
    preds = (decision_scores > threshold).astype(int)
    return decision_scores, preds


######################
# Run pipeline

# ====================================================
# DATA
# ====================================================

train_tensor = create_anomaly_dataset(0.0, 200)

x_train = train_tensor[0][0]
y_train = train_tensor[0][1]

test_tensor = create_anomaly_dataset(0.1, 200)

x_test = test_tensor[1][0]
y_test = test_tensor[1][1]


# ====================================================
# FEATURE EXTRACTION
# ====================================================

train_features = extract_image_features(x_train).cpu().numpy()
test_features = extract_image_features(x_train).cpu().numpy()

print("Train features:", train_features.shape)
print("Test features:", test_features.shape)


# ====================================================
# PCA + scaler -> 8 DIMENSIONS
# ====================================================

train_pca = feature_reductor(train_features, num_features=10)
test_pca = feature_reductor(train_features, num_features=10)


# ====================================================
# QUANTUM KERNEL
# ====================================================

train_kernel, test_kernel = create_quantum_kernels(x_train=train_pca, x_test=test_pca)


# ====================================================
# ONE-CLASS SVM
# ====================================================
# Unlike standard SVMs that require data from two classes to find a separating boundary, a One-Class SVM trains only on "normal" data to learn its distribution, flagging anything that deviates significantly from this norm
ocsvm = create_and_train_classifier(train_kernel)


# ====================================================
# ANOMALY SCORES
# ====================================================

decision_scores, preds = detect_anomaly(ocsvm, test_kernel)


# ====================================================
# EVALUATION
# ====================================================

auc = roc_auc_score(y_test, decision_scores)

print(f"\nROC-AUC: {auc:.4f}")


# ====================================================
# SAMPLE OUTPUTS
# ====================================================

for i in range(len(y_test)):
    print(
        f"Sample {i:03d} | " f"Score={decision_scores[i]:.6f} | " f"Label={y_test[i]}"
    )

# ====================================================
# CONFUSION MATRIX COMPONENTS
# ====================================================

TP = np.sum((preds == 1) & (y_test == 1))
FP = np.sum((preds == 1) & (y_test == 0))
TN = np.sum((preds == 0) & (y_test == 0))
FN = np.sum((preds == 0) & (y_test == 1))

print("\n==============================")
print("CONFUSION MATRIX SUMMARY")
print("==============================")
print(f"True Positives  (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives  (TN): {TN}")
print(f"False Negatives (FN): {FN}")


# ====================================================
# DETECTED ANOMALIES (CORRECT)
# ====================================================

print("\n==============================")
print("DETECTED ANOMALIES (TP)")
print("==============================")

for i in range(len(y_test)):
    if preds[i] == 1 and y_test[i] == 1:
        print(f"[TP] Sample {i:03d} | Score={decision_scores[i]:.6f}")


# ====================================================
# FALSE ALARMS (FP)
# ====================================================

print("\n==============================")
print("FALSE POSITIVES (Normal flagged as anomaly)")
print("==============================")

for i in range(len(y_test)):
    if preds[i] == 1 and y_test[i] == 0:
        print(f"[FP] Sample {i:03d} | Score={decision_scores[i]:.6f}")


# ====================================================
# MISSED ANOMALIES (FN)
# ====================================================

print("\n==============================")
print("MISSED ANOMALIES (FN)")
print("==============================")

for i in range(len(y_test)):
    if preds[i] == 0 and y_test[i] == 1:
        print(f"[FN] Sample {i:03d} | Score={decision_scores[i]:.6f}")


# ====================================================
# OPTIONAL: ACCURACY
# ====================================================

accuracy = (preds == y_test).mean()
print("\nAccuracy:", accuracy)
