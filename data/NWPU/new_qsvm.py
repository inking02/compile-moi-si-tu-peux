import torch
import torchvision.models as models
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from dataset_generator import create_anomaly_dataset

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

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Remove classification layer
resnet.fc = torch.nn.Identity()

resnet.eval()

with torch.no_grad():
    train_features = resnet(x_train).cpu().numpy()

with torch.no_grad():
    test_features = resnet(x_test).cpu().numpy()

print("Train features:", train_features.shape)
print("Test features:", test_features.shape)


# ====================================================
# PCA -> 8 DIMENSIONS
# ====================================================

pca = PCA(n_components=10)

train_pca = pca.fit_transform(train_features)
test_pca = pca.transform(test_features)

print("PCA shape:", train_pca.shape)


# ====================================================
# SCALE TO [0, 2π]
# ====================================================

scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))

train_pca = scaler.fit_transform(train_pca)
test_pca = scaler.transform(test_pca)


# ====================================================
# QUANTUM KERNEL
# ====================================================

feature_map = ZZFeatureMap(feature_dimension=10, reps=2)

kernel = FidelityQuantumKernel(feature_map=feature_map)


# ====================================================
# KERNEL MATRICES
# ====================================================

print("Building training kernel matrix...")

K_train = kernel.evaluate(train_pca, train_pca)

print("Training kernel shape:", K_train.shape)

print("Building test kernel matrix...")

K_test = kernel.evaluate(test_pca, train_pca)

print("Test kernel shape:", K_test.shape)


# ====================================================
# ONE-CLASS SVM
# ====================================================
# Unlike standard SVMs that require data from two classes to find a separating boundary, a One-Class SVM trains only on "normal" data to learn its distribution, flagging anything that deviates significantly from this norm
ocsvm = OneClassSVM(kernel="precomputed", nu=0.1)

ocsvm.fit(K_train)


# ====================================================
# ANOMALY SCORES
# ====================================================

decision_scores = ocsvm.decision_function(K_test)

# Larger = more anomalous
anomaly_scores = decision_scores


# ====================================================
# EVALUATION
# ====================================================

auc = roc_auc_score(y_test, anomaly_scores)

print(f"\nROC-AUC: {auc:.4f}")


# ====================================================
# SAMPLE OUTPUTS
# ====================================================

for i in range(len(y_test)):
    print(f"Sample {i:03d} | " f"Score={anomaly_scores[i]:.6f} | " f"Label={y_test[i]}")


# ====================================================
# THRESHOLD (use your current strategy or adjust)
# ====================================================

threshold = np.percentile(anomaly_scores, 90)

preds = (anomaly_scores > threshold).astype(int)

y_test = np.array(y_test).reshape(-1)

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
        print(f"[TP] Sample {i:03d} | Score={anomaly_scores[i]:.6f}")


# ====================================================
# FALSE ALARMS (FP)
# ====================================================

print("\n==============================")
print("FALSE POSITIVES (Normal flagged as anomaly)")
print("==============================")

for i in range(len(y_test)):
    if preds[i] == 1 and y_test[i] == 0:
        print(f"[FP] Sample {i:03d} | Score={anomaly_scores[i]:.6f}")


# ====================================================
# MISSED ANOMALIES (FN)
# ====================================================

print("\n==============================")
print("MISSED ANOMALIES (FN)")
print("==============================")

for i in range(len(y_test)):
    if preds[i] == 0 and y_test[i] == 1:
        print(f"[FN] Sample {i:03d} | Score={anomaly_scores[i]:.6f}")


# ====================================================
# OPTIONAL: ACCURACY
# ====================================================

accuracy = (preds == y_test).mean()
print("\nAccuracy:", accuracy)
