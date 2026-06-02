import torch
import torchvision.models as models
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from dataset_generator import create_anomaly_dataset

# ====================================================
# DATA
# ====================================================

train_tensor = create_anomaly_dataset(0.0, 100)
x_train = train_tensor[0][0]
y_train = train_tensor[0][1]

test_tensor = create_anomaly_dataset(0.1, 100)
x_test = test_tensor[1][0]
y_test = test_tensor[1][1]


# ====================================================
# FEATURE EXTRACTOR (Frozen ResNet)
# ====================================================

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet.eval()

with torch.no_grad():
    train_features = resnet(x_train).cpu().numpy()

with torch.no_grad():
    test_features = resnet(x_test).cpu().numpy()

print("Train features:", train_features.shape)
print("Test features:", test_features.shape)


# ====================================================
# CLEAN FEATURE NORMALIZATION (IMPORTANT)
# ====================================================

train_features = normalize(train_features)
test_features = normalize(test_features)


# ====================================================
# DIMENSION REDUCTION (SMALL + STABLE)
# ====================================================

pca = PCA(n_components=8)

train_q = pca.fit_transform(train_features)
test_q = pca.transform(test_features)


# Re-normalize AFTER PCA (critical for quantum kernels)
train_q = normalize(train_q)
test_q = normalize(test_q)


# ====================================================
# QUANTUM KERNEL
# ====================================================

feature_map = ZZFeatureMap(feature_dimension=8, reps=1)

kernel = FidelityQuantumKernel(feature_map=feature_map)


# ====================================================
# KERNEL EVALUATION (NO SVM FIRST - DEBUG MODE)
# ====================================================

print("Computing kernel matrix...")

K = kernel.evaluate(test_q, train_q)

# Simple anomaly score: low similarity to normal set
scores = K.mean(axis=1)

# ====================================================
# EVALUATION
# ====================================================

auc = roc_auc_score(y_test, scores)

print("\nROC-AUC:", auc)

# Flip check (important sanity test)
auc_flipped = roc_auc_score(y_test, -scores)

print("Flipped ROC-AUC:", auc_flipped)


# ====================================================
# SAMPLE OUTPUT
# ====================================================

for i in range(len(y_test)):
    print(f"Sample {i:03d} | " f"Score={scores[i]:.6f} | " f"Label={y_test[i]}")
