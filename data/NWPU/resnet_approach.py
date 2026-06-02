"""
File: resnet_approach.py

Description: This script experiments with ResNet feature extraction, compactness
training, PCA projection, and a quantum kernel for NWPU anomaly scoring.
"""

import torch
import torchvision.models as models
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn

from dataset_generator import create_anomaly_dataset
import torch.nn.functional as F

# ====================================================
# Data
# ====================================================

train_tensor = create_anomaly_dataset(0.0, 500)
x_train = train_tensor[0][0]
y_train = train_tensor[0][1]

test_tensor = create_anomaly_dataset(0.1, 500)
x_test = test_tensor[1][0]
y_test = test_tensor[1][1]

# ====================================================
# Feature extractor
# ====================================================
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Remove classifier
resnet.fc = torch.nn.Identity()

from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(x_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet.train()


with torch.no_grad():
    train_features = resnet(x_train)
    train_features = F.normalize(train_features, dim=1)
    center = train_features.mean(dim=0)
    center = F.normalize(center, dim=0)

optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-4)


def compactness_loss(z):
    """
    Computes feature compactness around the batch mean.

    Args:
        z (torch.Tensor): Feature tensor for a batch of images.

    Returns:
        torch.Tensor: Mean squared distance from each feature to the batch mean.
    """
    return ((z - z.mean(dim=0)) ** 2).mean()


epochs = 20
for epoch in range(epochs):

    with torch.no_grad():
        train_features = F.normalize(resnet(x_train), dim=1)
        center = train_features.mean(dim=0)
        center = F.normalize(center, dim=0)

    for (x,) in train_loader:
        z = F.normalize(resnet(x), dim=1)

        loss = ((z - center) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")


with torch.no_grad():
    train_features = F.normalize(resnet(x_train), dim=1)

train_distances = torch.norm(train_features - center, dim=1).cpu().numpy()

print(train_distances.mean())
print(train_distances.std())
print(train_distances.min(), train_distances.max())


with torch.no_grad():
    test_features = F.normalize(resnet(x_test), dim=1)

test_distances = torch.norm(test_features - center, dim=1).cpu().numpy()

normal_scores = test_distances[y_test == 0]
anomaly_scores = test_distances[y_test == 1]

print("Normal mean:", normal_scores.mean())
print("Anomaly mean:", anomaly_scores.mean())

resnet.eval()
for param in resnet.parameters():
    param.requires_grad = False

# ====================================================
# Extract train features
# ====================================================

with torch.no_grad():
    train_features = resnet(x_train)

train_features = F.normalize(train_features, dim=1).cpu().numpy()

print(train_features.shape)
# (N, 512)


from sklearn.metrics.pairwise import cosine_similarity

score = np.linalg.norm(train_features[:10] - center.cpu().numpy(), axis=1)

print(score.mean())

with torch.no_grad():
    test_features = resnet(x_test)

score = np.linalg.norm(test_features[0:1] - center.cpu().numpy(), axis=1)
print(score)

# ====================================================
# PCA -> 8 dimensions
# ====================================================

pca = PCA(n_components=8)

normal_pca = pca.fit_transform(train_features)

scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))

normal_pca = scaler.fit_transform(normal_pca)

print(normal_pca.shape)
# (N, 8)

# ====================================================
# Quantum kernel
# ====================================================

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

feature_map = ZZFeatureMap(feature_dimension=8, reps=2)

kernel = FidelityQuantumKernel(feature_map=feature_map)

# ====================================================
# Score every test image
# ====================================================

with torch.no_grad():
    test_features = resnet(x_test)

test_features = F.normalize(test_features, dim=1).cpu().numpy()


test_pca = pca.transform(test_features)
test_pca = scaler.transform(test_pca)

scores = []

for i, sample in enumerate(test_pca):

    sample = sample.reshape(1, -1)

    # Similarity against all normal training images
    K = kernel.evaluate(sample, normal_pca)

    similarity_score = np.mean(np.sort(K[0])[-10:])

    scores.append(similarity_score)
    print(f"Sample {i} got score {scores[-1]}. Should be a {y_test[i]}")

scores = np.array(scores)

# ====================================================
# Evaluation
# ====================================================

# Higher similarity = more normal
# Convert to anomaly score

anomaly_scores = 1.0 - scores

auc = roc_auc_score(y_test, anomaly_scores)

print(f"ROC-AUC: {auc:.4f}")

# Example threshold
threshold = np.percentile(scores, 5)

preds = (scores < threshold).astype(int)

accuracy = (preds == y_test).mean()

print(f"Accuracy: {accuracy:.4f}")
