from qiskit_algorithms.optimizers import COBYLA, ADAM
from qiskit_machine_learning.neural_networks import SamplerQNN
import numpy as np

from qiskit import QuantumCircuit
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

from dataset_generator import create_anomaly_dataset

from sklearn.decomposition import PCA

from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
import time

SEED = 8398

nb_features = 16

"""
optimizer = ADAM(maxiter=num_iter)

np.random.seed(SEED)

ansatz = EfficientSU2(
    num_qubits=nb_features,
    su2_gates=["ry", "rz"],
    entanglement="circular",
    reps=2
    )

emb_circuit = ZZFeatureMap(nb_features, reps=1, entanglement="circular")

qc = QuantumCircuit(nb_features)

qc.compose(emb_circuit, inplace=True)
qc.compose(ansatz, inplace=True)


def interpret(x):
    return x % 4

qnn=SamplerQNN(
    circuit=qc,  
    input_params=emb_circuit.parameters,
    weight_params=ansatz.parameters,
    interpret = interpret,
    output_shape=4  # Reshape by the number of classical registers
)



initial_weights = np.random.rand(ansatz.num_parameters) 

circuit_classifier = NeuralNetworkClassifier(neural_network=qnn,optimizer=optimizer,initial_point=initial_weights)
"""

# Experiment Parameters

nb_classes = 4
anomaly_ratio = 1

tensor = create_anomaly_dataset(anomaly_ratio, 400)

# Splitting the data in training set and test set
x_train = tensor[0][0].detach().numpy()
y_train = tensor[0][2].detach().numpy()

x_test = tensor[1][0].detach().numpy()
y_test = tensor[1][2].detach().numpy()

"""
pca = PCA(n_components=nb_features)

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

pca.fit(x_train_flat)

x_train = pca.transform(x_train_flat)
x_test = pca.transform(x_test_flat)
"""

unique = np.unique(y_train)
label_map = {old: new for new, old in enumerate(unique)}
y_train = np.vectorize(label_map.get)(y_train)
y_test = np.vectorize(label_map.get)(y_test)

import merlin as ml
import torch.nn as nn
import torch
import torch.optim as optim


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


builder = ml.CircuitBuilder(n_modes=11)

for _ in range(3):
    builder.add_entangling_layer()

builder.add_angle_encoding(modes=list(range(10)))

for _ in range(3):
    builder.add_entangling_layer()

reservoir = ml.QuantumLayer(input_size=10, builder=builder, n_photons=5)


class QuantumReservoirNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = CNNEncoder()

        self.reservoir = reservoir

        self.classifier = nn.Sequential(
            nn.Linear(self.reservoir.output_size, 32), nn.ReLU(), nn.Linear(32, 4)
        )

    def forward(self, x):
        x = self.encoder(x)

        x = torch.tanh(x)
        x = self.reservoir(x)

        x = self.classifier(x)

        return x


model = QuantumReservoirNet()


# -----------------------
# Torch conversion
# -----------------------
x_train_t = torch.tensor(x_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

x_test_t = torch.tensor(x_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)


# -----------------------
# Model, loss, optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    list(model.encoder.parameters()) + list(model.classifier.parameters()), lr=1e-3
)

epochs = 100
batch_size = 16


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean().item()
    return acc


# -----------------------
# Training loop
# -----------------------
model.train()

n = x_train_t.shape[0]

for epoch in range(epochs):
    perm = torch.randperm(n)

    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]

        xb = x_train_t[idx]
        yb = y_train_t[idx]

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    train_acc_epoch = accuracy(model, x_train_t, y_train_t)
    test_acc_epoch = accuracy(model, x_test_t, y_test_t)

    print(
        f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | "
        f"Train acc: {train_acc_epoch:.3f} | Test acc: {test_acc_epoch:.3f}"
    )

# -----------------------
# Final evaluation
# -----------------------
train_acc = accuracy(model, x_train_t, y_train_t)
test_acc = accuracy(model, x_test_t, y_test_t)

print(
    f">\n> Accuracy on the training set: {train_acc}\n"
    f"> Accuracy on the test set: {test_acc}\n>"
)
