from qiskit_algorithms.optimizers import COBYLA,ADAM
from qiskit_machine_learning.neural_networks import SamplerQNN
import numpy as np
from classifier import create_classifier_circuit

from qiskit import QuantumCircuit
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

from data.NWPU.dataset_generator import create_anomaly_dataset

from sklearn.decomposition import PCA

from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
import time



SEED = 8398

nb_features = 4

features = np.random.rand(nb_features)


num_iter = 50
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

# Experiment Parameters

nb_classes = 4
anomaly_ratio = 1

tensor = create_anomaly_dataset(anomaly_ratio, 100)

# Splitting the data in training set and test set
x_train = tensor[0][0].detach().numpy()
y_train = tensor[0][2].detach().numpy()

x_test = tensor[1][0].detach().numpy()
y_test = tensor[1][2].detach().numpy()

pca = PCA(n_components=nb_features)
x_train_red = pca.fit_transform(x_train[:,0,:,:].reshape(x_train.shape[0],-1))
x_train_green = pca.fit_transform(x_train[:,1,:,:].reshape(x_train.shape[0],-1))
x_train_blue = pca.fit_transform(x_train[:,2,:,:].reshape(x_train.shape[0],-1))
x_train=np.sum([x_train_red,x_train_green,x_train_blue],axis=0)/3
x_test_red = pca.fit_transform(x_test[:,0,:,:].reshape(x_test.shape[0],-1))
x_test_green = pca.fit_transform(x_test[:,1,:,:].reshape(x_test.shape[0],-1))
x_test_blue = pca.fit_transform(x_test[:,2,:,:].reshape(x_test.shape[0],-1))
x_test=np.sum([x_test_red,x_test_green,x_test_blue],axis=0)/3

start_time=time.time()
print("Running the training loop...")
circuit_classifier.fit(x_train,y_train)
print(f"Training completed. Took {time.time()-start_time:.2f} seconds.")

y_train_pred=circuit_classifier.predict(x_train)
y_test_pred=circuit_classifier.predict(x_test)

train_acc = circuit_classifier.score(x_train, y_train)
test_acc = circuit_classifier.score(x_test, y_test)

print(f">\n> Accuracy on the training set: {train_acc}\n> Accuracy on the test set: {test_acc}\n>")