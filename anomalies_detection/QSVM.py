"""
File: QSVM.py

Description: This module implements a Quantum Support Vector Machine (QSVM) model 
using Qiskit for quantum kernel computation. It leverages a classical SVM 
with a quantum feature map to perform classification tasks.
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC
import numpy as np
import torch
from math import log2

try:
    from .algo_utils import print_iter
    from .utils import Embedding
    from .utils import SUSPICIOUS_EVENT_LABELS
except ImportError:
    from algo_utils import print_iter
    from utils import Embedding
    from utils import SUSPICIOUS_EVENT_LABELS


EMBEDDING_ALIASES = {
    "amplitude": Embedding.AMPLITUDE,
    "angle_x": Embedding.ANGLE_X,
    "x": Embedding.ANGLE_X,
    "angle_y": Embedding.ANGLE_Y,
    "y": Embedding.ANGLE_Y,
    "angle_z": Embedding.ANGLE_Z,
    "z": Embedding.ANGLE_Z,
    "image_zz": Embedding.IMAGE_ZZ,
    "zz": Embedding.IMAGE_ZZ,
    "image_reupload": Embedding.IMAGE_REUPLOAD,
    "reupload": Embedding.IMAGE_REUPLOAD,
}


class QsvmModel:
    """
    Quantum Support Vector Machine model class.

    Attributes:
        svm: The classical SVM model used for classification.
        embedding_type: Type of embedding used.
        trained: Boolean indicating if the model has been trained.
        num_qubits: Number of qubits used in the quantum circuit.
    """

    def __init__(self, config=None) -> None:
        """
        Initializes a QSVM model with optional configuration overrides.

        Args:
            config (dict or None): Optional model configuration.
        """
        self.svm = SVC(kernel='precomputed', probability=True)  # Initialize SVM with a precomputed kernel
        self.embedding_type = Embedding.IMAGE_REUPLOAD  # Default embedding type
        self.embedding_reps = 2
        self.num_qubits_override = None
        self.trained = False
        self.X_train = None
        self.label_names = SUSPICIOUS_EVENT_LABELS.copy()
        self.positive_label = None
        self.default_tolerance = 0.6
        self.verbose = False

        if config:
            self._handle_config(config)

    def train(self, X_train, y_train, X_test, y_test) -> tuple[list, list]:
        """
        Trains the QSVM model using the provided training data.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Testing features.
            y_test: Testing labels.

        Returns:
            A tuple containing training and test accuracies.
        """
        if self.trained:
            raise ValueError("The model has already been trained.")

        X_train = self._to_feature_tensor(X_train)
        X_test = self._to_feature_tensor(X_test)
        self._pre_train_init(X_train)  # Initialize before training
        
        # Compute the kernel matrices for training and testing
        K_train = self._compute_kernel_matrix(X_train, X_train)
        K_test = self._compute_kernel_matrix(X_test, X_train)

        # Fit the SVM model on the training kernel matrix
        self.svm.fit(self._to_numpy(K_train), self._to_numpy(y_train))
        self.X_train = X_train

        # Compute accuracies on both training and testing data
        test_accuracy = self.svm.score(self._to_numpy(K_test), self._to_numpy(y_test))
        train_accuracy = self.svm.score(self._to_numpy(K_train), self._to_numpy(y_train))

        self.trained = True

        return train_accuracy, test_accuracy

    def predict(self, X, tolerance=None):
        """
        Predicts anomaly labels and confidence scores.

        The model flags a prediction only when its confidence is at or above the
        tolerance threshold. Otherwise, it returns "incertain" so the sample can
        be reviewed instead of treated as a confirmed event.

        Args:
            X: Feature vectors extracted from photos/images.
            tolerance: Minimum confidence required to flag an anomaly.

        Returns:
            A list of dictionaries containing the predicted label, confidence,
            threshold decision, and all class scores.
        """
        if not self.trained or self.X_train is None:
            raise ValueError("The model must be trained before calling predict.")

        tolerance = self.default_tolerance if tolerance is None else tolerance
        if not 0 <= tolerance <= 1:
            raise ValueError("tolerance must be between 0 and 1.")

        X = self._to_feature_tensor(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        K = self._compute_kernel_matrix(X, self.X_train)
        K_numpy = self._to_numpy(K)
        raw_predictions = self.svm.predict(K_numpy)
        scores = self._predict_scores(K_numpy)
        classes = list(self.svm.classes_)

        results = []
        positive_label = self.positive_label
        if positive_label is None and "anomalie" in self.label_names.values():
            positive_label = "anomalie"

        for prediction, class_scores in zip(raw_predictions, scores):
            class_index = classes.index(prediction)
            confidence = float(class_scores[class_index])
            label = self._label_name(prediction)
            confident = confidence >= tolerance
            if positive_label is not None:
                flagged = confident and label == positive_label
            else:
                flagged = confident

            results.append({
                "label": label if confident else "incertain",
                "predicted_label": label,
                "confidence": confidence,
                "tolerance": float(tolerance),
                "flagged": bool(flagged),
                "scores": {
                    self._label_name(class_id): float(score)
                    for class_id, score in zip(classes, class_scores)
                },
            })

        return results

    def detect_events(self, X, tolerance=None):
        """Alias for predict(), with domain-specific naming."""
        return self.predict(X, tolerance=tolerance)

    # PRIVATE METHODS:

    def _quantum_kernel(self, data1, data2) -> float:
        """
        Computes the quantum kernel between two sets of data.

        Args:
            data1: First set of data points.
            data2: Second set of data points.

        Returns:
            The computed kernel value.
        """
        state1 = self._statevector(data1)
        state2 = self._statevector(data2)
        return float(abs(np.vdot(state2.data, state1.data)) ** 2)

    def _compute_kernel_matrix(self, X1, X2):
        """
        Computes the kernel matrix for two datasets.

        Args:
            X1: First dataset.
            X2: Second dataset.

        Returns:
            The computed kernel matrix.
        """
        n_samples_1 = len(X1)
        n_samples_2 = len(X2)
        kernel_matrix = torch.zeros((n_samples_1, n_samples_2), dtype=torch.float64)
        states_1 = [self._statevector(x) for x in X1]
        states_2 = [self._statevector(x) for x in X2]

        for i in range(n_samples_1):
            for j in range(n_samples_2):
                if j == 0:  # Reduce print frequency
                    current_iteration = i * n_samples_2 + j
                    if self.verbose:
                        print_iter(current_iteration, n_samples_1 * n_samples_2, None, None)
                kernel_matrix[i, j] = float(abs(np.vdot(states_2[j].data, states_1[i].data)) ** 2)
        if self.verbose:
            print_iter(n_samples_1 * n_samples_2, n_samples_1 * n_samples_2, None, None)

        return kernel_matrix

    def _statevector(self, x):
        """Builds a Qiskit circuit for one feature vector and returns its statevector."""
        circuit = self._embedding_circuit(self._to_feature_tensor(x))
        return Statevector.from_instruction(circuit)

    def _embedding_circuit(self, x):
        """
        Builds the embedding circuit for one feature vector.

        Args:
            x (torch.Tensor): Feature vector to encode.

        Returns:
            QuantumCircuit: Circuit containing the selected embedding.

        Raises:
            ValueError: If an embedding type is unsupported or cannot encode the
            provided vector.
        """
        circuit = QuantumCircuit(self.num_qubits)

        if self.embedding_type == Embedding.AMPLITUDE:
            norm = torch.linalg.norm(x)
            if norm.item() == 0:
                raise ValueError("Amplitude embedding cannot encode an all-zero feature vector.")
            circuit.initialize(self._to_numpy(x / norm), range(self.num_qubits))
        elif self.embedding_type in {Embedding.ANGLE_X, Embedding.ANGLE_Y, Embedding.ANGLE_Z}:
            self._apply_angle_embedding(circuit, x)
        elif self.embedding_type == Embedding.IMAGE_ZZ:
            self._apply_image_zz_embedding(circuit, x)
        elif self.embedding_type == Embedding.IMAGE_REUPLOAD:
            self._apply_image_reupload_embedding(circuit, x)
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")

        return circuit

    def _apply_angle_embedding(self, circuit, x):
        """
        Applies single-qubit angle embedding to a circuit.

        Args:
            circuit (QuantumCircuit): Circuit to modify in place.
            x (torch.Tensor): Feature values used as rotation angles.
        """
        for qubit, value in enumerate(x):
            angle = float(value.item())
            if self.embedding_type == Embedding.ANGLE_X:
                circuit.rx(angle, qubit)
            elif self.embedding_type == Embedding.ANGLE_Y:
                circuit.ry(angle, qubit)
            elif self.embedding_type == Embedding.ANGLE_Z:
                circuit.rz(angle, qubit)

    def _apply_image_zz_embedding(self, circuit, x):
        """
        Image feature map for PCA-compressed visual features.

        Each PCA component is encoded with local rotations, then neighboring
        components interact through ZZ-style phases. The ring entanglement helps
        the kernel compare feature combinations such as object/scene structure,
        not only independent feature values.
        """
        values = [float(value.item()) for value in x]

        for _ in range(self.embedding_reps):
            for qubit, value in enumerate(values):
                circuit.ry(value, qubit)
                circuit.rz(value * value / np.pi, qubit)

            if self.num_qubits > 1:
                for qubit in range(self.num_qubits):
                    target = (qubit + 1) % self.num_qubits
                    self._apply_zz_phase(
                        circuit,
                        qubit,
                        target,
                        values[qubit],
                        values[target],
                    )

    def _apply_zz_phase(self, circuit, control_qubit, target_qubit, x1, x2):
        """
        Applies a ZZ-style phase interaction between two qubits.

        Args:
            circuit (QuantumCircuit): Circuit to modify in place.
            control_qubit (int): Control qubit index.
            target_qubit (int): Target qubit index.
            x1 (float): First feature value.
            x2 (float): Second feature value.
        """
        circuit.cx(control_qubit, target_qubit)
        circuit.rz((np.pi - x1) * (np.pi - x2) / np.pi, target_qubit)
        circuit.cx(control_qubit, target_qubit)

    def _apply_image_reupload_embedding(self, circuit, x):
        """
        Compact image feature map for PCA16-style vectors.

        Features are repeatedly uploaded onto a smaller qubit register. This can
        keep PCA16 information while using, for example, 8 qubits instead of 16.
        """
        values = [float(value.item()) for value in x]

        for rep in range(self.embedding_reps):
            for feature_index, value in enumerate(values):
                qubit = feature_index % self.num_qubits
                phase = value * (feature_index + 1) / len(values)
                circuit.ry(value, qubit)
                circuit.rz(phase, qubit)

            if self.num_qubits > 1:
                for qubit in range(self.num_qubits):
                    target = (qubit + 1) % self.num_qubits
                    left = values[qubit % len(values)]
                    right = values[(qubit + rep + 1) % len(values)]
                    self._apply_zz_phase(circuit, qubit, target, left, right)

    def _predict_scores(self, kernel_matrix):
        """
        Converts SVM outputs to class-like scores in [0, 1].

        These are confidence scores derived from the SVM decision function. They
        are useful for tolerance filtering, but they are not calibrated legal or
        forensic probabilities.
        """
        decision = np.asarray(self.svm.decision_function(kernel_matrix))
        if decision.ndim == 1:
            positive = 1 / (1 + np.exp(-decision))
            return np.vstack([1 - positive, positive]).T

        shifted = decision - np.max(decision, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _label_name(self, class_id):
        """Returns a readable event label for a class id."""
        return self.label_names.get(class_id, str(class_id))
    
    def _pre_train_init(self, X_train) -> None:
        """
        Prepares the model for training by initializing parameters.

        Args:
            X_train: Training features.
        """
        if self.embedding_type == Embedding.AMPLITUDE:
            n = log2(X_train.shape[1])
            if n % 1 != 0:
                raise ValueError(f"The number of features {X_train.shape[1]} is not a power of 2 for amplitude embedding. "
                                 f"Therefore, {n} is not valid for the number of qubits.")
            self.num_qubits = int(n)
        elif self.embedding_type == Embedding.IMAGE_REUPLOAD and self.num_qubits_override is not None:
            if self.num_qubits_override < 1:
                raise ValueError("num_qubits must be at least 1.")
            self.num_qubits = min(self.num_qubits_override, X_train.shape[1])
        else:
            self.num_qubits = X_train.shape[1]

    def _to_feature_tensor(self, values) -> torch.Tensor:
        """
        Converts feature values to a detached float64 torch tensor.

        Args:
            values: Tensor, pandas object, NumPy array, or array-like values.

        Returns:
            torch.Tensor: Converted feature tensor.
        """
        if isinstance(values, torch.Tensor):
            return values.detach().to(dtype=torch.float64)
        if hasattr(values, "to_numpy"):
            values = values.to_numpy()
        return torch.as_tensor(values, dtype=torch.float64)

    def _to_numpy(self, values):
        """
        Converts tensor, pandas, or array-like values to a NumPy array.

        Args:
            values: Values to convert.

        Returns:
            np.ndarray: Converted NumPy array.
        """
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy()
        if hasattr(values, "to_numpy"):
            return values.to_numpy()
        return np.asarray(values)

    def _handle_config(self, config: dict) -> None:
        """
        Handles configuration options for the model.

        Args:
            config: Configuration settings for the model.
        """
        self.svm = config.get("svm", self.svm)
        self.label_names = config.get("label_names", self.label_names)
        self.positive_label = config.get("positive_label", self.positive_label)
        self.default_tolerance = config.get("tolerance", self.default_tolerance)
        self.verbose = config.get("verbose", self.verbose)
        self.embedding_reps = config.get("embedding_reps", self.embedding_reps)
        self.num_qubits_override = config.get("num_qubits", self.num_qubits_override)

        embedding_choice = config.get("embedding")
        if isinstance(embedding_choice, str):
            embedding_choice = EMBEDDING_ALIASES.get(embedding_choice.lower())

        if embedding_choice in Embedding.__members__.values():
            self.embedding_type = embedding_choice
        elif embedding_choice is not None:
            valid_embeddings = ", ".join(sorted(EMBEDDING_ALIASES))
            raise ValueError(f"Invalid embedding choice. Must be one of: {valid_embeddings}.")
