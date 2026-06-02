import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit


def encode_feature_vector_to_angles(feature_vector: NDArray) -> NDArray:
    """
    Encode a feature vector (1D array of floats in [0,1]) into angles for quantum gates.
    For simplicity, we can use a linear mapping: angle = feature_value * π/2
    This maps 0 to 0 radians and 1 to π/2 radians.
    """
    return feature_vector * (np.pi / 2)


def apply_angle_encoding(circuit: QuantumCircuit, angles: NDArray, qubits: list[int]):
    """
    Apply angle encoding to the given quantum circuit.
    For each feature (angle), apply an RY gate to the corresponding qubit.
    """
    assert len(angles) == len(qubits), "Number of angles must match number of qubits."
    for angle, qubit in zip(angles, qubits):
        circuit.ry(angle, qubit)
    
    return circuit