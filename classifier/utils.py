import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator




def encode_feature_vector_to_angles(feature_vector: NDArray) -> NDArray:
    """
    Encode a feature vector (1D array of floats in [0,1]) into angles for quantum gates.
    For simplicity, we can use a linear mapping: angle = feature_value * π/2
    This maps 0 to 0 radians and 1 to π/2 radians.
    """
    return feature_vector * (np.pi / 2)


def angle_embedding(angles: NDArray):
    """
    Apply angle encoding to the given quantum circuit.
    For each feature (angle), apply an RY gate to the corresponding qubit.
    """

    circuit = QuantumCircuit(len(angles))
    for i, angle in enumerate(angles):
        circuit.ry(angle, i)
    
    return circuit

def classify(circuit: QuantumCircuit) -> int:
    """
    Simulate the given quantum circuit and return a classification result with 4 outcomes.   
    """

    sim = AerSimulator()

    result = sim.run(
        circuit,
        shots=1000
    ).result()
    counts = result.get_counts()
    
    # Get the most probable bitstring
    most_common_outcome = max(counts, key=counts.get)   

    # Map the outcome to its corresponding class
    class_mapping = {
        '00': "Airplane",
        '01': "Ships",
        '10': "Harbor",
        '11': "Airport"
    }

    predicted_class = class_mapping.get(most_common_outcome, "Error in mapping after measurement")  # Return -1 if outcome is unexpected

    return predicted_class