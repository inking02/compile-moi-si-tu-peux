from utils import encode_feature_vector_to_angles, angle_embedding
from qiskit import QuantumCircuit
import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit_aer import AerSimulator


def create_classifier_circuit(feature_vector: np.ndarray) -> QuantumCircuit:
    """
    Create a quantum circuit that encodes the given feature vector using angle encoding.
    The number of qubits is determined by the length of the feature vector.
    """

    # Encode feature vector to angles
    angles = encode_feature_vector_to_angles(feature_vector)
     
    # Angle embedding circuit
    qc = angle_embedding(angles)

    # Ansatz circuit
    ansatz = EfficientSU2(
    num_qubits=len(angles),
    su2_gates=["ry", "rz"],
    entanglement="circular",
    reps=2
    )

    qc.compose(ansatz, inplace=True)
    
    print(f"Classifier circuit created with {len(angles)} qubits.")
    print(f"Depth: {qc.depth()}, Size: {qc.size()} gates")
    
    return qc


def classify_with_circuit(circuit: QuantumCircuit) -> int:
    """
    Simulate the given quantum circuit and return a classification result with 4 outcomes.
    """

    # For simplicity, we will just measure all qubits and return the most common outcome as the class.
    # In a real implementation, you would design the circuit to produce specific output states corresponding to classes.
    
    # add measurement of first two qubits for 4 classes classification
    circuit.measure([0, 1], [0, 1])
    
    # Simulate the circuit
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