import numpy as np
import math
from qiskit import QuantumCircuit


def _apply_x_mask(
    qc: QuantumCircuit, address_bitstring: str, position_qubits: list[int]
):
    """
    Apply an X-mask to emulate controls-on-0 for an address.

    Why:
    - A standard MCX triggers only when all control qubits are |1>.
    - If an address bit is 0, we temporarily flip the corresponding qubit with X
      so that MCX can trigger, then we undo the mask later.

    Args:
    - qc: QuantumCircuit to modify.
    - address_bitstring: "natural" binary string (MSB on the left), e.g. "0101".
    - position_qubits: ordered list of position-qubit indices (q0 = LSB).
      Note: if position_qubits == [0, 1, 2, ...], then position_qubits[i] == i.
    """
    # Reverse so index 0 aligns with q0 (LSB)
    for bit_idx, bit in enumerate(reversed(address_bitstring)):
        if bit == "0":
            qc.x(position_qubits[bit_idx])


def encode_frqi_simple(binary_image: np.ndarray) -> tuple[QuantumCircuit, int]:
    """
    Pedagogical FRQI encoder for a binary (black/white) image.

    Convention used here:
    - Pixel index i is converted to a "natural" bitstring (MSB on the left) with format(...).
    - position qubits are q0..q(k-1), where q0 is the LSB.
    - The helper _apply_x_mask(...) reverses the bitstring to align bits with q0..

    Steps:
    1) Put all positions in superposition (H on all position qubits).
    2) For each black pixel address, apply an X-mask + MCX + undo the mask.
    """
    flat = (
        np.array(binary_image).astype(int).flatten()
    )  # flatten to 1D array row-major order: row0, row1, ...
    n_pixels = len(flat)
    if n_pixels == 0:
        raise ValueError("Empty image.")

    n_pos = math.ceil(math.log2(n_pixels))  # number of position qubits
    n_total = n_pos + 1  # +1 color qubit
    pos_qubits = list(range(n_pos))  # q0..q(k-1)
    color_qubit = n_pos  # qk

    # Warn if there are more addresses than pixels
    if 2**n_pos > n_pixels:
        print(f"Note: 2^k = {2**n_pos} > N = {n_pixels} (extra addresses stay white)")

    # Create circuit
    qc = QuantumCircuit(n_total)

    # Step 1) Superposition over addresses
    for q in pos_qubits:
        qc.h(q)
    # can also be written as: qc.h(pos_qubits)

    # Step 2) Encode black pixels with MCX

    for i, pixel_val in enumerate(flat):
        if int(pixel_val) != 1:  # only encode black pixels (value 1)
            continue

        address_bitstring = format(i, f"0{n_pos}b")  # MSB..LSB (natural)

        _apply_x_mask(
            qc, address_bitstring, pos_qubits
        )  # apply X-mask for controls-on-0
        qc.mcx(
            pos_qubits, color_qubit
        )  # Multi-controlled X: flip color qubit when ALL position qubits are |1⟩
        _apply_x_mask(qc, address_bitstring, pos_qubits)  # undo (X is its own inverse)

        qc.barrier()  # barrier for visual clarity

        # Note: MCX can apply the sandwiching X gates automatically with the ctrl_state option,

    print(f" FRQI circuit created!")
    print(f"   Depth: {qc.depth()}, Size: {qc.size()} gates")

    return qc, n_pos  # return circuit and number of position qubits


def encode_frqi_grayscale(grayscale_image: np.ndarray) -> tuple[QuantumCircuit, int]:

    flat = (
        np.array(grayscale_image).astype(float).flatten()
    )  # flatten to 1D array row-major order: row0, row1, ...
    n_pixels = len(flat)
    if n_pixels == 0:
        raise ValueError("Empty image.")

    n_pos = math.ceil(math.log2(n_pixels))  # number of position qubits
    n_total = n_pos + 1  # +1 color qubit
    pos_qubits = list(range(n_pos))  # q0..q(k-1)
    color_qubit = n_pos  # qk

    # Warn if there are more addresses than pixels
    if 2**n_pos > n_pixels:
        print(f"Note: 2^k = {2**n_pos} > N = {n_pixels} (extra addresses stay white)")

    # Create circuit
    qc = QuantumCircuit(n_total)

    # Step 1) Superposition over addresses
    for q in pos_qubits:
        qc.h(q)
    # can also be written as: qc.h(pos_qubits)

    # Step 2) Encode black pixels with MCX

    for i, pixel_val in enumerate(flat):

        address_bitstring = format(i, f"0{n_pos}b")  # MSB..LSB (natural)

        angle = 2 * np.arcsin(np.sqrt(pixel_val / 255))

        _apply_x_mask(
            qc, address_bitstring, pos_qubits
        )  # apply X-mask for controls-on-0
        qc.mcry(angle, pos_qubits, color_qubit)
        _apply_x_mask(qc, address_bitstring, pos_qubits)  # undo (X is its own inverse)

        qc.barrier()  # barrier for visual clarity

    print(f" FRQI circuit created!")
    print(f"   Depth: {qc.depth()}, Size: {qc.size()} gates")

    return qc, n_pos  # return circuit and number of position qubits


def reconstruct_grayscale_from_frqi(
    counts: dict, n_position_qubits: int, image_shape: tuple[int, int]
) -> np.ndarray:
    n_pixels = image_shape[0] * image_shape[1]  # Total number of pixels

    # 1. Accumulate statistics per pixel (Address)
    total_counts = np.zeros(n_pixels, dtype=int)
    ones_counts = np.zeros(n_pixels, dtype=int)

    print("Reconstructing image from measurements...\n")

    # For each measurement outcome (e.g., '10101')
    for outcome, count in counts.items():
        # Remove spaces if any
        outcome = outcome.replace(" ", "")

        # --- VISUAL MAPPING (Left -> High Index) ---
        # The string typically looks like: "C P3 P2 P1 P0"
        # Leftmost character is the Color Qubit
        color_char = outcome[0]

        # The rest is the Position (binary string)
        position_string = outcome[1:]

        # Convert "1101" -> 13 (Standard binary: Left is MSB)
        position_idx = int(position_string, 2)

        # Only process valid pixel positions
        if position_idx < n_pixels:
            # Accumulate counts for this address
            total_counts[position_idx] += count

            # If color bit is 1, count it as a "black" measurement
            if int(color_char) == 1:
                ones_counts[position_idx] += count

    # 2. Compute Pixel Values based on Ratio
    reconstructed = np.zeros(n_pixels, dtype=int)

    for i in range(n_pixels):
        if total_counts[i] > 0:
            # Calculate proportion of 1s (Intensity)
            ratio = ones_counts[i] / total_counts[i]

            # For Binary: Threshold at 50%
            reconstructed[i] = ratio * 255

    # Reshape to original image dimensions
    reconstructed_image = reconstructed.reshape(
        image_shape
    )  # from 1D to 2D, row-major order

    return reconstructed_image


def reconstruct_from_frqi(counts, n_position_qubits, image_shape):
    """
    Reconstruct image from FRQI measurement counts using the 'Ratio' method.
    Args:
    - counts: dict from measurement outcomes to counts.
    - n_position_qubits: number of position qubits used in FRQI.
    - image_shape: tuple (height, width) of the original image.

    Returns:
    - reconstructed_image: 2D numpy array of the reconstructed image.
    """

    n_pixels = image_shape[0] * image_shape[1]  # Total number of pixels

    # 1. Accumulate statistics per pixel (Address)
    total_counts = np.zeros(n_pixels, dtype=int)
    ones_counts = np.zeros(n_pixels, dtype=int)

    print("Reconstructing image from measurements...\n")

    # For each measurement outcome (e.g., '10101')
    for outcome, count in counts.items():
        # Remove spaces if any
        outcome = outcome.replace(" ", "")

        # --- VISUAL MAPPING (Left -> High Index) ---
        # The string typically looks like: "C P3 P2 P1 P0"
        # Leftmost character is the Color Qubit
        color_char = outcome[0]

        # The rest is the Position (binary string)
        position_string = outcome[1:]

        # Convert "1101" -> 13 (Standard binary: Left is MSB)
        position_idx = int(position_string, 2)

        # Only process valid pixel positions
        if position_idx < n_pixels:
            # Accumulate counts for this address
            total_counts[position_idx] += count

            # If color bit is 1, count it as a "black" measurement
            if int(color_char) == 1:
                ones_counts[position_idx] += count

    # 2. Compute Pixel Values based on Ratio
    reconstructed = np.zeros(n_pixels, dtype=int)

    for i in range(n_pixels):
        if total_counts[i] > 0:
            # Calculate proportion of 1s (Intensity)
            ratio = ones_counts[i] / total_counts[i]

            # For Binary: Threshold at 50%
            if ratio > 0.5:
                reconstructed[i] = 1

    # Reshape to original image dimensions
    reconstructed_image = reconstructed.reshape(
        image_shape
    )  # from 1D to 2D, row-major order

    return reconstructed_image
