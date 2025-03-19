import numpy as np
import math as m
import matrix_functions as mf

from aux_functions import Vector, generalized_cnot
from gates import generateOperator

# -----------------------------------------------------------------------------
# Helper: Full single-qubit operator on an n-qubit system
# -----------------------------------------------------------------------------
def full_single_qubit_operator(num_qubits, gate_letter, target_index):
    """
    Constructs the full 2^n x 2^n operator for applying a single-qubit gate 
    (identified by gate_letter, e.g. 'I', 'X', 'Z', etc.) on the qubit at 
    position target_index (0 = leftmost), while applying identity on all others.
    """
    # Create a gate string with 'I' everywhere except gate_letter at target_index.
    gate_string = ''.join(gate_letter if i == target_index else 'I' for i in range(num_qubits))
    # Request sparse representation (this now returns an object with a multiply_vector method)
    return generateOperator(gate_string, SparseMatrix=True)

# -----------------------------------------------------------------------------
# Three-Qubit Bit-Flip Code Functions
# -----------------------------------------------------------------------------
def encode_three_qubit(initial):
    ancilla = Vector([1, 0])
    # Tensor the initial state with two ancillas.
    encoded = initial.vtensor(ancilla).vtensor(ancilla)
    # Apply CNOT from qubit 0 -> 1 and 0 -> 2 (using new generalized_cnot that returns a SquareMatrix).
    cnot_01 = generalized_cnot(3, 0, 1)
    encoded = cnot_01.multiply_vector(encoded)
    cnot_02 = generalized_cnot(3, 0, 2)
    encoded = cnot_02.multiply_vector(encoded)
    return encoded

def apply_bit_flip_error(encoded, target_index):
    op = full_single_qubit_operator(3, 'X', target_index)
    return op.multiply_vector(encoded)

def diagnose_bit_flip_error(encoded):
    # Find all nonzero amplitude indices.
    nonzero_indices = [i for i, amp in enumerate(encoded.values) if abs(amp) > 1e-6]
    # If only the |000> and |111> amplitudes remain, assume no correctable error.
    if set(nonzero_indices) == {0, 7}:
        return None
    for index in nonzero_indices:
        bitstr = format(index, '03b')
        if bitstr not in ["000", "111"]:
            d0 = sum(1 for b1, b2 in zip(bitstr, "000") if b1 != b2)
            d1 = sum(1 for b1, b2 in zip(bitstr, "111") if b1 != b2)
            if d0 == 1:
                return bitstr.index("1")
            elif d1 == 1:
                return bitstr.index("0")
    return None

def correct_bit_flip_error(encoded):
    error_qubit = diagnose_bit_flip_error(encoded)
    if error_qubit is None:
        return encoded
    op = full_single_qubit_operator(3, 'X', error_qubit)
    corrected = op.multiply_vector(encoded)
    return corrected

def decode_three_qubit(encoded):
    # For our three-qubit bit-flip code, the decoded state is stored in amplitudes 0 and 7.
    alpha = encoded.values[0]
    beta  = encoded.values[7]
    return Vector([alpha, beta])

# -----------------------------------------------------------------------------
# Shor's Nine-Qubit Code Functions
# -----------------------------------------------------------------------------
def encode_shor(initial):
    alpha = initial.values[0]
    beta = initial.values[1]
    # Construct the 8-dimensional states used for encoding.
    psi_plus = Vector([1/np.sqrt(2) if i in [0, 7] else 0 for i in range(8)])
    psi_minus = Vector([1/np.sqrt(2) if i == 0 else (-1/np.sqrt(2) if i == 7 else 0) for i in range(8)])
    # Encode into 9 qubits via three tensor products.
    zeroL = psi_plus.vtensor(psi_plus).vtensor(psi_plus)
    oneL  = psi_minus.vtensor(psi_minus).vtensor(psi_minus)
    encoded_values = [alpha * z + beta * o for z, o in zip(zeroL.values, oneL.values)]
    return Vector(encoded_values)

def apply_error_shor(encoded, error_qubit, error_type):
    if error_type == "X":
        op = full_single_qubit_operator(9, 'X', error_qubit)
    elif error_type == "Z":
        op = full_single_qubit_operator(9, 'Z', error_qubit)
    elif error_type in ["XZ", "ZX"]:
        # For a combined error, compose the operators (note: X and Z are self-inverse).
        opX = full_single_qubit_operator(9, 'X', error_qubit)
        opZ = full_single_qubit_operator(9, 'Z', error_qubit)
        op = opZ.matrix_multiply(opX)
    else:
        raise ValueError("Unknown error type")
    return op.multiply_vector(encoded)

def correct_error_shor(encoded, error_qubit, error_type):
    # Since the Pauli X and Z operators are self-inverse, applying the same operator
    # corrects the error.
    return apply_error_shor(encoded, error_qubit, error_type)

def decode_shor(encoded):
    psi_plus = Vector([1/np.sqrt(2) if i in [0, 7] else 0 for i in range(8)])
    psi_minus = Vector([1/np.sqrt(2) if i == 0 else (-1/np.sqrt(2) if i == 7 else 0) for i in range(8)])
    psi0 = psi_plus.vtensor(psi_plus).vtensor(psi_plus)
    psi1 = psi_minus.vtensor(psi_minus).vtensor(psi_minus)
    alpha = sum(np.conjugate(psi0.values[i]) * encoded.values[i] for i in range(len(encoded.values)))
    beta  = sum(np.conjugate(psi1.values[i]) * encoded.values[i] for i in range(len(encoded.values)))
    return Vector([alpha, beta])

# =============================================================================
# Test Cases
# =============================================================================
def test_three_qubit_bit_flip():
    print("=== Testing Three-Qubit Bit-Flip Code ===")
    # Prepare an initial state: |ψ> = (|0> + |1>)/√2
    alpha = 1/np.sqrt(2)
    beta  = 1/np.sqrt(2)
    initial = Vector([alpha, beta])
    print("Initial state:", initial)
    
    # (1) Encode the state
    encoded = encode_three_qubit(initial)
    print("Encoded state (3 qubits):")
    print(encoded)
    
    # (2) Decode with NO error to check perfect encoding/decoding.
    decoded_no_error = decode_three_qubit(encoded)
    print("Decoded state (no error):", decoded_no_error)
    
    # (3) Simulate an error: Apply a bit-flip error on qubit 1.
    errored = apply_bit_flip_error(encoded, 1)
    print("State after bit-flip error on qubit 1:")
    print(errored)
    
    # (4) Diagnose the error.
    error_qubit = diagnose_bit_flip_error(errored)
    print("Diagnosed error on qubit:", error_qubit)
    
    # (5) Correct the error.
    corrected = correct_bit_flip_error(errored)
    print("State after error correction:")
    print(corrected)
    
    # (6) Decode the corrected state.
    decoded = decode_three_qubit(corrected)
    print("Decoded state after correction:", decoded)
    print("\n")

def test_shor_code():
    print("=== Testing Shor's Nine-Qubit Code ===")
    # Prepare an initial state: |ψ> = (|0> + |1>)/√2
    alpha = 1/np.sqrt(2)
    beta  = 1/np.sqrt(2)
    initial = Vector([alpha, beta])
    print("Initial state:", initial)
    
    # (1) Encode the state using Shor's code.
    encoded = encode_shor(initial)
    print("Encoded state (Shor, 9 qubits): vector length =", len(encoded.values))
    
    # (2) Apply a bit-flip error (X) on qubit 4.
    errored_X = apply_error_shor(encoded, 4, "X")
    print("Applied bit-flip error (X) on qubit 4.")
    
    # (3) Correct the bit-flip error.
    corrected_X = correct_error_shor(errored_X, 4, "X")
    print("Corrected bit-flip error.")
    
    # (4) Decode the state.
    decoded_X = decode_shor(corrected_X)
    print("Decoded state after bit-flip correction:", decoded_X)
    
    # (5) Now apply a phase-flip error (Z) on qubit 2.
    errored_Z = apply_error_shor(encoded, 2, "Z")
    print("Applied phase-flip error (Z) on qubit 2.")
    
    # (6) Correct the phase-flip error.
    corrected_Z = correct_error_shor(errored_Z, 2, "Z")
    print("Corrected phase-flip error.")
    
    # (7) Decode the state.
    decoded_Z = decode_shor(corrected_Z)
    print("Decoded state after phase-flip correction:", decoded_Z)
    print("\n")

# =============================================================================
# Main: Run Test Cases
# =============================================================================
if __name__ == "__main__":
    test_three_qubit_bit_flip()
    test_shor_code()