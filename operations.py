import numpy as np
import quantum_states as qs
import math as m
from time import time
from scipy.sparse import csr_matrix, kron as sparse_kron, issparse

# Predefined single-qubit quantum gates
gates = {
    'H': 1 / m.sqrt(2) * np.array([[1, 1],
                                    [1, -1]]),
    'I': np.identity(2),
    'X': np.array([[0, 1],
                    [1, 0]]),
    'Y': np.array([[0, -1j],
                    [1j, 0]]),
    'Z': np.array([[1, 0],
                    [0, -1]])
}

# Predefined single-qubit gates in sparse format
sgates = {key: csr_matrix(value) for key, value in gates.items()}

### Matrix Operations   -------------------------------------------------------------------------------------------------

def matrixProduct(matA, matB):
    """ Computes the product of two matrices. """
    if issparse(matA) and issparse(matB):
        return matA @ matB
    else:
        return np.dot(matA, matB)

def vecMatProduct(mat, vec):
    """ Computes the product of a matrix and a vector, supporting both sparse and dense formats. """
    if issparse(mat):
        return mat @ vec  # Sparse matrix-vector multiplication
    else:
        return np.dot(mat, vec)  # Dense matrix-vector multiplication

def constructGate(code, Sparse=False):
    """ Constructs a quantum gate matrix based on a specified sequence. """
    if code.isdigit():  # Ensure numeric prefixes are not mistaken as gate names
        raise ValueError(f"Invalid gate name: {code}")
    
    matrix = np.array([[1]]) if not Sparse else csr_matrix(np.array([[1]]))
    
    for char in code:
        if char not in gates:
            raise ValueError(f"Unknown gate: {char}")
        gate = sgates[char] if Sparse else gates[char]
        if issparse(matrix) and issparse(gate):
            matrix = sparse_kron(matrix, gate, format="csr")
        else:
            matrix = np.kron(matrix.toarray() if issparse(matrix) else matrix,
                             gate.toarray() if issparse(gate) else gate)
    
    return matrix