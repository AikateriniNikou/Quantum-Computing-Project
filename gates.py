""" Creating the gates needed for the project
Last updated: 23/02/25"""

import numpy as np

# DenseMatrix Class
class DenseMatrix:
    
    # Creates a square matrix with all elements set to 0
    def __init__(self, dimension):
        self.dimension = dimension
        self.data = [[0 for i in range(dimension)] for i in range(dimension)]
    
    # Gets the value of an element from row and column
    def __getitem__(self, key):
        row, col = key
        return self.data[row][col]
    
    # Sets the value of an element from row and column
    def __setitem__(self, key, value):
        row, col = key
        self.data[row][col] = value
    
    # Output when printed
    def __repr__(self):
        s = ""
        for row in self.data:
            s += str(row) + "\n"
        return s


# Gate Definitions
# ----------------

# Hadamard Gate
def hadamard_gate():
    H = DenseMatrix(2)
    factor = 1 / (2 ** 0.5)
    H[0, 0] = factor
    H[0, 1] = factor
    H[1, 0] = factor
    H[1, 1] = -factor
    return H

# Identity Gate
def identity_gate():
    I = DenseMatrix(2)
    I[0, 0] = 1
    I[1, 1] = 1
    return I

# C-NOT Gate
def cnot_gate():
    C = DenseMatrix(4)
    C[0, 0] = 1
    C[1, 1] = 1
    C[2, 3] = 1
    C[3, 2] = 1
    return C

# Phase Shift Gate
def phase_gate(phi):
    P = DenseMatrix(2)
    P[0, 0] = 1
    P[1, 1] = np.exp(1j * phi)
    return P

# C-V Gate
def cv_gate():
    CV = DenseMatrix(4)
    CV[0, 0] = 1
    CV[1, 1] = 1
    CV[2, 2] = 1
    CV[3, 3] = 1j
    return CV

# Toffoli Gate
def toffoli_gate():
    T = DenseMatrix(8)
    T[0, 0] = 1
    T[1, 1] = 1
    T[2, 2] = 1
    T[3, 3] = 1
    T[4, 4] = 1
    T[5, 5] = 1
    T[6, 7] = 1
    T[7, 6] = 1
    return T