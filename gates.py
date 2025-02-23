""" Creating the gates needed for the project
Last updated: 23/02/25"""


# DenseMatrix Class
class DenseMatrix:
    def __init__(self, dimension):
        self.dimension = dimension
        self.data = [[0 for i in range(dimension)] for i in range(dimension)]
    
    def __getitem__(self, key):
        row, col = key
        return self.data[row][col]
    
    def __setitem__(self, key, value):
        row, col = key
        self.data[row][col] = value
    
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
    C[0, 0] = 1; C[0, 1] = 0; C[0, 2] = 0; C[0, 3] = 0
    C[1, 0] = 0; C[1, 1] = 1; C[1, 2] = 0; C[1, 3] = 0
    C[2, 0] = 0; C[2, 1] = 0; C[2, 2] = 0; C[2, 3] = 1
    C[3, 0] = 0; C[3, 1] = 0; C[3, 2] = 1; C[3, 3] = 0
    return C