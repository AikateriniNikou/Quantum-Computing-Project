""" Creating the gates needed for the project
Last updated: 26/02/25"""

from matrix_functions import *

# Hadamard Gate
def H_GATE():
    H = SquareMatrix(2)
    factor = 1 / (2 ** 0.5)
    H[0, 0] = factor
    H[0, 1] = factor
    H[1, 0] = factor
    H[1, 1] = -factor
    return H

# Identity Gate
def I_GATE():
    I = SquareMatrix(2)
    I[0, 0] = 1
    I[1, 1] = 1
    return I

# C-NOT Gate
def CNOT_GATE():
    CNOT = SquareMatrix(4)
    CNOT[0, 0] = 1
    CNOT[1, 1] = 1
    CNOT[2, 3] = 1
    CNOT[3, 2] = 1
    return CNOT

# Phase Shift Gate
def P_GATE(phi):
    P = SquareMatrix(2)
    P[0, 0] = 1
    P[1, 1] = np.exp(1j * phi)
    return P

# C-V Gate
def CV_GATE():
    CV = SquareMatrix(4)
    CV[0, 0] = 1
    CV[1, 1] = 1
    CV[2, 2] = 1
    CV[3, 3] = 1j
    return CV

# Toffoli Gate
def T_GATE():
    T = SquareMatrix(8)
    T[0, 0] = 1
    T[1, 1] = 1
    T[2, 2] = 1
    T[3, 3] = 1
    T[4, 4] = 1
    T[5, 5] = 1
    T[6, 7] = 1
    T[7, 6] = 1
    return T