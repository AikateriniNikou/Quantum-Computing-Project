import numpy as np
import sparse as sp
import math as m
import matrix_functions as mf

# Single qubit gates
gates = {

    # Identity Gate
    'I' : np.array([[1, 0],
                    [0, 1]]),
    
    # Hadamard Gate
    'H' : 1/m.sqrt(2)*np.array([[1,1],
                                [1,-1]]),
    
    # Pauli-X Gate
    'X' : np.array([[0,1],
                    [1,0]]),
    
    # Pauli-Y Gate
    'Y' : np.array([[0,-1j],
                    [1j,0]]),
    
    # Pauli-Z Gate
    'Z' : np.array([[1,0],
                    [0,-1]])
}

# SparseMatrix single quibit gates
sgates = {

    'I' : sp.SparseMatrix(gates['I']),
    'H' : sp.SparseMatrix(gates['H']),
    'X' : sp.SparseMatrix(gates['X']),
    'Y' : sp.SparseMatrix(gates['Y']),
    'Z' : sp.SparseMatrix(gates['Z'])

}

def generateOperator(code, SparseMatrix = False):
    """ Function constructing matrix representing gate dynamically

    Works by parsing a carefully formatted string (code), with characters representing the gate
    at each qubit and returns the operation as a matrix.
    First character is the gate to be applied to the most significant qubit, etc.
    i.e. the code "HHI" represents the operation HxHxI(qubit1xqubit2xqubit3)
    where x denotes the tensor product"""

    gate_matrix = np.array([[1]])        # This is starts by making a 1x1 identity matrix so the first kronecker product is the first gate.
    if SparseMatrix:
        gate_matrix = sp.SparseMatrix(gate_matrix)  # If sparse makes the matrix sparse.
    TofN = 0                        # This is for storing the control number, number of qubits that are connected to the controlled gate eg: CCNot Gate => 3X.
    for char in code:
        if char.isdigit():          # Sets the control number.
            TofN = int(str(TofN)+char)
        elif TofN != 0:             # If a control number was set this creatses the controlled gate matrix
            if SparseMatrix:              # Two methods for sparse or not.
                gate = sgates[char]             # Gets the sparse gate matrix from dictioanary.
                l = 2**TofN-gate.size[0]        # These two lines create and identity sparse matrix but then force it to be 2x2 longer.
                Tof = sp.SparseMatrix(np.identity(l), (l+gate.size[0],l+gate.size[0]))    # sp.SparseMatrix takes two parameters; a matrix and a double this being the shape.
                for pos in gate.matrixDict:                                         # This part adds the sparse gate matrix to the new forced sparse identiy.
                    Tof.matrixDict[((Tof.size[0])-(gate.size[0])+pos[0]%(gate.size[0]) \
                     , (Tof.size[1])-(gate.size[1])+pos[1]%(gate.size[1]))] \
                      = gate.matrixDict[(pos[0]%(gate.size[0]),pos[1]%(gate.size[1]))]
            else:
                Tof = np.identity(2**TofN)          # For non sparse we start with an identity.
                gate = gates[char]                  # Gets gate from dictionary
                for x in range(len(gates)):         # This adds the 2x2 gate matrix to the end of the identity. 
                    for y in range(len(gates)):
                        Tof[len(Tof)-len(gate)+x%len(gate)][len(Tof)-len(gate) \
                        +y%len(gate)] = gate[x%len(gate)][y%len(gate)]
            gate_matrix = mf.kroneckerProduct(gate_matrix,Tof)       # Whether sparse or not this does the kronecker product of the existing matrix with the new controlled gate matrix.
            TofN = 0
        else:                   # This is the main part if there is no control element.. 
            if SparseMatrix:          # This changes the gate dictionary depending on whether we are using sparse matrices or not.
                gate_matrix = mf.kroneckerProduct(gate_matrix,sgates[char])      # Then whether we are sparse or not it does the kronecker product on the matrix.
            else:
                gate_matrix = mf.kroneckerProduct(gate_matrix,gates[char])
    return gate_matrix

""" Constructs an oracle gate dynamically. """
def Oracle(nq, s, SparseMatrix=False):
    binary_target = bin(s)[2:].zfill(nq)  # Convert target state to binary
    gate_list = "".join("X" if i == '0' else "I" for i in binary_target)
    L = generateOperator(gate_list, SparseMatrix)
    Z = generateOperator(f"{nq}Z", SparseMatrix)
    return mf.matrixProduct(mf.matrixProduct(L, Z), L)

def Hadamard(nq, SparseMatrix=False):
    return generateOperator('H' * nq, SparseMatrix)

def Diffuser(nq, SparseMatrix=False):
    L = generateOperator("X" * nq, SparseMatrix)
    Z = generateOperator(f"{nq}Z", SparseMatrix)
    return mf.matrixProduct(mf.matrixProduct(L, Z), L)