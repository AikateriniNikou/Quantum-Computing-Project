import numpy as np
import sparse as sp
import math as m
import matrix_functions as mf

# Construct single quibit gates dictionary
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

# Construct SparseMatrix single quibit gates dictionary
sparse_gates = {

    'I' : sp.SparseMatrix(gates['I']),
    'H' : sp.SparseMatrix(gates['H']),
    'X' : sp.SparseMatrix(gates['X']),
    'Y' : sp.SparseMatrix(gates['Y']),
    'Z' : sp.SparseMatrix(gates['Z'])

}

def generateOperator(gate_letters, SparseMatrix = False):
    """ Function constructing matrix representing gate dynamically

    Works by parsing a carefully formatted string (gate_letters), with characters representing the gate
    at each qubit and returns the operation as a matrix.
    First character is the gate to be applied to the most significant qubit, etc.
    i.e. the gate_letters "HHI" represents the operation HxHxI(qubit1xqubit2xqubit3)
    where x denotes the tensor product"""

    gate_matrix = np.array([[1]])        # This is starts by making a 1x1 identity matrix so the first kronecker product is the first gate.
    if SparseMatrix:
        gate_matrix = sp.SparseMatrix(gate_matrix)  # If sparse makes the matrix sparse.
    control_number = 0                        # This is for storing the control number, number of qubits that are connected to the controlled gate eg: CCNot Gate => 3X.
    for letter in gate_letters:
        if letter.isdigit():          # Sets the control number.
            control_number = int(str(control_number)+letter)
        elif control_number != 0:             # If a control number was set this creatses the controlled gate matrix
            if SparseMatrix:              # Two methods for sparse or not.
                gate = sparse_gates[letter]             # Gets the sparse gate matrix from dictioanary.
                l = 2**control_number-gate.size[0]        # These two lines create and identity sparse matrix but then force it to be 2x2 longer.
                temp = sp.SparseMatrix(np.identity(l), (l+gate.size[0],l+gate.size[0]))    # sp.SparseMatrix takes two parameters; a matrix and a double this being the shape.
                for pos in gate.matrixDict:                                         # This part adds the sparse gate matrix to the new forced sparse identiy.
                    temp.matrixDict[((temp.size[0])-(gate.size[0])+pos[0]%(gate.size[0]) \
                     , (temp.size[1])-(gate.size[1])+pos[1]%(gate.size[1]))] \
                      = gate.matrixDict[(pos[0]%(gate.size[0]),pos[1]%(gate.size[1]))]
            else:
                temp = np.identity(2**control_number)          # For non sparse we start with an identity.
                gate = gates[letter]                  # Gets gate from dictionary
                for x in range(len(gates)):         # This adds the 2x2 gate matrix to the end of the identity. 
                    for y in range(len(gates)):
                        temp[len(temp)-len(gate)+x%len(gate)][len(temp)-len(gate) \
                        +y%len(gate)] = gate[x%len(gate)][y%len(gate)]
            gate_matrix = mf.kroneckerProduct(gate_matrix,temp)       # Whether sparse or not this does the kronecker product of the existing matrix with the new controlled gate matrix.
            control_number = 0
        else:                   # This is the main part if there is no control element.. 
            if SparseMatrix:          # This changes the gate dictionary depending on whether we are using sparse matrices or not.
                gate_matrix = mf.kroneckerProduct(gate_matrix,sparse_gates[letter])      # Then whether we are sparse or not it does the kronecker product on the matrix.
            else:
                gate_matrix = mf.kroneckerProduct(gate_matrix,gates[letter])
    return gate_matrix

def Hadamard(N, SparseMatrix=False):
    return generateOperator('H' * N, SparseMatrix)

def Oracle(N, s, SparseMatrix=False):
    binary_target = bin(s)[2:].zfill(N)  # Convert target state to binary
    gate_list = "".join("X" if i == '0' else "I" for i in binary_target)
    prep_gate = generateOperator(gate_list, SparseMatrix)
    phase_flip = generateOperator(f"{N}Z", SparseMatrix)
    return mf.matrixProduct(mf.matrixProduct(prep_gate, phase_flip), prep_gate)

def Diffuser(N, SparseMatrix=False):
    inversion_gate = generateOperator("X" * N, SparseMatrix)
    phase_reflect = generateOperator(f"{N}Z", SparseMatrix)
    return mf.matrixProduct(mf.matrixProduct(inversion_gate, phase_reflect), inversion_gate)