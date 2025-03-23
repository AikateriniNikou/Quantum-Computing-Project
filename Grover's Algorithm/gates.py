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

    'I' : sp.SparseMatrix(gates['I']), # Identity
    'H' : sp.SparseMatrix(gates['H']), # Hadamard
    'X' : sp.SparseMatrix(gates['X']), # Pauli-X
    'Y' : sp.SparseMatrix(gates['Y']), # Pauli-Y
    'Z' : sp.SparseMatrix(gates['Z'])  # Pauli-Z

}

def generateOperator(gate_letters, SparseMatrix = False):
    # Builds a gate matrix from a string code.  
    # Each letter stands for a gate on a qubit, starting from the most significant bit.  
    # Example: "HHI" means H on qubit1, H on qubit2, I on qubit3, combined with tensor products.  

    gate_matrix = np.array([[1]])
    if SparseMatrix:
        gate_matrix = sp.SparseMatrix(gate_matrix)  # Convert to sparse matrix
        
    control_number = 0                     
    for letter in gate_letters:
        
        if letter.isdigit():          
            control_number = int(str(control_number) + letter)
            
        elif control_number != 0:  
            
            if SparseMatrix:              
                gate = sparse_gates[letter] 
                L = 2**control_number - gate.size[0] 
                temp = sp.SparseMatrix(np.identity(L), (L + gate.size[0], L + gate.size[0]))    
                for value in gate.matrixDict: 
                    temp.matrixDict[((temp.size[0]) - (gate.size[0]) + value[0]%(gate.size[0]), (temp.size[1]) - (gate.size[1]) + value[1]%(gate.size[1]))] \
                      = gate.matrixDict[(value[0]%(gate.size[0]), value[1]%(gate.size[1]))]
            else:
                temp = np.identity(2**control_number)         
                gate = gates[letter]                  
                for x in range(len(gates)):         
                    for y in range(len(gates)):
                        temp[len(temp) - len(gate) + x%len(gate)][len(temp) - len(gate) + y%len(gate)] = gate[x%len(gate)][y%len(gate)]
            
            gate_matrix = mf.tensorProduct(gate_matrix, temp) 
            control_number = 0
            
        else:                   
            if SparseMatrix:  
                gate_matrix = mf.tensorProduct(gate_matrix, sparse_gates[letter])
            else:
                gate_matrix = mf.tensorProduct(gate_matrix, gates[letter])

    return gate_matrix

# Generate matrix to apply Hadamard gates to every qubit
def Hadamard(N, SparseMatrix=False):
    return generateOperator('H' * N, SparseMatrix)

# Generate Oracle
def Oracle(N, s, SparseMatrix=False):
    binary_target = bin(s)[2:].zfill(N)  # Convert target state to binary
    gate_list = "".join("X" if i == '0' else "I" for i in binary_target) # Create appropriate code based on target
    prep_gate = generateOperator(gate_list, SparseMatrix)
    phase_flip = generateOperator(f"{N}Z", SparseMatrix)
    return mf.matrixProduct(mf.matrixProduct(prep_gate, phase_flip), prep_gate)

# Generate Diffuser
def Diffuser(N, SparseMatrix=False):
    inversion_gate = generateOperator("X" * N, SparseMatrix)
    phase_reflect = generateOperator(f"{N}Z", SparseMatrix)
    return mf.matrixProduct(mf.matrixProduct(inversion_gate, phase_reflect), inversion_gate)