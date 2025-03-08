""" Functions for matrices"""

import matplotlib.pyplot as plt

# Vector Class
class Vector:
    
    # Create vector
    def __init__(self, values):
        self.values = list(values)
        self.dimension = len(values)
    
    # Gets vector values
    def __getitem__(self, index):
        return self.values[index]

    # Set vector values
    def __setitem__(self, index, value):
        self.values[index] = value
        
    # Tensor product between two Vectors
    def vtensor(self, other):
        dim_A, dim_B = self.dimension, other.dimension
        dim_result = dim_A * dim_B

        result = Vector([0] * dim_result)

        for i in range(dim_A):
            for j in range(dim_B):
                result[i * dim_B + j] = self[i] * other[j]
        return result

    # Output when printed
    def __repr__(self):
        return "[" + ", ".join(str(r) for r in self.values) + "]"

# qbit Subclass
class qbit(Vector):
    
    # Create qbit
    def __init__(self, alpha=1, beta=0):
        # State |0> when input = 0, |1> for input = 1
        if isinstance(alpha, int) and alpha in [0, 1]:
            super().__init__([1, 0] if alpha == 0 else [0, 1])
        else:
            super().__init__([alpha, beta])

    
# SquareMatrix Class
class SquareMatrix:
    
    # Creates a square matrix with all elements set to 0
    def __init__(self, dimension):
        self.dimension = dimension
        self.data = [[0 for i in range(dimension)] for i in range(dimension)]
    
    # Gets the value of an element from row (r) and column (c)
    def __getitem__(self, key):
        r, c = key
        return self.data[r][c]
    
    # Sets the value of an element from row (r) and column (c)
    def __setitem__(self, key, value):
        r, c = key
        self.data[r][c] = value
        
    # Output when printed
    def __repr__(self):
        s = ""
        for r in self.data:
            s += str(r) + "\n"
        return s
    
    # Tensor product of two square matrices
    def mtensor(self, other):

        dim_A, dim_B = self.dimension, other.dimension
        dim_result = dim_A * dim_B  

        result = SquareMatrix(dim_result)
        
        for i in range(dim_A):
            for j in range(dim_A):
                for k in range(dim_B):
                    for l in range(dim_B):
                        result[i * dim_B + k, j * dim_B + l] = self[i, j] * other[k, l]

        return result
    
    # Matrix multiply two square matrices
    def matrix_multiply(self, other):
  
        if self.dimension != other.dimension:
            raise ValueError("Matrix dimensions don't match")

        dim = self.dimension
        result = SquareMatrix(dim)

        for i in range(dim):
            for j in range(dim):
                result[i, j] = sum(self[i, k] * other[k, j] for k in range(dim))

        return result

    # Multiply matrix with vector
    def multiply_vector(self, vector):

        if self.dimension != vector.dimension:
            raise ValueError("Matrix and vector dimensions don't match")

        result_values = [sum(self[i, j] * vector[j] for j in range(self.dimension)) for i in range(self.dimension)]
        return Vector(result_values)

class qprogram(object):
    
    # Initialise quantum program
    def __init__(self, nqbits : int, custom : list = None):
        self.qbits = []
        self.nqbits = nqbits

        if not custom: self.qbits = [qbit(0) for i in range(nqbits)]
        else:
            for each in custom:
                self.qbits.append(qbit(each))
        self.gates = [[] for i in range(nqbits)]

    # Output when printed
    def __repr__(self) -> str:
        return "[" + ", ".join(str(r) for r in self.qbits) + "]"
    
    # Addgates function
    def addgates(self, qbitindex : int, gates : list):
        self.gates[qbitindex] += gates
    
    # Applies added gates
    def apply_gates(self):
        for i in range(self.nqbits):  # Iterate over all qubits
            for gate in self.gates[i]:  # Iterate over gates assigned to the qubit
                self.qbits[i] = gate.multiply_vector(self.qbits[i])  # Apply the gate to the qubit
                
    # Remove a gate at an index
    def removegate(self, qbitindex : int, gateindex : int = -1):
        del self.gates[qbitindex][gateindex]
        
    # Clear all gates at an index
    def cleargates(self, qbitindex : int):
        self.gates[qbitindex] = []
    
    # View matrix representation of gates
    def viewgates(self):
        print("\nQuantum Program Gates:")
        for qbitindex, gate_sequence in enumerate(self.gates):
            if gate_sequence:
                print(f"Qubit {qbitindex}: {gate_sequence}")
            else:
                print(f"Qubit {qbitindex}: No gates assigned")

    # Run program
    def run(self):
        
        # Apply all stored gates to the qubits
        self.apply_gates()  

        # Compute the tensor product of all qubits
        final_state = self.qbits[0]
        for qubit in self.qbits[1:]:
            final_state = final_state.vtensor(qubit)
            
        # Store the final quantum state
        self.final_state = final_state  
        
        # Plot the final state
        self.plot_final_state()
        print("Quantum Program Output:", self.final_state)
        
    # Plot final state
    def plot_final_state(self):
        num_qubits = len(self.qbits)
        state_vector = self.final_state.values  # Extract state amplitudes

        # Compute probabilities
        probabilities = [abs(amplitude) ** 2 for amplitude in state_vector]

        # Generate labels (binary representations of basis states)
        basis_states = [bin(i)[2:].zfill(num_qubits) for i in range(len(probabilities))]

        # Plot the bar chart
        plt.figure(figsize=(8, 5), dpi=120)
        plt.bar(basis_states, probabilities)
        plt.xlabel("Quantum Basis States")
        plt.ylabel("Probability")
        plt.title("Quantum State Probabilities")
        plt.ylim(0, 1)
        plt.show()



# Calculate tensor product
# def tensor_product(A, B):
    
#     # Calculate matrix sizes from rows (r) and columns (c)
#     r_A, c_A = len(A), len(A[0])
#     r_B, c_B = len(B), len(B[0])

#     # Create matrix
#     result = [[0] * (c_A * c_B) for i in range(r_A * r_B)]

#     # Calculate tensor product
#     for i in range(r_A):
#         for j in range(c_A):
#             for k in range(r_B):
#                 for l in range(c_B):
#                     result[i * r_B + k][j * c_B + l] = A[i][j] * B[k][l]

#     return result

# Caclulate matrix multiplication
# def matrix_multiply(A, B):
    
#     # Calculate matrix sizes from rows (r) and columns (c)
#     r_A, c_A = len(A), len(A[0])
#     r_B, c_B = len(B), len(B[0])

#     # Check multiplication validity
#     if c_A != r_B:
#         raise ValueError("Matrix dimensions do not match for multiplication!")

#     # Create matrix
#     result = [[0 for i in range(c_B)] for i in range(r_A)]

#     # Calculate multiplication
#     for i in range(r_A):
#         for j in range(c_B):
#             for k in range(c_A):  
#                 result[i][j] += A[i][k] * B[k][j]

#     return result