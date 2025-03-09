""" Functions for matrices"""

import matplotlib.pyplot as plt


# def gather(i, target_qubits):
#     """Extracts bits of integer i at positions in target_qubits and packs them in order."""
#     result = 0
#     for k, q in enumerate(sorted(target_qubits)):
#         bit = (i >> q) & 1
#         result |= (bit << k)
#     return result

# def scatter(j, target_qubits):
#     """Scatters bits of integer j into positions specified in target_qubits."""
#     result = 0
#     for k, q in enumerate(sorted(target_qubits)):
#         bit = (j >> k) & 1
#         result |= (bit << q)
#     return result

# Gets a cnot matrix given control qbit and target qbit
def generalized_cnot(num_qubits, control, target):
    
        size = 2 ** num_qubits  # Total number of states
        CNOT_matrix = SquareMatrix(size)  # Initialize a SquareMatrix object
        
        # Start with identity matrix
        for i in range(size):
            CNOT_matrix[i, i] = 1
        
        # Modify entries to represent CNOT operation
        for i in range(size):  # Iterate through all basis states
            binary = format(i, f'0{num_qubits}b')  # Convert index to binary string
            if binary[control] == '1':  # Check if control qubit is 1
                flipped = list(binary)
                flipped[target] = '1' if flipped[target] == '0' else '0'  # Flip the target qubit
                j = int(''.join(flipped), 2)  # Convert back to integer
                
                # Swap rows and columns in the matrix
                CNOT_matrix[i, i] = 0
                CNOT_matrix[j, j] = 0
                CNOT_matrix[i, j] = 1
                CNOT_matrix[j, i] = 1
        
        return CNOT_matrix

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
            
    # Multiply matrix with vector
    def multiply_vector(self, vector):

        if self.dimension != vector.dimension:
            raise ValueError("Matrix and vector dimensions don't match")

        result_values = [sum(self[i, j] * vector[j] for j in range(self.dimension)) for i in range(self.dimension)]
        return Vector(result_values)

    
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

# The Quantum Computing program
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
        self.quantum_state = []
        
    def I_GATE(self, n):
        I = SquareMatrix(n)
        for i in range(n):
            I[i, i] = 1
        return I
    
    # Output when printed
    def __repr__(self) -> str:
        return "[" + ", ".join(str(r) for r in self.qbits) + "]"
    
    def addgates(self, qbitindex, gates, control_positions=None):
        """ Add single-qubit or multi-qubit gates """
        if control_positions is None:
            self.gates[qbitindex] += gates
            
            # Apply identity gate to all other qubits
            for i in range(self.nqbits):
                if i != qbitindex:
                    self.gates[i].append(self.I_GATE(2))
                    
        else:
            self.gates[qbitindex].append([control_positions, gates])
            
            # Apply identity gate to all other qubits
            for i in range(self.nqbits):
                if i != qbitindex:
                    self.gates[i].append(self.I_GATE(gates[0].dimension))

        
    # Applies the gates 
    def apply_gates(self):
        num_gates = max(len(gates) for gates in self.gates)
        
        # Start with identity matrix for full system
        full_gate_matrix = SquareMatrix(2**self.nqbits)
        for i in range(2**self.nqbits):
            full_gate_matrix[i, i] = 1

        # Construct the full gate matrix by applying tensor products
        for gate_index, n in enumerate(range(num_gates)):
            tensor_product = self.gates[0][gate_index]
            for i in range(1, self.nqbits):
                if isinstance(tensor_product, list):
                    tensor_product = generalized_cnot(self.nqbits, 0, tensor_product[0][0])
                    break
                    #tensor_product = tensor_product[1][0].mtensor(self.gates[i][gate_index])
                else:
                    tensor_product = tensor_product.mtensor(self.gates[i][gate_index])

            
            # Multiply into full gate matrix
            full_gate_matrix = tensor_product.matrix_multiply(full_gate_matrix)
            
        # Apply the final constructed gate matrix to the quantum state
        full_state = self._get_full_state()

        new_state = full_gate_matrix.multiply_vector(full_state)
        self._set_full_state(new_state)
        self.quantum_state = new_state

    # Gets the quantum state
    def _get_full_state(self):
        state_all = self.qbits
        ini_state = state_all[0]
        for i in range(len(state_all)-1):
            ini_state = ini_state.vtensor(state_all[i+1])
            
        return ini_state
    
    # Sets the quantum state
    def _set_full_state(self, full_state):
        size = 2
        for i in range(self.nqbits):
            self.qbits[i] = full_state[:size]
            size *= 2        
    
    # # Remove a gate at an index
    # def removegate(self, qbitindex : int, gateindex : int = -1):
    #     del self.gates[qbitindex][gateindex]
        
    # # Clear all gates at an index
    # def cleargates(self, qbitindex : int):
    #     self.gates[qbitindex] = []
    
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
        final_state = self.quantum_state
            
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