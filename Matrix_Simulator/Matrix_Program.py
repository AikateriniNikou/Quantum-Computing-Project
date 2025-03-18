import numpy as np
import matplotlib.pyplot as plt
  
""" Gate definitions """
class GATES:
    
    # Identity Gate
    def I_GATE():
        I = np.array([[1, 0],
                      [0, 1]], dtype=complex)
        return I
    
    # Hadamard Gate
    def H_GATE():
        value = complex(1/np.sqrt(2), 0)
        H = np.array([[value, value],
                      [value, -value]], dtype=complex)
        return H
    
    # Phase Gate
    def PHASE_GATE(theta):
        phase = complex(np.cos(theta), np.sin(theta))
        P = np.array([[1, 0],
                      [0, phase]], dtype=complex)
        return P
    
    # Pauli X Gate
    def X_GATE():
        X = np.array([[0, 1],
                      [1, 0]], dtype=complex)
        return X
    
    # Pauli Y Gate
    def Y_GATE():
        Y = np.array([[0, complex(0,-1)],
                      [complex(0,1), 0]], dtype=complex)
        return Y
    
    # Pauli Z Gate
    def Z_GATE():
        Z = np.array([[1, 0],
                      [0, -1]], dtype=complex)
        return Z
    
    # X Rotation Gate
    def X_ROT_GATE(theta):
        sin = np.sin(theta/2)
        cos = np.cos(theta/2)
        X_ROT = np.array([[cos, -1j * sin], 
                          [-1j * sin, cos]], dtype=complex)
        return X_ROT
    
    # Y Rotation Gate
    def Y_ROT_GATE(theta):
        sin = np.sin(theta/2)
        cos = np.cos(theta/2)
        Y_ROT = np.array([[cos, -sin], 
                          [sin, cos]], dtype=complex)
        return Y_ROT
    
    # Z Rotation Gate
    def Z_ROT_GATE(theta):
        value = 0.5j * theta
        Z_ROT = np.array([[np.exp(-value), 0], 
                          [0, np.exp(value)]], dtype=complex)
        return Z_ROT
    
    # CNOT Gate
    def CNOT_GATE():
        CNOT = np.array([[1, 0, 0, 0], 
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
        return CNOT
    
"""Functions that create tensor product matrices"""
class Tensor_Gates:
    
    # Tensor together specified gate and correct number of identity gates
    def Build_Matrix(gate, q, N):
        tensor_state = np.eye(1,1)
        for i in range(0, N):
            if i == q:
                tensor_state = tensor_product(tensor_state, gate)
            else:
                tensor_state = tensor_product(tensor_state, GATES.I_GATE())
        return tensor_state
    
    # Identity Gate Tensor
    def Tensor_Identity(q, N):
        return np.array(np.eye(2**N),dtype=complex)
    
    # Hadamard Gate Tensor
    def Tensor_H(q, N):
        return Tensor_Gates.Build_Matrix(GATES.H_GATE(), q, N)
    
    # Phase Gate Tensor
    def Tensor_PHASE(theta, q, N):
        return Tensor_Gates.Build_Matrix(GATES.PHASE_GATE(theta), q, N)
    
    # Pauli X Gate Tensor
    def Tensor_X(q, N):
        return Tensor_Gates.Build_Matrix(GATES.X_GATE(), q, N)
    
    # Pauli Y Gate Tensor
    def Tensor_Y(q, N):
        return Tensor_Gates.Build_Matrix(GATES.Y_GATE(), q, N)
    
    # Pauli Z Gate Tensor
    def Tensor_Z(q, N):
        return Tensor_Gates.Build_Matrix(GATES.Z_GATE(), q, N)
    
    # X Rotation Gate Tensor
    def Tensor_X_ROT(theta, q, N):
        return Tensor_Gates.Build_Matrix(GATES.X_ROT_GATE(theta), q, N)
    
    # Y Rotation Gate Tensor
    def Tensor_Y_ROT(theta, q, N):
        return Tensor_Gates.Build_Matrix(GATES.Y_ROT_GATE(theta), q, N)
    
    # Z Rotation Gate Tensor
    def Tensor_Z_ROT(theta, q, N):
        return Tensor_Gates.Build_Matrix(GATES.Z_ROT_GATE(theta), q, N)
    
    # CNOT Gate Tensor
    def Tensor_CNOT(control, target, N):
        new_a = QUANTUM.ket_bra_outer(2,0,0)
        new_b = QUANTUM.ket_bra_outer(2,1,1)
        tensor_state_0 = np.eye(1,1)
        tensor_state_1 = np.eye(1,1)
        for i in range (0, N):
            if control == i:
                tensor_state_0 = tensor_product(tensor_state_0, new_a)
                tensor_state_1 = tensor_product(tensor_state_1, new_b)
            elif target == i:
                tensor_state_0 = tensor_product(tensor_state_0, GATES.I_GATE())
                tensor_state_1 = tensor_product(tensor_state_1, GATES.X_GATE())
            else:
                tensor_state_0 = tensor_product(tensor_state_0, GATES.I_GATE())
                tensor_state_1 = tensor_product(tensor_state_1, GATES.I_GATE())
        return tensor_state_0 + tensor_state_1
    
    # Toffoli Gate Tensor
    def Tensor_Toffoli(control1, control2, target, N):
        if control2 != 0 and target != 0:
            tensor_state_0 = Tensor_Gates.Tensor_Swap(control1, 0, N)
            tensor_state_1 = Tensor_Gates.Tensor_CNOT(control2-1, target-1, N-1)
            tensor_state_3 = Tensor_Gates.Tensor_Operation(tensor_state_1)
            return np.dot(np.dot(tensor_state_0, tensor_state_3), tensor_state_0)
        elif control2 == 0:
            return Tensor_Gates.Tensor_Toffoli(control2, control1, target, N)
        else:
            state1 = Tensor_Gates.Tensor_Swap(control1, target, N)
            state2 = Tensor_Gates.Tensor_Toffoli(target, control2, control1, N)
            return np.dot(np.dot(state1, state2), state1)
    
    # Function for Toffoli Gate Tensor
    def Tensor_Operation(operation):
        identity = np.eye(*operation.shape)
        new_a = QUANTUM.ket_bra_outer(2,0,0)
        new_b = QUANTUM.ket_bra_outer(2,1,1)
        state1 = tensor_product(new_a, identity)
        state2 = tensor_product(new_b, operation)
        return state1 + state2
    
    # Swap Gate Tensor
    def Tensor_Swap(a, b, N):
        state1 = Tensor_Gates.Tensor_CNOT(a, b, N)
        state2 = Tensor_Gates.Tensor_CNOT(b, a, N)
        return np.dot(np.dot(state1, state2), state1)
        
"""Computes the inner product of arrays."""
def inner(a, b):
    return [np.array(sum(a[i] * b[i] for i in range(len(a))))]

"""Computes the outer product of arrays."""
def outer(a, b):
    m, n = len(a), len(b)
    outer_product = [[a[i] * b[j] for j in range(n)] for i in range(m)]
    
    return np.array(outer_product)[0]

"""Computes the tensor product of two arrays."""
def tensor_product(A, B):

    shape_A = np.array(A.shape)
    shape_B = np.array(B.shape)

    tensor_shape = tuple(shape_A * shape_B)

    tensor_product = np.zeros(tensor_shape, dtype=complex)

    for index_A in np.ndindex(A.shape):
        for index_B in np.ndindex(B.shape):
            new_index = tuple(np.array(index_A) * shape_B + np.array(index_B))
            tensor_product[new_index] = A[index_A] * B[index_B]

    return tensor_product

"""Produces bra and ket vectors."""
class QUANTUM:
    
    # Ket with N zeros and 1 in position a
    def basis_ket(N, a):
        value = np.zeros((N, 1))
        value[a, 0] = 1
        return value
    
    # Bra with N zeros and 1 in position a
    def basis_bra(N, a):
        value = np.zeros((1, N))
        value[0, a] = 1
        return value
    
    # Inner product between bra-ket
    def bra_ket_inner(N, a, b):
        value_b = QUANTUM.basis_bra(N, a)
        value_k = QUANTUM.basis_ket(N, b)
        return [[inner(value_b[0], value_k.T[0])]]
    
    # Outer product between bra-ket
    def ket_bra_outer(N, a, b):
        value_k = QUANTUM.basis_ket(N, a)
        value_b = QUANTUM.basis_bra(N, b)
        return outer(value_b, value_k)

"""Basis States"""
class Basis_Vector:
    
    # Generate basis state vector
    def __init__(self, N):
        
        # If int argumenet, create basis corresponding to all 0 qubits
        if isinstance(N, int):
            self.N = N
            self.ini_state = N
            self.index = 0
            self.basis = np.zeros((2**self.N, 1), dtype=complex)
            self.basis[self.index] = 1  
        
        # If list argumenet, create basis corresponding to custom list of qubits
        elif isinstance(N, list):
            self.N = int(np.log2(len(N)))  
            self.ini_state = N
            self.basis = np.array(self.ini_state, dtype=complex).reshape(-1, 1)  # Store state
        
    # Apply gate to basis state vector
    def apply(self, gate):
        self.basis = np.transpose(np.array(inner(gate, self.basis)))
    
    # Get the current basis state vectors
    def get_quantum_state(self):
        return self.basis
    
    # Turn states into strings
    def get_state_as_string(self):
        return state_as_string(self.index, self.N)
    
    # Print function
    def print(self):
        for i, value in enumerate(self.basis):
            print(f"{state_as_string(i, self.N)} - {value[0]}")                 

"""Our quantum program"""

class qprogram:
    
    def __init__(self, init_state):
        
        # Check if argument is int or list
        if isinstance(init_state, int):
            self.N = init_state
            self.ini_state = init_state
            self.basis = Basis_Vector(self.ini_state)

        elif isinstance(init_state, list):
            self.N = len(init_state)
            self.ini_state = qprogram.tensor_product_qubits(init_state)
            self.basis = Basis_Vector(self.ini_state)

        self.quantum_states = [self.basis.get_quantum_state()]
        self.names = []
        self.gates = []
    
    # Tensor product the qubits in list
    def tensor_product_qubits(qubit_list):
            ket0 = np.array([[1], [0]], dtype=complex)
            ket1 = np.array([[0], [1]], dtype=complex)

            value = ket0 if qubit_list[0] == 0 else ket1

            for qubit in qubit_list[1:]:
                next_ket = ket0 if qubit == 0 else ket1
                new_value = np.zeros((value.shape[0] * next_ket.shape[0], 1), dtype=complex)
                index = 0
                for i in range(value.shape[0]):
                    for j in range(next_ket.shape[0]):
                        new_value[index, 0] = value[i, 0] * next_ket[j, 0]
                        index += 1
                value = new_value

            return value.flatten().tolist()
    
    # Apply Identity Gate
    def i(self, q):
        tensor_state = Tensor_Gates.Tensor_Identity(q, self.N)
        self.names.append(f"Identity gate applied to qubit {q}:")
        self.gates.append(tensor_state)
     
    # Apply Hadamard Gate
    def h(self, q):
        tensor_state = Tensor_Gates.Tensor_H(q, self.N)
        self.names.append(f"Hadamard gate applied to qubit {q}:")
        self.gates.append(tensor_state)
    
    # Apply Phase Gate
    def phase(self, theta, q):
        tensor_state = Tensor_Gates.Tensor_PHASE(theta, q, self.N)
        self.names.append(f"Phase gate applied to qubit {q}")
        self.gates.append(tensor_state)
    
    # Apply Pauli X Gate
    def x(self, q):
        tensor_state = Tensor_Gates.Tensor_X(q, self.N)
        self.names.append(f"Pauli X gate applied to qubit {q}:")
        self.gates.append(tensor_state)

    # Apply Pauli Y Gate
    def y(self, q):
        tensor_state = Tensor_Gates.Tensor_Y(q, self.N)
        self.names.append(f"Pauli Y gate applied to qubit {q}:")
        self.gates.append(tensor_state)
    
    # Apply Pauli Z Gate
    def z(self, q):
        tensor_state = Tensor_Gates.Tensor_Z(q, self.N)
        self.names.append(f"Pauli Z gate applied to qubit {q}:")
        self.gates.append(tensor_state)
    
    # Apply X Rotation Gate
    def rot_x(self, theta, q):
        tensor_state = Tensor_Gates.Tensor_X_ROT(theta, q, self.N)
        self.names.append(f"Rotate X gate applied to qubit {q}")
        self.gates.append(tensor_state)
    
    # Apply Y Rotation Gate
    def rot_y(self, theta, q):
        tensor_state = Tensor_Gates.Tensor_Y_ROT(theta, q, self.N)
        self.names.append(f"Rotate Y gate applied to qubit {q}")
        self.gates.append(tensor_state)
    
    # Apply Z Rotation Gate
    def rot_z(self, theta, q):
        tensor_state = Tensor_Gates.Tensor_Z_ROT(theta, q, self.N)
        self.names.append(f"Rotate Z gate applied to qubit {q}")
        self.gates.append(tensor_state)

    # Apply CNOT Gate
    def cnot(self, control, target):
        tensor_state = Tensor_Gates.Tensor_CNOT(control, target, self.N)
        self.names.append(f"CNOT with control qubit {control} and target qubit {target}:")
        self.gates.append(tensor_state)
    
    # Apply Toffoli Gate
    def t(self, control1, control2, target2):
        tensor_state = Tensor_Gates.Tensor_Toffoli(control1, control2, target2, self.N)
        self.names.append(f"Toffoli with control qubit {control1}, control qubit {control2} and target qubit {target2}:")
        self.gates.append(tensor_state)
    
    # Run program
    def run(self, show_states=False, plot=False):
        self.basis = Basis_Vector(self.ini_state)
        
        # Print the basis states and the gate matrices
        if show_states:
            print("--------------------")
            print("Initial basis state:")
            print("--------------------")
            self.basis.print()
            print("")
        for gate, name in zip(self.gates, self.names):
            self.basis.apply(gate)
            self.quantum_states.append(self.basis.get_quantum_state())
            
            if show_states:
                print(name)
                print(gate)
                print("")
                print("------------")
                print("Basis state:")
                print("------------")
                self.basis.print()
                print("")
        
        # Plot probability graph
        if plot:

            # Compute probabilities
            probabilities = np.abs(self.quantum_states[-1].flatten())**2

            # X-axis indices for each basis state
            x = np.arange(len(probabilities))
            xticks = []
            for i in x:
                xticks.append(state_as_string(i,self.N))
                
            # Create bar plot
            plt.figure(figsize=(8,3), dpi=300)
            plt.bar(x, probabilities)

            # Labels and title
            plt.xlabel("Basis State Index")
            plt.xticks(x, xticks)
            plt.ylabel("Probability")
            plt.title("Quantum State Probability Distribution")

            # Show plot
            plt.show()

    def get_state_as_string(self):
        return self.basis.get_state_as_string()
    
"""Get quantum state as a string"""
def state_as_string(i,N):  
    binary_string = bin(i)
    string = binary_string[2:]
    string = string.zfill(N)
    return "|" + string + ">"


import time
total_time = 0
total_time_history = []
qp = qprogram([0, 0, 0])

for i in range(3):
    qp.h(i)

for y in range(300):
    if y % 25 == 0:
        print(y)
    start_time = time.time()  # Record the start time
    for x in range(2):
        # Oracle
        qp.h(2)
        qp.t(0, 1, 2)
        qp.h(2)
        
        for i in range(3):
            qp.h(i)

        for i in range(4):
            qp.x(i)

        qp.h(2)
        qp.t(0, 1, 2)
        qp.h(2)

        for i in range(4):
            qp.x(i)

        for i in range(3):
            qp.h(i)

        qp.run(show_states=False, plot=False)
    end_time = time.time()  # Record the start time
    total_time += end_time - start_time
    total_time_history.append(total_time)


import csv

# Save to CSV
with open("data_gate_300.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(total_time_history)  # Write the list as a single row