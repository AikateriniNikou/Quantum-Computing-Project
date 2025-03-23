import numpy as np
import matrix_functions as mf
from random import random as rnd

# Describes a multi-qubit basis state in binary form.  
class State():
    
    def __init__(self, values, ket=True):

        self.values = values
        self.d = 2**values[1]    
        self.den = values[0]
        self.vals = values
        self.ket = ket
        self.strRep = ""

        # Create basis state vector
        vr = np.zeros(self.d)
        vr[self.den] = 1
        self.vec = np.array(vr)

        bi = bin(self.den)
        self.bin = np.array([int(n) for n in bi[2:].zfill(values[1])])

        for val in self.bin:
            self.strRep += str(val)

    def __str__(self):
        return f"|{self.strRep}>" if self.ket else f"<{self.strRep}|"
    
# Defines QuantumRegister.  
# Starts from a basic state and lets you apply gates or measure results.  
class QuantumRegister():

    def __init__(self, input, SparseMatrix=False):

        self.qR = input  # Stores the quantum register in binary form.
        self.stateVector = self.qR.vec  # Creates the state vector
        self.qbitVector = np.array([State((i, self.qR.values[1])) for i in range(self.qR.d)])

    def applyGate(self, gate, SparseMatrix=False): # Apply quantum gates
        self.stateVector = mf.vecMatProduct(gate, self.stateVector)

    def measure(self): # Measure the system
        r = rnd()  # Generate a random number between 0 and 1.
        cumulative_prob = 0
        for i in range(self.qR.d):
            amp = self.stateVector[i]
            cumulative_prob += amp.real**2 + amp.imag**2  
            if r <= cumulative_prob:  
                return f"{self.qbitVector[i]}"