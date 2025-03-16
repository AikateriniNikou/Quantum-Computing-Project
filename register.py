""" This module defines the Register class, which represents a quantum system at any given time.

The Register class maintains a quantum state initialized as a non-superposed basis state and provides methods for applying quantum gates and performing measurements.
"""

import numpy as np
import quantum_states as qs
import math as m
import operations as op
from random import random as rnd
import sparse as sp

class Register():
    """ Represents a quantum register (system) at a given time.
    
    The register is initialized with a basis state and stores the state as a vector of amplitudes.
    
    Parameters
    ----------
    input : quantum_states.State
        A quantum state initialized elsewhere.
    
    Attributes
    ----------
    qR : quantum_states.State
        Stores the quantum state of the register.
    stateVector : numpy array
        Amplitudes of the basis states.
    qbitVector : numpy array
        Basis states representing individual qubits.
    
    Methods
    -------
    applyGate(self, gate)
        Applies a quantum gate to the state vector.
    measure(self)
        Returns a randomly selected basis state based on probability amplitudes.
    measure_collapse(self)
        Performs a measurement that collapses the state vector.
    __str__(self)
        Returns a string representation of the quantum register.
    
    Examples
    --------
    If |ψ> = -0.2|0> + 0.9|1>, then:
    - self.stateVector = [-0.2, 0.9]
    - self.qbitVector = [Register(0,1), Register(1,1)] (list of basis states [|0>, |1>])
    - __str__ prints: '-0.2|0> + 0.9|1>'
    """

    def __init__(self, input, Sparse=False):
        """ Initializes the quantum register with a given basis state.
        
        Parameters
        ----------
        input : quantum_states.State
            A quantum basis state initialized elsewhere.
        """
        
        self.qR = input  # Stores the quantum register.
        self.stateVector = self.qR.vec  # Initializes the state vector.
        self.qbitVector = np.array([qs.State((i, self.qR.values[1])) for i in range(self.qR.d)])

    def applyGate(self, gate, Sparse=False):
        """ Applies a quantum gate to the register.
        
        Uses matrix-vector multiplication to update the state vector.
        
        Parameters
        ----------
        gate : numpy array
            The matrix representing the quantum gate.
        """
        self.stateVector = op.vecMatProduct(gate, self.stateVector)

    def measure(self):
        """ Measures the quantum register, selecting a basis state probabilistically.
        
        Uses a Monte Carlo approach to determine the measured state without collapsing the register.
        
        Returns
        -------
        str
            String representation of the measured basis state.
        """
        
        r = rnd()  # Generate a random number between 0 and 1.
        cumulative_prob = 0
        for i in range(self.qR.d):
            amp = self.stateVector[i]
            cumulative_prob += amp.real**2 + amp.imag**2  # Compute probability of occurrence.
            if r <= cumulative_prob:  # Select state based on probability interval.
                return f"{self.qbitVector[i]}"

    def measure_collapse(self):
        """ Measures the quantum register and collapses the state.
        
        The system collapses to a specific basis state based on the probability distribution.
        
        Returns
        -------
        str
            String representation of the collapsed basis state.
        """
        
        r = rnd()  # Generate a random number between 0 and 1.
        cumulative_prob = 0
        for i in range(self.qR.d):
            amp = self.stateVector[i]
            cumulative_prob += amp.real**2 + amp.imag**2  # Compute probability of occurrence.
            if r <= cumulative_prob:  # Collapse to the chosen state.
                self.stateVector = np.zeros(self.qR.d)
                self.stateVector[i] = 1
                return f"{self.qbitVector[i]}"

    def __str__(self):
        """ Returns a string representation of the quantum register.
        
        Constructs a formatted output representing the superposition of states.
        
        Returns
        -------
        str
            The register state as a superposition of basis states.
        """
        
        output = ""
        for i in range(self.qR.d):
            sign = " +" if self.stateVector[i] >= 0 else " "
            output += f"{sign}{round(self.stateVector[i], 5)}{self.qbitVector[i]}"
        return output
