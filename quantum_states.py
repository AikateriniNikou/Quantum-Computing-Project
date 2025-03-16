""" Interface for representing basis states in quantum computing.
"""

from abc import ABC, abstractmethod

import numpy as np
from errors import InputError


class QubitState(ABC):
    """ Abstract base class for quantum states, acting as an interface.
    
    All methods are abstract and should be implemented in derived classes.

    Parameters
    ----------
    values : tuple
        A tuple containing (state, number of qubits used for representation).
    ket : bool, optional (default=True)
        Specifies whether to interpret the state in bra or ket notation,
        which determines its vector and string representation.

    Attributes
    ----------
    vals : tuple
        A tuple containing (state, number of qubits).
    strRep : str
        A string representation of the quantum state in superposition form.
    ket : bool
        Determines whether the state is represented as a ket or bra.

    Abstract Methods
    ----------------
    flip(self)
        Implemented in subclasses to switch between ket and bra notation.
    __str__(self)
        Implemented in subclasses to return the appropriate string representation.
    """

    @abstractmethod
    def __init__(self, values, ket=True):
        """ Should be overridden to handle binary or decimal representation. """
        self.vals = values
        self.ket = ket
        self.strRep = ""

    @abstractmethod
    def flip(self):
        """ Switch between ket and bra representation. """
        pass
    
    @abstractmethod
    def __str__(self):
        """ Return an appropriate string representation of the state. """
        pass


class State(QubitState):
    """ Represents a multi-qubit ket (basis state) in the computational (Z) basis.
    
    Note: The binary convention follows standard ordering, where the least significant
    qubit is on the right. For example, |00001> has a 1 in the least significant position.

    Parameters
    ----------
    values : tuple
        A tuple containing (state, number of qubits used for representation).
    ket : bool, optional (default=True)
        Specifies whether to interpret the state in bra or ket notation,
        which determines its vector and string representation.

    Attributes
    ----------
    values : tuple
        A tuple containing (state, number of qubits).
        - The first element represents the state in decimal notation.
        - The second element specifies the number of qubits.
        Example: (1, 4) corresponds to |0001>.
    strRep : str
        String representation of the quantum state.
    ket : bool
        Indicates whether the state is represented as a ket or bra.
    d : int
        Total number of basis states corresponding to the given number of qubits.
    den : int
        Decimal representation of the state, e.g., |011> corresponds to den = 3.
    vec : numpy array
        State vector representation as an array, e.g., |01> corresponds to [0,1,0,0].
    bin : str
        Binary representation of the state, prefixed with '0b', e.g., |3> corresponds to "0b11".
    strRep : str
        String representation of the state in Dirac notation (e.g., |state>).

    Methods
    -------
    flip(self)
        Switch between ket and bra representation.
    dotWith(self, ket)
        Compute the dot product between the state's vector representation and another object.
    __str__(self)
        Return the string representation of the state in Dirac notation.
    """

    def __init__(self, values, ket=True):
        super().__init__(values, ket)
        self.values = values
        self.d = 2**values[1]    # Total number of basis states
        self.den = values[0]     # Decimal representation of the state

        if self.den > self.d - 1:
            raise InputError("State cannot be represented with the given number of qubits.")

        # Construct vector representation (e.g., |0> = [1,0], |3> = [0,0,0,1])
        vr = np.zeros(self.d)
        vr[self.den] = 1
        self.vec = np.array(vr)

        bi = bin(self.den)
        self.bin = np.array([int(n) for n in bi[2:].zfill(values[1])])

        for val in self.bin:
            self.strRep += str(val)

    def flip(self):
        """ Switch between ket and bra representation. """
        self.ket = not self.ket

    def __str__(self):
        """ Return the Dirac notation representation of the state. """
        return f"|{self.strRep}>" if self.ket else f"<{self.strRep}|"
