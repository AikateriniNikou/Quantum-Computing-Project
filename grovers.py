""" This module provides functions for constructing and applying quantum gates, running Grover's algorithm, and simulating quantum measurements.

It enables dynamic gate creation, quantum state evolution, and probability-based measurement simulations.
"""

import operations as op
import register as re
import quantum_states as qs
import numpy as np
import matplotlib.pyplot as plt
import time

def Oracle(nq, s, Sparse=False):
    """ Constructs an oracle gate dynamically. """
    Tr = bin(s)[2:].zfill(nq)  # Convert target state to binary
    Neg = "".join("X" if i == '0' else "I" for i in Tr)
    L = op.constructGate(Neg, Sparse)
    Z = op.constructGate("Z" * nq, Sparse)
    return op.matrixProduct(op.matrixProduct(L, Z), L)


def Hadamard(nq, Sparse=False):
    """ Constructs an `nq`-qubit Hadamard gate.
    
    Parameters
    ----------
    nq : int
        Number of qubits in the register.
    
    Returns
    -------
    numpy array or sp.Sparse
        The `nq`-dimensional Hadamard gate matrix.
    """
    #return op.constructGate('H' * nq, Sparse)
    return op.constructGate('H' * nq, Sparse)


def Diffuser(nq, Sparse=False):
    """ Constructs the Grover diffusion operator.
    
    Parameters
    ----------
    nq : int
        Number of qubits in the register.
    
    Returns
    -------
    numpy array or sp.Sparse
        The `nq`-dimensional diffusion operator matrix.
    """
    L = op.constructGate("X" * nq, Sparse)
    #Z = op.constructGate(f"{nq}Z", Sparse)
    Z = op.constructGate("Z" * nq, Sparse)  # Correct way to create an nq-qubit Z gate

    return op.matrixProduct(op.matrixProduct(L, Z), L)

def Grovers(nq, s, cOut, Sparse=False):
    """ Implements Grover's search algorithm.
    
    The function dynamically adapts quantum gates based on the target state and the number of qubits.
    
    Parameters
    ----------
    nq : int
        Number of qubits in the register.
    s : int
        Decimal representation of the target state.
    cOut : bool
        If True, prints progress messages.
    
    Returns
    -------
    register.Register, float
        The quantum register after applying Grover's algorithm and the execution time.
    """
    if cOut:
        print("\n-------Constructing Gates------:")
    H = Hadamard(nq, Sparse)
    Orac = Oracle(nq, s, Sparse)
    Diff = Diffuser(nq, Sparse)
    
    R = re.Register(qs.State((0, nq)))
    start_time = time.time()
    R.applyGate(H, Sparse)
    
    it = int(np.pi / (4 * np.arcsin(1 / np.sqrt(2 ** nq))))  # Optimal iterations
    if cOut:
        print(f"\nRunning Grover's algorithm {it} times:")
    for _ in range(it):
        R.applyGate(Orac, Sparse)
        R.applyGate(H, Sparse)
        R.applyGate(Diff, Sparse)
        R.applyGate(H, Sparse)
    
    return R, time.time() - start_time

def FrequencyPlot(freq, States):
    """ Plots the observed frequency of basis states.
    
    Parameters
    ----------
    freq : list of int
        Number of occurrences of each state.
    States : list of str
        Basis states in Dirac notation.
    """
    xaxis = list(range(len(States)))
    plt.bar(xaxis, freq, tick_label=States)
    plt.ylabel("Frequency")
    plt.xlabel("Basis States")
    plt.xticks(rotation=90)
    plt.title("Measurement Frequency for Basis States")
    for i, f in enumerate(freq):
        plt.annotate(f, xy=(i, f), ha='center', va='bottom')
    plt.savefig("Measurement_Frequency_Plot.png")
    plt.show()

def Observe_System(R, k, nq):
    """ Simulates multiple observations of a quantum register.
    
    Instead of rerunning Grover's algorithm each time, this function simulates final measurements using a Monte Carlo approach.
    
    Parameters
    ----------
    R : register.Register
        The quantum register to observe.
    k : int
        Number of observations.
    nq : int
        Number of qubits in the register.
    
    Returns
    -------
    float
        The highest observed probability of any state.
    """
    Obs = [R.measure() for _ in range(k)]
    States = [f"|{bin(i)[2:].zfill(nq)}>" for i in range(2 ** nq)]
    freq = [Obs.count(s) / k for s in States]
    
    print(f"\nObservation results after {k} measurements:")
    for state, probability in zip(States, freq):
        print(f"{state}: {probability}")
    
    if nq <= 5:
        FrequencyPlot(freq, States)
    
    return max(freq)
