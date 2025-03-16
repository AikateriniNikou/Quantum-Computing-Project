import numpy as np
import matplotlib.pyplot as plt
import time
import gates as gate
import quantum_register as q


def Grovers(nq, s, runs, verbose, SparseMatrix=False):
    """ Implements Grover's search algorithm.
    
    The function dynamically adapts quantum gates based on the target state and the number of qubits."""
    
    if verbose:
        print("\n--------------")
        print("Creating Gates")
        print("--------------")
    H = gate.Hadamard(nq, SparseMatrix)
    Orac = gate.Oracle(nq, s, SparseMatrix)
    Diff = gate.Diffuser(nq, SparseMatrix)
    
    R = q.QuantumRegister(q.State((0, nq)))
    start_time = time.time()
    R.applyGate(H, SparseMatrix)
    
    if runs == 0:
        it = int(np.pi / (4 * np.arcsin(1 / np.sqrt(2 ** nq))))  # Optimal iterations
    
    if runs != 0:
        it = runs  # Custom iterations
        
    if verbose:
        print("\n*******************************")
        print(f"{it} Grover's Algorithm Iterations")
        print("*******************************")
    for _ in range(it):
        R.applyGate(Orac, SparseMatrix)
        R.applyGate(H, SparseMatrix)
        R.applyGate(Diff, SparseMatrix)
        R.applyGate(H, SparseMatrix)
    
    return R, time.time() - start_time

def FrequencyPlot(freq, States):
    """ Plots the observed frequency of basis states.

    """
    xaxis = list(range(len(States)))
    plt.bar(xaxis, freq, tick_label=States)
    plt.ylabel("Frequency")
    plt.xlabel("Basis States")
    plt.xticks(rotation=90)
    plt.title("Measurement Frequency for Basis States")
    for i, f in enumerate(freq):
        plt.annotate(f, xy=(i, f), ha='center', va='bottom')
    plt.savefig("Measurement_Frequency_Plot.png", dpi=300, bbox_inches='tight')
    plt.show()

def Observe_System(R, k, nq):
    """ Simulates multiple observations of a quantum register.
    
    Instead of rerunning Grover's algorithm each time, this function simulates final measurements using a Monte Carlo approach.

    """
    Obs = [R.measure() for _ in range(k)]
    States = [f"|{bin(i)[2:].zfill(nq)}>" for i in range(2 ** nq)]
    freq = [Obs.count(s) / k for s in States]
    
    print(f"\nMeasurement Percentage After {k} Measurements:\n")
    for state, probability in zip(States, freq):
        probability = probability * 100
        print(f"{state}: {probability}%")
    
    if nq <= 4:
        FrequencyPlot(freq, States)
    
    return max(freq)