import numpy as np
import matplotlib.pyplot as plt
import gates as gate
import quantum_register as q
import time

def Grovers(N, target, runs, verbose, SparseMatrix=False):
    # Runs Grover’s search.
    # Builds the quantum circuit step by step
    # Adjusts gates to match the desired solution and the number of qubits used.
    
    start_time = time.time()  # Record the start time
    
    if runs == 0:
        it = int(np.pi / (4 * np.arcsin(1 / np.sqrt(2 ** N))))  # Optimal Grover iterations
    
    if runs != 0:
        it = runs  # Custom Grover iterations
    
    if verbose:
        print("\n--------------")
        print("Creating Gates")
        print("--------------")
    H = gate.Hadamard(N, SparseMatrix) # Create Hadamard Gate
    O = gate.Oracle(N, target, SparseMatrix) # Create Oracle
    D = gate.Diffuser(N, SparseMatrix) # Create Diffuser
    
    if verbose:
        print("\n-------------------------")
        print("Creating Quantum Register")
        print("-------------------------")
    Reg = q.QuantumRegister(q.State((0, N))) # Generate quantum register of N qubits in state |0>
    
    Reg.applyGate(H, SparseMatrix) # Apply Hadamard gates to register
    
    if verbose:
        print("\n********************************")
        print(f"Grover's Algorithm Iterations: {it}")
        print("********************************")
    for _ in range(it):
        print("")
        Reg.applyGate(O, SparseMatrix) # Apply Oracle to register
        Reg.applyGate(H, SparseMatrix) # Apply Hadamard gates to register
        Reg.applyGate(D, SparseMatrix) # Apply Diffuser to register
        Reg.applyGate(H, SparseMatrix) # Apply Hadamard gates to register
    
    end_time = time.time()  # Record the end time
    
    return Reg, end_time - start_time

def measurementPlot(freq, States):
    # Draws a bar chart showing how often each outcome showed up in the results.

    xaxis = list(range(len(States)))
    
    plt.figure(figsize=(10,5))
    plt.bar(xaxis, freq, tick_label=States)
    plt.ylabel("Frequency")
    plt.xlabel("Basis States")
    plt.xticks(rotation=90)
    plt.title("Measurement Frequency for Basis States")
    for i, f in enumerate(freq): # Label probability of states
        plt.annotate(f, xy=(i, f), ha='center', va='bottom')
    plt.savefig("Results/Measurement_Frequency_Plot.png", dpi=500, bbox_inches='tight')
    plt.show()

def measureSystem(R, measurements, N):
    # Generates repeated measurement results from a quantum state without running the circuit again.
    
    Obs = [R.measure() for _ in range(measurements)]
    States = [f"|{bin(i)[2:].zfill(N)}>" for i in range(2 ** N)]
    frequency = [Obs.count(s) / measurements for s in States]
    
    print(f"Measurement Percentage After {measurements} Measurements:\n")
    for state, prob in zip(States, frequency):
        prob = prob * 100
        print(f"{state}: {prob}%")
    
    measurementPlot(frequency, States)
    
    return max(frequency)