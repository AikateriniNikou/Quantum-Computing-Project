import grovers_algorithm as gr

""" Grover's Algorithm Inputs"""
N = 3 # Number of qubits in system

target = 0 # Target state (0 for first state, 1 for second state, etc.)

runs = 0 # Number of Grover algorithm iterations (Leave at 0 for optimal amount)

measurements = 100000 # Number of measurements of the final basis state

use_sparse = True # Use sparse (True) or dense (False) matrix method

verbose = False # Show algorithm progress

save_graphs = False # Save graphs to disk


""" Run Grover's Algorithm"""
R, dt = gr.Grovers(N, target, runs, verbose, use_sparse)
    
# Measure the system
gr.measureSystem(R, measurements, N) 

print(f"\nUses SparseMatrix: {use_sparse}")
print(f"Time Taken = {dt} Seconds")
