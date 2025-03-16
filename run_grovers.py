

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Ensures Python finds local modules
import grovers  # Import again


import grovers as gr  # Import the module that implements Grover's algorithm

k = 10000  # Number of times the system will be measured to estimate success probability

# Prompt user for target state and number of qubits
s = int(input('\n' + "Target state: "))  # Target state in decimal representation
nq = int(input("number of qubits: "))  # Number of qubits in the quantum register

# Run Grover's algorithm to search for the target state
R, Dt = gr.Grovers(nq, s, True, True)  # Execute Grover's algorithm and store the register and execution time

# Simulate multiple measurements of the system to estimate the probability of measuring the correct state
success_rate = gr.Observe_System(R, k, nq)

# Print the estimated probability of successfully measuring the target state
print(f"success rate = {success_rate}")



