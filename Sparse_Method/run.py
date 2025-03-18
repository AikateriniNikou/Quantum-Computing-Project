import grovers_algorithm as gr
import matplotlib.pyplot as plt
import numpy as np
import time
import csv

# with open("data_gate_300.csv", "r") as file:
#     reader = csv.reader(file)
#     data = list(reader)[0]  # Read first row
#     data = [float(i) for i in data]  # Convert strings to integers
    
# with open("data_bitwise_300.csv", "r") as file:
#     reader = csv.reader(file)
#     data_bitwise = list(reader)[0]  # Read first row
#     data_bitwise = [float(i) for i in data_bitwise]  # Convert strings to integers
    
measurements = 10000 # Number of measurements of the algorithm
#runs = np.arange(1, 100) # Number of iterations of Grover's algorithm
runs = [0]
s = 12 # Target state
N = 5 # Number of qubits
#use_sparse = [True, False]
use_sparse = [True]
verbose = True
save_graphs = False

suc_rates = []
total_time = 0
total_time_sparse = [[],[]]

for i, use_sparse_true_false in enumerate(use_sparse):
    total_time = 0
    for a_run in runs:
    
        start_time = time.time()  # Record the start time

        # Run Grover's algorithm for the given parameters
        R, Dt = gr.Grovers(N, s, a_run, verbose, use_sparse_true_false)
    
        #Simulate the measurement of the System, AS IF you run Grover's each time
        success_rate = gr.Observe_System(R, measurements, N, verbose)
        success_rate_percent = success_rate * 100
    
        end_time = time.time()  # Record the end time

        time_taken = end_time - start_time  # Calculate elapsed time
        total_time += time_taken
        print(f"\nChance of Success - {success_rate_percent}%")
        print(f"Time Taken = {time_taken} Seconds")
        print(f"Uses SparseMatrix: {use_sparse_true_false}")
        suc_rates.append(success_rate)
        total_time_sparse[i].append(total_time)
    
# print(f"\nTotal Time = {total_time} Seconds")
# plt.figure(figsize=(10,5))
# plt.plot(suc_rates)
# plt.xlabel("Number of algorithm iterations")
# plt.ylabel("Success Rate")
# plt.title("Success Rate as a function of Grover iterations")
# if save_graphs:
#     plt.savefig("Results/Success_rate.png", dpi=500, bbox_inches='tight')
# plt.show()

# plt.figure(figsize=(10,5))
# plt.plot(total_time_sparse[0], label="Sparse Matrix Method")
# plt.plot(total_time_sparse[1], label="Dense Matrix Method")
# plt.plot(data, label="Gate Method")
# plt.plot(data_bitwise, label="Bitwise Method")
# plt.xlabel("Algorithm iterations")
# plt.ylabel("Time Taken (Seconds)")
# plt.title("Algorithm Time Comparisons")
# plt.legend()
# if save_graphs:
#     plt.savefig("Results/Algorithm_Comparison.png", dpi=500, bbox_inches='tight')
# plt.show()