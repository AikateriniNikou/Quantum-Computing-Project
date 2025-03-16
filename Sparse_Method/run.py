import grovers_algorithm as gr
import matplotlib.pyplot as plt
import numpy as np
import time

k = 100 # Number of measurements
runs = np.arange(1, 8) # Number of iterations of Grover's algorithm
s = 0 # Target state
nq = 2 # Number of qubits
use_sparse = True
verbose = True

suc_rates = []
total_time = 0
for a_run in runs:
    
    start_time = time.time()  # Record the start time

    # Run Grover's algorithm for the given parameters
    R, Dt = gr.Grovers(nq, s, a_run, verbose, use_sparse)
    
    #Simulate the measurement of the System, AS IF you run Grover's each time
    success_rate = gr.Observe_System(R, k, nq)
    success_rate_percent = success_rate * 100
    
    end_time = time.time()  # Record the end time

    time_taken = end_time - start_time  # Calculate elapsed time
    total_time += time_taken
    print(f"\nChance of Success - {success_rate_percent}%")
    print(f"Time Taken = {time_taken} Seconds")
    suc_rates.append(success_rate)
    
print(f"\nTotal Time = {total_time} Seconds")
plt.plot(runs, suc_rates)
plt.xlabel("Number of algorithm iterations")
plt.ylabel("Success Rate")
plt.title("Success Rate as a function of Grover iterations")
plt.savefig("Success_rate.png", dpi=300, bbox_inches='tight')
plt.show()
