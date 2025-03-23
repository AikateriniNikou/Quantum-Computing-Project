import numpy as np

# Class to handle sparse matrices by keeping only non-zero values in a dictionary.
class SparseMatrix():
    
    def __init__(self, m, s=None):
        
        if isinstance(m, np.ndarray): # Dense method
            self.matrixDict = {}
            for x in range(len(m)):
                for y in range(len(m[0])):
                    if m[x][y] != 0:
                        self.matrixDict[(x, y)] = m[x][y]

        elif isinstance(m, dict): # Sparse method
            self.matrixDict = m

        self.len = len(self.matrixDict) # Number of values
        
        
        if s is not None: # Deals with the Oracle
            self.size = s
        elif isinstance(m, np.ndarray):
            self.size = (len(m), len(m[0]))
        else:
            self.size = (0, 0)
            for pos in m:
                self.size = (max(self.size[0], pos[0] + 1), max(self.size[1], pos[1] + 1))
                
        self.dimension = self.size[0]
                
    def multiply_vector(self, vector):
        result_values = [0] * self.dimension
        # Multiply only over nonzero elements.
        for (i, j), value in self.matrixDict.items():
            result_values[i] += value * vector.values[j]
        # Return a new Vector with the computed result.
        from aux_functions import Vector  # Import here if needed
        return Vector(result_values)