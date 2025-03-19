""" This module provides a custom Sparse class for handling sparse matrices efficiently.

The Sparse class represents sparse matrices using a dictionary-based format, allowing efficient storage and retrieval of non-zero elements while preserving matrix operations.
"""

import numpy as np

class SparseMatrix():
    """ A class for efficient representation and manipulation of sparse matrices.
    
    This class converts standard matrices into a sparse dictionary format, reducing memory usage by storing only non-zero elements.
    
    Parameters
    ----------
    m : numpy array or dictionary
        A matrix (including zeros) to be converted into a sparse representation.
    s : tuple, optional
        Defines the size of the matrix (width, height). Default is None.
    
    Attributes
    ----------
    matrixDict : dict
        A dictionary storing the non-zero elements of the matrix, where keys are coordinate tuples and values are the corresponding matrix entries.
    len : int
        The number of non-zero elements in the matrix.
    size : tuple
        The dimensions of the original matrix (width, height), including zero elements.
    
    Methods
    -------
    asMatrix(self)
        Reconstructs and returns the full matrix as a numpy array, including zero elements.
    __str__(self)
        Returns a string representation of the reconstructed matrix.
    """

    def __init__(self, m, s=None):
        if isinstance(m, np.ndarray):
            self.matrixDict = {}
            for x in range(len(m)):
                for y in range(len(m[0])):
                    if m[x][y] != 0:
                        self.matrixDict[(x, y)] = m[x][y]
        elif isinstance(m, dict):
            self.matrixDict = m
        else:
            raise TypeError("Sparse matrix must be initialized with a numpy array or dictionary.")
            
        self.len = len(self.matrixDict)
        
        if s is not None:
            self.size = s
        elif isinstance(m, np.ndarray):
            self.size = (len(m), len(m[0]))
        else:
            self.size = (0, 0)
            for pos in self.matrixDict:
                self.size = (max(self.size[0], pos[0] + 1), max(self.size[1], pos[1] + 1))
                
        # Define dimension (assumes a square matrix)
        self.dimension = self.size[0]
        
    def __getitem__(self, key):
        # Return the stored value or 0 if the key is not present.
        return self.matrixDict.get(key, 0)
    
    def multiply_vector(self, vector):
        if self.dimension != vector.dimension:
            raise ValueError("Matrix and vector dimensions don't match")
        result_values = [0] * self.dimension
        # Multiply only over nonzero elements.
        for (i, j), value in self.matrixDict.items():
            result_values[i] += value * vector.values[j]
        # Return a new Vector with the computed result.
        from aux_functions import Vector  # Import here if needed
        return Vector(result_values)