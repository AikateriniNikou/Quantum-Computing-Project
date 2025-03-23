import numpy as np
import sparse as sp

# Multiply matrices
def matrixProduct(matrix_A, matrix_B):

    if isinstance(matrix_A, np.ndarray) & isinstance(matrix_B, np.ndarray): # Dense Method
        if matrix_A.shape[1] != matrix_B.shape[0]:
            print("Matrix shapes are not correct")
        else:
            new_matrix = np.zeros((matrix_A.shape[0], matrix_B.shape[1]))
            for i in range(new_matrix.shape[0]):
                for j in range(new_matrix.shape[1]):
                    for n in range(matrix_A.shape[1]):
                        new_matrix[i][j] += matrix_A[i][n] * matrix_B[n][j]
            return new_matrix
        
    if isinstance(matrix_A, sp.SparseMatrix) & isinstance(matrix_B, sp.SparseMatrix): # Sparse Method
        new_matrix = {}
        for a in matrix_A.matrixDict:
            for b in matrix_B.matrixDict:
                if a[0] == b[1]:
                    if (b[0], a[1]) in new_matrix:
                        new_matrix[(b[0], a[1])] += matrix_A.matrixDict[a] * matrix_B.matrixDict[b]
                    else:
                        new_matrix[(b[0], a[1])] = matrix_A.matrixDict[a] * matrix_B.matrixDict[b]
        return sp.SparseMatrix(new_matrix, (matrix_A.size[0], matrix_B.size[1]))


# Tensor product
def tensorProduct(matrix_A, matrix_B):
    
    if isinstance(matrix_A, np.ndarray) & isinstance(matrix_B, np.ndarray): # Dense Method
        new_matrix = np.zeros((matrix_A.shape[0] * matrix_B.shape[0], matrix_A.shape[1] * matrix_B.shape[1]))
        for i in range(new_matrix.shape[0]):
            for j in range(new_matrix.shape[1]):
                new_matrix[i][j] = matrix_A[i//matrix_B.shape[0]][j//matrix_B.shape[1]] * matrix_B[i%matrix_B.shape[0]][j%matrix_B.shape[1]]
        return new_matrix
    
    if isinstance(matrix_A, sp.SparseMatrix) & isinstance(matrix_B, sp.SparseMatrix): # Sparse Method
        new_matrix = {}
        for a in matrix_A.matrixDict:
            for b in matrix_B.matrixDict:
                new_matrix[(b[0] + a[0] * matrix_B.size[0], b[1] + a[1] * matrix_B.size[1])] \
                    = matrix_A.matrixDict[a] * matrix_B.matrixDict[b]
        return sp.SparseMatrix(new_matrix, (matrix_A.size[0] * matrix_B.size[0], matrix_A.size[1] * matrix_B.size[1]))


# Matrix * Vector product
def vecMatProduct(matrix, vector):
    
    if isinstance(matrix, np.ndarray): # Dense Method
        vecR = np.resize(vector, (len(vector), 1))
        return matrixProduct(matrix, vecR)[:,0]
    
    if isinstance(matrix, sp.SparseMatrix): # Sparse Method
        V = [0] * len(vector)
        for value in matrix.matrixDict:
            V[value[0]] += matrix.matrixDict[value] * vector[value[1]]
        return np.array(V)