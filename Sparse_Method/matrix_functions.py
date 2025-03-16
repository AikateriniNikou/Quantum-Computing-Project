import numpy as np
import sparse as sp

# Multiply matrices
def matrixProduct(matA,matB):

    if isinstance(matA, np.ndarray) & isinstance(matB, np.ndarray):
        if matA.shape[1] != matB.shape[0]:
            print(f"Non axN Nxb matching matrices : {matA.shape[0]}x{matA.shape[1]} and {matB.shape[0]}x{matB.shape[1]}")
        else:
            matZ = np.zeros((matA.shape[0],matB.shape[1]))
            for i in range(matZ.shape[0]):
                for j in range(matZ.shape[1]):
                    for n in range(matA.shape[1]):
                        matZ[i][j] += matA[i][n]*matB[n][j]
            return matZ
        
    if isinstance(matA, sp.SparseMatrix) & isinstance(matB, sp.SparseMatrix):
        matZ = {}
        for a in matA.matrixDict:
            for b in matB.matrixDict:
                if a[0] == b[1]:
                    if (b[0],a[1]) in matZ:
                        matZ[(b[0],a[1])] += matA.matrixDict[a]*matB.matrixDict[b]
                    else:
                        matZ[(b[0],a[1])] = matA.matrixDict[a]*matB.matrixDict[b]
        return sp.SparseMatrix(matZ, (matA.size[0],matB.size[1]))


# Tensor product
def tensorProduct(vecA,vecB):

    lA = len(vecA)
    lB = len(vecB)
    T = np.zeros(lA*lB)
    for i in range (lA):
        for j in range (lB):
            T[i*lB+j] = vecA[i]*vecB[j]
    return T


# Kronecker product
def kroneckerProduct(matA,matB):
    
    if isinstance(matA, np.ndarray) & isinstance(matB, np.ndarray):
        matZ = np.zeros((matA.shape[0]*matB.shape[0], matA.shape[1]*matB.shape[1]))
        for i in range(matZ.shape[0]):
            for j in range(matZ.shape[1]):
                matZ[i][j] = matA[i//matB.shape[0]][j//matB.shape[1]]*matB[i%matB.shape[0]][j%matB.shape[1]]
        return matZ
    
    if isinstance(matA, sp.SparseMatrix) & isinstance(matB, sp.SparseMatrix):
        matZ = {}
        for a in matA.matrixDict:
            for b in matB.matrixDict:
                matZ[( b[0]+a[0]*matB.size[0] , b[1]+a[1]*matB.size[1] )] = matA.matrixDict[a]*matB.matrixDict[b]
        return sp.SparseMatrix(matZ, (matA.size[0]*matB.size[0],matA.size[1]*matB.size[1]))


def vecMatProduct(mat,vec):
    """ Takes a matrix and a single array vector and formats them for the matrixProduct() function."""
    
    if isinstance(mat, np.ndarray):
        vecR = np.resize(vec,(len(vec),1))
        return matrixProduct(mat,vecR)[:,0]
    
    if isinstance(mat, sp.SparseMatrix):
        V = [0]*len(vec)
        for pos in mat.matrixDict:
            V[pos[0]] += mat.matrixDict[pos]*vec[pos[1]]
        return np.array(V)