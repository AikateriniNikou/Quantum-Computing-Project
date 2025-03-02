import numpy as np

def tensor_product(A,B):
    A_x, A_y = A.shape  # access the shape of the first matrix A
    B_x, B_y = B.shape  # access the shape of the second matrix B

    """
    print(A.shape)
    print(B.shape)
    """

    X_size = A_x*B_x
    Y_size = A_y*B_y

    result = np.zeros((X_size, Y_size))
    
    """
    print(result.shape)
    """

    """ Loop through both the matrices and calculate the element at each result matrix position"""
    for i in range(0,A_x):
        for j in range(0,A_y):
            for k in range(0,B_x):
                for l in range(0,B_y):
                    result[i*B_x + k][j*B_y + l] = A[i][j]*B[k][l]

    return result

"""
A = np.array([[1, 2, 3, 4]])

B = np.array([[9,2,4]])

result = tensor_product(A, B)

print(result)

correct_result = np.kron(A,B)
print(correct_result)
"""

