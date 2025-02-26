""" Functions for matrices
Last updated: 26/02/25"""

class Vector:
    def __init__(self, values):
        self.values = list(values)
        self.dimension = len(values)

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def __repr__(self):
        s = "["
        for r in self.values:
            s += str(r) + " "
        s = s + "]"
        return s

# SquareMatrix Class
class SquareMatrix:
    
    # Creates a square matrix with all elements set to 0
    def __init__(self, dimension):
        self.dimension = dimension
        self.data = [[0 for i in range(dimension)] for i in range(dimension)]
    
    # Gets the value of an element from row (r) and column (c)
    def __getitem__(self, key):
        r, c = key
        return self.data[r][c]
    
    # Sets the value of an element from row (r) and column (c)
    def __setitem__(self, key, value):
        r, c = key
        self.data[r][c] = value
    
    # Output when printed
    def __repr__(self):
        s = ""
        for r in self.data:
            s += str(r) + "\n"
        return s
    
    # Tensor product of two square matrices
    def tensor_product(self, other):

        dim_A, dim_B = self.dimension, other.dimension
        dim_result = dim_A * dim_B  

        result = SquareMatrix(dim_result)
        
        for i in range(dim_A):
            for j in range(dim_A):
                for k in range(dim_B):
                    for l in range(dim_B):
                        result[i * dim_B + k, j * dim_B + l] = self[i, j] * other[k, l]

        return result
    
    # Matrix multiply two square matrices
    def matrix_multiply(self, other):
  
        if self.dimension != other.dimension:
            raise ValueError("Matrix dimensions don't match")

        dim = self.dimension
        result = SquareMatrix(dim)

        for i in range(dim):
            for j in range(dim):
                result[i, j] = sum(self[i, k] * other[k, j] for k in range(dim))

        return result

    def multiply_vector(self, vector):
        """Multiplies the square matrix with a vector."""
        if self.dimension != vector.dimension:
            raise ValueError("Matrix and vector dimensions must match!")

        result_values = [sum(self[i, j] * vector[j] for j in range(self.dimension)) for i in range(self.dimension)]
        return Vector(result_values)


# Calculate tensor product
# def tensor_product(A, B):
    
#     # Calculate matrix sizes from rows (r) and columns (c)
#     r_A, c_A = len(A), len(A[0])
#     r_B, c_B = len(B), len(B[0])

#     # Create matrix
#     result = [[0] * (c_A * c_B) for i in range(r_A * r_B)]

#     # Calculate tensor product
#     for i in range(r_A):
#         for j in range(c_A):
#             for k in range(r_B):
#                 for l in range(c_B):
#                     result[i * r_B + k][j * c_B + l] = A[i][j] * B[k][l]

#     return result

# Caclulate matrix multiplication
# def matrix_multiply(A, B):
    
#     # Calculate matrix sizes from rows (r) and columns (c)
#     r_A, c_A = len(A), len(A[0])
#     r_B, c_B = len(B), len(B[0])

#     # Check multiplication validity
#     if c_A != r_B:
#         raise ValueError("Matrix dimensions do not match for multiplication!")

#     # Create matrix
#     result = [[0 for i in range(c_B)] for i in range(r_A)]

#     # Calculate multiplication
#     for i in range(r_A):
#         for j in range(c_B):
#             for k in range(c_A):  
#                 result[i][j] += A[i][k] * B[k][j]

#     return result