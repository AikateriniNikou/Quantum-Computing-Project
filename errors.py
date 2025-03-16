""" This module defines custom error classes for handling exceptions related to quantum computations.

It provides a structured approach to managing errors by defining base and specific exception classes.
"""

class Error(Exception):
    """Base class for exceptions in this module.
    
    Parameters
    ----------
    expression : str
        Error message or trace.
    
    Attributes
    ----------
    expression : str
        Stores the error message or trace.
    """
    pass

class InputError(Error):
    """Exception raised for invalid input values.
    
    Parameters
    ----------
    expression : str
        Error message or trace.
    
    Attributes
    ----------
    expression : str
        Stores the error message or trace.
    message : str
        Descriptive explanation of the error.
    """
    
    def __init__(self, expression):
        self.expression = expression
        self.message = "Invalid input format."

class MatrixError(Error):
    """Exception raised for issues related to matrix operations.
    
    Parameters
    ----------
    expression : str
        Error message or trace.
    
    Attributes
    ----------
    expression : str
        Stores the error message or trace.
    message : str
        Descriptive explanation of the matrix-related error.
    """
    
    def __init__(self, expression):
        self.expression = expression
        self.message = "Unexpected matrix format or structure."
