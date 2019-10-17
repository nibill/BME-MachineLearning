from tanh import tanh
import numpy as np


def cost_function(theta, X, y):
    """
    Computes the cost of using theta as the parameter for regression

    Args:
        theta: Parameters of shape [num_features]
        X: Data matrix of shape [num_data, num_features]
        y: Labels corresponding to X of size [num_data, 1]

    Returns:
        l: The cost for regression

    """

    l = None
    #######################################################################
    # TODO:                                                               #
    # Compute and return the cost l of a particular choice of   #
    # theta.                                                              #
    #                                                                     #
    #######################################################################
    
    l = 0
    m = len(X)
    
    sq_error = (tanh() - y)**2
    
        
    l = 1.0/m 
    
    
    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return l
