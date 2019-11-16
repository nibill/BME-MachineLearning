import numpy as np


def svm_loss(w, b, X, y, C):
    """
    Computes the loss of a linear SVM w.r.t. the given data and parameters

    Args:
        w: Parameters of shape [num_features]
        b: Bias (a scalar)
        X: Data matrix of shape [num_data, num_features]
        y: Labels corresponding to X of size [num_data, 1]
        C: SVM hyper-parameter

    Returns:
        l: The value of the objective for a linear SVM

    """

    l = 0
    #######################################################################
    # TODO:                                                               #
    # Compute and return the value of the unconstrained SVM objective     #
    #                                                                     #
    #######################################################################

    zeros = np.zeros(y.shape[0])
    f_x = (np.dot(X, w) + b)
    max_inner = 1.0 - np.multiply(y, f_x)
    after_max = np.maximum(zeros, max_inner)
    summed = (np.sum(after_max))
    l = np.divide(np.linalg.norm(w) ** 2.0, 2.0 * C) + np.divide(summed, y.shape[0])

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return l
