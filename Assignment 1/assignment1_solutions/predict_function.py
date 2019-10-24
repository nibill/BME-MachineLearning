from tanh import tanh
import numpy as np


def predict_function(theta, X, y=None):
    """
    Compute predictions on X using the parameters theta. If y is provided
    computes and returns the accuracy of the classifier as well.

    """

    preds = None
    accuracy = None
    #######################################################################
    # TODO:                                                               #
    # Compute predictions on X using the parameters theta.                #
    # If y is provided compute the accuracy of the classifier as well.    #
    #                                                                     #
    #######################################################################
    
    preds = ( tanh( np.dot(theta, np.transpose(X)) ) >= 0.5 ) * 1
    
    if y is None:
        accuracy = "Ground truth missing. Cannot compute accuracy."
    else :
        accuracy = float( np.sum( np.equal(preds, y) * 1) ) / y.size
    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return preds, accuracy