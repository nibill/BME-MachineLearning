from cost_function import cost_function
from gradient_function import gradient_function
import numpy as np
import time


def tanh_GD(X, y, num_iter=10000, alpha=0.01):
    """
    Perform regression with gradient descent.

    Args:
        theta_0: Initial value for parameters of shape [num_features]
        X: Data matrix of shape [num_train, num_features]
        y: Labels corresponding to X of size [num_train, 1]
        num_iter: Number of iterations of GD
        alpha: The learning rate

    Returns:
        theta: The value of the parameters after regression

    """

    theta = np.zeros(X.shape[1])
    losses = []
    for i in range(num_iter):
        start = time.time()
        #######################################################################
        # TODO:                                                               #
        # Perform one step of gradient descent:                               #
        #   - Select a single training example at random                      #
        #   - Update theta based on alpha and using gradient_function         #
        #                                                                     #
        #######################################################################

        pass

        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################
        if i % 1000 == 0:
            exec_time = time.time() - start
            loss = cost_function(theta, X, y)
            losses.append(loss)
            print('Iter {}/{}: cost = {}  ({}s)'.format(i, num_iter, loss, exec_time))
            alpha *= 0.9

    return theta, losses