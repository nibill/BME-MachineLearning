import numpy as np
import time
from matplotlib import pyplot as plt

def getDistance(centroid, points):
    return np.linalg.norm(points - centroid, axis=1)

def kmeans(X, k, max_iter=100):
    """
    Perform k-means clusering on the data X with k number of clusters.

    Args:
        X: The data to be clustered of shape [n, num_features]
        k: The number of cluster centers to be used

    Returns:
        centers: A matrix of the computed cluster centers of shape [k, num_features]
        assign: A vector of cluster assignments for each example in X of shape [n]

    """

    centers = None
    assign = np.zeros(len(X))

    start = time.time()

    #######################################################################
    # TODO:                                                               #
    # Perfom k-means clustering of the input data X and store the         #
    # resulting cluster-centers as well as cluster assignments.           #
    #                                                                     #
    #######################################################################

    # 1st step: Chose k random rows of X as initial cluster centers
    centers = X[np.random.choice(np.arange(len(X)), k), :]
    error = np.zeros([X.shape[0], k], dtype=np.float64)

    for i in range(max_iter):
        # 2nd step: Update the cluster assignment
        for i, c in enumerate(centers):
            error[:, i] = getDistance(c, X)

        # 3rd step: Check for convergence
        assign = np.argmin(error, axis = 1)

        # 4th step: Update the cluster centers based on the new assignment
        for c in range(k):
            centers[c] = np.mean(X[assign == c], 0)

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    exec_time = time.time()-start
    print('Number of iterations: {}, Execution time: {}s'.format(i+1, exec_time))

    return centers, assign
