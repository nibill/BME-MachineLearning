import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils import shuffle

import time


def em_segmentation(img, k, max_iter=20):
    """
    Learns a MoG model using the EM-algorithm for image-segmentation.

    Args:
        img: The input color image of shape [h, w, 3]
        k: The number of gaussians to be used

    Returns:
        label_img: A matrix of labels indicating the gaussian of size [h, w]

    """

    label_img = None

    #######################################################################
    # TODO:                                                               #
    # 1st: Augment the pixel features with their 2D coordinates to get    #
    #      features of the form RGBXY (see np.meshgrid)                   #
    # 2nd: Fit the MoG to the resulting data using                        #
    #      sklearn.mixture.GaussianMixture                                #
    # 3rd: Predict the assignment of the pixels to the gaussian and       #
    #      generate the label-image                                       #
    #######################################################################

    h = img.shape[0]
    w = img.shape[1]
    cols = img.shape[2]

    xgrid, ygrid = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1))

    coords = np.stack((ygrid, xgrid), axis=2)
    img = np.concatenate((img, coords), axis=2)
    img = np.reshape(img, (h * w, cols + 2))

    moG = GaussianMixture(n_components=k, max_iter = max_iter).fit(img)

    label_img = moG.predict(img)

    means = np.delete(moG.means_, [3, 4], axis=1).astype('uint8')

    img_temp = np.take(means, label_img, axis=0)
    label_img = np.reshape(img_temp, (h, w, cols))

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return label_img
