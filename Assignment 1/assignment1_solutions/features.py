from skimage.feature import hog
import numpy as np

def hog_features(X):
    """
    Extract HOG features from input images

    Args:
        X: Data matrix of shape [num_train, 577]

    Returns:
        hogs: Extracted hog features

    """
    
    hog_list = []
    
    for i in range(X.shape[0]):
        #######################################################################
        # TODO:                                                               #
        # Extract HOG features from each image and append them to the         #
        # hog_list                                                            #
        #                                                                     #
        # Hint: Make sure that you reshape the imput features to size (24,24) #
        #                                                                     #
        #######################################################################

        # Reshape image
        X_img = np.reshape(X[i,1:], [24, 24], order='F')
        
        # Extract HOG features
        # Higher accuracy, but slower computations
        #hog_feat = hog(X_img, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(3, 3)) 
        
        # Compromise to keep execution time for SGD under 2s
        hog_feat = hog(X_img, orientations=6, pixels_per_cell=(3, 3), cells_per_block=(1, 1)) 
        
        # Add intercept term
        hog_feat = np.append([1,], np.asarray(hog_feat))
        
        # Append result to hog_list
        hog_list.append(hog_feat)

        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################
        
    hogs = np.stack(hog_list,axis=0)
    hogs = np.concatenate((np.ones((X.shape[0], 1)), np.reshape(hogs,(X.shape[0],-1))), axis=1)

    return hogs