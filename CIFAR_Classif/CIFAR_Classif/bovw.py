import cv2
import numpy as np

def get_SIFT_kp_and_desc(imgs, dim_out=64):
    """given a list of images, return the keypoints and descriptors of each image

    Args:
        imgs list: list of images

    Returns:
        (list, list): list of keypoints, list of descriptors
    """
    # defining feature extractor that we want to use (SIFT)
    extractor = cv2.SIFT_create()
    keypoints = []
    descriptors = []
    for img in imgs:
        # extract keypoints and descriptors for each image
        img_keypoints, img_descriptors = extractor.detectAndCompute(img, None)
        img_descriptors = img_descriptors if img_descriptors is not None else np.array([]).reshape(0, 128)
        img_descriptors = reduce_descs_dimensions(img_descriptors, n_components=dim_out)
        keypoints.append(img_keypoints)
        descriptors.append(img_descriptors)
        
    return keypoints, descriptors

def reduce_descs_dimensions(X, n_components=64):
    X = X.astype(np.float32)
    train_mean = np.mean(X)
    X = X - train_mean # zero-center

    train_cov = np.dot(X.T, X)
    eigvals, eigvecs = np.linalg.eig(train_cov)
    perm = eigvals.argsort() # sort by increasing eigenvalue
    pca_transform = eigvecs[:, perm[128 - n_components:128]] # eigenvectors for the n_components last eigenvalues
    return X @ pca_transform

def get_bovw_features(imgs, bag):
    """Quite explicit function name

    Args:
        imgs (list): list of images
        bag (KMeans): KMeans object, must be fitted
    Returns:
        list: list of feature vectors
    """
    N = len(imgs) # Number of images
    K = bag.n_clusters # Number of visual words
    
    feature_vector = np.zeros((N, K))

    for i in range(N):
        feature_vector_curr = np.zeros(K, dtype=np.float128)
        
        word_vector = bag.predict(imgs[i].astype(np.float128).reshape(1, -1)) # Predicts the visual word for each image and converts it to const double
        # For each unique visual word
        for word in np.unique(bag.cluster_centers_):
            res = list(word_vector).count(word)
            feature_vector_curr[word] = res # Increments histogram for that word

        # Normalizes the current histogram
        cv2.normalize(feature_vector_curr, feature_vector_curr, norm_type=cv2.NORM_L2)
        feature_vector[i] = feature_vector_curr # Assigned the current histogram to the feature vector

    return feature_vector