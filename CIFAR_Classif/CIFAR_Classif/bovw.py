import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_SIFT_kp_and_desc(imgs):
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
        keypoints.append(img_keypoints)
        descriptors.append(img_descriptors)
    return keypoints, descriptors


def get_bovw_features(kps, descs, n_clusters):
    """Quite explicit function name

    Args:
        kps: list of keypoints
        descs : list of descriptors
        n_clusters : hyperparameter for KMeans

    Returns:
        list: list of feature vectors
    """
    
    descs_without_none = [desc for desc in descs if desc is not None]
    
    descs_stack = np.vstack(descs_without_none, dtype=np.float32)
    bag = KMeans(n_clusters=n_clusters, random_state=42).fit(descs_stack)
    
    N = len(kps) # Number of images
    K = bag.n_clusters # Number of visual words
    
    feature_vector = np.zeros((N, K))
    visual_word_pos = 0 # Position of the visual word

    for i in range(N):
        if isinstance(descs[i], type(None)):
            descs[i] = [[0 for _ in range(128)]]
        feature_vector_curr = np.zeros(bag.n_clusters, dtype=np.float32)
        curr_desc = np.asarray(descs[i], dtype=np.float32)
        word_vector = bag.predict(curr_desc)
        
        # For each unique visual word
        for word in np.unique(word_vector):
            res = list(word_vector).count(word)
            feature_vector_curr[word] = res # Increments histogram for that word
        
        # Normalizes the current histogram
        cv2.normalize(feature_vector_curr, feature_vector_curr, norm_type=cv2.NORM_L2)

        feature_vector[visual_word_pos] = feature_vector_curr # Assigned the current histogram to the feature vector
        visual_word_pos += 1 # Increments the position of the visual word

    return feature_vector