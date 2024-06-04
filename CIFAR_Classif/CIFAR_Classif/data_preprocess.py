import pickle
from skimage.color import rgb2gray
import cv2

def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

def preprocess(X):
    new_X = []
    for idx in range(len(X)):
        img = X[idx].reshape(3, 32, 32).transpose(1, 2, 0) # Reshape the image to 32x32x3
        img = rgb2gray(img)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') # Used to compute SIFT features
        new_X.append(img) # Convert the image to grayscale
    return new_X

def load_data(path_list, apply_preprocess=False):
    """Load the data from the given paths and preprocess it.

    Args:
        path_list (list): A list of path strings.

    Returns:
        X: Target data.
        y: Target labels.
    """
    data_dict = {}
    for path in path_list:
        data_dict = data_dict | unpickle(path)
    X = data_dict[b"data"]
    y = data_dict[b"labels"]
    if apply_preprocess:
        X = preprocess(X)
    return X, y
