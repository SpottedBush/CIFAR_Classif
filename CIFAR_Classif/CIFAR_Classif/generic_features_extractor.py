import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from skimage.color import rgb2gray


class GenericFeaturesExtractor:
    def __init__(self, kernel="hog"):
        """Initialize the features extractor with the given kernel.
        Args:
            kernel (str, optional): Chose between ["hog", "lbp"]. Defaults to "hog".
        """
        if kernel == "hog":
            self.kernel = (lambda img: hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, channel_axis=-1))
        elif kernel == "lbp":
            self.kernel = (lambda img: local_binary_pattern(rgb2gray(img), 8, 1, method="uniform"))
        else:
            raise ValueError("Invalid kernel. Choose from: ['hog', 'lbp']")

    def extract_features(self, X, features_selection=False, threshold=0.1):
        """
        X: list of images
        dim_reduction: bool, default=False
        features_selection: bool, default=False
        threshold: float, default=0.1
        n_components: int, default=2

        You only need to specify n_components if dim_reduction is True. \n
        You only need to specify the threshold if features_selection is True.
        """
        extracted_features = []
        for img in X:
            extracted_features.append(self.kernel(img))
            
        if features_selection:
            extracted_features = apply_feature_selection(extracted_features, threshold=threshold)

        return extracted_features
    
def apply_feature_selection(X, threshold=0.1):
    # Calculate the variance of each feature
    variances = np.var(X, axis=0)
    # Find the indices of features with low variance
    low_variance_indices = np.where(variances < threshold)[0]
    # Remove the features with low variance
    return np.delete(X, low_variance_indices, axis=1)