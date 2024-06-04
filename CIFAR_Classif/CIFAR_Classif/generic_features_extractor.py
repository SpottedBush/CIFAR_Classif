import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA


class GenericFeaturesExtractor:
    def __init__(self, kernel="hog"):
        """Initialize the features extractor with the given kernel.
        Args:
            kernel (str, optional): Chose between ["hog", "lbp"]. Defaults to "hog".
        """
        if kernel == "hog":
            self.kernel = (lambda img: hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1)))
        elif kernel == "lbp":
            self.kernel = (lambda img: local_binary_pattern(img, 8, 1, method="uniform"))
        else:
            raise ValueError("Invalid kernel. Choose from: ['hog', 'lbp']")


    def extract_features(self, X, dim_reduction=False, features_selection=False, threshold=0.1, n_components=2):
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

        if dim_reduction:
            pca = PCA(n_components=n_components)
            extracted_features = pca.fit_transform(extracted_features)

        if features_selection:
            # Calculate the variance of each feature
            variances = np.var(extracted_features, axis=0)
            # Find the indices of features with low variance
            low_variance_indices = np.where(variances < threshold)[0]
            # Remove the features with low variance
            extracted_features = np.delete(extracted_features, low_variance_indices, axis=1)

        return extracted_features
