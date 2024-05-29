import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA


class GenericFeaturesExtractor:
    def __init__(self, kernel="hog"):
        self.str_kernel = kernel

    def extract_features(
        self, X, dim_reduction=False, features_selection=False, threshold=0.1, n_components=2
    ):
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
        if self.str_kernel == "hog":
            for img in X:
                # Converting the image to grayscale to extract HOG features
                fd = hog(
                    rgb2gray(img), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1)
                )
                extracted_features.append(fd)

        if self.str_kernel == "lbp":
            for img in X:
                # Converting the image to grayscale to extract LBP features
                lbp = local_binary_pattern(rgb2gray(img), 8, 1, method="uniform")
                extracted_features.append(lbp)

        else:
            raise ValueError("Invalid kernel. Choose from: ['hog', 'lbp']")

        if dim_reduction:
            pca = PCA(
                n_components=n_components
            )  # Specify the number of components you want to keep
            extracted_features = pca.fit_transform(extracted_features)

        if features_selection:
            # Calculate the variance of each feature
            variances = np.var(extracted_features, axis=0)
            # Find the indices of features with low variance
            low_variance_indices = np.where(variances < threshold)[0]
            # Remove the features with low variance
            extracted_features = np.delete(extracted_features, low_variance_indices, axis=1)

        return extracted_features
