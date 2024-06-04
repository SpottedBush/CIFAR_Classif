"""
Just some simple metrics module to compare the different computed models
"""

import matplotlib.pyplot as plt
import seaborn as sns

from CIFAR_Classif.generic_classifier import GenericClassifier
from CIFAR_Classif.generic_features_extractor import GenericFeaturesExtractor

# ------Benchmarks------

def benchmark_feature_extractors(X_train, X_test, feature_extractor_list = ["hog", 'lbp']):
    """Benchmark different feature extractors for a specific dataset.

    Args:
        X_train (pd.Dataframe): Training data.
        y_train (pd.Dataframe): Training labels.
        X_test (pd.Dataframe): Testing data.
        y_test (pd.Dataframe): Testing labels.
        feature_extractor_list ([strings]): Feature extractor to use. default=["hog", 'lbp']. Must be one of the following list: ["hog", 'lbp'].
    """
    X_train_features = {}
    X_test_features = {}
    for feature_extractor in feature_extractor_list:
        generic_features_extractor = GenericFeaturesExtractor(kernel=feature_extractor)
        X_train_features[feature_extractor] = generic_features_extractor.extract_features(X_train)
        X_test_features[feature_extractor] = generic_features_extractor.extract_features(X_test)
    return X_train_features, X_test_features 

def benchmark_models(X_train, y_train, X_test, y_test, model_list = ["svc", 'logistic_regression', 'knn']):
    """Benchmark different classifiers for a specific dataset.

    Args:
        X_train (pd.Dataframe): Training data.
        y_train (pd.Dataframe): Training labels.
        X_test (pd.Dataframe): Testing data.
        y_test (pd.Dataframe): Testing labels.
        model_list ([strings]): Model to use. default=["svc", 'logistic_regression', 'knn']. Must be one of the following list: ["svc", 'logistic_regression', 'random_forest', 'knn', 'decision_tree', 'gradient_boosting'].
    """
    for model in model_list:
        generic_classifier = GenericClassifier(kernel=model)
        # Training and scoring
        print(f"------{model}------")
        generic_classifier.training_score(X_train, y_train, X_test, y_test, verbose=True)

# ------Quantitative metrics------


def plot_feature_correlation(feature_correlation_data):
    sns.heatmap(feature_correlation_data, annot=True)
    plt.xlabel("Feature_x")
    plt.ylabel("Feature_y")
    plt.title("Feature Correlation")
    plt.show()


def plot_feature_importance(feature_names, importance_values):
    plt.bar(feature_names, importance_values)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importance")
    plt.xticks(rotation=90)
    plt.show()


def plot_confusion_matrix(confusion_matrix_data):
    sns.heatmap(confusion_matrix_data, annot=True, fmt="d")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc_curve(false_positive_rate, true_positive_rate):
    plt.plot(false_positive_rate, true_positive_rate)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()


def plot_precision_recall_curve(recall, precision):
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()


def plot_learning_curve(train_sizes, train_scores_mean, test_scores_mean):
    plt.plot(train_sizes, train_scores_mean, label="Training Accuracy")
    plt.plot(train_sizes, test_scores_mean, label="Validation Accuracy")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.show()


def plot_accuracy(accuracy_values):
    plt.plot(accuracy_values)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.show()


# ------Qualitative metrics------


def plot_decision_boundary(x, y, z):
    plt.contourf(x, y, z, cmap="RdYlBu")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.colorbar()
    plt.show()


# ------Other metrics------


def plot_feature_distribution(feature_distribution_data):
    for feature in feature_distribution_data:
        sns.histplot(feature_distribution_data[feature], kde=True)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {feature}")
        plt.show()


def plot_feature_density(feature_density_data):
    for feature in feature_density_data:
        sns.kdeplot(feature_density_data[feature])
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Density Plot of {feature}")
        plt.show()


def plot_feature_outliers(feature_outliers_data):
    for feature in feature_outliers_data:
        sns.boxplot(feature_outliers_data[feature])
        plt.xlabel("Feature")
        plt.ylabel("Value")
        plt.title(f"Outliers of {feature}")
        plt.show()
