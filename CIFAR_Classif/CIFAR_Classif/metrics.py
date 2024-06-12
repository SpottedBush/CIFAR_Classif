"""
Just some simple metrics module to compare the different computed models
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

from CIFAR_Classif.generic_classifier import GenericClassifier
from CIFAR_Classif.generic_features_extractor import GenericFeaturesExtractor
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

# ------Benchmarks------
def benchmark_feature_extractors(X_train, y_train, X_test, y_test, feature_extractor_list = ["hog", "flat"], compare_models=True, verbose=True):
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
        print(f"\n\n------Feature extractor tested: {feature_extractor}------")
        accuracy, _ = benchmark_models(X_train_features[feature_extractor], y_train, X_test_features[feature_extractor], y_test, model_list = ["svc", 'logistic_regression', 'knn'], verbose=False)
        print(f"\tAccuracy for svc: {accuracy[0]}")
        print(f"\tAccuracy for logistic_regression: {accuracy[1]}")
        print(f"\tAccuracy for knn: {accuracy[2]}")
    return X_train_features, X_test_features 

def benchmark_models(X_train, y_train, X_test, y_test, model_list = ["svc", 'logistic_regression', 'knn'], verbose=True):
    """Benchmark different classifiers for a specific dataset.

    Args:
        X_train (pd.Dataframe): Training data.
        y_train (pd.Dataframe): Training labels.
        X_test (pd.Dataframe): Testing data.
        y_test (pd.Dataframe): Testing labels.
        model_list ([strings]): Model to use. default=["svc", 'logistic_regression', 'knn']. Must be one of the following list: ["svc", 'logistic_regression', 'random_forest', 'knn', 'decision_tree', 'gradient_boosting'].
    """
    accuracy_list, report_list = [], []
    for model in model_list:
        generic_classifier = GenericClassifier(kernel=model)
        # Training and scoring
        if verbose:
            print(f"------{model}------")
        accuracy, report = generic_classifier.training_score(X_train, y_train, X_test, y_test, verbose=verbose)
        accuracy_list.append(accuracy)
        report_list.append(report)
    return accuracy_list, report_list

# ------Quantitative metrics------

def plot_class_distribution(y_true, labels=None):
    freqs = Counter(y_true)
    xvals = range(len(freqs.values()))
    colors = plt.get_cmap('tab10').colors
    plt.figure(figsize=[13,5])
    plt.bar(xvals, freqs.values(), tick_label=labels, color=colors)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.title("Class Distribution")
    plt.show()

def plot_feature_correlation(X):
    feature_correlation_data = X.corr()
    sns.heatmap(feature_correlation_data, annot=True)
    plt.xlabel("Feature_x")
    plt.ylabel("Feature_y")
    plt.title("Feature Correlation")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None):
    confusion_matrix_data = confusion_matrix(y_true, y_pred)
    sns.heatmap(confusion_matrix_data, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc_curve(y_true, y_pred, labels=None):
    n_classes = len(np.unique(y_true)) # Supposed to be 10 for CIFAR-10
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=i)
        plt.plot(fpr, tpr, label=f"Class {labels[i] if labels else i}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    

def plot_precision_recall_curve(y_true, y_pred, labels=None):
    n_classes = len(np.unique(y_true)) # Supposed to be 10 for CIFAR-10
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=i)
        plt.plot(recall, precision, label=f"Class {labels[i] if labels else i}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()

# In the end those were not very useful for CIFAR-10 dataset
# ------Qualitative metrics------
def plot_decision_boundary(x, y, z): # We did not went deep on decision trees...
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
