"""
Just a simple metrics module to compare the different computed models
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from CIFAR_Classif.generic_classifier import GenericClassifier

def benchmark_model(df, y_col):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(y_col, axis=1), df[y_col], test_size=0.2, random_state=42)
    svc = GenericClassifier(X_train, y_train, X_test, y_test, kernel='svc')
    logistic_reg = GenericClassifier(X_train, y_train, X_test, y_test, kernel='logistic_regression')
    random_forest = GenericClassifier(X_train, y_train, X_test, y_test, kernel='random_forest')
    knn = GenericClassifier(X_train, y_train, X_test, y_test, kernel='knn')
    decision_tree = GenericClassifier(X_train, y_train, X_test, y_test, kernel='decision_tree')
    gradient_boosting = GenericClassifier(X_train, y_train, X_test, y_test, kernel='gradient_boosting')
    
    # Training and scoring
    print("------SVC------")
    svc.training_score(verbose=True)
    print("------Logistic Regression------")
    logistic_reg.training_score(verbose=True)
    print("------Random Forest------")
    random_forest.training_score(verbose=True)
    print("------KNN------")
    knn.training_score(verbose=True)
    print("------Decision Tree------")
    decision_tree.training_score(verbose=True)
    print("------Gradient Boosting------")
    gradient_boosting.training_score(verbose=True)

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
