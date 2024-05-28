"""
Just a simple metrics module to compare the different computed models
"""


import seaborn as sns

import matplotlib.pyplot as plt

def plot_roc_curve(false_positive_rate, true_positive_rate):
    plt.plot(false_positive_rate, true_positive_rate)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

def plot_precision_recall_curve(recall, precision):
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

def plot_confusion_matrix(confusion_matrix_data):
    sns.heatmap(confusion_matrix_data, annot=True, fmt='d')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_classification_report(classification_report_data):
    print(classification_report_data)

def plot_learning_curve(train_sizes, train_scores_mean, test_scores_mean):
    plt.plot(train_sizes, train_scores_mean, label='Training Accuracy')
    plt.plot(train_sizes, test_scores_mean, label='Validation Accuracy')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.show()

def plot_feature_importance(feature_names, importance_values):
    plt.bar(feature_names, importance_values)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.xticks(rotation=90)
    plt.show()

def plot_decision_boundary(x, y, z):
    plt.contourf(x, y, z, cmap='RdYlBu')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.colorbar()
    plt.show()

def plot_calibration_curve(probabilities, true_probabilities, predicted_probabilities):
    plt.plot(probabilities, true_probabilities, label='True Probabilities')
    plt.plot(probabilities, predicted_probabilities, label='Predicted Probabilities')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve')
    plt.legend()
    plt.show()

def plot_residuals(predicted_values, residuals):
    plt.scatter(predicted_values, residuals)
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual')
    plt.title('Residual Plot')
    plt.show()

def plot_feature_correlation(feature_correlation_data):
    sns.heatmap(feature_correlation_data, annot=True)
    plt.xlabel('Feature')
    plt.ylabel('Feature')
    plt.title('Feature Correlation')
    plt.show()

def plot_feature_distribution(feature_distribution_data):
    for feature in feature_distribution_data:
        sns.histplot(feature_distribution_data[feature], kde=True)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {feature}')
        plt.show()

def plot_feature_density(feature_density_data):
    for feature in feature_density_data:
        sns.kdeplot(feature_density_data[feature])
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'Density Plot of {feature}')
        plt.show()

def plot_feature_outliers(feature_outliers_data):
    for feature in feature_outliers_data:
        sns.boxplot(feature_outliers_data[feature])
        plt.xlabel('Feature')
        plt.ylabel('Value')
        plt.title(f'Outliers of {feature}')
        plt.show()
