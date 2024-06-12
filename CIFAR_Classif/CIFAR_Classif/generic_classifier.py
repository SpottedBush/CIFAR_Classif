from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


class GenericClassifier:
    """A generic classifier class that can be used to train and evaluate different classifiers."""
    def __init__(self, kernel="svc"):
        """Initialize the classifier with the given data and kernel.

        Args:
            X (pd.Dataframe): Data.
            y (pd.Dataframe): Labels.
            kernel (str, optional): Must be one of the following list: ["svc", 'logistic_regression', 'random_forest', 'knn', 'decision_tree', 'gradient_boosting']. Defaults to "svc".
        """
        if kernel not in ["svc", 'logistic_regression', 'random_forest', 'knn', 'decision_tree', 'gradient_boosting']:
            raise ValueError("Invalid kernel. Choose from: ['svc', 'logistic_regression', 'random_forest', 'knn', 'decision_tree', 'gradient_boosting']")
        self.str_kernel = kernel
        if kernel == "svc":
            self.kernel = SVC(decision_function_shape='ovr')
        elif kernel == 'logistic_regression':
            self.kernel = LogisticRegression(multi_class='ovr')
        elif kernel == 'random_forest':
            self.kernel = RandomForestClassifier()
        elif kernel == 'knn':
            self.kernel = KNeighborsClassifier()
        elif kernel == 'decision_tree':
            self.kernel = DecisionTreeClassifier()
        elif kernel == 'gradient_boosting':
            self.kernel = GradientBoostingClassifier()
        else:
            raise ValueError("Invalid kernel")

    def set_parameters(self, param_dict):
        """Set the parameters of the current kernel with the given dictionary."""
        self.kernel.set_params(**param_dict)
        
    # Classic classifier methods
    def fit(self, X_train, y_train):
        return self.kernel.fit(X_train, y_train)

    def transform(self, X_test):
        return self.kernel.transform(X_test)

    def fit_transform(self, X_train, y_train):
        return self.kernel.fit_transform(X_train, y_train)

    def predict(self, X_test):
        return self.kernel.predict(X_test)

    def compute_accuracy(self, X_test, y_test, verbose=False):
        y_pred = self.kernel.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if verbose:
            print(f"{self.str_kernel} accuracy is:", accuracy)
        return accuracy

    def grid_search(self, param_grid, X_train, y_train, verbose=False):
        """Perform grid search on the classifier."""
        grid_search = GridSearchCV(self.kernel, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        if verbose:
            print("Best Parameters:", best_params)
            print("Best Score:", best_score)
        return grid_search

    def training_score(self, X_train, y_train, X_test, y_test, verbose: bool = True):
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        if verbose:
            print(f"Accuracy: {accuracy}")
            print("Classification Report:")
            print(report)
        return accuracy, report