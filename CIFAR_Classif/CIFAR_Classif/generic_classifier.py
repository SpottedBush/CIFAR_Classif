from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier



class GenericClassifier:
    def __init__(self, X, y, kernel="SVC"):
        if kernel not in ["svc", 'logistic_regression', 'random_forest', 'knn', 'decision_tree', 'gradient_boosting']:
            raise ValueError("Invalid kernel. Choose from: ['svc', 'logistic_regression', 'random_forest', 'knn', 'decision_tree', 'gradient_boosting']")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.str_kernel = kernel
        if kernel == "svc":
            self.kernel = SVC()
        elif kernel == 'logistic_regression':
            self.model = LogisticRegression()
        elif kernel == 'random_forest':
            self.model = RandomForestClassifier()
        elif kernel == 'knn':
            self.model = KNeighborsClassifier()
        elif kernel == 'decision_tree':
            self.model = DecisionTreeClassifier()
        elif kernel == 'gradient_boosting':
            self.model = GradientBoostingClassifier()
        else:
            raise ValueError("Invalid kernel")

    # Classic classifier methods
    def fit(self):
        return self.kernel.fit(self.X_train, self.y_train)

    def transform(self):
        return self.kernel.transform(self.X_test)

    def fit_transform(self):
        return self.kernel.fit_transform(self.X_train, self.y_train)

    def predict(self):
        return self.kernel.predict(self.X_test)

    def compute_accuracy(self, verbose=False):
        y_pred = self.kernel.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        if verbose:
            print(f"{self.str_kernel} accuracy is:", accuracy)
        return accuracy

    def grid_search(self, param_grid, verbose=False):
        grid_search = GridSearchCV(self.kernel, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        if verbose:
            print("Best Parameters:", best_params)
            print("Best Score:", best_score)
        return best_params, best_score

    def training_score(self, verbose: bool = True):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        if verbose:
            print(f"Accuracy: {accuracy}")
            print("Classification Report:")
            print(report)
        return accuracy, report