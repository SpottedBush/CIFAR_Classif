from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class GenericClassifier:
    def __init__(self, X, y, kernel="SVC"):
        if kernel not in ["SVC", "KNN", "LR"]:
            raise ValueError("Invalid kernel. Choose from: ['SVC', 'KNN', 'LR']")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.str_kernel = kernel
        if kernel == "SVC":
            self.kernel = SVC()
        if kernel == "KNN":
            self.kernel = KNeighborsClassifier()
        if kernel == "LR":
            self.kernel = LogisticRegression(max_iter=1000)

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
