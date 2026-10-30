import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class BatchGradientDescentLogisticRegression(
    ClassifierMixin, BaseEstimator
):
    def __init__(self, learning_rate=0.01, epochs=2000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("This model requires exactly two classes.")

        y_binary = (y == self.classes_[1]).astype(float)
        self.weights_ = np.zeros(X.shape[1])
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.epochs):
            scores = np.clip(
                X @ self.weights_ + self.bias_, -500, 500
            )
            probabilities = 1 / (1 + np.exp(-scores))
            error = probabilities - y_binary

            dw = (X.T @ error) / len(y)
            db = np.mean(error)

            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

            p = np.clip(probabilities, 1e-9, 1 - 1e-9)
            loss = -np.mean(
                y_binary * np.log(p)
                + (1 - y_binary) * np.log(1 - p)
            )
            self.loss_history_.append(loss)

        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["weights_", "bias_"])
        X = check_array(X)
        scores = np.clip(
            X @ self.weights_ + self.bias_, -500, 500
        )
        probabilities = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - probabilities, probabilities])

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return self.classes_[(probabilities >= 0.5).astype(int)]
