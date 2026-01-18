"""
evaluation.py
Model evaluation utilities
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score
)


def evaluate_classification(model, X_test, y_test):
    """
    Evaluate classification model
    """
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }


def evaluate_regression(model, X_test, y_test):
    """
    Evaluate regression model
    """
    y_pred = model.predict(X_test)

    return {
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2_score": r2_score(y_test, y_pred)
    }
