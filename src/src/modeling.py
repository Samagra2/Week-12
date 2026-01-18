"""
modeling.py
Contains model training and hyperparameter tuning functions
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


def train_churn_model(X_train, y_train):
    """
    Train Random Forest classifier for churn prediction
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def tune_churn_model(X_train, y_train):
    """
    Hyperparameter tuning for churn model
    """
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="f1"
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_


def train_house_price_model(X_train, y_train):
    """
    Train Random Forest regressor for house price prediction
    """
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_sales_model(X_train, y_train):
    """
    Train Linear Regression model for sales prediction
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
