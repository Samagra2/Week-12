"""
preprocessing.py
Handles data cleaning, feature engineering, and encoding
"""

import pandas as pd
import numpy as np


def preprocess_churn(df):
    """
    Preprocess customer churn dataset
    - Encode categorical variables
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    return df


def preprocess_house_prices(df):
    """
    Preprocess house prices dataset
    - Drop Property_ID
    - One-hot encode categorical variables
    """
    df = df.copy()

    df.drop(columns=["Property_ID"], inplace=True)

    df = pd.get_dummies(
        df,
        columns=["Location", "Property_Type"],
        drop_first=True
    )

    return df


def preprocess_sales(df):
    """
    Preprocess sales dataset
    - Convert Date to datetime
    - Extract Day, Month, Year
    - Encode categorical variables
    """
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    df.drop(columns=["Date"], inplace=True)

    categorical_cols = df.select_dtypes(include="object").columns

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    return df
