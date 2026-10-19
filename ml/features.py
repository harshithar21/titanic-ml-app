"""Feature engineering used both in training and in the API.

Keeping this in one module guarantees that the data the model sees in
production is transformed exactly like the data it was trained on.
"""
import pandas as pd

RAW_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "HasCabin"]
FARE_CAP = 250  # about the 99th percentile of Titanic fares (see Week 1 outlier analysis)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add FamilySize and IsAlone, and cap extreme fares."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Fare"] = df["Fare"].clip(lower=0, upper=FARE_CAP)
    return df
