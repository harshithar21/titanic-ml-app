import pandas as pd

from ml.features import add_features


def test_family_features():
    df = pd.DataFrame({"SibSp": [0, 2], "Parch": [0, 1], "Fare": [10.0, 10.0]})
    out = add_features(df)
    assert out["FamilySize"].tolist() == [1, 4]
    assert out["IsAlone"].tolist() == [1, 0]


def test_fare_is_capped_and_input_not_modified():
    df = pd.DataFrame({"SibSp": [0], "Parch": [0], "Fare": [512.3]})
    assert add_features(df)["Fare"].iloc[0] == 250
    assert df["Fare"].iloc[0] == 512.3
