# Imports
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from catboost import CatBoostClassifier, Pool

from preprocess import Preprocessor


def train_fit_predict(X: pd.DataFrame, y: pd.Series, test_df: pd.DataFrame) -> np.ndarray:
    model = CatBoostClassifier(iterations=2000, task_type="GPU")
    model.fit(X, y, verbose_eval=100)

    return model.predict(test_df)


def main(input_path: Path):
    orig_train = pd.read_csv(input_path / "train.csv")
    orig_test = pd.read_csv(input_path / "test.csv")

    test_passenger_ids = orig_test["PassengerId"].copy()  # Store PassengerId before preprocessing

    preprocessor = Preprocessor(
        encoder=LabelEncoder(), imputer=KNNImputer(n_neighbors=5)
    )

    train = preprocessor.preprocess_data(orig_train)
    test = preprocessor.preprocess_data(orig_test)

    y_train = train["Transported"]
    X_train = train.drop(columns=["Transported"])

    y_pred = train_fit_predict(X_train, y_train, test)

    result = pd.DataFrame({"PassengerId": test_passenger_ids, "Transported": y_pred})

    # result["Transported"] = result["Transported"].apply(lambda x: x in [1, True])

    result.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    # parser.add_argument('-o', type=str, required=True)
    args = parser.parse_args()

    main(Path(args.i))
