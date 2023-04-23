# Imports
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, confusion_matrix
from catboost import CatBoostClassifier, Pool

from preprocess import preprocess_data


def train_fit_predict(X: pd.DataFrame, y: pd.Series, test_df: pd.DataFrame) -> np.ndarray:
    model = CatBoostClassifier(iterations=2000, task_type="GPU")
    model.fit(X, y, verbose_eval=100)

    return model.predict(test_df)


def main(input_path: Path):
    orig_train = pd.read_csv(input_path / "train.csv")
    orig_test = pd.read_csv(input_path / "test.csv")

    train = preprocess_data(orig_train)
    test = preprocess_data(orig_test)

    # selector = VarianceThreshold()
    # print(f"orig shape: {train.shape}")
    # new_X = selector.fit_transform(train)
    # print(f"selected shape: {new_X.shape}")

    y_train = train["Transported"]
    X_train = train.drop(columns=["Transported"])

    y_pred = train_fit_predict(X_train, y_train, test)

    result = pd.DataFrame({"PassengerId": orig_test["PassengerId"], "Transported": y_pred})

    # result["Transported"] = result["Transported"].apply(lambda x: x in [1, True])

    result.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    # parser.add_argument('-o', type=str, required=True)
    args = parser.parse_args()

    main(Path(args.i))
