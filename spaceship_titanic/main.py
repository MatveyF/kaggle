# Imports
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostClassifier, Pool

from preprocess import preprocess_data


def train_fit_predict(X: pd.DataFrame, y: pd.Series, test_df: pd.DataFrame) -> np.ndarray:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    err = []
    y_pred = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y[train_index], y[test_index]
        _train = Pool(X_train, label=y_train)
        _valid = Pool(X_val, label=y_val)

        cb = CatBoostClassifier()

        cb.fit(_train, eval_set=_valid, use_best_model=True, verbose_eval=100)

        p = cb.predict(X_val)
        print("err: ", mean_squared_log_error(y_val, [i == "True" for i in p]))
        err.append(mean_squared_log_error(y_val, [i == "True" for i in p]))
        pred = cb.predict(test_df)
        y_pred.append(pred)

    new_y_pred = np.ndarray(shape=(5, 4277))
    for i, arr in enumerate(y_pred):
        new_y_pred[i] = [j == "True" for j in arr]

    return np.mean(new_y_pred, 0)


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

    y_pred = [i >= 0.5 for i in y_pred]

    result = pd.DataFrame({"PassengerId": orig_test["PassengerId"], "Transported": y_pred})

    # result["Transported"] = result["Transported"].apply(lambda x: x in [1, True])

    result.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    # parser.add_argument('-o', type=str, required=True)
    args = parser.parse_args()

    main(Path(args.i))
