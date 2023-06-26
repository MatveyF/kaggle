# Imports
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from catboost import CatBoostClassifier

from preprocess import Preprocessor
from data_loading import CSVDataLoader, DataLoader


def train_fit_predict(X: pd.DataFrame, y: pd.Series, test_df: pd.DataFrame) -> np.ndarray:
    model = CatBoostClassifier(iterations=2000, task_type="GPU")
    model.fit(X, y, verbose_eval=100)

    return model.predict(test_df)


def main(input_path: Path):

    loader = CSVDataLoader(input_path)
    train, test = loader.load_data()

    test_passenger_ids = test["PassengerId"].copy()  # Store PassengerId before preprocessing

    preprocessor = Preprocessor(
        encoder=LabelEncoder(), imputer=KNNImputer(n_neighbors=5)
    )

    train = preprocessor.preprocess_data(train)
    test = preprocessor.preprocess_data(test)

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
