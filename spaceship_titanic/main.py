# Imports
import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from catboost import CatBoostClassifier

from preprocess import Preprocessor
from data_loading import CSVDataLoader, DataLoader


SUBMISSION_FILE = "submission.csv"


class Pipeline:
    def __init__(
        self,
        preprocessor: Preprocessor,
        loader: DataLoader,
        predictor: Any = None,
        output_path: Path = Path(SUBMISSION_FILE),
    ):
        self.preprocessor = preprocessor
        self.loader = loader
        self.output_path = output_path

        if predictor is None:
            self.predictor = CatBoostClassifier(iterations=2000, task_type="GPU")
        else:
            self.predictor = predictor

    def run(self):
        train, test = self.loader.load_data()

        test_passenger_ids = test["PassengerId"].copy()

        train = self.preprocessor.preprocess_data(train)
        test = self.preprocessor.preprocess_data(test)

        y_train = train["Transported"]
        X_train = train.drop(columns=["Transported"])

        self.predictor.fit(X_train, y_train, verbose_eval=100)
        y_pred = self.predictor.predict(test)

        result = self._process_output(test_passenger_ids, y_pred)

        result.to_csv(self.output_path, index=False)

    @staticmethod
    def _process_output(passenger_ids: pd.Series, predictions: pd.Series) -> pd.DataFrame:
        if predictions.isin([0, 1]).all():
            return pd.DataFrame({"PassengerId": passenger_ids, "Transported": predictions})

        return pd.DataFrame({"PassengerId": passenger_ids, "Transported": predictions.astype(int)})


def main(input_path: Path, output_path: Path):

    pipeline = Pipeline(
        preprocessor=Preprocessor(
            encoder=LabelEncoder(), imputer=KNNImputer(n_neighbors=5)
        ),
        loader=CSVDataLoader(input_path),
        predictor=CatBoostClassifier(iterations=2000, task_type="GPU"),
        output_path=output_path,
    )

    pipeline.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=False, default=SUBMISSION_FILE)
    args = parser.parse_args()

    main(Path(args.i), Path(args.o))
