# Imports
import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

from preprocess import Preprocessor
from data_loading import CSVDataLoader, DataLoader


SUBMISSION_FILE = "submission.csv"


class Pipeline:
    """ Pipeline class to train and predict on the data.

    Args:
        preprocessor:
            The preprocessor to use to transform the data.
        loader:
            The data loader to use to load the data.
        predictor:
            The model to train and to make predictions.
        output_path:
            The path to save the predictions to.
    """
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

        # Fit the preprocessor on the training data
        self.preprocessor.fit(train)

        # Transform both training and test data
        train = self.preprocessor.transform(train)
        test = self.preprocessor.transform(test)

        y_train = train["Transported"]
        X_train = train.drop(columns=["Transported"])

        self.predictor.fit(X_train, y_train)
        y_pred = self.predictor.predict(test)

        result = pd.DataFrame({"PassengerId": test_passenger_ids, "Transported": y_pred})

        result.to_csv(self.output_path, index=False)


def main(input_path: Path, output_path: Path):
    # maybe try target encoding?
    encoders = {
        "cabin_deck": LabelEncoder(),
        "cabin_side": LabelEncoder(),
    }

    imputers = {
        "Age": KNNImputer(n_neighbors=5),
        "RoomService": KNNImputer(n_neighbors=5),
        "FoodCourt": KNNImputer(n_neighbors=5),
        "ShoppingMall": KNNImputer(n_neighbors=5),
        "Spa": KNNImputer(n_neighbors=5),
        "VRDeck": KNNImputer(n_neighbors=5),
        "CryoSleep": KNNImputer(n_neighbors=5),
        "AmountBilled": KNNImputer(n_neighbors=5),
    }

    features_to_drop = [
        "PassengerId", "Name", "Cabin", "HomePlanet", "Destination", "FoodCourt", "ShoppingMall", "cabin_num"
    ]

    pipeline = Pipeline(
        preprocessor=Preprocessor(
            scalers=None, encoders=encoders, imputers=imputers, features_to_drop=features_to_drop
        ),
        loader=CSVDataLoader(input_path),
        predictor=CatBoostClassifier(
            iterations=1993,
            depth=10,
            learning_rate=0.09982799280243655,
            random_strength=1,
            bagging_temperature=0.45672029947151016,
            od_type="Iter",
            od_wait=11,
            task_type="GPU",
        ),
        output_path=output_path,
    )

    pipeline.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=False, default=SUBMISSION_FILE)
    args = parser.parse_args()

    main(Path(args.i), Path(args.o))
