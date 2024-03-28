from pathlib import Path
import logging

import pandas as pd
from catboost import CatBoostClassifier

from data_loader import DataLoader
from preprocess import Preprocessor
from optimisation import Optimiser
from utils import Model


logger = logging.getLogger(__name__)


class Pipeline:
    """Pipeline class to chain together the data loading, preprocessing, and modeling steps.

    Args:
        data_loader (DataLoader):
            An instance of the DataLoader class.
        preprocessor (Preprocessor):
            An instance of the Preprocessor class.
        model (Model):
            ML model.
        output_path (Path):
            The path to save the submissions.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        preprocessor: Preprocessor,
        model: Model,
        optimiser: Optimiser,
        output_path: Path = Path("submission.csv"),
    ):
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.model = model or CatBoostClassifier(iterations=2500)
        self.optimiser = optimiser
        self.output_path = output_path

        self._target_col_names: list[str] = [
            "Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"
        ]

    def run(self) -> None:
        train, test = self.data_loader.load_data()

        test_id = test["id"]

        logger.info("Fitting the preprocessor...")
        self.preprocessor.fit(train)

        logger.info("Transforming the training data.")
        train = self.preprocessor.transform(train)

        logger.info("Transforming the test data.")
        test = self.preprocessor.transform(test)

        y_train = train[self._target_col_names]
        X_train = train.drop(columns=self._target_col_names)

        results = pd.DataFrame({"id": test_id})

        logger.info("Fitting the model...")
        for target_col in self._target_col_names:
            self.optimiser.fit(X_train, y_train[target_col])
            optimised_model = self.model(**self.optimiser.best_params)
            optimised_model.fit(X_train, y_train[target_col])
            results[target_col] = optimised_model.predict_proba(test)[:, 1]

        results.to_csv(self.output_path, index=False)
        logger.info("Submission saved to %s", self.output_path)
