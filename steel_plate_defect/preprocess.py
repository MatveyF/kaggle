from typing import Any
from logging import getLogger

import pandas as pd
from sklearn.impute import KNNImputer

from utils import NotFittedError


logger = getLogger(__name__)


class Preprocessor:
    """Preprocessor for the steel plate fault dataset

    Args:
        scalers:
            A dictionary of scalers for numerical features.
            The keys are the names of the numerical features and the values are the scalers.
        imputers:
            A dictionary of imputers for numerical features.
            The keys are the names of the numerical features and the values are the imputers.
        encoders:
            A dictionary of encoders for categorical features.
            The keys are the names of the categorical features and the values are the encoders.
        feature_engineering:
            Whether to apply feature engineering to the data.
            Default is False.
    """
    def __init__(
        self,
        scalers: dict[str, Any],
        imputers: dict[str, Any],
        encoders: dict[str, Any],
        feature_engineering: bool = False,
    ):
        self.scalers = scalers
        self.imputers = imputers
        self.encoders = encoders
        self.feature_engineering = feature_engineering

        self._to_drop: list[str] = ["id"]  # literally row number

        self._numerical_feature_names: list[str] = [
            "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas", "X_Perimeter", "Y_Perimeter",
            "Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity", "Length_of_Conveyer",
            "Steel_Plate_Thickness", "Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index", "Edges_X_Index",
            "Edges_Y_Index", "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index", "Orientation_Index",
            "Luminosity_Index", "SigmoidOfAreas"
        ]

        # Both binary. But they are not mutually exclusive (both can be 0 & both can be 1)
        self._categorical_feature_names: list[str] = ["TypeOfSteel_A300", "TypeOfSteel_A400"]

        # All binary. Not mutually exclusive, 21 samples have 2 or more faults present
        self._target_col_names: list[str] = [
            "Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"
        ]

        self._is_fitted: bool = False

    @staticmethod
    def _feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
        data["X_range"] = data["X_Maximum"] - data["X_Minimum"]
        data["Y_range"] = data["Y_Maximum"] - data["Y_Minimum"]

        data.drop(["X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum"], axis=1, inplace=True)

        return data

    def fit(self, data: pd.DataFrame) -> None:

        for col in self._numerical_feature_names:
            # impute first. Can try re-ordering?
            if col not in self.imputers and data[col].isnull().sum() > 0:
                logger.warning(
                    f"Column '{col}' has missing values but there is no imputer specified. Using KNN imputer."
                )
                self.imputers[col] = KNNImputer(n_neighbors=5)

            if col in self.imputers:
                self.imputers[col].fit(data[col])

            if col in self.scalers:
                self.scalers[col].fit(data[col])

        for col in self._categorical_feature_names:
            if col in self.encoders:
                self.encoders[col].fit(data[col])

        self.is_fitted = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        if not self.is_fitted:
            message = "The preprocessor has not been fitted, please use the `fit` method first."
            logger.error(message)
            raise NotFittedError(message)

        data.drop(self._to_drop, axis=1, inplace=True)

        if self.feature_engineering:
            data = self._feature_engineering(data)

        for col in self._numerical_feature_names:
            if col in data.columns:
                if col in self.imputers:
                    data[col] = self.imputers[col].transform(data[col])

                if col in self.scalers:
                    data[col] = self.scalers[col].transform(data[col])

        for col in self._categorical_feature_names:
            if col in self.encoders and col in data.columns:
                data[col] = self.encoders[col].transform(data[col])

        return data

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value: bool) -> None:
        self._is_fitted = value
