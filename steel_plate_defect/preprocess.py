import pandas as pd

from utils import NotFittedError


class Preprocessor:
    def __init__(self):

        self._is_fitted: bool = False

    def fit(self, data: pd.DataFrame) -> None:

        self.is_fitted = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        if not self.is_fitted:
            raise NotFittedError("The preprocessor has not been fitted, please use the `fit` method first.")

        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data=data)
        return self.transform(data=data)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value: bool) -> None:
        self._is_fitted = value
