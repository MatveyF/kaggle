from typing import Protocol

import pandas as pd


class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        ...

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        ...


class NotFittedError(Exception):
    pass
