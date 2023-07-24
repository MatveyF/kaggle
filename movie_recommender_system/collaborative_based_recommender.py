from typing import Protocol

import pandas as pd
from surprise import Reader, Dataset, Trainset, Prediction

from .data_loader import DataLoader


class SurpriseModel(Protocol):

    def fit(self, trainset: Trainset) -> None:
        ...

    def predict(self, uid: int, iid: int) -> Prediction:
        ...


class CollaborativeBasedRecommender:
    def __init__(self, loader: DataLoader, model: SurpriseModel):
        self.model = model

        df: pd.DataFrame = loader.load_data()

        reader = Reader()
        data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
        self.trainset = data.build_full_trainset()

    def fit(self) -> None:
        self.model.fit(self.trainset)

    def get_recommendations(self, user_id: int, item_id: int) -> float:
        return self.model.predict(uid=user_id, iid=item_id).est
