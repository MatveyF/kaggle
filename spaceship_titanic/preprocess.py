# Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer


class Preprocessor:
    def __init__(self, scalers = None, encoders = None, imputers = None):
        self.scalers = scalers if scalers is not None else {}
        self.encoders = encoders if encoders is not None else {}
        self.imputers = imputers if imputers is not None else {}

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # extract group information (and cast to int)
        df["PassengerGroup"] = df["PassengerId"].str.split("_").str[0].apply(lambda x: int(x))

        # expand the cabin information
        df[["cabin_deck", "cabin_num", "cabin_side"]] = df["Cabin"].str.split("/", expand=True)

        df["AmountBilled"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)

        return df

    def fit(self, df: pd.DataFrame):
        df = self._generate_features(df)

        # Fit the encoder and imputer on the training data
        for col_name in ["cabin_deck", "cabin_side"]:
            if col_name not in self.encoders:
                self.encoders[col_name] = LabelEncoder()
            self.encoders[col_name].fit(df[col_name])

        for col_name in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CryoSleep",
                         "AmountBilled"]:
            if col_name not in self.imputers:
                self.imputers[col_name] = KNNImputer(n_neighbors=5)
            self.imputers[col_name].fit(df[[col_name]])

        for col_name in ["Age", "RoomService", "Spa", "VRDeck", "AmountBilled"]:
            if col_name in self.scalers:
                self.scalers[col_name].fit(df[[col_name]])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._generate_features(df)

        # assume cabin_side missing values are "Port"
        df["cabin_side"] = df["cabin_side"].fillna("P")

        # assume vip missing values are False
        df["VIP"] = df["VIP"].fillna(False)

        # encode the cabin information
        for col_name in ["cabin_deck", "cabin_side"]:
            df[col_name] = self.encoders[col_name].transform(df[col_name])

        for col_name in [
            "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CryoSleep", "AmountBilled"
        ]:
            df[col_name] = self.imputers[col_name].transform(df[col_name].to_numpy().reshape(-1, 1))

        df.drop(
            columns=[
                "Name", "Cabin", "PassengerId", "HomePlanet", "Destination", "FoodCourt", "ShoppingMall", "cabin_num"
            ],
            inplace=True
        )

        # If a scaler is provided, scale the numerical features
        for col_name in ["Age", "RoomService", "Spa", "VRDeck", "AmountBilled"]:
            if col_name in self.scalers:
                df[col_name] = self.scalers[col_name].transform(df[[col_name]])

        return df
