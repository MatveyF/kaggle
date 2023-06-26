# Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer


class Preprocessor:
    def __init__(
        self,
        scaler = None,
        encoder = LabelEncoder(),
        imputer = KNNImputer(),
    ):
        self.scaler = scaler
        self.encoder = encoder
        self.imputer = imputer

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # extract group information (and cast to int)
        df["PassengerGroup"] = df["PassengerId"].str.split("_").str[0].apply(lambda x: int(x))

        # expand the cabin information
        df[["cabin_deck", "cabin_num", "cabin_side"]] = df["Cabin"].str.split("/", expand=True)

        # assume cabin_side missing values are "P"
        df["cabin_side"] = df["cabin_side"].apply(lambda x: None if x is np.nan else x == "P")

        # assume vip missing values are False
        df["VIP"] = df["VIP"].fillna(False)

        # encode the cabin information
        for col_name in ["cabin_deck", "cabin_side"]:
            df[col_name] = self.encoder.fit_transform(df[col_name])

        for col_name in [
            "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CryoSleep"
        ]:
            df[col_name] = self.imputer.fit_transform(df[col_name].to_numpy().reshape(-1, 1))

        df["AmountBilled"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)

        df.drop(
            columns=[
                "Name", "Cabin", "PassengerId", "HomePlanet", "Destination", "FoodCourt", "ShoppingMall", "cabin_num"
            ],
            inplace=True
        )

        # If a scaler is provided, scale the numerical features
        if self.scaler is not None:
            numerical_features = ["Age", "RoomService", "Spa", "VRDeck", "AmountBilled"]
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])

        return df
