# Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    df = _extract_features(df)
    df = _drop_features(df)
    df = _clean_features(df)

    return df


def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    # Fill missing values
    imputer = KNNImputer()

    for colname in [
        "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "VIP", "CryoSleep", "cabin_side"
    ]:
        df[colname] = imputer.fit_transform(df[colname].to_numpy().reshape(-1, 1))

    df["AmountBilled"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
    df.drop(columns=["FoodCourt", "ShoppingMall", "cabin_num"], inplace=True)

    return df


def _drop_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["Name", "Cabin", "PassengerId", "HomePlanet", "Destination"])


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    # extract group information (and covert to int)
    df["PassengerGroup"] = df["PassengerId"].str.split("_").str[0].apply(lambda x: int(x))

    # expand the cabin information
    df[['cabin_deck', 'cabin_num', 'cabin_side']] = df["Cabin"].str.split("/", expand=True)

    # encode the cabin information
    encoder = LabelEncoder()
    df["cabin_deck"] = encoder.fit_transform(df["cabin_deck"])

    df["cabin_side"] = df["cabin_side"].apply(lambda x: None if x is np.nan else x == "P")

    return df
