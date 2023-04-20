# Imports
import argparse

import pandas as pd
import numpy as np


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    df = _extract_features(df)
    df = _drop_features(df)
    df = _clean_features(df)

    return df


def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    # one hot encode
    df = pd.get_dummies(data=df, columns=["HomePlanet"])
    df = pd.get_dummies(data=df, columns=["Destination"])

    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].mean())

    df["RoomService"] = df["RoomService"].fillna(df["RoomService"].median())
    df["FoodCourt"] = df["FoodCourt"].fillna(df["FoodCourt"].median())
    df["ShoppingMall"] = df["ShoppingMall"].fillna(df["ShoppingMall"].median())
    df["Spa"] = df["Spa"].fillna(df["Spa"].median())
    df["VRDeck"] = df["VRDeck"].fillna(df["VRDeck"].median())

    df["AmountBilled"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
    df.drop(columns=["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"], inplace=True)

    df["VIP"] = df["VIP"].fillna(False)
    df["CryoSleep"] = df["CryoSleep"].fillna(False)

    df.drop(columns=["cabin_num"], inplace=True)
    df["cabin_side"] = df["cabin_side"].fillna(False)

    return df


def _drop_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["Name", "Cabin", "PassengerId"])


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    # extract group information (and covert to int)
    df["PassengerGroup"] = df["PassengerId"].str.split("_").str[0].apply(lambda x: int(x))

    # expand the cabin information
    df[['cabin_deck', 'cabin_num', 'cabin_side']] = df["Cabin"].str.split("/", expand=True)

    # df = df[df["cabin_num"].isnull() == False]  # drop rows with no cabin information ?

    # encode the cabin information
    df = pd.get_dummies(data=df, columns=["cabin_deck"])

    df["cabin_side"] = df["cabin_side"].apply(lambda x: None if x is np.nan else x == "P")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.i)
    data = preprocess_data(data)
    data.to_csv(args.o, index=False)
