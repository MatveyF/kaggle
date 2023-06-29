# Imports
from typing import Dict, Optional, List

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer


class Preprocessor:
    """
    Preprocessor class to transform the data into a format that can be used by the model.

    Args:
        scalers:
            A dictionary of scalers to use for each column. The keys should be the column names and the values should
            be the scaler to use. If a column is not present in the dictionary, no scaling will be applied.
        encoders:
            A dictionary of encoders to use for each column. The keys should be the column names and the values should
            be the encoder to use.
        imputers:
            A dictionary of imputers to use for each column. The keys should be the column names and the values should
            be the imputer to use.
        features_to_drop:
            A list of feature names to drop from the data. May contain generated feature names.
    """
    def __init__(
        self,
        scalers: Optional[Dict] = None,
        encoders: Optional[Dict] = None,
        imputers: Optional[Dict] = None,
        features_to_drop: Optional[List[str]] = None,
    ):
        self.scalers = scalers if scalers is not None else {}
        self.encoders = encoders if encoders is not None else {}
        self.imputers = imputers if imputers is not None else {}
        self.features_to_drop = features_to_drop

        # Note that these can contain generated features
        self._numerical_features = [
            "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CryoSleep", "AmountBilled",
            "FoodCourt", "ShoppingMall"
        ]
        self._categorical_features = ["cabin_deck", "cabin_side", "HomePlanet", "Destination"]

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # extract group information (and cast to int)
        df["PassengerGroup"] = df["PassengerId"].str.split("_").str[0].apply(lambda x: int(x))

        # expand the cabin information
        df[["cabin_deck", "cabin_num", "cabin_side"]] = df["Cabin"].str.split("/", expand=True)

        df["AmountBilled"] = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)

        return df

    def fit(self, df: pd.DataFrame):
        """ Fit the preprocessor on the training data.

        Args:
            df: The training data to fit the preprocessor on.
        """
        df = self._generate_features(df)

        # Fit the encoder on the training categorical data
        for col_name in self._categorical_features:
            if col_name not in self.encoders:
                self.encoders[col_name] = LabelEncoder()
            self.encoders[col_name].fit(df[col_name])

        # Fit the imputer and scaler on the training numerical data
        for col_name in self._numerical_features:
            if col_name not in self.imputers:
                self.imputers[col_name] = KNNImputer(n_neighbors=5)
            self.imputers[col_name].fit(df[[col_name]])

            if col_name in self.scalers:
                self.scalers[col_name].fit(df[[col_name]])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Transform the data into a format that can be used by the model.

        Args:
            df: The data to transform.

        Returns:
            pd.DataFrame: The transformed data.
        """
        df = self._generate_features(df)

        # assume cabin_side missing values are "Port"
        df["cabin_side"] = df["cabin_side"].fillna("P")

        # assume vip missing values are False
        df["VIP"] = df["VIP"].fillna(False)

        # Drop undesired features
        if self.features_to_drop is not None:
            df.drop(columns=self.features_to_drop, inplace=True)

        # Encoding the categorical features
        for col_name in self._categorical_features:
            if col_name in df.columns:
                df[col_name] = self.encoders[col_name].transform(df[col_name])

        # Impute the missing values for the numerical features, if a scaler is provided, scale it
        for col_name in self._numerical_features:
            if col_name in df.columns:
                df[col_name] = self.imputers[col_name].transform(df[col_name].to_numpy().reshape(-1, 1))

                if col_name in self.scalers:
                    df[col_name] = self.scalers[col_name].transform(df[[col_name]])

        return df
