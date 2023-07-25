from pathlib import Path
from ast import literal_eval
from typing import Optional

import pandas as pd


INAVLID_ROWS = [19730, 29503, 35587]  # these three samples have a date instead of id in the id column


class Preprocessor:
    """Preprocesses the movie data for the recommender system.

    This class takes in three dataframes containing movie metadata, credits, and keywords,
    and preprocesses them by merging the dataframes, extracting the director's name, and
    combining the features. The preprocessed data is stored in the processed_data attribute.

    Attributes
    ----------
    processed_data : pd.DataFrame
        A DataFrame containing the preprocessed data. This attribute is None until the preprocess method is called.

    Methods
    -------
    preprocess():
        Preprocesses the data and stores the result in the processed_data attribute.
    _preprocess() -> pd.DataFrame:
        Performs the actual preprocessing and returns the preprocessed data.
    _get_director(row: pd.Series) -> str:
        Extracts the director's name from a row of the crew data.
    _combine_features(row: pd.Series) -> str:
        Combines the features into a single string for each movie.
    save_to_csv(path: Path):
        Saves the preprocessed data to a CSV file. Raises an exception if preprocess has not been called.
    """
    def __init__(self, movies_metadata_df: pd.DataFrame, credits_df: pd.DataFrame, keywords_df: pd.DataFrame):
        self.movies_metadata_df = movies_metadata_df
        self.credits_df = credits_df
        self.keywords_df = keywords_df

        self.processed_data: Optional[pd.DataFrame] = None

    def preprocess(self) -> None:
        self.processed_data = self._preprocess()

    def save_to_csv(self, path: Path) -> None:
        """Saves the preprocessed data to a CSV file.

        Args:
            path (Path): The path to the CSV file to save the data to

        Raises:
            ValueError: If preprocess method has not been called prior to calling this method
        """
        if self.processed_data is None:
            raise ValueError("Preprocess the data before saving it to a CSV file.")
        self.processed_data.to_csv(path, index=False)

    def _preprocess(self) -> pd.DataFrame:
        """Preprocesses the data by merging the dataframes, extracting the
        director's name, and combining the features.

        Returns:
            pd.DataFrame: A dataframe containing the preprocessed data
        """
        self.movies_metadata_df.drop(INAVLID_ROWS, axis=0, inplace=True)

        self.movies_metadata_df.drop(
            ["adult", "belongs_to_collection", "homepage", "imdb_id", "original_title", "poster_path",
             "production_companies", "production_countries", "status", "video", "budget", "revenue"],
            axis=1, inplace=True
        )

        self.keywords_df["id"] = self.keywords_df["id"].astype(int)
        self.credits_df["id"] = self.credits_df["id"].astype(int)
        self.movies_metadata_df["id"] = self.movies_metadata_df["id"].astype(int)

        combined = self.movies_metadata_df.merge(self.credits_df, on="id", suffixes=('', '_credits'))
        combined = combined.merge(self.keywords_df, on="id", suffixes=('', '_keywords'))

        for column in ["genres", "cast", "crew", "keywords"]:
            combined[column] = combined[column].apply(literal_eval)

        combined["director"] = combined["crew"].apply(self._get_director)
        combined["genres"] = combined["genres"].apply(lambda row: [i["name"] for i in row])

        for column in ["cast", "keywords"]:
            combined[column] = combined[column].apply(lambda row: [str.lower(i["name"].replace(" ", "")) for i in row])

        combined["combined"] = combined.apply(self._combine_features, axis=1)

        return combined

    @staticmethod
    def _get_director(row: pd.Series) -> str:
        """Extracts the director's name from a row of the crew data."""
        return next((str.lower(i["name"].replace(" ", "")) for i in row if i["job"] == "Director"), "")

    @staticmethod
    def _combine_features(row: pd.Series) -> str:
        """Combines the features into a single string for each movie."""
        return ' '.join(row['keywords']) + ' ' + ' '.join(row['cast']) + ' ' + str(row['director']) + ' ' + ' '.join(
            row['genres'])
