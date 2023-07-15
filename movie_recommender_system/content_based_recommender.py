from enum import Enum

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, euclidean_distances

from .data_loader import DataLoader


class SimilarityMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class ContentBasedRecommender:
    def __init__(self, loader: DataLoader, similarity_metric: SimilarityMetric = SimilarityMetric.COSINE):
        self.df: pd.DataFrame = loader.load_data()
        self.similarity_metric = similarity_metric

        self._preprocess()

    def _preprocess(self):
        self.df["tagline"] = self.df["tagline"].fillna('')
        self.df["overview"] = self.df["overview"].fillna('')

        if "combined" in self.df.columns:
            self.df["combined"] = self.df["combined"] + self.df["overview"] + self.df["tagline"]
        else:
            self.df["combined"] = self.df["overview"] + self.df["tagline"]

        self.df["combined"] = self.df["combined"].fillna('')

        cv_matrix = CountVectorizer(stop_words="english").fit_transform(self.df["combined"])

        if self.similarity_metric == SimilarityMetric.COSINE:
            self.similarities = linear_kernel(cv_matrix, cv_matrix)
        elif self.similarity_metric == SimilarityMetric.EUCLIDEAN:
            self.similarities = euclidean_distances(cv_matrix, cv_matrix)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

    def get_recommendations(self, title: str, n: int = 10) -> pd.DataFrame:
        """ Get the top n movies

        Args:
            title (str): Title of the movie
            n (int): Number of movies to return

        Returns:
            pd.DataFrame: A dataframe with the top n movies

        """
        idx = self.df[self.df['title'] == title].index[0]

        # Get the pairwise similarity scores for all movies and sort them
        similarity_scores = list(enumerate(self.similarities[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        top_movies_indices = [i[0] for i in similarity_scores[1:n + 1]]

        return self.df['title'].iloc[top_movies_indices]
