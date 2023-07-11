from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from data_loader import DataLoader, CSVDataLoader


class ContentBasedRecommender:
    def __init__(self, loader: DataLoader):
        self.df: pd.DataFrame = loader.load_data()

        self._preprocess()

    def _preprocess(self):
        self.df["tagline"] = self.df["tagline"].fillna('')
        self.df["overview"] = self.df["overview"] + self.df["tagline"]
        self.df["overview"] = self.df["overview"].fillna('')  # This is important because tfidf cannot handle NaNs

        tv_matrix = TfidfVectorizer(stop_words="english").fit_transform(self.df["overview"].dropna())
        self.cosine_similarities = linear_kernel(tv_matrix, tv_matrix)

    def get_recommendations(self, title: str, n: int = 10):
        """ Get the top n movies

        Args:
            title (str): Title of the movie
            n (int): Number of movies to return

        Returns:
            pd.DataFrame: A dataframe with the top n movies

        """
        idx = self.df[self.df['title'] == title].index[0]

        # Get the pairwise similarity scores for all movies and sort them
        similarity_scores = list(enumerate(self.cosine_similarities[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        top_movies_indices = [i[0] for i in similarity_scores[1:n + 1]]

        return self.df['title'].iloc[top_movies_indices]


def main():

    recommender = ContentBasedRecommender(
        loader=CSVDataLoader(path=Path("data/movies_metadata.csv"))
    )
    print(recommender.get_recommendations("The Godfather", 25))


if __name__ == '__main__':
    main()
