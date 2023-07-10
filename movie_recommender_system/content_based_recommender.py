from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from data_loader import DataLoader, CSVDataLoader


class ContentBasedRecommender:
    def __init__(self, loader: DataLoader):
        self.df: pd.DataFrame = loader.load_data()

        tv_matrix = TfidfVectorizer(stop_words="english").fit_transform(self.df["overview"].dropna())  # by overview
        self.cosine_similarities = linear_kernel(tv_matrix, tv_matrix)

    def get_recommendations(self, title: str, n: int = 10):
        """ Get the top n movies

        Args:
            title (str): Title of the movie
            n (int): Number of movies to return

        Returns:
            pd.DataFrame: A dataframe with the top n movies

        """
        # Get the index of the movie that matches the title
        idx = self.df[self.df['title'] == title].index[0]

        # Get the pairwise similarity scores for all movies
        similarity_scores = list(enumerate(self.cosine_similarities[idx]))

        # Sort movies based on similarity scores
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Get top n similar movies
        top_movies_indices = [i[0] for i in similarity_scores[1:n + 1]]

        return self.df['title'].iloc[top_movies_indices]


def main():

    recommender = ContentBasedRecommender(
        loader=CSVDataLoader(path=Path("data/movies_metadata.csv"))
    )
    print(recommender.get_recommendations("Shrek 2"))


if __name__ == '__main__':
    main()
