# In this file I will implement a simple movie recommender system using the demographic filtering technique
import pandas as pd


class SimpleRecommender:
    def __init__(self, percentile: float = 0.9):

        self.percentile = percentile

        self._preprocess()

    def _preprocess(self):
        movies_metadata_df = pd.read_csv('data/movies_metadata.csv')
        movies_metadata_df.drop(
            ['adult', 'belongs_to_collection', 'homepage', 'imdb_id', 'original_language', 'original_title', 'overview',
             'poster_path', 'production_companies', 'production_countries', 'spoken_languages', 'status', 'tagline',
             'video', 'budget', 'revenue'], axis=1, inplace=True
        )

        vote_counts = movies_metadata_df[movies_metadata_df['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = movies_metadata_df[movies_metadata_df['vote_average'].notnull()]['vote_average'].astype('int')

        self.C = vote_averages.mean()
        self.m = vote_counts.quantile(self.percentile)

        movies_metadata_df["weighted_rating"] = movies_metadata_df.apply(self._weighted_rating, axis=1)

        self._preprocessed_data = movies_metadata_df.sort_values('weighted_rating', ascending=False).dropna()

    def _weighted_rating(self, row: pd.Series) -> float:
        """ Calculate the weighted rating of each movie

        This function uses IMDB's weighted rating formula to calculate the weighted
        rating of each movie. The formula is as follows:

            (v / (v + m) * R) + (m / (m + v) * C)

        where:
            v = number of votes for the movie
            m = minimum votes required to be listed in the chart
            R = average rating of the movie
            C = mean vote across the whole report

        Args:
            row (pd.Series):
                A single row with the following keys:
                    vote_count (int): Number of votes for the movie
                    vote_average (float): Average rating of the movie

        """
        v = row['vote_count']
        R = row['vote_average']

        return (v / (v + self.m) * R) + (self.m / (self.m + v) * self.C)

    def get_top_n_movies(self, n: int = 10) -> pd.DataFrame:
        """ Get the top n movies

        Args:
            n (int): Number of movies to return

        Returns:
            pd.DataFrame: A dataframe with the top n movies

        """
        return self._preprocessed_data[['title', 'vote_count', 'vote_average', 'weighted_rating']].head(n)


def main():
    simple_recommender = SimpleRecommender()

    print(simple_recommender.get_top_n_movies())


if __name__ == '__main__':
    main()
