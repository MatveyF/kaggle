from surprise import SVD

from movie_recommender_system.data_loader import PostgresDataLoader
from movie_recommender_system.collaborative_based_recommender import CollaborativeBasedRecommender
from movie_recommender_system.config import DATABASE_CONFIG


def main():
    recommender = CollaborativeBasedRecommender(
        loader=PostgresDataLoader(
            dbname=DATABASE_CONFIG["dbname"],
            user=DATABASE_CONFIG["user"],
            password=DATABASE_CONFIG["password"],
            host=DATABASE_CONFIG["host"],
            port=DATABASE_CONFIG["port"],
            table="movies_metadata",
        ),
        model=SVD(),
    )

    recommender.fit()

    print(recommender.get_recommendations(1, 1))


if __name__ == '__main__':
    main()
