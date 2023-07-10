from data_loader import PostgresDataLoader
from simple_recommender import SimpleRecommender
from config import DATABASE_CONFIG


def main():
    simple_recommender = SimpleRecommender(
        loader=PostgresDataLoader(
            dbname=DATABASE_CONFIG["dbname"],
            user=DATABASE_CONFIG["user"],
            password=DATABASE_CONFIG["password"],
            host=DATABASE_CONFIG["host"],
            port=DATABASE_CONFIG["port"],
            table="movies_metadata",
        ),
        percentile=0.80,
    )
    print(simple_recommender.get_recommendations(25))


if __name__ == '__main__':
    main()
