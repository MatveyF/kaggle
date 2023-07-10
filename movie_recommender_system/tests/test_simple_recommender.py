import pytest
import pandas as pd

from movie_recommender_system.data_loader import DataLoader
from movie_recommender_system.simple_recommender import SimpleRecommender


@pytest.fixture
def mocked_data_loader(mocker):

    class MockedDataLoader(DataLoader):
        def load_data(self):
            data = {
                'title': ['Movie 1', 'Movie 2', 'Movie 3'],
                'vote_count': [100, 200, 300],
                'vote_average': [7.5, 7.3, 5.5]
            }
            return pd.DataFrame(data)

    mocked_data_loader = MockedDataLoader()
    mocker.patch.object(mocked_data_loader, 'load_data', return_value=mocked_data_loader.load_data())

    return mocked_data_loader


@pytest.mark.parametrize(
    "n, expected_titles", [
        (3, ["Movie 2", "Movie 1", "Movie 3"]),
        (2, ["Movie 2", "Movie 1"])
    ]
)
def test_simple_recommender_preprocess(mocked_data_loader, n, expected_titles):
    recommender = SimpleRecommender(mocked_data_loader)

    assert expected_titles == recommender.get_recommendations(n)["title"].values.tolist()
