from ast import literal_eval

import pytest
import pandas as pd

from movie_recommender_system.preprocessor import Preprocessor


@pytest.fixture
def sample_data():
    movies_metadata_df = pd.DataFrame({
        "id": [1, 2, 3],
        "title": [f"movie title {i}" for i in range(1, 4)],
        "genres": ['[{"id": 1, "name": "Action"}]', '[{"id": 2, "name": "Comedy"}]', '[{"id": 3, "name": "Drama"}]'],
    })

    credits_df = pd.DataFrame({
        "id": [1, 2, 3],
        "crew": [f'[{{"job": "Director", "name": "Director{i}"}}]' for i in range(1, 4)],
        "cast": [f'[{{"name": "Actor{i}"}}]' for i in range(1, 4)],
    })

    keywords_df = pd.DataFrame({
        "id": [1, 2, 3],
        "keywords": [f'[{{"id": {i}, "name": "Keyword{i}"}}]' for i in range(1, 4)],
    })

    return movies_metadata_df, credits_df, keywords_df

def test_preprocess(sample_data, mocker):
    movies_metadata_df, credits_df, keywords_df = sample_data
    preprocessor = Preprocessor(movies_metadata_df, credits_df, keywords_df)

    # Mock the drop method to do nothing
    mocker.patch.object(pd.DataFrame, "drop", return_value=None)
    preprocessor.preprocess()

    assert preprocessor.processed_data is not None
    assert "director" in preprocessor.processed_data.columns
    assert "combined" in preprocessor.processed_data.columns

def test_save_to_csv_without_preprocessing(sample_data, tmp_path):
    movies_metadata_df, credits_df, keywords_df = sample_data
    preprocessor = Preprocessor(movies_metadata_df, credits_df, keywords_df)

    file_path = tmp_path / "test.csv"

    with pytest.raises(ValueError, match="Preprocess the data before saving it to a CSV file."):
        preprocessor.save_to_csv(file_path)

def test_get_director(sample_data):
    _, credits_df, _ = sample_data

    expected_directors = pd.Series(["director1", "director2", "director3"], name="crew")
    directors = credits_df["crew"].apply(literal_eval).apply(Preprocessor._get_director)

    pd.testing.assert_series_equal(directors, expected_directors)

def test_combine_features():
    row = pd.Series({
        "keywords": ["keyword1", "keyword2"],
        "cast": ["actor1", "actor2"],
        "director": "director",
        "genres": ["genre1", "genre2"]
    })

    expected_output = "keyword1 keyword2 actor1 actor2 director genre1 genre2"
    assert Preprocessor._combine_features(row) == expected_output
