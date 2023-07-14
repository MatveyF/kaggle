import pandas as pd
import pytest

from movie_recommender_system.content_based_recommender import ContentBasedRecommender, SimilarityMetric
from movie_recommender_system.data_loader import DataLoader

@pytest.fixture
def mock_data():
    return pd.DataFrame({
        "title": ["The Godfather", "The Dark Knight", "Pulp Fiction"],
        "overview": [
            "The aging patriarch of an organized crime dynasty...", "When the menace known as the Joker...",
            "A burger-loving hit man..."
        ],
        "tagline": ["An offer you can\'t refuse.", "Why so serious?", "Just because you are a character..."],
    })

@pytest.fixture
def mock_loader(mocker, mock_data):
    mock = mocker.Mock(spec=DataLoader)
    mock.load_data.return_value = mock_data
    return mock

def test_preprocess(mock_loader, mock_data):
    recommender = ContentBasedRecommender(loader=mock_loader, similarity_metric=SimilarityMetric.COSINE)

    assert "combined" in recommender.df.columns
    assert recommender.df["combined"].equals(mock_data["overview"] + mock_data["tagline"])

def test_get_recommendations(mock_loader):
    recommender = ContentBasedRecommender(loader=mock_loader, similarity_metric=SimilarityMetric.COSINE)
    recommendations = recommender.get_recommendations("The Godfather", 2)

    assert len(recommendations) == 2
    assert "The Godfather" not in recommendations

def test_unknown_similarity_metric(mock_loader):
    with pytest.raises(ValueError):
        ContentBasedRecommender(loader=mock_loader, similarity_metric="unknown")
