import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient

import main


def setup_module():
    main.load_artifacts()
    main.state["ratings_by_user"], main.state["rated_items"] = main.load_ratings_data()


client = TestClient(main.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["backend"] == "running"
    assert data["models_loaded"] is True
    assert data["error"] is None


def test_recommendations_known_user():
    response = client.get("/recommendations/1")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert data["known_user"] is True
    assert len(data["recommendations"]) == 10
    for item in data["recommendations"]:
        assert "movie_id" in item
        assert "title" in item
        assert "genres" in item
        assert "score" in item


def test_recommendations_unknown_user():
    response = client.get("/recommendations/999999")
    assert response.status_code == 200
    data = response.json()
    assert data["known_user"] is False
    assert len(data["recommendations"]) == 10


def test_popular():
    response = client.get("/recommendations/popular")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 10
    assert all("score" in item for item in data)


def test_movies_search_and_genre():
    response = client.get("/movies", params={"search": "Toy Story", "limit": 5})
    assert response.status_code == 200
    data = response.json()
    assert any("Toy Story" in item["title"] for item in data)

    response = client.get("/movies", params={"genre": "Comedy", "limit": 5})
    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 5
    assert all("Comedy" in item["genres"] for item in data)


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["relevance_threshold"] == 4.0
    assert data["top_k"] == 10
    for key in ("popularity", "collaborative_filtering", "content_based", "hybrid"):
        assert key in data["models"]
        assert set(data["models"][key]) == {"rmse", "precision_at_10", "recall_at_10"}
    assert len(data["model_list"]) == 4
    assert all(
        isinstance(item["rmse"], (int, float))
        and isinstance(item["precision_at_10"], (int, float))
        and isinstance(item["recall_at_10"], (int, float))
        for item in data["model_list"]
    )


def test_diverse_recommendations():
    response = client.get("/recommendations/diverse/1", params={"diversity": 0.5, "limit": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert data["known_user"] is True
    assert data["diversity"] == 0.5
    assert len(data["recommendations"]) == 5
    for item in data["recommendations"]:
        assert "movie_id" in item
        assert "title" in item
        assert "genres" in item
        assert "score" in item
        assert "relevance_score" in item
        assert "mmr_score" in item
        assert "explanation" in item


def test_diverse_parameter_validation():
    assert client.get("/recommendations/diverse/1", params={"diversity": 1.5}).status_code == 422
    assert client.get("/recommendations/diverse/1", params={"diversity": -0.1}).status_code == 422
    assert client.get("/recommendations/diverse/1", params={"limit": 0}).status_code == 422
    assert client.get("/recommendations/diverse/1", params={"limit": 51}).status_code == 422


def test_diverse_excludes_rated_movies():
    rated = main.state["rated_items"][1]
    response = client.get("/recommendations/diverse/1", params={"diversity": 0.5, "limit": 10})
    assert response.status_code == 200
    returned = {item["movie_id"] for item in response.json()["recommendations"]}
    assert returned.isdisjoint(rated)


def test_diverse_zero_matches_relevance_order():
    diverse = client.get("/recommendations/diverse/1", params={"diversity": 0.0, "limit": 5}).json()
    normal = client.get("/recommendations/1", params={"limit": 5}).json()
    assert [item["movie_id"] for item in diverse["recommendations"]] == [
        item["movie_id"] for item in normal["recommendations"]
    ]


def test_diverse_higher_diversity_not_less_genre_coverage():
    low = client.get("/recommendations/diverse/1", params={"diversity": 0.0, "limit": 8}).json()
    high = client.get("/recommendations/diverse/1", params={"diversity": 1.0, "limit": 8}).json()

    def genre_count(payload):
        return len({genre for item in payload["recommendations"] for genre in item["genres"].split("|") if genre})

    assert genre_count(high) >= genre_count(low)


def test_missing_artifacts():
    original_models = main.state["models"]
    original_error = main.state["error"]
    main.state["models"] = None
    main.state["error"] = "Missing model artifacts: hybrid_model.joblib"
    try:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["models_loaded"] is False

        response = client.get("/recommendations/1")
        assert response.status_code == 503

        response = client.get("/recommendations/popular")
        assert response.status_code == 503

        response = client.get("/recommendations/diverse/1")
        assert response.status_code == 503

        response = client.get("/movies")
        assert response.status_code == 503
    finally:
        main.state["models"] = original_models
        main.state["error"] = original_error
