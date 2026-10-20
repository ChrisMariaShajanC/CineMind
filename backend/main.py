import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
DATA_DIR = BACKEND_DIR / "data" / "raw" / "ml-1m"

CORS_ORIGINS = [
    "http://localhost:8443",
    "http://127.0.0.1:8443",
]

state: dict[str, Any] = {
    "models": None,
    "ratings_by_user": {},
    "rated_items": {},
    "error": None,
}


def load_ratings_data() -> tuple[dict[int, pd.DataFrame], dict[int, set[int]]]:
    ratings_by_user: dict[int, pd.DataFrame] = {}
    rated_items: dict[int, set[int]] = {}
    try:
        from src.data.load_dataset import load_ratings

        ratings = load_ratings()
        for uid, group in ratings.groupby("UserID"):
            uid = int(uid)
            ratings_by_user[uid] = group[["MovieID", "Rating"]].copy()
            rated_items[uid] = set(int(x) for x in group["MovieID"].tolist())
    except Exception as exc:  # noqa: BLE001
        state["error"] = f"Ratings data unavailable: {exc}"
    return ratings_by_user, rated_items


def load_artifacts() -> None:
    state["error"] = None
    required = ["hybrid_model.joblib", "movie_metadata.joblib"]
    missing = [name for name in required if not (MODELS_DIR / name).is_file()]
    if missing:
        state["models"] = None
        state["error"] = "Missing model artifacts: " + ", ".join(missing)
        return
    try:
        hybrid = joblib.load(MODELS_DIR / "hybrid_model.joblib")
        metadata = joblib.load(MODELS_DIR / "movie_metadata.joblib")
        popularity = None
        popularity_path = MODELS_DIR / "popularity_model.joblib"
        if popularity_path.is_file():
            popularity = joblib.load(popularity_path)
        state["models"] = {
            "hybrid": hybrid,
            "metadata": metadata,
            "popularity": popularity,
        }
    except Exception as exc:  # noqa: BLE001
        state["models"] = None
        state["error"] = f"Failed to load model artifacts: {exc}"


def metadata_lookup(metadata: pd.DataFrame) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for _, row in metadata.iterrows():
        result[int(row["MovieID"])] = {
            "movie_id": int(row["MovieID"]),
            "title": str(row["Title"]),
            "genres": str(row["Genres"]),
        }
    return result


def content_user_profile(genre_matrix, movie_to_idx, user_ratings, global_mean):
    if user_ratings is None or user_ratings.empty:
        return None
    vec = np.zeros(genre_matrix.shape[1], dtype=float)
    weight_sum = 0.0
    for _, row in user_ratings.iterrows():
        idx = movie_to_idx.get(int(row["MovieID"]))
        if idx is None:
            continue
        weight = float(row["Rating"]) - global_mean
        vec += weight * genre_matrix[idx].toarray().ravel()
        weight_sum += abs(weight)
    if weight_sum == 0:
        return None
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return vec / norm


def recommend_for_user(user_id: int, top_k: int = 10) -> tuple[list[int], list[float]]:
    models = state["models"]
    if not models:
        raise RuntimeError("Models are not loaded.")
    hybrid = models["hybrid"]
    movie_ids = hybrid["movie_ids"]
    user_to_idx = hybrid["user_to_idx"]
    movie_to_idx = hybrid["movie_to_idx"]
    user_factors = hybrid["user_factors"]
    movie_factors = hybrid["movie_factors"]
    genre_matrix = hybrid["genre_matrix"]
    cf_global = hybrid["global_mean"]
    content_global = hybrid.get("pop_global_mean", cf_global)
    rated = state["rated_items"].get(user_id, set())
    candidates = [mid for mid in movie_ids if mid not in rated]
    if not candidates:
        candidates = list(movie_ids)

    user_ratings = state["ratings_by_user"].get(user_id)
    u = user_to_idx.get(user_id)
    if u is not None:
        all_cf = cf_global + user_factors[u] @ movie_factors.T
        cf_vals = np.array([float(all_cf[movie_to_idx[mid]]) for mid in candidates], dtype=float)
    else:
        cf_vals = np.full(len(candidates), cf_global, dtype=float)

    profile = content_user_profile(genre_matrix, movie_to_idx, user_ratings if user_ratings is not None else pd.DataFrame(), content_global)
    if profile is not None:
        content_arr = np.asarray(genre_matrix @ profile).ravel()
        ct_vals = np.array([float(content_arr[movie_to_idx[mid]]) for mid in candidates], dtype=float)
    else:
        ct_vals = np.zeros(len(candidates), dtype=float)

    cf_norm = (cf_vals - cf_vals.min()) / (cf_vals.max() - cf_vals.min() + 1e-12)
    ct_norm = (ct_vals - ct_vals.min()) / (ct_vals.max() - ct_vals.min() + 1e-12)
    scores = hybrid["cf_weight"] * cf_norm + hybrid["content_weight"] * ct_norm
    order = np.argsort(scores)[::-1][:top_k]
    rec_ids = [candidates[i] for i in order]
    rec_scores = [float(scores[i]) for i in order]
    return rec_ids, rec_scores


def _movie_genres(metadata: pd.DataFrame, movie_id: int) -> list[str]:
    rows = metadata[metadata["MovieID"] == movie_id]
    if rows.empty:
        return []
    return [genre for genre in str(rows.iloc[0]["Genres"]).split("|") if genre]


def candidate_scores(user_id: int) -> list[tuple[int, float]]:
    models = state["models"]
    if not models:
        raise RuntimeError("Models are not loaded.")
    hybrid = models["hybrid"]
    metadata = models["metadata"]
    movie_ids = hybrid["movie_ids"]
    user_to_idx = hybrid["user_to_idx"]
    movie_to_idx = hybrid["movie_to_idx"]
    user_factors = hybrid["user_factors"]
    movie_factors = hybrid["movie_factors"]
    genre_matrix = hybrid["genre_matrix"]
    cf_global = hybrid["global_mean"]
    content_global = hybrid.get("pop_global_mean", cf_global)
    rated = state["rated_items"].get(user_id, set()) if user_id in user_to_idx else set()
    candidates = [mid for mid in movie_ids if mid not in rated]
    if not candidates:
        candidates = list(movie_ids)
    if not candidates:
        return []

    user_ratings = state["ratings_by_user"].get(user_id)
    u = user_to_idx.get(user_id)
    if u is not None:
        all_cf = cf_global + user_factors[u] @ movie_factors.T
        cf_vals = np.array([float(all_cf[movie_to_idx[mid]]) for mid in candidates], dtype=float)
    else:
        cf_vals = np.full(len(candidates), cf_global, dtype=float)

    profile = content_user_profile(
        genre_matrix,
        movie_to_idx,
        user_ratings if user_ratings is not None else pd.DataFrame(),
        content_global,
    )
    if profile is not None:
        content_arr = np.asarray(genre_matrix @ profile).ravel()
        ct_vals = np.array([float(content_arr[movie_to_idx[mid]]) for mid in candidates], dtype=float)
    else:
        ct_vals = np.zeros(len(candidates), dtype=float)

    cf_norm = (cf_vals - cf_vals.min()) / (cf_vals.max() - cf_vals.min() + 1e-12)
    ct_norm = (ct_vals - ct_vals.min()) / (ct_vals.max() - ct_vals.min() + 1e-12)
    scores = hybrid["cf_weight"] * cf_norm + hybrid["content_weight"] * ct_norm
    return [(mid, float(score)) for mid, score in zip(candidates, scores)]


def diverse_recommendations(user_id: int, diversity: float, limit: int) -> list[dict[str, Any]]:
    scored = candidate_scores(user_id)
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[1], item[0]))
    pool_size = min(len(scored), max(limit * 5, 50))
    pool = scored[:pool_size]
    metadata = state["models"]["metadata"]
    genre_sets = {mid: set(_movie_genres(metadata, mid)) for mid, _ in pool}

    def genre_similarity(left: int, right: int) -> float:
        left_genres = genre_sets.get(left, set())
        right_genres = genre_sets.get(right, set())
        union = left_genres | right_genres
        if not union:
            return 0.0
        return len(left_genres & right_genres) / len(union)

    selected_ids: list[int] = []
    selected: list[dict[str, Any]] = []
    remaining = list(pool)
    while len(selected) < limit and remaining:
        best_index = 0
        best_mmr = None
        best_similarity = 0.0
        for index, (movie_id, relevance) in enumerate(remaining):
            if not selected_ids:
                mmr = (1.0 - diversity) * relevance
                max_similarity = 0.0
            else:
                similarities = [genre_similarity(movie_id, selected_id) for selected_id in selected_ids]
                max_similarity = max(similarities) if similarities else 0.0
                mmr = (1.0 - diversity) * relevance - diversity * max_similarity
            if (
                best_mmr is None
                or mmr > best_mmr
                or (mmr == best_mmr and movie_id < remaining[best_index][0])
            ):
                best_index = index
                best_mmr = mmr
                best_similarity = max_similarity
        movie_id, relevance = remaining.pop(best_index)
        if diversity > 0 and best_similarity > 0:
            most_similar = max(selected_ids, key=lambda selected_id: genre_similarity(movie_id, selected_id))
            explanation = (
                f"MMR balances relevance ({relevance:.3f}) against genre overlap "
                f"({best_similarity:.2f}) with movie {most_similar}."
            )
        else:
            explanation = "Selected by relevance; no diversity penalty applied."
        selected_ids.append(movie_id)
        selected.append(
            {
                "movie_id": movie_id,
                "relevance_score": relevance,
                "mmr_score": float(best_mmr if best_mmr is not None else relevance),
                "explanation": explanation,
            }
        )
    return selected


def popular_movies(top_k: int = 10) -> tuple[list[int], list[float]]:
    models = state["models"]
    if not models:
        raise RuntimeError("Models are not loaded.")
    hybrid = models["hybrid"]
    pop_stats = hybrid.get("pop_stats")
    if pop_stats is None and models.get("popularity"):
        pop_stats = models["popularity"].get("pop_stats")
    if pop_stats is None:
        movie_ids = hybrid["movie_ids"][:top_k]
        return movie_ids, [0.0] * len(movie_ids)
    stats = pop_stats.sort_values(["rating_mean", "rating_count"], ascending=[False, False])
    ids = stats["MovieID"].head(top_k).tolist()
    scores = [float(x) for x in stats["rating_mean"].head(top_k).tolist()]
    return [int(x) for x in ids], scores


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    state["ratings_by_user"], state["rated_items"] = load_ratings_data()
    yield


app = FastAPI(title="CineMind Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = state["models"] is not None
    return {
        "status": "healthy" if loaded else "degraded",
        "backend": "running",
        "models_loaded": loaded,
        "error": state["error"],
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    if not METRICS_PATH.is_file():
        raise HTTPException(status_code=404, detail="Evaluation metrics are not available.")
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
            return json.load(metrics_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation metrics: {exc}") from exc


@app.get("/movies")
def list_movies(
    search: str | None = Query(default=None),
    genre: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict[str, Any]]:
    models = state["models"]
    if not models:
        raise HTTPException(status_code=503, detail=state["error"] or "Models not loaded.")
    metadata = models["metadata"]
    df = metadata.copy()
    if search:
        df = df[df["Title"].astype(str).str.contains(search, case=False, na=False)]
    if genre:
        df = df[df["Genres"].astype(str).str.contains(genre, case=False, na=False)]
    df = df.head(limit)
    lookup = metadata_lookup(df)
    return list(lookup.values())


@app.get("/recommendations/popular")
def recommendations_popular(limit: int = Query(default=10, ge=1, le=50)) -> list[dict[str, Any]]:
    models = state["models"]
    if not models:
        raise HTTPException(status_code=503, detail=state["error"] or "Models not loaded.")
    metadata = models["metadata"]
    lookup = metadata_lookup(metadata)
    ids, scores = popular_movies(limit)
    result = []
    for mid, score in zip(ids, scores):
        info = lookup.get(mid, {"movie_id": mid, "title": None, "genres": None})
        result.append({**info, "score": score})
    return result


@app.get("/recommendations/{user_id}")
def recommendations(user_id: int, limit: int = Query(default=10, ge=1, le=50)) -> dict[str, Any]:
    models = state["models"]
    if not models:
        raise HTTPException(status_code=503, detail=state["error"] or "Models not loaded.")
    metadata = models["metadata"]
    lookup = metadata_lookup(metadata)
    hybrid = models["hybrid"]
    user_to_idx = hybrid["user_to_idx"]
    known = user_id in user_to_idx
    try:
        ids, scores = recommend_for_user(user_id, limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}") from exc
    items = []
    for mid, score in zip(ids, scores):
        info = lookup.get(mid, {"movie_id": mid, "title": None, "genres": None})
        items.append({**info, "score": score})
    return {
        "user_id": user_id,
        "known_user": known,
        "recommendations": items,
    }


@app.get("/recommendations/diverse/{user_id}")
def recommendations_diverse(
    user_id: int,
    diversity: float = Query(default=0.5, ge=0.0, le=1.0),
    limit: int = Query(default=10, ge=1, le=50),
) -> dict[str, Any]:
    models = state["models"]
    if not models:
        raise HTTPException(status_code=503, detail=state["error"] or "Models not loaded.")
    metadata = models["metadata"]
    lookup = metadata_lookup(metadata)
    hybrid = models["hybrid"]
    user_to_idx = hybrid["user_to_idx"]
    known = user_id in user_to_idx
    try:
        diverse = diverse_recommendations(user_id, diversity, limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Diverse recommendation failed: {exc}") from exc
    items = []
    for item in diverse:
        movie_id = int(item["movie_id"])
        info = lookup.get(movie_id, {"movie_id": movie_id, "title": None, "genres": None})
        items.append(
            {
                **info,
                "score": item["relevance_score"],
                "relevance_score": item["relevance_score"],
                "mmr_score": item["mmr_score"],
                "explanation": item["explanation"],
            }
        )
    return {
        "user_id": user_id,
        "known_user": known,
        "diversity": diversity,
        "recommendations": items,
    }
