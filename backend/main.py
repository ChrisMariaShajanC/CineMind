import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

import auth
import database
import tmdb
from database import User, UserPreference, Favourite, WatchlistItem, UserRating, get_db, init_db

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


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: str
    password: str


class PreferenceRequest(BaseModel):
    languages: list[str] = []
    genres: list[str] = []
    moods: list[str] = []
    release_period: str = ""
    duration: str = ""
    onboarding_completed: bool = False


class CatalogueItemRequest(BaseModel):
    source: str = "tmdb"
    external_id: str
    title: str
    poster_url: str | None = None
    genres: list[str] = []


class RatingRequest(BaseModel):
    source: str = "tmdb"
    external_id: str
    rating: float = Field(ge=0.5, le=5.0)


bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    user_id = auth.decode_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    user = db.get(User, int(user_id))
    if user is None:
        raise HTTPException(status_code=401, detail="User not found.")
    return user


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
    init_db()
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

# --- Authentication ---


@app.post("/auth/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered.")
    user = User(name=payload.name.strip(), email=email, password_hash=auth.hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    prefs = UserPreference(user_id=user.id)
    db.add(prefs)
    db.commit()
    token = auth.create_access_token(str(user.id))
    return {"access_token": token, "token_type": "bearer", "user": {"id": user.id, "name": user.name, "email": user.email}}


@app.post("/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if user is None or not auth.verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = auth.create_access_token(str(user.id))
    return {"access_token": token, "token_type": "bearer", "user": {"id": user.id, "name": user.name, "email": user.email}}


@app.post("/auth/logout")
def logout():
    return {"ok": True}


@app.get("/auth/me")
def me(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "name": current_user.name, "email": current_user.email, "created_at": current_user.created_at.isoformat()}


# --- Preferences ---


@app.get("/preferences")
def get_preferences(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    prefs = current_user.preferences
    if prefs is None:
        prefs = UserPreference(user_id=current_user.id)
        db.add(prefs)
        db.commit()
        db.refresh(prefs)
    return prefs.as_dict()


@app.put("/preferences")
def update_preferences(payload: PreferenceRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    prefs = current_user.preferences
    if prefs is None:
        prefs = UserPreference(user_id=current_user.id)
        db.add(prefs)
    prefs.languages = json.dumps(payload.languages)
    prefs.genres = json.dumps(payload.genres)
    prefs.moods = json.dumps(payload.moods)
    prefs.release_period = payload.release_period
    prefs.duration = payload.duration
    prefs.onboarding_completed = 1 if payload.onboarding_completed else prefs.onboarding_completed
    db.commit()
    db.refresh(prefs)
    return prefs.as_dict()


# --- Favourites ---


@app.get("/me/favourites")
def list_favourites(current_user: User = Depends(get_current_user)):
    return [{"id": f.id, "source": f.source, "external_id": f.external_id, "title": f.title, "poster_url": f.poster_url, "genres": json.loads(f.genres or "[]"), "created_at": f.created_at.isoformat()} for f in current_user.favourites]


@app.post("/me/favourites")
def add_favourite(payload: CatalogueItemRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    existing = db.query(Favourite).filter(Favourite.user_id == current_user.id, Favourite.source == payload.source, Favourite.external_id == payload.external_id).first()
    if existing:
        return {"ok": True, "id": existing.id}
    fav = Favourite(user_id=current_user.id, source=payload.source, external_id=payload.external_id, title=payload.title, poster_url=payload.poster_url, genres=json.dumps(payload.genres))
    db.add(fav)
    db.commit()
    db.refresh(fav)
    return {"ok": True, "id": fav.id}


@app.delete("/me/favourites/{source}/{external_id}")
def remove_favourite(source: str, external_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    fav = db.query(Favourite).filter(Favourite.user_id == current_user.id, Favourite.source == source, Favourite.external_id == external_id).first()
    if fav:
        db.delete(fav)
        db.commit()
    return {"ok": True}


# --- Watchlist ---


@app.get("/me/watchlist")
def list_watchlist(current_user: User = Depends(get_current_user)):
    return [{"id": w.id, "source": w.source, "external_id": w.external_id, "title": w.title, "poster_url": w.poster_url, "genres": json.loads(w.genres or "[]"), "created_at": w.created_at.isoformat()} for w in current_user.watchlist]


@app.post("/me/watchlist")
def add_watchlist(payload: CatalogueItemRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    existing = db.query(WatchlistItem).filter(WatchlistItem.user_id == current_user.id, WatchlistItem.source == payload.source, WatchlistItem.external_id == payload.external_id).first()
    if existing:
        return {"ok": True, "id": existing.id}
    item = WatchlistItem(user_id=current_user.id, source=payload.source, external_id=payload.external_id, title=payload.title, poster_url=payload.poster_url, genres=json.dumps(payload.genres))
    db.add(item)
    db.commit()
    db.refresh(item)
    return {"ok": True, "id": item.id}


@app.delete("/me/watchlist/{source}/{external_id}")
def remove_watchlist(source: str, external_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    item = db.query(WatchlistItem).filter(WatchlistItem.user_id == current_user.id, WatchlistItem.source == source, WatchlistItem.external_id == external_id).first()
    if item:
        db.delete(item)
        db.commit()
    return {"ok": True}


# --- User ratings on catalogue movies ---


@app.post("/me/ratings")
def add_user_rating_catalogue(payload: RatingRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    existing = db.query(UserRating).filter(UserRating.user_id == current_user.id, UserRating.source == payload.source, UserRating.external_id == payload.external_id).first()
    if existing:
        existing.rating = payload.rating
    else:
        db.add(UserRating(user_id=current_user.id, source=payload.source, external_id=payload.external_id, rating=payload.rating))
    db.commit()
    return {"ok": True}


@app.get("/me/ratings")
def list_user_ratings(current_user: User = Depends(get_current_user)):
    return [{"source": r.source, "external_id": r.external_id, "rating": r.rating, "created_at": r.created_at.isoformat()} for r in current_user.ratings]


# --- TMDB catalogue endpoints ---


def _tmdb_guard():
    if not tmdb.TMDB_API_KEY:
        raise HTTPException(status_code=503, detail="TMDB_API_KEY is not configured.")


@app.get("/catalogue/search")
def catalogue_search(q: str = Query(..., min_length=1), language: str | None = None, genre: int | None = None, year: int | None = None, page: int = Query(default=1, ge=1)):
    _tmdb_guard()
    try:
        return tmdb.search_movies(q, language=language, genre=genre, year=year, page=page)
    except tmdb.TmdbError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/catalogue/popular")
def catalogue_popular(language: str | None = None, page: int = Query(default=1, ge=1)):
    _tmdb_guard()
    try:
        return tmdb.popular_movies(language=language, page=page)
    except tmdb.TmdbError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/catalogue/trending")
def catalogue_trending(time_window: str = Query(default="week", pattern="^(day|week)$"), page: int = Query(default=1, ge=1)):
    _tmdb_guard()
    try:
        return tmdb.trending_movies(time_window=time_window, page=page)
    except tmdb.TmdbError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/catalogue/discover")
def catalogue_discover(language: str | None = None, genre: int | None = None, year: int | None = None, page: int = Query(default=1, ge=1)):
    _tmdb_guard()
    try:
        return tmdb.discover_movies(language=language, genre=genre, year=year, page=page)
    except tmdb.TmdbError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/catalogue/movie/{external_id}")
def catalogue_movie(external_id: str):
    _tmdb_guard()
    try:
        return tmdb.movie_details(int(external_id))
    except (ValueError, tmdb.TmdbError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/catalogue/genres")
def catalogue_genres():
    return tmdb.GENRE_MAP


@app.get("/catalogue/languages")
def catalogue_languages():
    return tmdb.SUPPORTED_LANGUAGES


# --- Cold-start personalized recommendations ---


def _score_tmdb_movie(movie: dict, prefs: dict, fav_genres: set[str], rated_ids: set[str]) -> float:
    score = 0.0
    langs = set(prefs.get("languages", []))
    genres = set(prefs.get("genres", []))
    moods = set(prefs.get("moods", []))
    if langs and movie.get("original_language") in langs:
        score += 2.0
    movie_genres = set(movie.get("genres") or [])
    score += len(genres & movie_genres) * 1.5
    score += len(fav_genres & movie_genres) * 1.0
    va = movie.get("vote_average")
    if va is not None:
        score += float(va) / 2.0
    pop = movie.get("popularity")
    if pop is not None:
        score += min(float(pop), 100.0) / 100.0
    if movie.get("poster_url"):
        score += 0.2
    return score


@app.get("/me/recommendations")
def personalized_recommendations(limit: int = Query(default=10, ge=1, le=50), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    prefs = current_user.preferences
    pref_dict = prefs.as_dict() if prefs else {"languages": [], "genres": [], "moods": [], "release_period": "", "duration": "", "onboarding_completed": False}
    fav_genres: set[str] = set()
    for fav in current_user.favourites:
        try:
            fav_genres |= set(json.loads(fav.genres or "[]"))
        except Exception:
            pass
    rated_ids = {r.external_id for r in current_user.ratings}

    candidates: list[dict] = []
    if tmdb.TMDB_API_KEY:
        try:
            for lang in (pref_dict.get("languages") or [None]):
                data = tmdb.popular_movies(language=lang, page=1) if lang else tmdb.popular_movies(page=1)
                candidates.extend(data.get("results", []))
            data = tmdb.trending_movies(page=1)
            candidates.extend(data.get("results", []))
        except tmdb.TmdbError:
            pass

    seen: set[str] = set()
    unique: list[dict] = []
    for m in candidates:
        eid = m.get("external_id")
        if eid in seen or eid in rated_ids:
            continue
        seen.add(eid)
        unique.append(m)

    scored = sorted(unique, key=lambda m: _score_tmdb_movie(m, pref_dict, fav_genres, rated_ids), reverse=True)
    pool = scored[: max(limit * 5, 50)]

    def genre_sim(a: dict, b: dict) -> float:
        ga = set(a.get("genres") or [])
        gb = set(b.get("genres") or [])
        union = ga | gb
        return len(ga & gb) / len(union) if union else 0.0

    selected: list[dict] = []
    remaining = list(pool)
    diversity = 0.5
    while len(selected) < limit and remaining:
        best_idx = 0
        best_mmr = None
        best_sim = 0.0
        for idx, m in enumerate(remaining):
            rel = _score_tmdb_movie(m, pref_dict, fav_genres, rated_ids)
            if not selected:
                mmr_val = (1.0 - diversity) * rel
                max_s = 0.0
            else:
                sims = [genre_sim(m, s) for s in selected]
                max_s = max(sims) if sims else 0.0
                mmr_val = (1.0 - diversity) * rel - diversity * max_s
            if best_mmr is None or mmr_val > best_mmr:
                best_idx = idx
                best_mmr = mmr_val
                best_sim = max_s
        m = remaining.pop(best_idx)
        selected.append(m)

    items = []
    for m in selected:
        items.append({
            "source": "tmdb",
            "external_id": m.get("external_id"),
            "movie_id": None,
            "title": m.get("title"),
            "genres": "|".join(m.get("genres") or []),
            "poster_url": m.get("poster_url"),
            "overview": m.get("overview"),
            "release_date": m.get("release_date"),
            "original_language": m.get("original_language"),
            "vote_average": m.get("vote_average"),
            "score": _score_tmdb_movie(m, pref_dict, fav_genres, rated_ids),
        })
    return {"user_id": current_user.id, "recommendations": items}
