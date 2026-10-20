import json
import os
import time
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from database import SessionLocal, TmdbCache

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
TMDB_BASE_URL = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")
TMDB_IMAGE_BASE = os.getenv("TMDB_IMAGE_BASE", "https://image.tmdb.org/t/p/w500")
CACHE_TTL_SECONDS = int(os.getenv("TMDB_CACHE_TTL", "3600"))

SUPPORTED_LANGUAGES = {
    "en": "English",
    "ml": "Malayalam",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ko": "Korean",
    "ja": "Japanese",
}


class TmdbError(Exception):
    pass


def _cache_get(key: str) -> Any | None:
    try:
        db = SessionLocal()
        row = db.get(TmdbCache, key)
        if row is None:
            db.close()
            return None
        if row.expires_at < datetime.utcnow():
            db.delete(row)
            db.commit()
            db.close()
            return None
        payload = json.loads(row.payload)
        db.close()
        return payload
    except Exception:
        return None


def _cache_set(key: str, payload: Any, ttl: int = CACHE_TTL_SECONDS) -> None:
    try:
        db = SessionLocal()
        expires = datetime.utcnow() + timedelta(seconds=ttl)
        row = db.get(TmdbCache, key)
        if row is None:
            row = TmdbCache(cache_key=key, payload=json.dumps(payload), expires_at=expires)
            db.add(row)
        else:
            row.payload = json.dumps(payload)
            row.expires_at = expires
        db.commit()
        db.close()
    except Exception:
        pass


def _request(path: str, params: dict | None = None) -> dict:
    if not TMDB_API_KEY:
        raise TmdbError("TMDB_API_KEY is not configured.")
    url = f"{TMDB_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    query = {"api_key": TMDB_API_KEY, "language": "en-US"}
    if params:
        query.update(params)
    last_exc = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url, params=query)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "1"))
                time.sleep(min(retry_after, 5))
                continue
            if resp.status_code >= 400:
                raise TmdbError(f"TMDB request failed: {resp.status_code} {resp.text[:200]}")
            return resp.json()
        except httpx.HTTPError as exc:
            last_exc = exc
            time.sleep(0.5 * (attempt + 1))
    raise TmdbError(f"TMDB network error: {last_exc}")


def _poster_url(path: str | None) -> str | None:
    if not path:
        return None
    if path.startswith("http"):
        return path
    return f"{TMDB_IMAGE_BASE}{path}"


def _normalize_movie(raw: dict) -> dict:
    genre_ids = raw.get("genre_ids") or []
    genres = raw.get("genres")
    if genres and isinstance(genres, list) and genres and isinstance(genres[0], dict):
        genre_names = [g.get("name") for g in genres if g.get("name")]
    else:
        genre_names = []
    return {
        "source": "tmdb",
        "external_id": str(raw.get("id")),
        "title": raw.get("title") or raw.get("name") or "Untitled",
        "overview": raw.get("overview") or "",
        "poster_url": _poster_url(raw.get("poster_path")),
        "backdrop_url": _poster_url(raw.get("backdrop_path")),
        "release_date": raw.get("release_date") or raw.get("first_air_date"),
        "original_language": raw.get("original_language"),
        "genres": genre_names if genre_names else [],
        "genre_ids": genre_ids,
        "vote_average": raw.get("vote_average"),
        "popularity": raw.get("popularity"),
        "vote_count": raw.get("vote_count"),
    }


def search_movies(query: str, *, language: str | None = None, genre: int | None = None,
                 year: int | None = None, page: int = 1) -> dict:
    cache_key = f"search:{query}:{language}:{genre}:{year}:{page}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    params: dict[str, Any] = {"query": query, "page": page, "include_adult": False}
    if language:
        params["with_original_language"] = language
    if year:
        params["primary_release_year"] = year
    if genre:
        params["with_genres"] = genre
    data = _request("/search/movie", params)
    result = {
        "page": data.get("page", page),
        "total_pages": data.get("total_pages", 1),
        "total_results": data.get("total_results", 0),
        "results": [_normalize_movie(m) for m in data.get("results", [])],
    }
    _cache_set(cache_key, result)
    return result


def popular_movies(*, language: str | None = None, page: int = 1) -> dict:
    cache_key = f"popular:{language}:{page}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    params: dict[str, Any] = {"page": page}
    if language:
        params["with_original_language"] = language
    data = _request("/movie/popular", params)
    result = {
        "page": data.get("page", page),
        "total_pages": data.get("total_pages", 1),
        "total_results": data.get("total_results", 0),
        "results": [_normalize_movie(m) for m in data.get("results", [])],
    }
    _cache_set(cache_key, result)
    return result


def trending_movies(*, time_window: str = "week", page: int = 1) -> dict:
    cache_key = f"trending:{time_window}:{page}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    data = _request(f"/trending/movie/{time_window}", {"page": page})
    result = {
        "page": data.get("page", page),
        "total_pages": data.get("total_pages", 1),
        "total_results": data.get("total_results", 0),
        "results": [_normalize_movie(m) for m in data.get("results", [])],
    }
    _cache_set(cache_key, result)
    return result


def movie_details(tmdb_id: int) -> dict:
    cache_key = f"movie:{tmdb_id}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    data = _request(f"/movie/{tmdb_id}")
    result = _normalize_movie(data)
    _cache_set(cache_key, result, ttl=86400)
    return result


def discover_movies(*, language: str | None = None, genre: int | None = None,
                    year: int | None = None, page: int = 1, sort_by: str = "popularity.desc") -> dict:
    cache_key = f"discover:{language}:{genre}:{year}:{page}:{sort_by}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    params: dict[str, Any] = {"page": page, "sort_by": sort_by, "include_adult": False}
    if language:
        params["with_original_language"] = language
    if genre:
        params["with_genres"] = genre
    if year:
        params["primary_release_year"] = year
    data = _request("/discover/movie", params)
    result = {
        "page": data.get("page", page),
        "total_pages": data.get("total_pages", 1),
        "total_results": data.get("total_results", 0),
        "results": [_normalize_movie(m) for m in data.get("results", [])],
    }
    _cache_set(cache_key, result)
    return result


GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
}
