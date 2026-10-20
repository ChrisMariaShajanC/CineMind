import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.load_dataset import load_movies, load_ratings, load_users

DATA_DIR = BACKEND_DIR / "data" / "raw" / "ml-1m"
MODELS_DIR = BACKEND_DIR / "models"
RELEVANCE_THRESHOLD = 4.0
TEST_RATIO = 0.2
MIN_TRAIN_RATINGS = 5
MIN_TOTAL_FOR_EVAL = 10
MIN_POPULARITY_COUNT = 50
N_COMPONENTS = 50
TOP_K = 10
RANDOM_STATE = 42
HYBRID_CF_WEIGHT = 0.6
HYBRID_CONTENT_WEIGHT = 0.4


def validate_data(movies, ratings, users):
    if movies.empty or ratings.empty or users.empty:
        raise ValueError("One or more datasets are empty.")
    for col in ("MovieID", "Title", "Genres"):
        if col not in movies.columns:
            raise ValueError(f"movies.dat missing column: {col}")
    for col in ("UserID", "MovieID", "Rating", "Timestamp"):
        if col not in ratings.columns:
            raise ValueError(f"ratings.dat missing column: {col}")
    for col in ("UserID", "Gender", "Age", "Occupation", "Zip-code"):
        if col not in users.columns:
            raise ValueError(f"users.dat missing column: {col}")
    ratings = ratings.dropna(subset=["UserID", "MovieID", "Rating"]).copy()
    ratings["UserID"] = ratings["UserID"].astype(int)
    ratings["MovieID"] = ratings["MovieID"].astype(int)
    ratings["Rating"] = ratings["Rating"].astype(float)
    movies = movies.dropna(subset=["MovieID", "Title"]).copy()
    movies["MovieID"] = movies["MovieID"].astype(int)
    movies["Genres"] = movies["Genres"].fillna("")
    users = users.dropna(subset=["UserID"]).copy()
    users["UserID"] = users["UserID"].astype(int)
    return movies, ratings, users


def per_user_split(ratings):
    train_parts = []
    test_parts = []
    for _, group in ratings.groupby("UserID", sort=False):
        group = group.sort_values("Timestamp", kind="mergesort")
        n = len(group)
        if n < MIN_TOTAL_FOR_EVAL:
            train_parts.append(group)
            continue
        n_test = max(1, int(round(n * TEST_RATIO)))
        if n - n_test < MIN_TRAIN_RATINGS:
            n_test = n - MIN_TRAIN_RATINGS
        if n_test < 1:
            train_parts.append(group)
            continue
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=ratings.columns)
    return train_df, test_df


def build_popularity(train_ratings):
    stats = train_ratings.groupby("MovieID")["Rating"].agg(["count", "mean"]).reset_index()
    stats = stats.rename(columns={"count": "rating_count", "mean": "rating_mean"})
    stats = stats[stats["rating_count"] >= MIN_POPULARITY_COUNT].copy()
    global_mean = float(train_ratings["Rating"].mean())
    return stats, global_mean


def build_content(movies):
    movies = movies.copy()
    movies["genre_text"] = movies["Genres"].fillna("").astype(str).str.replace("|", " ", regex=False)
    vectorizer = TfidfVectorizer()
    genre_matrix = vectorizer.fit_transform(movies["genre_text"])
    movie_cosine = cosine_similarity(genre_matrix).astype(np.float32)
    return vectorizer, genre_matrix, movie_cosine


def build_cf(train_ratings, movie_ids, user_ids):
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    rows = train_ratings["UserID"].map(user_to_idx)
    cols = train_ratings["MovieID"].map(movie_to_idx)
    mask = rows.notna() & cols.notna()
    rows = rows[mask].astype(int).to_numpy()
    cols = cols[mask].astype(int).to_numpy()
    data = train_ratings.loc[mask, "Rating"].to_numpy(dtype=float)
    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
    global_mean = float(train_ratings["Rating"].mean())
    centered = matrix.copy()
    centered.data = centered.data - global_mean
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    user_factors = svd.fit_transform(centered)
    movie_factors = svd.components_.T
    return svd, user_factors, movie_factors, user_to_idx, movie_to_idx, global_mean


def user_train_items(train_ratings):
    return train_ratings.groupby("UserID")["MovieID"].apply(set).to_dict()


def recommend_popularity(stats, global_mean, seen_items, n=TOP_K):
    candidates = stats[~stats["MovieID"].isin(seen_items)].copy()
    if candidates.empty:
        return []
    candidates = candidates.sort_values(["rating_mean", "rating_count"], ascending=[False, False])
    return candidates["MovieID"].head(n).tolist()


def content_user_profile(genre_matrix, movie_to_idx, user_ratings_df, global_mean):
    if user_ratings_df.empty:
        return None
    vec = np.zeros(genre_matrix.shape[1], dtype=float)
    weight_sum = 0.0
    for _, row in user_ratings_df.iterrows():
        mid = row["MovieID"]
        idx = movie_to_idx.get(mid)
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


def recommend_content(movies, genre_matrix, movie_to_idx, user_profile, seen_items, n=TOP_K):
    if user_profile is None:
        return []
    scores = genre_matrix @ user_profile
    scores = np.asarray(scores).ravel()
    order = np.argsort(scores)[::-1]
    result = []
    for idx in order:
        mid = int(movies.iloc[idx]["MovieID"])
        if mid in seen_items:
            continue
        result.append(mid)
        if len(result) >= n:
            break
    return result


def predict_content(user_ratings_df, movie_id, movie_cosine, movie_to_idx, global_mean):
    idx = movie_to_idx.get(movie_id)
    if idx is None or user_ratings_df.empty:
        return global_mean
    pairs = []
    for _, row in user_ratings_df.iterrows():
        ridx = movie_to_idx.get(int(row["MovieID"]))
        if ridx is not None:
            pairs.append((ridx, float(row["Rating"])))
    if not pairs:
        return global_mean
    rated_indices = [p[0] for p in pairs]
    ratings = np.array([p[1] for p in pairs], dtype=float)
    sims = movie_cosine[idx, rated_indices]
    denom = np.sum(np.abs(sims))
    if denom == 0:
        return float(user_ratings_df["Rating"].mean())
    return float(np.sum(sims * ratings) / denom)


def predict_cf(user_id, movie_id, user_to_idx, movie_to_idx, user_factors, movie_factors, global_mean):
    u = user_to_idx.get(user_id)
    m = movie_to_idx.get(movie_id)
    if u is None or m is None:
        return global_mean
    pred = global_mean + float(user_factors[u] @ movie_factors[m])
    return float(np.clip(pred, 1.0, 5.0))


def predict_popularity(movie_id, pop_stats, global_mean):
    row = pop_stats[pop_stats["MovieID"] == movie_id]
    if row.empty:
        return global_mean
    return float(row.iloc[0]["rating_mean"])


def predict_hybrid(user_id, movie_id, user_ratings_df, movie_cosine, movie_to_idx,
                   user_to_idx, movie_factors, user_factors, pop_stats, global_mean, content_global_mean):
    cf_pred = predict_cf(user_id, movie_id, user_to_idx, movie_to_idx, user_factors, movie_factors, global_mean)
    content_pred = predict_content(user_ratings_df, movie_id, movie_cosine, movie_to_idx, content_global_mean)
    # weighted combination of available predictions
    has_cf = user_to_idx.get(user_id) is not None and movie_to_idx.get(movie_id) is not None
    has_content = not user_ratings_df.empty
    if has_cf and has_content:
        return HYBRID_CF_WEIGHT * cf_pred + HYBRID_CONTENT_WEIGHT * content_pred
    if has_cf:
        return cf_pred
    if has_content:
        return content_pred
    return predict_popularity(movie_id, pop_stats, global_mean)


def recommend_cf(user_id, movie_ids, user_to_idx, movie_to_idx, user_factors, movie_factors, global_mean, seen_items, n=TOP_K):
    u = user_to_idx.get(user_id)
    if u is None:
        return []
    scores = global_mean + user_factors[u] @ movie_factors.T
    order = np.argsort(scores)[::-1]
    result = []
    for idx in order:
        mid = int(movie_ids[idx])
        if mid in seen_items:
            continue
        result.append(mid)
        if len(result) >= n:
            break
    return result


def recommend_hybrid(movies, user_id, user_ratings_df, train_user_items, stats, global_mean,
                     genre_matrix, movie_to_idx, user_profile, movie_ids, user_to_idx, user_factors,
                     movie_factors, movie_cosine, n=TOP_K):
    seen_items = train_user_items.get(user_id, set())
    candidates = [mid for mid in movie_ids if mid not in seen_items]
    if not candidates:
        return []
    # CF scores
    u = user_to_idx.get(user_id)
    if u is not None:
        all_cf = global_mean + user_factors[u] @ movie_factors.T
        cf_scores = {mid: float(all_cf[movie_to_idx[mid]]) for mid in candidates}
    else:
        cf_scores = {mid: cf_global_mean for mid in candidates}
    # content scores
    if user_profile is not None:
        content_scores_arr = np.asarray(genre_matrix @ user_profile).ravel()
        ct_vals = np.array([float(content_scores_arr[movie_to_idx[mid]]) for mid in candidates], dtype=float)
    else:
        ct_vals = np.zeros(len(candidates), dtype=float)
    cf_vals = np.array([cf_scores[mid] for mid in candidates], dtype=float)
    cf_norm = (cf_vals - cf_vals.min()) / (cf_vals.max() - cf_vals.min() + 1e-12)
    ct_norm = (ct_vals - ct_vals.min()) / (ct_vals.max() - ct_vals.min() + 1e-12)
    hybrid = HYBRID_CF_WEIGHT * cf_norm + HYBRID_CONTENT_WEIGHT * ct_norm
    order = np.argsort(hybrid)[::-1]
    return [candidates[i] for i in order[:n]]


def rmse_popularity(test_df, pop_stats, global_mean):
    pop_map = dict(zip(pop_stats["MovieID"], pop_stats["rating_mean"]))
    preds = np.array([pop_map.get(int(m), global_mean) for m in test_df["MovieID"]], dtype=float)
    actuals = test_df["Rating"].to_numpy(dtype=float)
    return float(np.sqrt(np.mean((preds - actuals) ** 2)))


def rmse_cf(test_df, user_to_idx, movie_to_idx, user_factors, movie_factors, global_mean):
    preds = np.full(len(test_df), global_mean, dtype=float)
    # group by user for vectorized prediction
    test_df = test_df.reset_index(drop=True)
    for uid, idx in test_df.groupby("UserID").groups.items():
        u = user_to_idx.get(int(uid))
        if u is None:
            continue
        mids = test_df.loc[idx, "MovieID"].to_numpy()
        midx = [movie_to_idx.get(int(m)) for m in mids]
        valid = [(i, mi) for i, mi in zip(idx, midx) if mi is not None]
        if not valid:
            continue
        ii = [v[0] for v in valid]
        mm = [v[1] for v in valid]
        preds[ii] = global_mean + user_factors[u] @ movie_factors[mm].T
    actuals = test_df["Rating"].to_numpy(dtype=float)
    preds = np.clip(preds, 1.0, 5.0)
    return float(np.sqrt(np.mean((preds - actuals) ** 2)))


def rmse_content(test_df, train_user_ratings, movie_cosine, movie_to_idx, global_mean):
    preds = np.full(len(test_df), global_mean, dtype=float)
    test_df = test_df.reset_index(drop=True)
    for uid, idx in test_df.groupby("UserID").groups.items():
        ur = train_user_ratings.get(int(uid))
        train_pairs = []
        if ur is not None and not ur.empty:
            for _, r in ur.iterrows():
                mi = movie_to_idx.get(int(r["MovieID"]))
                if mi is not None:
                    train_pairs.append((mi, float(r["Rating"])))
        if not train_pairs:
            continue
        train_indices = np.array([p[0] for p in train_pairs], dtype=int)
        train_ratings = np.array([p[1] for p in train_pairs], dtype=float)
        mids = test_df.loc[idx, "MovieID"].to_numpy()
        test_indices = [movie_to_idx.get(int(m)) for m in mids]
        valid = [(i, ti) for i, ti in zip(idx, test_indices) if ti is not None]
        if not valid:
            continue
        ii = [v[0] for v in valid]
        tt = [v[1] for v in valid]
        sim = movie_cosine[tt][:, train_indices]
        denom = np.sum(np.abs(sim), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pred_vals = np.where(denom > 0, (sim @ train_ratings) / np.where(denom > 0, denom, 1.0), train_ratings.mean())
        preds[ii] = pred_vals
    actuals = test_df["Rating"].to_numpy(dtype=float)
    preds = np.clip(preds, 1.0, 5.0)
    return float(np.sqrt(np.mean((preds - actuals) ** 2)))


def rmse_hybrid(test_df, train_user_ratings, movie_cosine, movie_to_idx, user_to_idx,
                user_factors, movie_factors, pop_stats, cf_global_mean, content_global_mean):
    preds = np.full(len(test_df), cf_global_mean, dtype=float)
    test_df = test_df.reset_index(drop=True)
    pop_map = dict(zip(pop_stats["MovieID"], pop_stats["rating_mean"]))
    for uid, idx in test_df.groupby("UserID").groups.items():
        ur = train_user_ratings.get(int(uid))
        train_pairs = []
        if ur is not None and not ur.empty:
            for _, r in ur.iterrows():
                mi = movie_to_idx.get(int(r["MovieID"]))
                if mi is not None:
                    train_pairs.append((mi, float(r["Rating"])))
        train_indices = np.array([p[0] for p in train_pairs], dtype=int) if train_pairs else np.array([], dtype=int)
        train_ratings = np.array([p[1] for p in train_pairs], dtype=float) if train_pairs else np.array([])
        u = user_to_idx.get(int(uid))
        mids = test_df.loc[idx, "MovieID"].to_numpy()
        midx = [movie_to_idx.get(int(m)) for m in mids]
        cf_pred = np.full(len(idx), cf_global_mean, dtype=float)
        content_pred = np.full(len(idx), content_global_mean, dtype=float)
        valid_positions = []
        valid_midx = []
        for pos, mi in enumerate(midx):
            if mi is not None:
                valid_positions.append(pos)
                valid_midx.append(mi)
        if valid_midx:
            if u is not None:
                cf_pred[valid_positions] = cf_global_mean + user_factors[u] @ movie_factors[valid_midx].T
            if train_pairs:
                sim = movie_cosine[valid_midx][:, train_indices]
                denom = np.sum(np.abs(sim), axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    cp = np.where(denom > 0, (sim @ train_ratings) / np.where(denom > 0, denom, 1.0), train_ratings.mean())
                content_pred[valid_positions] = cp
        # fallback popularity for missing movies
        for pos, mid in enumerate(mids):
            if midx[pos] is None:
                cf_pred[pos] = pop_map.get(int(mid), cf_global_mean)
                content_pred[pos] = pop_map.get(int(mid), content_global_mean)
        has_cf = u is not None
        has_content = bool(train_pairs)
        if has_cf and has_content:
            combined = 0.6 * cf_pred + 0.4 * content_pred
        elif has_cf:
            combined = cf_pred
        elif has_content:
            combined = content_pred
        else:
            combined = np.array([pop_map.get(int(m), cf_global_mean) for m in mids], dtype=float)
        preds[list(idx)] = combined
    actuals = test_df["Rating"].to_numpy(dtype=float)
    preds = np.clip(preds, 1.0, 5.0)
    return float(np.sqrt(np.mean((preds - actuals) ** 2)))


def evaluate_recommendations(recommend_fn, test_df, train_user_items, relevant_by_user):
    precisions = []
    recalls = []
    for uid, group in test_df.groupby("UserID"):
        relevant = relevant_by_user.get(uid, set())
        recs = recommend_fn(uid)
        hits = len(set(recs) & relevant)
        precisions.append(hits / TOP_K)
        if relevant:
            recalls.append(hits / len(relevant))
        else:
            recalls.append(0.0)
    if not precisions:
        return 0.0, 0.0
    return float(np.mean(precisions)), float(np.mean(recalls))


def main():
    print("Loading datasets...")
    try:
        movies = load_movies()
        ratings = load_ratings()
        users = load_users()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return
    movies, ratings, users = validate_data(movies, ratings, users)
    print(f"movies: {movies.shape}, ratings: {ratings.shape}, users: {users.shape}")

    print("Splitting ratings per user...")
    train_ratings, test_ratings = per_user_split(ratings)
    print(f"train ratings: {len(train_ratings)}, test ratings: {len(test_ratings)}")

    relevant_by_user = (
        test_ratings[test_ratings["Rating"] >= RELEVANCE_THRESHOLD]
        .groupby("UserID")["MovieID"]
        .apply(set)
        .to_dict()
    )
    train_user_items = user_train_items(train_ratings)
    train_user_ratings = {
        uid: g[["MovieID", "Rating"]] for uid, g in train_ratings.groupby("UserID")
    }

    print("Building popularity model...")
    pop_stats, global_mean = build_popularity(train_ratings)
    print(f"popular movies above threshold: {len(pop_stats)}")

    print("Building content model...")
    vectorizer, genre_matrix, movie_cosine = build_content(movies)
    movie_ids = sorted(movies["MovieID"].tolist())
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    print("Building collaborative filtering model...")
    user_ids = sorted(train_ratings["UserID"].unique().tolist())
    svd, user_factors, movie_factors, user_to_idx, cf_movie_to_idx, cf_global_mean = build_cf(
        train_ratings, movie_ids, user_ids
    )
    # unify movie_to_idx with cf mapping (same order)
    movie_to_idx = cf_movie_to_idx

    print("Training complete. Evaluating...")

    # RMSE evaluation
    def pop_predict(uid, mid, ur):
        return predict_popularity(mid, pop_stats, global_mean)

    def cf_predict(uid, mid, ur):
        return predict_cf(uid, mid, user_to_idx, movie_to_idx, user_factors, movie_factors, cf_global_mean)

    def content_predict(uid, mid, ur):
        return predict_content(ur, mid, movie_cosine, movie_to_idx, global_mean)

    def hybrid_predict(uid, mid, ur):
        return predict_hybrid(uid, mid, ur, movie_cosine, movie_to_idx, user_to_idx, movie_factors,
                            user_factors, pop_stats, cf_global_mean, global_mean)

    print("Computing RMSE: popularity...", flush=True)
    pop_rmse = rmse_popularity(test_ratings, pop_stats, global_mean)
    print("Computing RMSE: CF...", flush=True)
    cf_rmse = rmse_cf(test_ratings, user_to_idx, movie_to_idx, user_factors, movie_factors, cf_global_mean)
    print("Computing RMSE: content...", flush=True)
    content_rmse = rmse_content(test_ratings, train_user_ratings, movie_cosine, movie_to_idx, global_mean)
    print("Computing RMSE: hybrid...", flush=True)
    hybrid_rmse = rmse_hybrid(test_ratings, train_user_ratings, movie_cosine, movie_to_idx, user_to_idx,
                             user_factors, movie_factors, pop_stats, cf_global_mean, global_mean)
    print("RMSE done.", flush=True)

    # Recommendation evaluation
    def make_recommend_popularity(uid):
        seen = train_user_items.get(uid, set())
        return recommend_popularity(pop_stats, global_mean, seen, TOP_K)

    def make_recommend_cf(uid):
        seen = train_user_items.get(uid, set())
        return recommend_cf(uid, movie_ids, user_to_idx, movie_to_idx, user_factors, movie_factors, cf_global_mean, seen, TOP_K)

    def make_recommend_content(uid):
        seen = train_user_items.get(uid, set())
        ur = train_user_ratings.get(uid, pd.DataFrame(columns=["MovieID", "Rating"]))
        profile = content_user_profile(genre_matrix, movie_to_idx, ur, global_mean)
        return recommend_content(movies, genre_matrix, movie_to_idx, profile, seen, TOP_K)

    def make_recommend_hybrid(uid):
        seen = train_user_items.get(uid, set())
        ur = train_user_ratings.get(uid, pd.DataFrame(columns=["MovieID", "Rating"]))
        profile = content_user_profile(genre_matrix, movie_to_idx, ur, global_mean)
        return recommend_hybrid(movies, uid, ur, train_user_items, pop_stats, cf_global_mean,
                                genre_matrix, movie_to_idx, profile, movie_ids, user_to_idx,
                                user_factors, movie_factors, movie_cosine, TOP_K)

    print("Evaluating recommendations: popularity...", flush=True)
    pop_p, pop_r = evaluate_recommendations(make_recommend_popularity, test_ratings, train_user_items, relevant_by_user)
    print("Evaluating recommendations: CF...", flush=True)
    cf_p, cf_r = evaluate_recommendations(make_recommend_cf, test_ratings, train_user_items, relevant_by_user)
    print("Evaluating recommendations: content...", flush=True)
    content_p, content_r = evaluate_recommendations(make_recommend_content, test_ratings, train_user_items, relevant_by_user)
    print("Evaluating recommendations: hybrid...", flush=True)
    hybrid_p, hybrid_r = evaluate_recommendations(make_recommend_hybrid, test_ratings, train_user_items, relevant_by_user)
    print("Recommendation eval done.", flush=True)

    print("\nEvaluation metrics (relevance threshold: rating >= {:.1f})".format(RELEVANCE_THRESHOLD))
    print(f"Popularity  RMSE={pop_rmse:.4f}  P@10={pop_p:.4f}  R@10={pop_r:.4f}")
    print(f"CF (SVD)    RMSE={cf_rmse:.4f}  P@10={cf_p:.4f}  R@10={cf_r:.4f}")
    print(f"Content     RMSE={content_rmse:.4f}  P@10={content_p:.4f}  R@10={content_r:.4f}")
    print(f"Hybrid      RMSE={hybrid_rmse:.4f}  P@10={hybrid_p:.4f}  R@10={hybrid_r:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "top_k": TOP_K,
        "test_ratio": TEST_RATIO,
        "models": {
            "popularity": {
                "rmse": pop_rmse,
                "precision_at_10": pop_p,
                "recall_at_10": pop_r,
            },
            "collaborative_filtering": {
                "rmse": cf_rmse,
                "precision_at_10": cf_p,
                "recall_at_10": cf_r,
            },
            "content_based": {
                "rmse": content_rmse,
                "precision_at_10": content_p,
                "recall_at_10": content_r,
            },
            "hybrid": {
                "rmse": hybrid_rmse,
                "precision_at_10": hybrid_p,
                "recall_at_10": hybrid_r,
            },
        },
        "model_list": [
            {"model": "Popularity", "rmse": pop_rmse, "precision_at_10": pop_p, "recall_at_10": pop_r},
            {"model": "Collaborative Filtering", "rmse": cf_rmse, "precision_at_10": cf_p, "recall_at_10": cf_r},
            {"model": "Content-Based", "rmse": content_rmse, "precision_at_10": content_p, "recall_at_10": content_r},
            {"model": "Hybrid", "rmse": hybrid_rmse, "precision_at_10": hybrid_p, "recall_at_10": hybrid_r},
        ],
    }
    metrics_path = MODELS_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(f"  {metrics_path}")

    # Save artifacts
    popularity_artifact = {
        "pop_stats": pop_stats,
        "global_mean": global_mean,
        "min_count": MIN_POPULARITY_COUNT,
    }
    content_artifact = {
        "vectorizer": vectorizer,
        "genre_matrix": genre_matrix,
        "movie_cosine": movie_cosine,
        "movie_ids": movie_ids,
        "movie_to_idx": movie_to_idx,
        "global_mean": global_mean,
    }
    cf_artifact = {
        "svd": svd,
        "user_factors": user_factors,
        "movie_factors": movie_factors,
        "user_ids": user_ids,
        "user_to_idx": user_to_idx,
        "movie_ids": movie_ids,
        "movie_to_idx": movie_to_idx,
        "global_mean": cf_global_mean,
        "n_components": N_COMPONENTS,
    }
    hybrid_artifact = {
        "cf_weight": HYBRID_CF_WEIGHT,
        "content_weight": HYBRID_CONTENT_WEIGHT,
        "user_factors": user_factors,
        "movie_factors": movie_factors,
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx,
        "movie_ids": movie_ids,
        "global_mean": cf_global_mean,
        "genre_matrix": genre_matrix,
        "movie_cosine": movie_cosine,
        "pop_stats": pop_stats,
        "pop_global_mean": global_mean,
    }
    id_mappings = {
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx,
    }
    movie_metadata = movies[["MovieID", "Title", "Genres"]].copy()

    paths = []
    def dump(obj, name):
        path = MODELS_DIR / name
        joblib.dump(obj, path)
        paths.append(path)

    dump(popularity_artifact, "popularity_model.joblib")
    dump(content_artifact, "content_model.joblib")
    dump(cf_artifact, "cf_model.joblib")
    dump(hybrid_artifact, "hybrid_model.joblib")
    dump(id_mappings, "id_mappings.joblib")
    dump(movie_metadata, "movie_metadata.joblib")

    print("\nSaved artifacts:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
