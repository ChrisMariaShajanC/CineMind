const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

export type ApiMovie = {
  movie_id: number;
  title: string;
  genres: string;
  score?: number | null;
  explanation?: string | null;
  relevance_score?: number | null;
  mmr_score?: number | null;
};

export type RecommendationResponse = {
  user_id: number;
  known_user: boolean;
  recommendations: ApiMovie[];
};

export type HealthResponse = {
  status: string;
  backend: string;
  models_loaded: boolean;
  error: string | null;
};

export type ModelMetrics = {
  model: string;
  rmse: number;
  precision_at_10: number;
  recall_at_10: number;
};

export type MetricsResponse = {
  relevance_threshold: number;
  top_k: number;
  test_ratio: number;
  models: Record<string, {
    rmse: number;
    precision_at_10: number;
    recall_at_10: number;
  }>;
  model_list: ModelMetrics[];
};

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    throw new Error(`API request failed: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export const api = {
  baseUrl: API_BASE_URL,
  health: () => fetchJson<HealthResponse>("/health"),
  popularMovies: (limit = 10) =>
    fetchJson<ApiMovie[]>(`/recommendations/popular?limit=${limit}`),
  searchMovies: (params: { search?: string; genre?: string; limit?: number }) => {
    const query = new URLSearchParams();
    if (params.search) query.set("search", params.search);
    if (params.genre) query.set("genre", params.genre);
    if (params.limit) query.set("limit", String(params.limit));
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return fetchJson<ApiMovie[]>(`/movies${suffix}`);
  },
  recommendations: (userId: number, limit = 10) =>
    fetchJson<RecommendationResponse>(`/recommendations/${userId}?limit=${limit}`),
  diverseRecommendations: (userId: number, diversity = 0.5, limit = 10) =>
    fetchJson<RecommendationResponse>(`/recommendations/diverse/${userId}?diversity=${diversity}&limit=${limit}`),
  metrics: () => fetchJson<MetricsResponse>("/metrics"),
};

import type { Movie } from "../data/catalogue";

export function apiMovieToMovie(apiMovie: ApiMovie): Movie {
  return {
    id: apiMovie.movie_id,
    title: apiMovie.title,
    year: 0,
    runtime: "—",
    genres: apiMovie.genres ? apiMovie.genres.split("|").filter(Boolean) : [],
    image: "/favicon.svg",
    score: apiMovie.score != null ? Number(apiMovie.score).toFixed(1) : "—",
    match: 0,
    overview: "Movie metadata from the trained backend model.",
    language: "Unknown",
    runtimeMinutes: 0,
    popularity: 0,
    moods: [],
    tags: [],
  };
}
