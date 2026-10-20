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

export type AuthUser = { id: number; name: string; email: string };

export type AuthResponse = { access_token: string; token_type: string; user: AuthUser };

export type PreferencePayload = {
  languages: string[];
  genres: string[];
  moods: string[];
  release_period: string;
  duration: string;
  onboarding_completed: boolean;
};

export type CatalogueMovie = {
  source: string;
  external_id: string;
  title: string;
  overview?: string | null;
  poster_url?: string | null;
  backdrop_url?: string | null;
  release_date?: string | null;
  original_language?: string | null;
  genres: string[];
  vote_average?: number | null;
  popularity?: number | null;
};

export type CatalogueResponse = {
  page: number;
  total_pages: number;
  total_results: number;
  results: CatalogueMovie[];
};

function authHeaders(token: string | null): HeadersInit | undefined {
  return token ? { Authorization: `Bearer ${token}` } : undefined;
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

  // Auth
  register: (name: string, email: string, password: string) =>
    fetchJson<AuthResponse>("/auth/register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, email, password }) }),
  login: (email: string, password: string) =>
    fetchJson<AuthResponse>("/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }) }),
  me: (token: string) => fetchJson<AuthUser>("/auth/me", { headers: authHeaders(token) }),

  // Preferences
  getPreferences: (token: string) => fetchJson<PreferencePayload>("/preferences", { headers: authHeaders(token) }),
  updatePreferences: (token: string, payload: PreferencePayload) =>
    fetchJson<PreferencePayload>("/preferences", { method: "PUT", headers: { ...authHeaders(token), "Content-Type": "application/json" }, body: JSON.stringify(payload) }),

  // Favourites / watchlist
  favourites: (token: string) => fetchJson<any[]>("/me/favourites", { headers: authHeaders(token) }),
  addFavourite: (token: string, item: any) =>
    fetchJson<any>("/me/favourites", { method: "POST", headers: { ...authHeaders(token), "Content-Type": "application/json" }, body: JSON.stringify(item) }),
  removeFavourite: (token: string, source: string, externalId: string) =>
    fetch(`/me/favourites/${source}/${externalId}`, { method: "DELETE", headers: authHeaders(token) }),
  watchlist: (token: string) => fetchJson<any[]>("/me/watchlist", { headers: authHeaders(token) }),
  addWatchlist: (token: string, item: any) =>
    fetchJson<any>("/me/watchlist", { method: "POST", headers: { ...authHeaders(token), "Content-Type": "application/json" }, body: JSON.stringify(item) }),
  removeWatchlist: (token: string, source: string, externalId: string) =>
    fetch(`/me/watchlist/${source}/${externalId}`, { method: "DELETE", headers: authHeaders(token) }),

  // Catalogue
  catalogueSearch: (params: { q: string; language?: string; genre?: number; year?: number; page?: number }) => {
    const query = new URLSearchParams({ q: params.q });
    if (params.language) query.set("language", params.language);
    if (params.genre) query.set("genre", String(params.genre));
    if (params.year) query.set("year", String(params.year));
    if (params.page) query.set("page", String(params.page));
    return fetchJson<CatalogueResponse>(`/catalogue/search?${query.toString()}`);
  },
  cataloguePopular: (params: { language?: string; page?: number } = {}) => {
    const query = new URLSearchParams();
    if (params.language) query.set("language", params.language);
    if (params.page) query.set("page", String(params.page));
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return fetchJson<CatalogueResponse>(`/catalogue/popular${suffix}`);
  },
  catalogueTrending: (params: { time_window?: string; page?: number } = {}) => {
    const query = new URLSearchParams();
    if (params.time_window) query.set("time_window", params.time_window);
    if (params.page) query.set("page", String(params.page));
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return fetchJson<CatalogueResponse>(`/catalogue/trending${suffix}`);
  },
  catalogueGenres: () => fetchJson<Record<string, string>>("/catalogue/genres"),
  catalogueLanguages: () => fetchJson<Record<string, string>>("/catalogue/languages"),

  // Personalized
  personalized: (token: string, limit = 10) =>
    fetchJson<any>(`/me/recommendations?limit=${limit}`, { headers: authHeaders(token) }),
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
