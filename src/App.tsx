import { useEffect, useMemo, useRef, useState } from "react";
import { demoCatalogue, featuredMovie, type Movie } from "./data/catalogue";
import { api, apiMovieToMovie } from "./lib/api";
import type { MetricsResponse } from "./lib/api";
import { defaultDiscovery, defaultProfile, explainRecommendation, rankMovies, rerankMoviesForDiversity, type DiscoverySettings, type Feedback, type RankedMovie, type RecommendationContext, type TasteProfile } from "./lib/recommender";

type Page = "Discover" | "For You" | "Beyond Your Taste" | "Mood & Context" | "My Collection" | "ML Insights";
type FiltersState = { genre: string; language: string; year: string; runtime: string };

const movies = demoCatalogue.getAll();
const allGenres = demoCatalogue.getGenres();
const allLanguages = demoCatalogue.getLanguages();
const defaultFilters: FiltersState = { genre: "All", language: "All", year: "Any", runtime: "Any" };

function usePersistentState<T>(key: string, fallback: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(key);
      const parsed = stored ? JSON.parse(stored) : fallback;
      return parsed ?? fallback;
    } catch {
      return fallback;
    }
  });
  useEffect(() => {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* Local persistence is optional in this prototype. */ }
  }, [key, value]);
  return [value, setValue] as const;
}

function applyMovieFilters(items: Movie[], filters: FiltersState) {
  return items.filter((movie) => {
    const yearMatches = filters.year === "Any" || (filters.year === "2023–24" ? movie.year >= 2023 : filters.year === "2020–22" ? movie.year >= 2020 && movie.year <= 2022 : movie.year < 2020);
    const runtimeMatches = filters.runtime === "Any" || (filters.runtime === "Under 100 min" ? movie.runtimeMinutes < 100 : filters.runtime === "100–120 min" ? movie.runtimeMinutes >= 100 && movie.runtimeMinutes <= 120 : movie.runtimeMinutes > 120);
    return (filters.genre === "All" || movie.genres.includes(filters.genre)) && (filters.language === "All" || movie.language === filters.language) && yearMatches && runtimeMatches;
  });
}

const nav: { name: Page; icon: string }[] = [
  { name: "Discover", icon: "compass" }, { name: "For You", icon: "spark" },
  { name: "Beyond Your Taste", icon: "orbit" }, { name: "Mood & Context", icon: "mood" },
  { name: "My Collection", icon: "bookmark" }, { name: "ML Insights", icon: "chart" },
];

function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const paths: Record<string, React.ReactNode> = {
    compass: <><circle cx="12" cy="12" r="9" /><path d="m15.4 8.6-2.1 4.7-4.7 2.1 2.1-4.7 4.7-2.1Z" /></>,
    spark: <path d="m12 3 1.4 5.6L19 10l-5.6 1.4L12 17l-1.4-5.6L5 10l5.6-1.4L12 3Z" />,
    orbit: <><circle cx="12" cy="12" r="2.2" /><ellipse cx="12" cy="12" rx="9" ry="4.3" transform="rotate(-28 12 12)" /><circle cx="18.8" cy="8" r="1" fill="currentColor" /></>,
    mood: <><circle cx="12" cy="12" r="9" /><path d="M8.5 14.2c1.8 2 5.2 2 7 0M9 9h.01M15 9h.01" /></>,
    bookmark: <path d="M6 4.5A1.5 1.5 0 0 1 7.5 3h9A1.5 1.5 0 0 1 18 4.5V21l-6-3.8L6 21V4.5Z" />,
    chart: <><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></>,
    search: <><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></>,
    plus: <><path d="M12 5v14M5 12h14" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
    moon: <path d="M20 15.2A8.5 8.5 0 0 1 8.8 4 8.5 8.5 0 1 0 20 15.2Z" />,
    close: <path d="m6 6 12 12M18 6 6 18" />,
    play: <path d="m9 7 8 5-8 5V7Z" />,
    heart: <path d="M20.8 5.9c-1.7-1.8-4.5-1.8-6.3 0L12 8.4 9.5 5.9a4.4 4.4 0 0 0-6.3 6.3L12 21l8.8-8.8a4.4 4.4 0 0 0 0-6.3Z" />,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></>,
    arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
    menu: <path d="M4 7h16M4 12h16M4 17h16" />,
    star: <path d="m12 3 2.6 5.3 5.9.9-4.3 4.1 1 5.8-5.2-2.7-5.2 2.7 1-5.8-4.3-4.1 5.9-.9L12 3Z" />,
  };
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>;
}

function Button({ children, variant = "primary", icon, onClick, ariaLabel, className = "", disabled = false }: { children?: React.ReactNode; variant?: "primary" | "secondary" | "ghost" | "icon"; icon?: string; onClick?: () => void; ariaLabel?: string; className?: string; disabled?: boolean }) {
  return <button className={`btn btn-${variant} ${className}`} onClick={onClick} aria-label={ariaLabel} disabled={disabled}>{icon && <Icon name={icon} />}{children}</button>;
}

function PageHeader({ eyebrow, title, copy, action }: { eyebrow: string; title: string; copy: string; action?: React.ReactNode }) {
  return <header className="page-header"><div><span className="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{copy}</p></div>{action}</header>;
}

function Section({ title, kicker, children, action }: { title: string; kicker?: string; children: React.ReactNode; action?: React.ReactNode }) {
  return <section className="section"><div className="section-head"><div>{kicker && <span className="eyebrow">{kicker}</span>}<h2>{title}</h2></div>{action}</div>{children}</section>;
}

function MovieCard({ movie, onOpen, saved, onSave, favourite, onFavourite, explanation, showReason = false, rank, hideMatch = false }: { movie: Movie; onOpen: (m: Movie) => void; saved: boolean; onSave: (id: number) => void; favourite?: boolean; onFavourite?: (id: number) => void; explanation?: string; showReason?: boolean; rank?: number; hideMatch?: boolean }) {
  return <article className="movie-card">
    <button className="poster-button" onClick={() => onOpen(movie)} aria-label={`Open details for ${movie.title}`}>
      <img src={movie.image} alt="" />
      <span className="poster-shade" />
      {rank && <span className="rank">{String(rank).padStart(2, "0")}</span>}
      {!hideMatch && <span className="match">Curated for you</span>}
      <span className="poster-open"><Icon name="arrow" /></span>
    </button>
    <div className="movie-copy">
      <div className="movie-title-row"><div><h3>{movie.title}</h3><p>{movie.year} · {movie.runtime} · {movie.genres[0]}</p></div>
        <div className="card-actions">{onFavourite && <button className={`save ${favourite ? "saved" : ""}`} onClick={() => onFavourite(movie.id)} aria-label={favourite ? "Remove from favourites" : "Add to favourites"}><Icon name="heart" size={15} /></button>}<button className={`save ${saved ? "saved" : ""}`} onClick={() => onSave(movie.id)} aria-label={saved ? "Remove from watchlist" : "Add to watchlist"}><Icon name={saved ? "check" : "plus"} size={17} /></button></div>
      </div>
      {showReason && <p className="reason"><Icon name="spark" size={14} />{explanation || `Curated for its ${movie.moods[0]?.toLowerCase()} sample tag.`}</p>}
    </div>
  </article>;
}

function MovieGrid({ items, ...props }: { items: Movie[]; onOpen: (m: Movie) => void; saved: Set<number>; onSave: (id: number) => void; favourites?: Set<number>; onFavourite?: (id: number) => void; explanation?: (movie: Movie) => string; showReason?: boolean; ranked?: boolean; className?: string; hideMatch?: boolean }) {
  return <div className={`movie-grid ${props.className || ""}`}>{items.map((m, i) => <MovieCard key={m.id} movie={m} onOpen={props.onOpen} saved={props.saved.has(m.id)} onSave={props.onSave} favourite={props.favourites?.has(m.id)} onFavourite={props.onFavourite} explanation={props.explanation?.(m)} showReason={props.showReason} rank={props.ranked ? i + 1 : undefined} hideMatch={props.hideMatch} />)}</div>;
}

function Filters({ value, onChange, includeSort = false }: { value: FiltersState; onChange: (filters: FiltersState) => void; includeSort?: boolean }) {
  return <div className="filters">
    <select aria-label="Genre" value={value.genre} onChange={(e) => onChange({ ...value, genre: e.target.value })}><option value="All">All genres</option>{allGenres.map((genre) => <option key={genre}>{genre}</option>)}</select>
    <select aria-label="Language" value={value.language} onChange={(e) => onChange({ ...value, language: e.target.value })}><option value="All">All languages</option>{allLanguages.map((language) => <option key={language}>{language}</option>)}</select>
    <select aria-label="Release period" value={value.year} onChange={(e) => onChange({ ...value, year: e.target.value })}>{["Any", "2023–24", "2020–22", "Before 2020"].map((year) => <option key={year}>{year}</option>)}</select>
    <select aria-label="Runtime" value={value.runtime} onChange={(e) => onChange({ ...value, runtime: e.target.value })}>{["Any", "Under 100 min", "100–120 min", "Over 120 min"].map((runtime) => <option key={runtime}>{runtime}</option>)}</select>
    {includeSort && <span className="filter-note">Filters update instantly</span>}
  </div>;
}

function Discover(props: PageProps) {
  const [query, setQuery] = useState("");
  const [realSearch, setRealSearch] = useState<Movie[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [popularMovies, setPopularMovies] = useState<Movie[]>([]);
  const [popularLoading, setPopularLoading] = useState(false);
  const [popularError, setPopularError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    setPopularLoading(true);
    setPopularError(null);
    api.popularMovies(6).then((data) => {
      if (!cancelled) setPopularMovies(data.map(apiMovieToMovie));
    }).catch((error) => {
      if (!cancelled) setPopularError(error instanceof Error ? error.message : "Failed to load popular movies");
    }).finally(() => {
      if (!cancelled) setPopularLoading(false);
    });
    return () => { cancelled = true; };
  }, []);
  useEffect(() => {
    if (!query.trim()) {
      setRealSearch([]);
      setSearchError(null);
      setSearchLoading(false);
      return;
    }
    let cancelled = false;
    setSearchLoading(true);
    setSearchError(null);
    api.searchMovies({ search: query, limit: 20 }).then((data) => {
      if (!cancelled) setRealSearch(data.map(apiMovieToMovie));
    }).catch((error) => {
      if (!cancelled) setSearchError(error instanceof Error ? error.message : "Search failed");
    }).finally(() => {
      if (!cancelled) setSearchLoading(false);
    });
    return () => { cancelled = true; };
  }, [query]);
  const found = query.trim() ? realSearch : [];
  const context = recommendationContext(props);
  const picked = useMemo(() => rerankMoviesForDiversity(movies, context, 6).map((item) => item.movie), [props.profile, props.ratings, props.feedback, props.discovery]);
  const hiddenGems = useMemo(() => [...movies].sort((a, b) => a.popularity - b.popularity || Number(b.score) - Number(a.score)).slice(0, 4), []);
  const referenceMovie = useMemo(() => {
    const favouriteId = [...props.favourites][0] ?? props.profile.favouriteMovies[0];
    if (favouriteId) return demoCatalogue.getById(favouriteId);
    const highlyRatedId = Object.entries(props.ratings).filter(([, rating]) => rating >= 4).sort((a, b) => b[1] - a[1])[0]?.[0];
    return highlyRatedId ? demoCatalogue.getById(Number(highlyRatedId)) : undefined;
  }, [props.favourites, props.profile.favouriteMovies, props.ratings]);
  const related = useMemo(() => referenceMovie
    ? rerankMoviesForDiversity(movies.filter((movie) => movie.id !== referenceMovie.id && movie.genres.some((genre) => referenceMovie.genres.includes(genre))), context, 5).map((item) => item.movie)
    : [], [referenceMovie, props.profile, props.ratings, props.feedback, props.discovery]);
  return <>
    <div className="discover-intro">
      <span className="eyebrow">YOUR EVENING, CURATED</span>
      <h1>Stories worth<br /><em>getting lost in.</em></h1>
      <p>Recommendations shaped by your taste—not everyone else’s.</p>
      <div className="search" role="search"><Icon name="search" /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search films, genres, moods..." aria-label="Search the demonstration catalogue" aria-keyshortcuts="Control+K Meta+K" /><button type="button" onClick={props.openSearch} aria-label="Open catalogue search"><kbd>Ctrl / ⌘ K</kbd></button>{query && <button type="button" className="search-clear" onClick={() => setQuery("")} aria-label="Clear search"><Icon name="close" size={15} /></button>}</div>
    </div>
    {query ? <Section title={`Search results � ${found.length}`}>{searchLoading ? <div className="loading-grid" aria-label="Searching movies">{[1, 2, 3].map((n) => <i key={n} />)}</div> : searchError ? <EmptyState icon="info" title="Search unavailable" copy={searchError} button="Retry" onAction={() => setQuery(query)} /> : found.length ? <MovieGrid items={found} onOpen={props.open} saved={props.saved} onSave={props.save} /> : <EmptyState icon="search" title="No catalogue matches" copy="Try another title, genre, language, mood, or theme." button="Clear search" onAction={() => setQuery("")} />}</Section> : <>
      <section className="hero">
        <img src={demoCatalogue.getById(featuredMovie.id)?.image || featuredMovie.image} alt="A lone figure crossing a vast cinematic landscape" />
        <div className="hero-overlay" />
        <div className="hero-content"><span className="sample-pill">SAMPLE FEATURE</span><p className="hero-meta">{featuredMovie.year} · {featuredMovie.runtime} · {featuredMovie.genres.join(", ")}</p><h2>A Field<br />of Silence</h2><p className="hero-overview">{featuredMovie.overview}</p><div className="hero-actions"><Button icon="play" onClick={() => props.open(featuredMovie)}>View details</Button><Button variant="secondary" icon={props.saved.has(featuredMovie.id) ? "check" : "plus"} onClick={() => props.save(featuredMovie.id)}>Watchlist</Button></div></div>
        <div className="hero-match"><strong>Editor’s pick</strong><span>Personalized sample</span></div>
      </section>
      <Section kicker="PERSONALIZED · SIMULATED RANKING" title="Picked for you" action={<button className="text-link" onClick={() => props.navigate("For You")}>View all recommendations <Icon name="arrow" /></button>}><MovieGrid className="poster-row" items={picked} onOpen={props.open} saved={props.saved} onSave={props.save} favourites={props.favourites} onFavourite={props.favourite} /></Section>
      <Section kicker="LOWER SAMPLE POPULARITY FIRST" title="Hidden gems"><div className="landscape-row">{hiddenGems.map((m) => <button key={m.id} className="landscape-card" onClick={() => props.open(m)}><img src={m.image} alt="" /><span><small>{m.genres.join(" · ")}</small><strong>{m.title}</strong><em>Sample popularity · {m.popularity}/100</em></span></button>)}</div></Section>
      {referenceMovie && related.length ? <section className="because"><div><span className="eyebrow">BECAUSE YOU LOVED</span><h2>{referenceMovie.title}</h2><p>More sample films sharing {referenceMovie.genres.join(" and ")}, reranked with your current preferences.</p><Button variant="secondary" onClick={() => props.open(referenceMovie)}>Revisit this film <Icon name="arrow" /></Button></div><MovieGrid className="poster-row" items={related} onOpen={props.open} saved={props.saved} onSave={props.save} favourites={props.favourites} onFavourite={props.favourite} /></section> : <Section kicker="BECAUSE YOU LOVED" title="Teach CineMind what belongs here"><EmptyState icon="heart" title="No recommendation thread yet" copy="Favourite or rate a compatible sample film highly to generate this catalogue-based thread." button="Explore recommendations" onAction={() => props.navigate("For You")} /></Section>}
      <Section kicker="STEP OUTSIDE THE FAMILIAR" title="Explore different genres"><div className="genre-row">{["Sci-Fi", "Documentary", "Western", "Comedy", "History"].map((g, i) => <button key={g} className="genre-tile" onClick={() => setQuery(g)}><span>0{i + 1}</span><strong>{g}</strong><small>Explore collection <Icon name="arrow" size={14} /></small></button>)}</div></Section>
      <Section kicker="BACKEND POPULAR MOVIES" title="Popular from trained model">{popularLoading ? <div className="loading-grid" aria-label="Loading popular movies">{[1, 2, 3].map((n) => <i key={n} />)}</div> : popularError ? <EmptyState icon="info" title="Popular movies unavailable" copy={popularError} button="Retry" onAction={() => window.location.reload()} /> : popularMovies.length ? <MovieGrid className="poster-row" items={popularMovies} onOpen={props.open} saved={props.saved} onSave={props.save} hideMatch /> : <EmptyState icon="search" title="No popular movies returned" copy="The backend returned no popular movies." button="Retry" onAction={() => window.location.reload()} />}</Section>
    </>}
  </>;
}

type PageProps = {
  open: (m: Movie) => void;
  saved: Set<number>;
  save: (id: number) => void;
  favourites: Set<number>;
  favourite: (id: number) => void;
  ratings: Record<number, number>;
  rate: (id: number, rating: number) => void;
  feedback: Record<number, Feedback>;
  setMovieFeedback: (id: number, feedback: Feedback) => void;
  profile: TasteProfile;
  setProfile: React.Dispatch<React.SetStateAction<TasteProfile>>;
  discovery: DiscoverySettings;
  setDiscovery: React.Dispatch<React.SetStateAction<DiscoverySettings>>;
  notify: (message: string) => void;
  navigate: (page: Page) => void;
  openSearch: () => void;
};

function recommendationContext(props: PageProps): RecommendationContext {
  return { settings: props.discovery, profile: props.profile, feedback: props.feedback, ratings: props.ratings, catalogue: demoCatalogue };
}

function ForYou(props: PageProps) {
  const [sort, setSort] = useState("relevance");
  const [filters, setFilters] = useState(defaultFilters);
  const [loading, setLoading] = useState(false);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const [backendMovies, setBackendMovies] = useState<Movie[]>([]);
  const [backendLoading, setBackendLoading] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [backendRetry, setBackendRetry] = useState(0);
  useEffect(() => {
    let cancelled = false;
    setBackendLoading(true);
    setBackendError(null);
    api.recommendations(1, 8).then((data) => {
      if (!cancelled) setBackendMovies(data.recommendations.map(apiMovieToMovie));
    }).catch((error) => {
      if (!cancelled) setBackendError(error instanceof Error ? error.message : "Recommendations unavailable");
    }).finally(() => {
      if (!cancelled) setBackendLoading(false);
    });
    return () => { cancelled = true; };
  }, [backendRetry]);
  const context = recommendationContext(props);
  const sorted = useMemo<RankedMovie[]>(() => {
    const filtered = applyMovieFilters(movies, filters);
    const ranked = sort === "relevance" ? rerankMoviesForDiversity(filtered, context) : rankMovies(filtered, context);
    return ranked.sort((a, b) => {
      if (sort === "rating") return +b.movie.score - +a.movie.score;
      if (sort === "year") return b.movie.year - a.movie.year;
      if (sort === "runtime") return a.movie.runtimeMinutes - b.movie.runtimeMinutes;
      return 0;
    });
  }, [filters, sort, props.discovery, props.profile, props.feedback, props.ratings, refreshVersion]);
  const refresh = () => {
    setLoading(true);
    window.setTimeout(() => {
      setRefreshVersion((version) => version + 1);
      setLoading(false);
      props.notify("Demonstration recommendations recalculated from your current local preferences.");
    }, 550);
  };
  return <>
    <PageHeader eyebrow="PERSONALIZED FOR ALEX" title="Your cinema, refined." copy="Every recommendation includes the metadata signals behind it. Ranking and genre-diversity reranking are simulated locally for this prototype." action={<Button variant="secondary" icon="spark" onClick={refresh} disabled={loading}>{loading ? "Recalculating…" : "Refresh recommendations"}</Button>} />
    <div className="preference-summary"><span>YOUR CURRENT SIGNAL</span><strong>{props.profile.genres.length ? props.profile.genres.slice(0, 3).join(" · ") : "No genres selected yet"}</strong><p>{Object.keys(props.feedback).length} feedback signals · {Object.keys(props.ratings).length} personal ratings · Discovery set to {props.profile.discovery}</p></div>
    <div className="toolbar"><Filters value={filters} onChange={setFilters} /><div className="segmented">{[["relevance", "Relevance"], ["rating", "Rating"], ["year", "Year"], ["runtime", "Runtime"]].map(([value, label]) => <button key={value} className={sort === value ? "active" : ""} onClick={() => setSort(value)}>{label}</button>)}</div></div>
    <Section kicker="BACKEND RECOMMENDATIONS" title="From trained model (MovieLens user 1)">{backendLoading ? <div className="loading-grid" aria-label="Loading recommendations">{[1, 2, 3].map((n) => <i key={n} />)}</div> : backendError ? <EmptyState icon="info" title="Recommendations unavailable" copy={backendError} button="Retry" onAction={() => setBackendRetry((v) => v + 1)} /> : backendMovies.length ? <MovieGrid items={backendMovies} onOpen={props.open} saved={props.saved} onSave={props.save} hideMatch /> : <EmptyState icon="search" title="No recommendations returned" copy="The backend returned no recommendations." button="Retry" onAction={() => setBackendRetry((v) => v + 1)} />}</Section>
    {loading ? <div className="loading-grid" aria-label="Refreshing recommendations">{[1, 2, 3].map((n) => <i key={n} />)}</div> : sorted.length === 0 ? <EmptyState icon="search" title="No films match these filters" copy="Try broadening one of the filters to see more recommendations." button="Reset filters" onAction={() => setFilters(defaultFilters)} /> :
    <div className="recommendation-list">{sorted.slice(0, 8).map((ranked) => <article className="recommendation" key={ranked.movie.id}>
      <button className="rec-poster" onClick={() => props.open(ranked.movie)}><img src={ranked.movie.image} alt="" /><span>Demo ranking</span></button>
      <div className="rec-body"><div><span className="eyebrow">{ranked.movie.genres.join(" · ")}</span><h2>{ranked.movie.title}</h2><p>{ranked.movie.year} · {ranked.movie.runtime} · Sample rating {ranked.movie.score}</p></div><div className="why"><Icon name="spark" /><div><strong>Why recommended?</strong><p>{ranked.explanation}</p></div></div>
        <div className="rec-actions"><Button variant="secondary" icon="plus" onClick={() => props.save(ranked.movie.id)}>{props.saved.has(ranked.movie.id) ? "In watchlist" : "Watchlist"}</Button><button className={props.feedback[ranked.movie.id] === "more" ? "selected" : ""} onClick={() => props.setMovieFeedback(ranked.movie.id, "more")}>＋<i>More like this</i></button><button className={props.feedback[ranked.movie.id] === "less" ? "selected" : ""} onClick={() => props.setMovieFeedback(ranked.movie.id, "less")}>−<i>Less like this</i></button><button onClick={() => props.open(ranked.movie)}><Icon name="info" size={14} /><i>Why recommended?</i></button></div>
      </div>
    </article>)}</div>}
  </>;
}

function Beyond(props: PageProps) {
  const settings = props.discovery;
  const [backendMovies, setBackendMovies] = useState<Movie[]>([]);
  const [backendLoading, setBackendLoading] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [diverseItems, setDiverseItems] = useState<{ movie: Movie; explanation: string }[]>([]);
  const [diverseLoading, setDiverseLoading] = useState(false);
  const [diverseError, setDiverseError] = useState<string | null>(null);
  const [diverseRetry, setDiverseRetry] = useState(0);
  useEffect(() => {
    let cancelled = false;
    setBackendLoading(true);
    setBackendError(null);
    api.recommendations(1, 8).then((data) => {
      if (!cancelled) setBackendMovies(data.recommendations.map(apiMovieToMovie));
    }).catch((error) => {
      if (!cancelled) setBackendError(error instanceof Error ? error.message : "Recommendations unavailable");
    }).finally(() => {
      if (!cancelled) setBackendLoading(false);
    });
    return () => { cancelled = true; };
  }, []);
  useEffect(() => {
    let cancelled = false;
    setDiverseLoading(true);
    setDiverseError(null);
    api.diverseRecommendations(1, settings.diversity / 100, 8).then((data) => {
      if (!cancelled) setDiverseItems(data.recommendations.map((item) => ({
        movie: apiMovieToMovie(item),
        explanation: item.explanation || "MMR reranking balances relevance and genre diversity.",
      })));
    }).catch((error) => {
      if (!cancelled) setDiverseError(error instanceof Error ? error.message : "Diverse recommendations unavailable");
    }).finally(() => {
      if (!cancelled) setDiverseLoading(false);
    });
    return () => { cancelled = true; };
  }, [settings.diversity, diverseRetry]);
  const update = (key: keyof DiscoverySettings, value: number | string) => props.setDiscovery((current) => ({ ...current, [key]: value }));
  const context = recommendationContext(props);
  const selected = diverseItems.map((item) => item.movie);
  const genreCounts = selected.slice(0, 6).flatMap((movie) => movie.genres).reduce<Record<string, number>>((counts, genre) => ({ ...counts, [genre]: (counts[genre] || 0) + 1 }), {});
  const topGenres = Object.entries(genreCounts).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([genre]) => genre);
  const summary = settings.mode === "surprise"
    ? `Prioritizing ${settings.novelty > 60 ? "less-popular discoveries" : "balanced discoveries"} with ${settings.diversity > 60 ? "broader genre variety" : "a focused genre range"}.`
    : `Prioritizing familiar ${props.profile.genres.slice(0, 2).join(" and ") || "preferred"} stories while allowing ${settings.novelty}% novelty.`;
  return <div className="beyond-page">
    <PageHeader eyebrow="SIGNATURE DISCOVERY MODE" title="Beyond your taste." copy="Diversity reranking is provided by the trained backend model; other discovery controls remain a local preview." action={<div className="mode-switch"><button className={settings.mode === "taste" ? "active" : ""} onClick={() => update("mode", "taste")}>My Taste</button><button className={settings.mode === "surprise" ? "active" : ""} onClick={() => update("mode", "surprise")}><Icon name="spark" /> Surprise Me</button></div>} />
    <div className="discovery-summary"><Icon name="orbit" /><div><span>LIVE DISCOVERY SUMMARY</span><p>{summary} The list and genre distribution update from these settings.</p></div><button onClick={() => { props.setDiscovery(defaultDiscovery); props.notify("Discovery controls reset to prototype defaults."); }}>Reset preferences</button></div>
    <div className="beyond-hero">
      <div className="controls-panel"><div className="panel-intro"><span>DISCOVERY COMPASS</span><h2>How far should we wander?</h2><p>Fine-tune the balance between the familiar and the unexpected.</p></div>
        {[["Familiarity", "familiarity", "Less familiar", "Highly familiar"], ["Genre diversity", "diversity", "Focused", "Wide-ranging"], ["Novelty", "novelty", "Comfort", "Discovery"], ["Popularity", "popularity", "Hidden", "Mainstream"]].map(([label, key, left, right]) => <label className="range-control" key={key}><span><strong>{label}</strong><em>{settings[key as keyof DiscoverySettings] as number}%</em></span><input type="range" min="0" max="100" value={settings[key as keyof DiscoverySettings] as number} onChange={(e) => update(key as keyof DiscoverySettings, +e.target.value)} /><small><i>{left}</i><i>{right}</i></small></label>)}
      </div>
      <div className="diversity-viz"><div className="viz-head"><div><span className="eyebrow">LIVE GENRE DISTRIBUTION</span><h2>{topGenres.length} genres in the leading edit</h2></div><span className="score-ring">{topGenres.length}<small>Genres</small></span></div>
        <div className="orbit-map"><span className="orbit orbit-a" /><span className="orbit orbit-b" /><span className="orbit orbit-c" /><span className="planet core">{topGenres[0] || "Drama"}</span><span className="planet p1">{topGenres[1] || "Mystery"}</span><span className="planet p2">{topGenres[2] || "Adventure"}</span><span className="planet p3">{topGenres[3] || "History"}</span><span className="planet p4">{topGenres[4] || "Comedy"}</span><span className="planet p5">{selected[0]?.language || "World"}</span></div>
        <div className="viz-legend"><span><i className="dot current" />Your comfort zone</span><span><i className="dot new" />New territory</span></div>
      </div>
    </div>
    <Section kicker="BACKEND RECOMMENDATIONS" title="From trained model (MovieLens user 1)">{backendLoading ? <div className="loading-grid" aria-label="Loading recommendations">{[1, 2, 3].map((n) => <i key={n} />)}</div> : backendError ? <EmptyState icon="info" title="Recommendations unavailable" copy={backendError} button="Retry" onAction={() => window.location.reload()} /> : backendMovies.length ? <MovieGrid items={backendMovies} onOpen={props.open} saved={props.saved} onSave={props.save} hideMatch /> : <EmptyState icon="search" title="No recommendations returned" copy="The backend returned no recommendations." button="Retry" onAction={() => window.location.reload()} />}</Section>
    <Section kicker="DIVERSE RECOMMENDATIONS" title="MMR reranking from trained model">{diverseLoading ? <div className="loading-grid" aria-label="Loading diverse recommendations">{[1, 2, 3].map((n) => <i key={n} />)}</div> : diverseError ? <EmptyState icon="info" title="Diverse recommendations unavailable" copy={diverseError} button="Retry" onAction={() => setDiverseRetry((value) => value + 1)} /> : diverseItems.length ? <div className="beyond-grid">{diverseItems.map((item) => <div className="beyond-card" key={item.movie.id}><MovieCard movie={item.movie} onOpen={props.open} saved={props.saved.has(item.movie.id)} onSave={props.save} favourite={props.favourites.has(item.movie.id)} onFavourite={props.favourite} hideMatch /><div className="different"><span>WHY THIS RANKED HERE</span><p>{item.explanation}</p></div></div>)}</div> : <EmptyState icon="search" title="No diverse recommendations returned" copy="The backend returned no diverse recommendations." button="Retry" onAction={() => setDiverseRetry((value) => value + 1)} />}</Section>
    <p className="filter-note">Genre diversity drives backend MMR reranking; other discovery controls remain a local preview.</p>
  </div>;
}

function Mood(props: PageProps) {
  const moods = ["Feel-good", "Suspenseful", "Thought-provoking", "Relaxing", "Adventurous", "Emotional"];
  const [mood, setMood] = useState("Thought-provoking");
  const [filters, setFilters] = useState(defaultFilters);
  const results = applyMovieFilters(movies.filter((movie) => movie.moods.includes(mood)), filters);
  return <>
    <PageHeader eyebrow="CURATED CONTEXT · DEMONSTRATION TAGS" title="What kind of evening is this?" copy="Choose a curated mood tag and context. This prototype does not infer emotions or use an emotion model. Backend movie metadata is available, but no mood model is implemented." />
    <div className="mood-grid">{moods.map((m, i) => <button onClick={() => setMood(m)} className={mood === m ? "active" : ""} key={m}><span className="mood-symbol">{["☼", "◒", "◇", "∿", "↗", "○"][i]}</span><strong>{m}</strong><small>{["Warm, hopeful stories", "Tension that builds", "Ideas that linger", "Gentle, easy pacing", "A sense of wonder", "Stories that move you"][i]}</small></button>)}</div>
    <div className="toolbar mood-toolbar"><Filters value={filters} onChange={setFilters} /><span className="result-count">{results.length} curated result{results.length === 1 ? "" : "s"}</span></div>
    <Section kicker={`CURATED FOR · ${mood.toUpperCase()}`} title={`${mood} films for right now`}>{results.length ? <MovieGrid items={results} onOpen={props.open} saved={props.saved} onSave={props.save} favourites={props.favourites} onFavourite={props.favourite} showReason ranked hideMatch /> : <EmptyState icon="mood" title="No films fit every choice" copy="Broaden a context filter to reveal more curated results." button="Reset filters" onAction={() => setFilters(defaultFilters)} />}</Section>
  </>;
}

function Collection(props: PageProps) {
  const [tab, setTab] = useState("Watchlist");
  const [filters, setFilters] = useState(defaultFilters);
  const [query, setQuery] = useState("");
  const tabs = ["Watchlist", "Favourites", "Rated movies", "Preferred genres"];
  const source = tab === "Watchlist" ? movies.filter((m) => props.saved.has(m.id)) : tab === "Favourites" ? movies.filter((m) => props.favourites.has(m.id)) : movies.filter((m) => props.ratings[m.id]);
  const items = applyMovieFilters(source, filters).filter((movie) => movie.title.toLowerCase().includes(query.toLowerCase()));
  const average = Object.values(props.ratings).length ? (Object.values(props.ratings).reduce((sum, value) => sum + value, 0) / Object.values(props.ratings).length).toFixed(1) : "—";
  return <>
    <PageHeader eyebrow="YOUR LOCAL PROTOTYPE LIBRARY" title="My collection" copy="The films you saved, loved, and rated on this device. Nothing is stored in a remote account." action={<div className="collection-stats"><span><strong>{props.saved.size}</strong>Saved</span><span><strong>{Object.keys(props.ratings).length}</strong>Rated</span><span><strong>{average}</strong>Avg. rating</span></div>} />
    <div className="tabs">{tabs.map((t) => <button className={tab === t ? "active" : ""} key={t} onClick={() => setTab(t)}>{t}</button>)}</div>
    {tab === "Preferred genres" ? <TasteProfile profile={props.profile} setProfile={props.setProfile} /> : <>
      <div className="toolbar"><Filters value={filters} onChange={setFilters} /><label className="mini-search"><Icon name="search" /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search collection" /></label></div>
      {items.length ? <MovieGrid items={items} onOpen={props.open} saved={props.saved} onSave={props.save} favourites={props.favourites} onFavourite={props.favourite} /> : <EmptyState icon={tab === "Favourites" ? "heart" : tab === "Rated movies" ? "star" : "bookmark"} title={`No ${tab.toLowerCase()} yet`} copy="Use movie cards or details to build this local prototype collection." button="Explore films" onAction={() => props.navigate("Discover")} />}
    </>}
  </>;
}

function TasteProfile({ profile, setProfile }: { profile: TasteProfile; setProfile: React.Dispatch<React.SetStateAction<TasteProfile>> }) {
  return <div className="taste-profile"><div className="profile-panel"><span className="eyebrow">YOUR TASTE, IN FOCUS</span><h2>{profile.genres.length ? profile.genres.slice(0, 2).join(" meets ") : "Your profile is open."}</h2><p>Preferred genres are used by the local demonstration ranker. Update them here or restart onboarding.</p><div className="taste-tags">{profile.genres.map((genre) => <span key={genre}>{genre}</span>)}</div></div><div className="profile-bars"><h3>Preferred genres</h3><div className="preferred-choices">{allGenres.map((genre) => <button key={genre} className={profile.genres.includes(genre) ? "active" : ""} onClick={() => setProfile((current) => { const genres = Array.isArray(current?.genres) ? current.genres : []; return { ...defaultProfile, ...(current || {}), genres: genres.includes(genre) ? genres.filter((item) => item !== genre) : [...genres, genre] }; })}>{genre}{profile.genres.includes(genre) && <Icon name="check" size={14} />}</button>)}</div></div></div>;
}

function Insights() {
  const [empty, setEmpty] = useState(true);
  const [preview, setPreview] = useState(false);
  const [selectedModel, setSelectedModel] = useState("Hybrid Recommendation");
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api.metrics().then((data) => {
      if (!cancelled) setMetrics(data);
    }).catch((caughtError) => {
      if (!cancelled) setError(caughtError instanceof Error ? caughtError.message : "Failed to load evaluation metrics.");
    }).finally(() => {
      if (!cancelled) setLoading(false);
    });
    return () => { cancelled = true; };
  }, [reloadToken]);

  const modelKeyByTab: Record<string, string> = {
    "Popularity Baseline": "popularity",
    "Content-Based Filtering": "content_based",
    "SVD Collaborative Filtering": "collaborative_filtering",
    "Hybrid Recommendation": "hybrid",
  };
  const rows = metrics?.model_list ?? (metrics?.models
    ? Object.entries(metrics.models).map(([key, value]) => ({ model: key, ...value }))
    : []);
  const selectedMetrics = metrics?.models?.[modelKeyByTab[selectedModel]];
  const connected = !!metrics && rows.length > 0;

  const modelInformation: Record<string, { status: string; summary: string; inputs: string }> = {
    "Popularity Baseline": { status: connected ? "Evaluation available" : "Design specification only", summary: "Ranks items from aggregate popularity signals without personalizing to an individual user.", inputs: "Requires item interaction counts or aggregate ratings." },
    "Content-Based Filtering": { status: connected ? "Evaluation available" : "Design specification only", summary: "Would compare movie metadata with a user preference profile built from genres and descriptions.", inputs: "Requires normalized movie features and user preference vectors." },
    "SVD Collaborative Filtering": { status: connected ? "Evaluation available" : "Awaiting trained model", summary: "Would learn latent user and item factors from a real user–movie rating matrix.", inputs: "Requires MovieLens-style user IDs, movie IDs, ratings, and a trained artifact." },
    "Hybrid Recommendation": { status: connected ? "Evaluation available" : "Awaiting experiment pipeline", summary: "Would combine collaborative and content signals, then optionally apply a diversity-aware reranker.", inputs: "Requires outputs from both component models and evaluation configuration." },
  };
  const model = modelInformation[selectedModel];
  const metricDefinitions = [
    ["RMSE", "Lower is better", "Penalizes larger rating prediction errors."],
    ["MAE", "Lower is better", "Average absolute rating prediction error."],
    ["Precision@K", "Higher is better", "Share of top-K recommendations that are relevant."],
    ["Recall@K", "Higher is better", "Share of relevant items recovered in the top K."],
    ["Intra-list diversity", "Higher is broader", "How dissimilar recommended items are to one another."],
    ["Novelty", "Higher is less familiar", "How much the list surfaces less-popular items."],
  ];
  return <>
    <PageHeader eyebrow="RECOMMENDER LAB" title="ML Insights" copy={connected ? "Measured evaluation metrics from the trained backend models." : "A transparent view of how recommendation approaches will be evaluated. The backend does not expose evaluation metrics; only model-loading status is available."} action={loading ? <span className="sample-pill">LOADING METRICS</span> : error ? <span className="sample-pill">METRICS UNAVAILABLE</span> : connected ? <span className="sample-pill">MEASURED · BACKEND CONNECTED</span> : <span className="sample-pill">PROTOTYPE · NO MODEL CONNECTED</span>} />
    <div className="model-tabs">{Object.keys(modelInformation).map((m, i) => <button key={m} className={selectedModel === m ? "active" : ""} onClick={() => setSelectedModel(m)} aria-pressed={selectedModel === m}><span>0{i + 1}</span>{m}</button>)}</div>
    <div className="model-information" aria-live="polite"><div><span>SELECTED APPROACH</span><h2>{selectedModel}</h2></div><p>{model.summary}</p><dl><div><dt>Status</dt><dd>{model.status}</dd></div><div><dt>Data needed</dt><dd>{model.inputs}</dd></div></dl></div>
    {loading ? <div className="insights-empty"><div className="empty-visual"><span className="axis axis-x" /><span className="axis axis-y" /><i className="ghost-bar b1" /><i className="ghost-bar b2" /><i className="ghost-bar b3" /><i className="ghost-line" /></div><div><span className="eyebrow">LOADING EVALUATION METRICS</span><h2>Fetching measured results.</h2><p>The ML Insights page is requesting evaluation metrics from the backend.</p></div></div> :
      error ? <EmptyState icon="info" title="Metrics unavailable" copy={error} button="Retry" onAction={() => setReloadToken((value) => value + 1)} /> :
      !connected ? (empty ? <div className="insights-empty"><div className="empty-visual"><span className="axis axis-x" /><span className="axis axis-y" /><i className="ghost-bar b1" /><i className="ghost-bar b2" /><i className="ghost-bar b3" /><i className="ghost-line" /></div><div><span className="eyebrow">AWAITING EXPERIMENT RESULTS</span><h2>Real metrics belong here.</h2><p>Once evaluation data is connected, this space will compare model accuracy, relevance, diversity, and novelty—without fabricated values.</p><Button onClick={() => setEmpty(false)} variant="secondary">Preview dashboard structure</Button></div></div> :
        <div className="metrics-preview"><div className="preview-note"><Icon name="info" /><span><strong>Illustrative structure only</strong> Values, bars, and outcomes are intentionally omitted until measured experiment output is connected.</span><button onClick={() => setEmpty(true)}>Return to empty state</button></div><div className="metric-grid">{metricDefinitions.map(([metric, direction]) => <div className="metric" key={metric}><span>{metric}</span><strong>—</strong><small>{direction} · no result</small></div>)}</div><div className="chart-grid">{["Standard hybrid vs. diversity-aware", "Relevance vs. diversity trade-off", "New-user cold start", "Recommended genre distribution", "Popular vs. less-popular", "Discovery preference effect"].map((c) => <div className="chart-placeholder" key={c}><div><span>{c}</span><small>Experiment data required</small></div><div className="chart-lines"><i /><i /><i /></div></div>)}</div></div>) :
      <div className="metrics-preview">
        <div className="preview-note"><Icon name="info" /><span><strong>Measured results</strong> RMSE, Precision@10, and Recall@10 from the trained backend models. Relevance threshold: {metrics?.relevance_threshold ?? "—"}; top K: {metrics?.top_k ?? "—"}.</span></div>
        <div className="metric-grid">
          <div className="metric"><span>RMSE</span><strong>{selectedMetrics?.rmse.toFixed(4) ?? "—"}</strong><small>Lower is better · {selectedModel}</small></div>
          <div className="metric"><span>Precision@10</span><strong>{selectedMetrics?.precision_at_10.toFixed(4) ?? "—"}</strong><small>Higher is better · {selectedModel}</small></div>
          <div className="metric"><span>Recall@10</span><strong>{selectedMetrics?.recall_at_10.toFixed(4) ?? "—"}</strong><small>Higher is better · {selectedModel}</small></div>
        </div>
      </div>}
    <Section kicker="MODEL COMPARISON" title="Measured evaluation results by model"><div className="measure-grid">{rows.map((row, index) => <article key={row.model}><span>{String(index + 1).padStart(2, "0")}</span><h3>{row.model}</h3><strong>RMSE {row.rmse.toFixed(4)}</strong><p>Precision@10 {row.precision_at_10.toFixed(4)} · Recall@10 {row.recall_at_10.toFixed(4)}</p></article>)}</div></Section>
    <Section kicker="EVALUATION DICTIONARY" title="How each metric should be read"><div className="metric-dictionary">{metricDefinitions.map(([metric, direction, description]) => <article key={metric}><div><h3>{metric}</h3><span>{direction}</span></div><p>{description}</p></article>)}</div></Section>
    <Section kicker="BACKEND-READY STRUCTURE" title="Experiment payloads to connect"><div className="measure-grid">{[["Accuracy", "RMSE & MAE", "Rating-prediction output by model and dataset split."], ["Ranking", "Precision@K & Recall@K", "Top-K relevance at consistent evaluation cutoffs."], ["Discovery", "Diversity & novelty", "Catalog coverage, popularity, and genre distribution outputs."]].map(([a, b, c], i) => <article key={a}><span>0{i + 1}</span><h3>{a}</h3><strong>{b}</strong><p>{c}</p></article>)}</div></Section>
  </>;
}

function EmptyState({ icon, title, copy, button, onAction }: { icon: string; title: string; copy: string; button: string; onAction: () => void }) {
  return <div className="empty-state"><Icon name={icon} size={28} /><h2>{title}</h2><p>{copy}</p><Button variant="secondary" onClick={onAction}>{button}</Button></div>;
}

function SearchOverlay({ close, openMovie }: { close: () => void; openMovie: (movie: Movie) => void }) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const results = demoCatalogue.search(query);
  useEffect(() => {
    inputRef.current?.focus();
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [close]);
  return <div className="search-overlay" role="dialog" aria-modal="true" aria-label="Search movie catalogue" onMouseDown={close}>
    <div className="search-dialog" onMouseDown={(event) => event.stopPropagation()}>
      <div className="search-dialog-input"><Icon name="search" /><input ref={inputRef} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search titles, genres, moods, languages…" aria-label="Search the demonstration movie catalogue" /><button onClick={close} aria-label="Close search"><Icon name="close" /></button></div>
      <div className="search-dialog-meta"><span>{query ? `${results.length} result${results.length === 1 ? "" : "s"}` : "Browse the demonstration catalogue"}</span><kbd>Esc to close</kbd></div>
      <div className="search-results">{results.length ? results.map((movie) => <button key={movie.id} onClick={() => { openMovie(movie); close(); }}><img src={movie.image} alt="" /><span><strong>{movie.title}</strong><small>{movie.year} · {movie.language} · {movie.genres.join(", ")}</small></span><Icon name="arrow" /></button>) : <div className="search-no-results"><Icon name="search" /><strong>No matching sample films</strong><span>Try a title, genre, mood, language, or theme.</span></div>}</div>
    </div>
  </div>;
}

function DetailModal({ movie, close, props }: { movie: Movie; close: () => void; props: PageProps }) {
  useEffect(() => {
    const handler = (event: KeyboardEvent) => event.key === "Escape" && close();
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [close]);
  const similar = movies.filter((candidate) => candidate.id !== movie.id && candidate.genres.some((genre) => movie.genres.includes(genre))).slice(0, 3);
  const explanation = explainRecommendation(movie, recommendationContext(props));
  return <div className="modal-backdrop" onMouseDown={close}><div className="detail-modal" onMouseDown={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-label={`${movie.title} details`}>
    <button className="modal-close" onClick={close} aria-label="Close details"><Icon name="close" /></button>
    <div className="detail-art"><img src={movie.image} alt="" /><div /></div>
    <div className="detail-content"><span className="sample-pill">SAMPLE MOVIE DATA</span><h2>{movie.title}</h2><p className="detail-meta">{movie.year} · {movie.runtime} · {movie.language} · {movie.genres.join(", ")} · Sample catalog rating {movie.score}/10</p><p className="detail-overview">{movie.overview}</p>
      <div className="detail-actions"><Button icon={props.saved.has(movie.id) ? "check" : "plus"} onClick={() => props.save(movie.id)}>{props.saved.has(movie.id) ? "In watchlist" : "Add to watchlist"}</Button><Button variant="secondary" icon="heart" onClick={() => props.favourite(movie.id)}>{props.favourites.has(movie.id) ? "Favourite" : "Add favourite"}</Button></div>
      <div className="detail-feedback"><div className="stars"><span>Your local rating</span>{[1, 2, 3, 4, 5].map((n) => <button aria-label={`Rate ${n} stars`} className={(props.ratings[movie.id] || 0) >= n ? "active" : ""} onClick={() => props.rate(movie.id, n)} key={n}><Icon name="star" /></button>)}</div><div className="feedback-pair"><button className={props.feedback[movie.id] === "more" ? "active" : ""} onClick={() => props.setMovieFeedback(movie.id, "more")}>More like this</button><button className={props.feedback[movie.id] === "less" ? "active" : ""} onClick={() => props.setMovieFeedback(movie.id, "less")}>Less like this</button></div></div>
      <div className="modal-reason"><Icon name="spark" /><div><strong>Why this was recommended</strong><p>{explanation} This explanation reflects the active demonstration ranking signals only.</p></div></div>
      <h3 className="similar-title">Similar sample films</h3><div className="similar-row">{similar.map((m) => <button key={m.id} onClick={() => props.open(m)}><img src={m.image} alt="" /><span>{m.title}</span></button>)}</div>
    </div>
  </div></div>;
}

function Onboarding({ close, complete }: { close: () => void; complete: (profile: TasteProfile, ratings: Record<number, number>) => void }) {
  const [step, setStep] = useState(1);
  const [genres, setGenres] = useState<string[]>([]);
  const [selectedMovies, setSelectedMovies] = useState<number[]>([]);
  const [ratings, setRatings] = useState<Record<number, number>>({});
  const [discovery, setDiscovery] = useState<"familiar" | "new" | null>(null);
  const [query, setQuery] = useState("");
  const options = ["Drama", "Mystery", "Thriller", "Sci-Fi", "Comedy", "Documentary", "Romance", "Adventure"];
  const movieOptions = movies.filter((movie) => `${movie.title} ${movie.genres.join(" ")}`.toLowerCase().includes(query.toLowerCase()));
  const temporaryProfile: TasteProfile = { genres, favouriteMovies: selectedMovies, discovery: discovery || "new" };
  const onboardingContext: RecommendationContext = { settings: { ...defaultDiscovery, mode: discovery === "familiar" ? "taste" : "surprise" }, profile: temporaryProfile, feedback: {}, ratings, catalogue: demoCatalogue };
  const initialRecommendations = rerankMoviesForDiversity(movies.filter((movie) => !selectedMovies.includes(movie.id)), onboardingContext, 4).map((item) => item.movie);
  const canContinue = step === 1 ? genres.length >= 2 : step === 2 ? selectedMovies.length > 0 : step === 4 ? discovery !== null : true;
  const finish = () => complete({ genres, favouriteMovies: selectedMovies, discovery: discovery || "new" }, ratings);
  return <div className="modal-backdrop onboarding-bg"><div className="onboarding" role="dialog" aria-modal="true">
    <div className="onboarding-top"><div className="brand-mark">C</div><span>BUILD YOUR TASTE PROFILE</span><button onClick={close}>Skip for now</button></div>
    <div className="progress">{[1, 2, 3, 4, 5].map((n) => <i key={n} className={step >= n ? "active" : ""} />)}</div>
    <div className="onboarding-content">
      {step === 1 && <><span className="eyebrow">STEP 1 OF 5</span><h2>What worlds pull you in?</h2><p>Choose at least two. You can change these any time.</p><div className="choice-grid">{options.map((g) => <button className={genres.includes(g) ? "active" : ""} onClick={() => setGenres(genres.includes(g) ? genres.filter((x) => x !== g) : [...genres, g])} key={g}>{g}{genres.includes(g) && <Icon name="check" />}</button>)}</div></>}
      {step === 2 && <><span className="eyebrow">STEP 2 OF 5</span><h2>Pick a few favourites.</h2><p>Search and select at least one sample film closest to your taste.</p><label className="onboarding-search"><Icon name="search" /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search sample movies or genres" /></label><div className="onboarding-posters">{movieOptions.map((m) => <button key={m.id} className={selectedMovies.includes(m.id) ? "active" : ""} onClick={() => setSelectedMovies((current) => current.includes(m.id) ? current.filter((id) => id !== m.id) : [...current, m.id])}><img src={m.image} alt="" /><span>{m.title}</span>{selectedMovies.includes(m.id) && <i><Icon name="check" /></i>}</button>)}</div></>}
      {step === 3 && <><span className="eyebrow">STEP 3 OF 5 · OPTIONAL</span><h2>Make the signal sharper.</h2><p>Rate your selected sample films. More feedback can improve future recommendations.</p><div className="rating-stack">{movies.filter((movie) => selectedMovies.includes(movie.id)).map((m) => <div key={m.id}><img src={m.image} alt="" /><span>{m.title}<small>{m.year}</small></span><div>{[1, 2, 3, 4, 5].map((n) => <button className={(ratings[m.id] || 0) >= n ? "active" : ""} onClick={() => setRatings({ ...ratings, [m.id]: n })} aria-label={`Rate ${m.title} ${n} stars`} key={n}><Icon name="star" /></button>)}</div></div>)}</div></>}
      {step === 4 && <><span className="eyebrow">STEP 4 OF 5</span><h2>Comfort, or discovery?</h2><p>Set your starting point. You’ll always be able to adjust it.</p><div className="discovery-choice"><button className={discovery === "familiar" ? "active" : ""} onClick={() => setDiscovery("familiar")}><Icon name="heart" size={28} /><strong>Keep it familiar</strong><span>More films close to what you already love.</span></button><button className={discovery === "new" ? "active" : ""} onClick={() => setDiscovery("new")}><Icon name="orbit" size={28} /><strong>Lead me somewhere new</strong><span>Broader genres, with a thread back to your taste.</span></button></div></>}
      {step === 5 && <><span className="eyebrow">YOUR FIRST DEMONSTRATION EDIT</span><h2>Your cinema starts here.</h2><p>These initial recommendations use your selected sample genres, films, and ratings. They can improve as you add more local feedback.</p><div className="onboarding-posters result">{initialRecommendations.map((m) => <article key={m.id}><img src={m.image} alt="" /><span>{m.title}</span><small>{m.genres.join(" · ")}</small></article>)}</div></>}
    </div>
    <div className="onboarding-footer"><span>{step === 1 ? `${genres.length} genres selected` : step === 2 ? `${selectedMovies.length} movies selected` : step === 5 ? "Local taste profile ready" : "Your choices remain editable"}</span><div>{step > 1 && <Button variant="ghost" onClick={() => setStep(step - 1)}>Back</Button>}<Button disabled={!canContinue} onClick={() => step === 5 ? finish() : setStep(step + 1)}>{step === 5 ? "Enter CineMind" : "Continue"} <Icon name="arrow" /></Button></div></div>
  </div></div>;
}

export default function App() {
  const [page, setPage] = useState<Page>("Discover");
  const [theme, setTheme] = usePersistentState<"dark" | "light">("cinemind-theme", "dark");
  const [selected, setSelected] = useState<Movie | null>(null);
  const [savedIds, setSavedIds] = usePersistentState<number[]>("cinemind-watchlist", [2, 4, 7, 10]);
  const [favouriteIds, setFavouriteIds] = usePersistentState<number[]>("cinemind-favourites", []);
  const [ratings, setRatings] = usePersistentState<Record<number, number>>("cinemind-ratings", {});
  const [feedback, setFeedback] = usePersistentState<Record<number, Feedback>>("cinemind-feedback", {});
  const [profile, setProfile] = usePersistentState<TasteProfile>("cinemind-profile", defaultProfile);
  const [discovery, setDiscovery] = usePersistentState<DiscoverySettings>("cinemind-discovery", defaultDiscovery);
  const [onboarding, setOnboarding] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [toast, setToast] = useState("");
  const safeSavedIds = Array.isArray(savedIds) ? savedIds.filter(Number.isFinite) : [];
  const safeFavouriteIds = Array.isArray(favouriteIds) ? favouriteIds.filter(Number.isFinite) : [];
  const safeRatings = ratings && typeof ratings === "object" && !Array.isArray(ratings) ? ratings : {};
  const safeFeedback = feedback && typeof feedback === "object" && !Array.isArray(feedback) ? feedback : {};
  const safeTheme = theme === "light" ? "light" : "dark";
  const saved = useMemo(() => new Set(safeSavedIds), [savedIds]);
  const favourites = useMemo(() => new Set(safeFavouriteIds), [favouriteIds]);
  const safeProfile: TasteProfile = {
    ...defaultProfile,
    ...(profile || {}),
    genres: Array.isArray(profile?.genres) ? profile.genres : defaultProfile.genres,
    favouriteMovies: Array.isArray(profile?.favouriteMovies) ? profile.favouriteMovies : [],
    discovery: profile?.discovery === "familiar" ? "familiar" : "new",
  };
  const safeDiscovery: DiscoverySettings = { ...defaultDiscovery, ...(discovery || {}) };
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setSearchOpen(true);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
  const notify = (message: string) => {
    setToast(message);
    window.setTimeout(() => setToast(""), 2600);
  };
  const save = (id: number) => {
    setSavedIds((current) => { const ids = Array.isArray(current) ? current : []; return ids.includes(id) ? ids.filter((item) => item !== id) : [...ids, id]; });
    notify(saved.has(id) ? "Removed from your local watchlist." : "Added to your local watchlist.");
  };
  const favourite = (id: number) => {
    setFavouriteIds((current) => { const ids = Array.isArray(current) ? current : []; return ids.includes(id) ? ids.filter((item) => item !== id) : [...ids, id]; });
    notify(favourites.has(id) ? "Removed from local favourites." : "Added to local favourites.");
  };
  const rate = (id: number, rating: number) => {
    setRatings((current) => ({ ...(current && typeof current === "object" ? current : {}), [id]: rating }));
    const movie = demoCatalogue.getById(id);
    if (movie && rating >= 4) setProfile((current) => ({ ...defaultProfile, ...(current || {}), genres: [...new Set([...(Array.isArray(current?.genres) ? current.genres : []), ...movie.genres])] }));
    notify(`Saved a ${rating}-star rating locally.`);
  };
  const setMovieFeedback = (id: number, value: Feedback) => {
    const removing = safeFeedback[id] === value;
    setFeedback((current) => {
      const next = { ...(current && typeof current === "object" ? current : {}) };
      if (next[id] === value) delete next[id]; else next[id] = value;
      return next;
    });
    const movie = demoCatalogue.getById(id);
    if (movie && !removing) setProfile((current) => {
      const currentGenres = Array.isArray(current?.genres) ? current.genres : [];
      return { ...defaultProfile, ...(current || {}), genres: value === "more" ? [...new Set([...currentGenres, ...movie.genres])] : currentGenres.filter((genre) => !movie.genres.includes(genre) || currentGenres.length <= 1) };
    });
    notify(removing ? "Movie feedback cleared." : `Local preference updated: ${value === "more" ? "more" : "less"} like this.`);
  };
  const navigate = (p: Page) => { setPage(p); setMobileMenu(false); window.scrollTo({ top: 0, behavior: "smooth" }); };
  const props: PageProps = { open: setSelected, saved, save, favourites, favourite, ratings: safeRatings, rate, feedback: safeFeedback, setMovieFeedback, profile: safeProfile, setProfile, discovery: safeDiscovery, setDiscovery, notify, navigate, openSearch: () => setSearchOpen(true) };
  const content = (() => {
    if (page === "Discover") return <Discover {...props} />;
    if (page === "For You") return <ForYou {...props} />;
    if (page === "Beyond Your Taste") return <Beyond {...props} />;
    if (page === "Mood & Context") return <Mood {...props} />;
    if (page === "My Collection") return <Collection {...props} />;
    return <Insights />;
  })();
  const profileProgress = Math.min(100, 20 + safeProfile.genres.length * 8 + safeProfile.favouriteMovies.length * 5 + Object.keys(safeRatings).length * 3);
  const completeOnboarding = (nextProfile: TasteProfile, onboardingRatings: Record<number, number>) => {
    setProfile(nextProfile);
    setRatings((current) => ({ ...(current && typeof current === "object" ? current : {}), ...onboardingRatings }));
    setFavouriteIds((current) => [...new Set([...(Array.isArray(current) ? current : []), ...nextProfile.favouriteMovies])]);
    setDiscovery((current) => ({ ...current, mode: nextProfile.discovery === "familiar" ? "taste" : "surprise" }));
    setOnboarding(false);
    setPage("For You");
    notify("Local taste profile updated. Your demonstration edit is ready.");
  };
  return <div className={`app theme-${safeTheme}`}>
    <aside className={`sidebar ${mobileMenu ? "mobile-open" : ""}`}>
      <button className="brand" onClick={() => navigate("Discover")}><span className="brand-mark">C</span><span><strong>CineMind</strong><small>KNOW YOUR TASTE</small></span></button>
      <nav>{nav.map((item) => <button key={item.name} className={page === item.name ? "active" : ""} onClick={() => navigate(item.name)}><Icon name={item.icon} /><span>{item.name}</span>{item.name === "Beyond Your Taste" && <i>NEW</i>}</button>)}</nav>
      <div className="sidebar-bottom">
        <button className="taste-progress" onClick={() => setOnboarding(true)}><span className="progress-ring">{profileProgress}</span><span><strong>Taste profile</strong><small>Refine local recommendations</small></span><Icon name="arrow" /></button>
        <button className="theme-toggle" onClick={() => setTheme(safeTheme === "dark" ? "light" : "dark")}><Icon name={safeTheme === "dark" ? "moon" : "sun"} /><span>{safeTheme === "dark" ? "Dark" : "Light"} theme</span><i><b /></i></button>
        <button className="profile" onClick={() => setOnboarding(true)}><span>AM</span><span><strong>Alex Morgan</strong><small>Refine taste profile</small></span><Icon name="arrow" /></button>
      </div>
    </aside>
    <header className="mobile-header"><button className="brand" onClick={() => setOnboarding(true)}><span className="brand-mark">C</span><strong>CineMind</strong></button><button onClick={() => setMobileMenu(!mobileMenu)} aria-label="Toggle menu"><Icon name={mobileMenu ? "close" : "menu"} /></button></header>
    <main>{content}<footer><div className="brand"><span className="brand-mark">C</span><strong>CineMind</strong></div><p>Know your taste. Discover beyond it.</p><span>Prototype · Sample movie data · No backend connected</span></footer></main>
    <nav className="mobile-nav" aria-label="Primary mobile navigation">{nav.map((item) => <button key={item.name} onClick={() => navigate(item.name)} className={page === item.name ? "active" : ""}><Icon name={item.icon} /><span>{item.name.replace("Beyond Your Taste", "Beyond").replace("Mood & Context", "Mood").replace("My Collection", "Collection").replace("ML Insights", "Insights")}</span></button>)}</nav>
    {searchOpen && <SearchOverlay close={() => setSearchOpen(false)} openMovie={setSelected} />}
    {selected && <DetailModal movie={selected} close={() => setSelected(null)} props={props} />}
    {onboarding && <Onboarding close={() => setOnboarding(false)} complete={completeOnboarding} />}
    {toast && <div className="toast" role="status"><Icon name="check" />{toast}<small>Stored on this device only</small></div>}
  </div>;
}
