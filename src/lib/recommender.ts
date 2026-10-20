import type { Movie, MovieCatalogueAdapter } from "../data/catalogue";

export type Feedback = "more" | "less";
export type DiscoverySettings = {
  mode: "taste" | "surprise";
  familiarity: number;
  diversity: number;
  novelty: number;
  popularity: number;
};
export type TasteProfile = {
  genres: string[];
  favouriteMovies: number[];
  discovery: "familiar" | "new";
};
export type RecommendationContext = {
  settings: DiscoverySettings;
  profile: TasteProfile;
  feedback: Record<number, Feedback>;
  ratings: Record<number, number>;
  catalogue: MovieCatalogueAdapter;
};
export type RankedMovie = { movie: Movie; score: number; explanation: string; diversityReason?: string };

export const defaultDiscovery: DiscoverySettings = { mode: "surprise", familiarity: 38, diversity: 66, novelty: 72, popularity: 44 };
export const defaultProfile: TasteProfile = { genres: ["Drama", "Mystery"], favouriteMovies: [], discovery: "new" };

function overlappingGenres(movie: Movie, ids: number[], catalogue: MovieCatalogueAdapter) {
  const referenceGenres = new Set(ids.flatMap((id) => catalogue.getById(id)?.genres || []));
  return movie.genres.filter((genre) => referenceGenres.has(genre));
}

export function getRecommendationSignals(movie: Movie, context: RecommendationContext) {
  const settings = { ...defaultDiscovery, ...(context.settings || {}) };
  const preferredGenres = Array.isArray(context.profile?.genres) ? context.profile.genres : [];
  const genreMatches = movie.genres.filter((genre) => preferredGenres.includes(genre));
  const favouriteMatches = overlappingGenres(movie, context.profile?.favouriteMovies || [], context.catalogue);
  const highRatedIds = Object.entries(context.ratings || {}).filter(([, rating]) => rating >= 4).map(([id]) => Number(id));
  const lowRatedIds = Object.entries(context.ratings || {}).filter(([, rating]) => rating <= 2).map(([id]) => Number(id));
  const ratingMatches = overlappingGenres(movie, highRatedIds, context.catalogue);
  const lowRatingMatches = overlappingGenres(movie, lowRatedIds, context.catalogue);
  const genreAffinity = genreMatches.length / Math.max(movie.genres.length, 1);
  const unfamiliarity = 1 - genreAffinity;
  const hiddenGem = 1 - movie.popularity / 100;
  const feedbackValue = context.feedback?.[movie.id] === "more" ? .18 : context.feedback?.[movie.id] === "less" ? -.28 : 0;
  const favouriteValue = favouriteMatches.length ? .14 : 0;
  const ratingValue = ratingMatches.length ? .16 : lowRatingMatches.length ? -.14 : 0;
  const modeValue = settings.mode === "surprise" ? unfamiliarity : genreAffinity;
  const genreContribution = genreAffinity * settings.familiarity / 100;
  const modeContribution = modeValue * .25;
  const noveltyContribution = hiddenGem * settings.novelty / 100;
  const popularityContribution = movie.popularity / 100 * settings.popularity / 100;
  const score =
    genreContribution +
    modeContribution +
    noveltyContribution +
    popularityContribution +
    Number(movie.score) / 10 * .1 +
    favouriteValue +
    ratingValue +
    feedbackValue;
  return { settings, genreMatches, favouriteMatches, ratingMatches, lowRatingMatches, hiddenGem, feedbackValue, genreContribution, modeContribution, noveltyContribution, popularityContribution, score };
}

export function explainRecommendation(movie: Movie, context: RecommendationContext) {
  const signals = getRecommendationSignals(movie, context);
  const reasons: string[] = [];
  if (signals.feedbackValue > 0) reasons.push("you asked for more like this film");
  if (signals.feedbackValue < 0) reasons.push("your negative feedback lowers this film’s position");
  if (signals.ratingMatches.length) reasons.push(`it shares ${signals.ratingMatches.join(" and ")} with movies you rated highly`);
  else if (signals.favouriteMatches.length) reasons.push(`it shares ${signals.favouriteMatches.join(" and ")} with a selected favourite`);
  else if (signals.genreMatches.length && (signals.genreContribution > 0 || signals.settings.mode === "taste" && signals.modeContribution > 0)) reasons.push(`it matches your ${signals.genreMatches.join(" and ")} preference`);
  if (signals.noveltyContribution > .2 && movie.popularity < 55) reasons.push("your novelty setting favors its lower sample popularity");
  if (signals.popularityContribution > .35 && movie.popularity >= 65) reasons.push("your popularity setting favors more widely known sample titles");
  if (signals.settings.mode === "surprise" && !signals.genreMatches.length) reasons.push(`Surprise Me introduces ${movie.genres.join(" and ")}`);
  if (!reasons.length) reasons.push("its catalogue rating provides a small quality signal in the demonstration ranking");
  return `${reasons.slice(0, 2).join("; ")}.`;
}

export function rankMovies(items: Movie[], context: RecommendationContext): RankedMovie[] {
  return items
    .map((movie) => ({ movie, score: getRecommendationSignals(movie, context).score, explanation: explainRecommendation(movie, context) }))
    .sort((a, b) => b.score - a.score || a.movie.id - b.movie.id);
}

function genreSimilarity(a: Movie, b: Movie) {
  const union = new Set([...a.genres, ...b.genres]);
  const overlap = a.genres.filter((genre) => b.genres.includes(genre)).length;
  return union.size ? overlap / union.size : 0;
}

export function rerankMoviesForDiversity(items: Movie[], context: RecommendationContext, limit = items.length): RankedMovie[] {
  const candidates = rankMovies(items, context);
  const selected: RankedMovie[] = [];
  const coveredGenres = new Set<string>();
  const strength = Math.max(0, Math.min(100, context.settings.diversity)) / 100;

  while (candidates.length && selected.length < limit) {
    let bestIndex = 0;
    let bestAdjustedScore = Number.NEGATIVE_INFINITY;
    let bestNewGenres: string[] = [];

    candidates.forEach((candidate, index) => {
      const maxSimilarity = selected.length ? Math.max(...selected.map((chosen) => genreSimilarity(candidate.movie, chosen.movie))) : 0;
      const newGenres = candidate.movie.genres.filter((genre) => !coveredGenres.has(genre));
      const varietyBonus = selected.length ? newGenres.length / Math.max(candidate.movie.genres.length, 1) * strength * .2 : 0;
      const repetitionPenalty = selected.length ? maxSimilarity * strength * .35 : 0;
      const adjustedScore = candidate.score + varietyBonus - repetitionPenalty;
      if (adjustedScore > bestAdjustedScore || (adjustedScore === bestAdjustedScore && candidate.movie.id < candidates[bestIndex].movie.id)) {
        bestIndex = index;
        bestAdjustedScore = adjustedScore;
        bestNewGenres = newGenres;
      }
    });

    const [chosen] = candidates.splice(bestIndex, 1);
    const diversityReason = selected.length && strength > 0 && bestNewGenres.length
      ? `Diversity reranking adds ${bestNewGenres.join(" and ")} to reduce repeated genres in this list.`
      : undefined;
    selected.push({ ...chosen, explanation: diversityReason ? `${chosen.explanation} ${diversityReason}` : chosen.explanation, diversityReason });
    chosen.movie.genres.forEach((genre) => coveredGenres.add(genre));
  }

  return selected;
}
