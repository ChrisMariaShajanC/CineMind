export type Movie = {
  id: number;
  title: string;
  year: number;
  runtime: string;
  genres: string[];
  image: string;
  score: string;
  match: number;
  overview: string;
  language: string;
  runtimeMinutes: number;
  popularity: number;
  moods: string[];
  tags: string[];
};

export type MovieCatalogueAdapter = {
  getAll: () => Movie[];
  getById: (id: number) => Movie | undefined;
  search: (query: string) => Movie[];
  getGenres: () => string[];
  getLanguages: () => string[];
};

const images = {
  portrait: "https://images.unsplash.com/photo-1609062757924-6c2d01b3b422?auto=format&fit=crop&w=900&q=85",
  blonde: "https://images.unsplash.com/photo-1568038479111-87bf80659645?auto=format&fit=crop&w=900&q=85",
  hallway: "https://images.unsplash.com/photo-1619213117400-cd7f8e40381f?auto=format&fit=crop&w=900&q=85",
  desert: "https://images.unsplash.com/photo-1506773090264-ac0b07293a64?auto=format&fit=crop&w=900&q=85",
  dunes: "https://images.unsplash.com/photo-1612599538248-4d7b1ff49a64?auto=format&fit=crop&w=900&q=85",
  horse: "https://images.unsplash.com/photo-1541516342299-a93196e274d8?auto=format&fit=crop&w=900&q=85",
  ocean: "https://images.unsplash.com/photo-1536183922588-166604504d5e?auto=format&fit=crop&w=900&q=85",
  stars: "https://images.unsplash.com/photo-1580659986392-440ea995857c?auto=format&fit=crop&w=900&q=85",
  tower: "https://images.unsplash.com/photo-1679056037984-6977d95bcb65?auto=format&fit=crop&w=900&q=85",
  bridge: "https://images.unsplash.com/photo-1780597257751-75c30466379b?auto=format&fit=crop&w=900&q=85",
  architecture: "https://images.unsplash.com/photo-1784890055131-fbb8a9746ff4?auto=format&fit=crop&w=900&q=85",
  mountain: "https://images.unsplash.com/photo-1765375522831-b01105fbab14?auto=format&fit=crop&w=900&q=85",
};

const demonstrationMovies: Movie[] = [
  { id: 1, title: "The Quiet Meridian", year: 2023, runtime: "1h 58m", runtimeMinutes: 118, genres: ["Drama", "Mystery"], language: "English", popularity: 72, moods: ["Thought-provoking", "Emotional"], tags: ["slow-burn", "character-led", "atmospheric"], image: images.portrait, score: "8.4", match: 96, overview: "A cartographer returns to her coastal hometown and uncovers a decades-old secret hidden in a set of impossible maps." },
  { id: 2, title: "Salt of the Moon", year: 2022, runtime: "2h 11m", runtimeMinutes: 131, genres: ["Drama", "Romance"], language: "French", popularity: 58, moods: ["Emotional", "Relaxing"], tags: ["poetic", "intimate", "night"], image: images.blonde, score: "8.1", match: 93, overview: "Two strangers meet at a remote observatory and spend one luminous night confronting what they left behind." },
  { id: 3, title: "Rooms Without Doors", year: 2024, runtime: "1h 46m", runtimeMinutes: 106, genres: ["Thriller", "Drama"], language: "English", popularity: 84, moods: ["Suspenseful", "Thought-provoking"], tags: ["contained", "tense", "architecture"], image: images.hallway, score: "7.9", match: 91, overview: "An architect wakes inside a building she designed, where each room seems to anticipate her next move." },
  { id: 4, title: "A Field of Silence", year: 2021, runtime: "1h 42m", runtimeMinutes: 102, genres: ["Adventure", "Drama"], language: "Spanish", popularity: 46, moods: ["Adventurous", "Thought-provoking"], tags: ["meditative", "landscape", "human connection"], image: images.desert, score: "8.0", match: 88, overview: "A solitary radio operator crosses an empty frontier to deliver one final message." },
  { id: 5, title: "The Pale Horizon", year: 2024, runtime: "2h 04m", runtimeMinutes: 124, genres: ["Sci-Fi", "Adventure"], language: "English", popularity: 88, moods: ["Adventurous", "Thought-provoking"], tags: ["contemplative", "space", "epic"], image: images.dunes, score: "8.3", match: 86, overview: "On a planet losing its shadows, a geologist follows a signal that may be the last evidence of home." },
  { id: 6, title: "North of Everywhere", year: 2020, runtime: "1h 55m", runtimeMinutes: 115, genres: ["Western", "Adventure"], language: "English", popularity: 35, moods: ["Adventurous", "Suspenseful"], tags: ["moral ambiguity", "landscape", "journey"], image: images.horse, score: "7.7", match: 82, overview: "A rider and a runaway cross a fractured landscape while pursued by competing versions of the truth." },
  { id: 7, title: "After the Tide", year: 2023, runtime: "1h 49m", runtimeMinutes: 109, genres: ["Fantasy", "Drama"], language: "Korean", popularity: 61, moods: ["Emotional", "Relaxing"], tags: ["magical realism", "grief", "coastal"], image: images.ocean, score: "8.2", match: 89, overview: "Every full moon, a flooded village reappears offshore—and one woman swims out to meet it." },
  { id: 8, title: "Constellations", year: 2019, runtime: "1h 37m", runtimeMinutes: 97, genres: ["Documentary"], language: "English", popularity: 29, moods: ["Thought-provoking", "Relaxing"], tags: ["factual", "visual", "human stories"], image: images.stars, score: "8.5", match: 79, overview: "Six astronomers around the world look upward while their lives undergo quiet, seismic change." },
  { id: 9, title: "Hours of Glass", year: 2022, runtime: "2h 06m", runtimeMinutes: 126, genres: ["History", "Drama"], language: "German", popularity: 42, moods: ["Thought-provoking", "Emotional"], tags: ["period", "moral complexity", "craft"], image: images.tower, score: "7.8", match: 84, overview: "A clockmaker must choose between precision and conscience as a city rewrites its own history." },
  { id: 10, title: "Understructure", year: 2024, runtime: "1h 40m", runtimeMinutes: 100, genres: ["Crime", "Thriller"], language: "English", popularity: 76, moods: ["Suspenseful"], tags: ["atmospheric", "urban", "minimal violence"], image: images.bridge, score: "7.6", match: 81, overview: "Beneath a sleeping city, a structural inspector finds evidence that someone is erasing entire streets." },
  { id: 11, title: "The Last Archive", year: 2021, runtime: "1h 52m", runtimeMinutes: 112, genres: ["Mystery", "History"], language: "Italian", popularity: 38, moods: ["Suspenseful", "Thought-provoking"], tags: ["patient", "tactile", "historical"], image: images.architecture, score: "8.0", match: 87, overview: "An archivist follows marginal notes through a ruined library toward a truth meant to disappear." },
  { id: 12, title: "Soft Geometry", year: 2023, runtime: "1h 33m", runtimeMinutes: 93, genres: ["Drama", "Comedy"], language: "Japanese", popularity: 51, moods: ["Feel-good", "Relaxing"], tags: ["observational", "gentle", "found family"], image: images.mountain, score: "7.5", match: 78, overview: "Three residents of an isolated modernist hotel slowly—and awkwardly—become a family." },
];

export function createCatalogueAdapter(source: Movie[]): MovieCatalogueAdapter {
  const movies = source.map((movie) => ({ ...movie, genres: [...movie.genres], moods: [...movie.moods], tags: [...movie.tags] }));
  return {
    getAll: () => movies,
    getById: (id) => movies.find((movie) => movie.id === id),
    search: (query) => {
      const normalized = query.trim().toLowerCase();
      if (!normalized) return movies;
      return movies.filter((movie) => `${movie.title} ${movie.genres.join(" ")} ${movie.language} ${movie.moods.join(" ")} ${movie.tags.join(" ")}`.toLowerCase().includes(normalized));
    },
    getGenres: () => [...new Set(movies.flatMap((movie) => movie.genres))].sort(),
    getLanguages: () => [...new Set(movies.map((movie) => movie.language))].sort(),
  };
}

export const demoCatalogue = createCatalogueAdapter(demonstrationMovies);
export const featuredMovie = demoCatalogue.getById(4)!;
