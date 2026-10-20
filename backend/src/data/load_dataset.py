from pathlib import Path

import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BACKEND_DIR / "data" / "raw" / "ml-1m"

MOVIE_COLUMNS = ["MovieID", "Title", "Genres"]
RATING_COLUMNS = ["UserID", "MovieID", "Rating", "Timestamp"]
USER_COLUMNS = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]

ENCODING = "latin-1"
DELIMITER = "::"


def _require_files() -> None:
    missing = [name for name in ("movies.dat", "ratings.dat", "users.dat") if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing MovieLens 1M files: "
            + ", ".join(missing)
            + f". Expected them in {DATA_DIR}. "
            "Download MovieLens 1M and place movies.dat, ratings.dat, and users.dat there."
        )


def load_movies() -> pd.DataFrame:
    _require_files()
    return pd.read_csv(
        DATA_DIR / "movies.dat",
        sep=DELIMITER,
        engine="python",
        header=None,
        names=MOVIE_COLUMNS,
        encoding=ENCODING,
    )


def load_ratings() -> pd.DataFrame:
    _require_files()
    return pd.read_csv(
        DATA_DIR / "ratings.dat",
        sep=DELIMITER,
        engine="python",
        header=None,
        names=RATING_COLUMNS,
        encoding=ENCODING,
    )


def load_users() -> pd.DataFrame:
    _require_files()
    return pd.read_csv(
        DATA_DIR / "users.dat",
        sep=DELIMITER,
        engine="python",
        header=None,
        names=USER_COLUMNS,
        encoding=ENCODING,
    )


def inspect() -> None:
    _require_files()
    frames = {
        "movies": load_movies(),
        "ratings": load_ratings(),
        "users": load_users(),
    }
    for name, df in frames.items():
        print(f"=== {name} ===")
        print("shape:", df.shape)
        print("columns:", list(df.columns))
        print("first 5 rows:")
        print(df.head(5).to_string(index=False))
        print("missing values:")
        print(df.isna().sum().to_string())
        print()


if __name__ == "__main__":
    inspect()
