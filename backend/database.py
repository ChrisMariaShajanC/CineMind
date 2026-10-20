import json
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

BACKEND_DIR = Path(__file__).resolve().parent
DB_PATH = BACKEND_DIR / "data" / "cinemind.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", "").strip() or f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def now() -> datetime:
    return datetime.utcnow()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=now, nullable=False)
    updated_at = Column(DateTime, default=now, onupdate=now, nullable=False)

    preferences = relationship("UserPreference", back_populates="user", uselist=False, cascade="all, delete-orphan")
    favourites = relationship("Favourite", back_populates="user", cascade="all, delete-orphan")
    watchlist = relationship("WatchlistItem", back_populates="user", cascade="all, delete-orphan")
    ratings = relationship("UserRating", back_populates="user", cascade="all, delete-orphan")


class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    languages = Column(Text, default="[]")
    genres = Column(Text, default="[]")
    moods = Column(Text, default="[]")
    release_period = Column(String(64), default="")
    duration = Column(String(64), default="")
    onboarding_completed = Column(Integer, default=0)
    updated_at = Column(DateTime, default=now, onupdate=now, nullable=False)

    user = relationship("User", back_populates="preferences")

    def as_dict(self) -> dict:
        return {
            "languages": json.loads(self.languages or "[]"),
            "genres": json.loads(self.genres or "[]"),
            "moods": json.loads(self.moods or "[]"),
            "release_period": self.release_period or "",
            "duration": self.duration or "",
            "onboarding_completed": bool(self.onboarding_completed),
        }


class Favourite(Base):
    __tablename__ = "favourites"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    source = Column(String(32), nullable=False, default="tmdb")
    external_id = Column(String(64), nullable=False)
    title = Column(String(500), nullable=False)
    poster_url = Column(String(1000), nullable=True)
    genres = Column(Text, default="[]")
    created_at = Column(DateTime, default=now, nullable=False)
    user = relationship("User", back_populates="favourites")
    __table_args__ = (UniqueConstraint("user_id", "source", "external_id", name="uq_fav_user_source_ext"),)


class WatchlistItem(Base):
    __tablename__ = "watchlist"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    source = Column(String(32), nullable=False, default="tmdb")
    external_id = Column(String(64), nullable=False)
    title = Column(String(500), nullable=False)
    poster_url = Column(String(1000), nullable=True)
    genres = Column(Text, default="[]")
    created_at = Column(DateTime, default=now, nullable=False)
    user = relationship("User", back_populates="watchlist")
    __table_args__ = (UniqueConstraint("user_id", "source", "external_id", name="uq_wl_user_source_ext"),)


class UserRating(Base):
    __tablename__ = "user_ratings"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    source = Column(String(32), nullable=False, default="tmdb")
    external_id = Column(String(64), nullable=False)
    rating = Column(Float, nullable=False)
    created_at = Column(DateTime, default=now, nullable=False)
    user = relationship("User", back_populates="ratings")
    __table_args__ = (UniqueConstraint("user_id", "source", "external_id", name="uq_rating_user_source_ext"),)


class TmdbCache(Base):
    __tablename__ = "tmdb_cache"
    cache_key = Column(String(500), primary_key=True)
    payload = Column(Text, nullable=False)
    expires_at = Column(DateTime, nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
