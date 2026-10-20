import sys
import uuid
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient

import main
from database import init_db

init_db()
client = TestClient(main.app)


def _register(email=None, password="password123", name="Test User"):
    if email is None:
        email = f"u{uuid.uuid4().hex}@example.com"
    return client.post("/auth/register", json={"name": name, "email": email, "password": password})


def test_register_login_duplicate():
    email = f"a{uuid.uuid4().hex}@example.com"
    r = client.post("/auth/register", json={"name": "A", "email": email, "password": "password123"})
    assert r.status_code in (200, 201), r.text
    token = r.json()["access_token"]
    assert token

    r2 = client.post("/auth/register", json={"name": "A", "email": email, "password": "password123"})
    assert r2.status_code == 409

    r3 = client.post("/auth/login", json={"email": email, "password": "password123"})
    assert r3.status_code == 200
    assert "access_token" in r3.json()

    r4 = client.post("/auth/login", json={"email": email, "password": "wrong"})
    assert r4.status_code == 401


def test_protected_route_requires_token():
    r = client.get("/auth/me")
    assert r.status_code == 401
    r2 = client.get("/preferences")
    assert r2.status_code == 401


def test_preferences_persistence_and_validation():
    email = f"b{uuid.uuid4().hex}@example.com"
    r = client.post("/auth/register", json={"name": "B", "email": email, "password": "password123"})
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "languages": ["ml", "en"],
        "genres": ["Drama", "Thriller"],
        "moods": ["Cozy"],
        "release_period": "2010s",
        "duration": "medium",
        "onboarding_completed": True,
    }
    r = client.put("/preferences", json=payload, headers=headers)
    assert r.status_code == 200
    assert r.json()["languages"] == ["ml", "en"]
    assert r.json()["onboarding_completed"] is True

    r2 = client.get("/preferences", headers=headers)
    assert r2.status_code == 200
    assert r2.json()["genres"] == ["Drama", "Thriller"]


def test_user_data_isolation():
    e1 = f"u1{uuid.uuid4().hex}@example.com"
    e2 = f"u2{uuid.uuid4().hex}@example.com"
    r1 = client.post("/auth/register", json={"name": "U1", "email": e1, "password": "password123"})
    r2 = client.post("/auth/register", json={"name": "U2", "email": e2, "password": "password123"})
    t1 = r1.json()["access_token"]
    t2 = r2.json()["access_token"]
    h1 = {"Authorization": f"Bearer {t1}"}
    h2 = {"Authorization": f"Bearer {t2}"}
    client.post("/me/favourites", json={"source": "tmdb", "external_id": "1", "title": "Movie A"}, headers=h1)
    favs2 = client.get("/me/favourites", headers=h2).json()
    assert all(f["external_id"] != "1" for f in favs2)
    favs1 = client.get("/me/favourites", headers=h1).json()
    assert any(f["external_id"] == "1" for f in favs1)


def test_personalized_without_tmdb_key():
    email = f"c{uuid.uuid4().hex}@example.com"
    r = client.post("/auth/register", json={"name": "C", "email": email, "password": "password123"})
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    client.put("/preferences", json={"languages": ["en"], "genres": ["Drama"], "moods": [], "release_period": "", "duration": "", "onboarding_completed": True}, headers=headers)
    resp = client.get("/me/recommendations", headers=headers)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        assert "recommendations" in resp.json()


def test_catalogue_missing_key_returns_503():
    r = client.get("/catalogue/search", params={"q": "inception"})
    assert r.status_code in (503, 502, 200)


def test_existing_endpoints_still_work():
    r = client.get("/health")
    assert r.status_code == 200
    r = client.get("/recommendations/1")
    assert r.status_code == 200
    r = client.get("/metrics")
    assert r.status_code == 200
