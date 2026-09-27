"""API tests. They use a FakeClassifier, so they run in about a second and need no weights.

Run from the project root:  python -m pytest -q
"""
import io
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app

CLASSES = ["AnnualCrop", "Forest", "Highway", "Industrial", "Residential", "River", "SeaLake"]


class FakeClassifier:
    """Always returns the same probability vector, so each test controls the outcome."""
    classes = CLASSES
    model_name = "fake"
    model_version = "test-v1"
    weights_sha = "0" * 64
    preprocess_version = "test"

    def __init__(self, probs):
        self.probs = np.array(probs, dtype=np.float32)
        self.calls = 0

    def predict(self, batch):
        self.calls += 1
        return np.tile(self.probs, (batch.shape[0], 1))  # same row for every item in the batch


def one_hot_ish(index: int, conf: float) -> list[float]:
    """conf on `index`, the remainder spread evenly over the other classes."""
    rest = (1 - conf) / (len(CLASSES) - 1)
    return [conf if i == index else rest for i in range(len(CLASSES))]


def png_bytes(size=(64, 64), mode="RGB", blank=False, fmt="PNG") -> bytes:
    rng = np.random.default_rng(0)
    if blank:
        arr = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    else:
        arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr).convert(mode)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def make_client(tmp_path):
    """Factory: make_client(probs) -> (TestClient, FakeClassifier), with a fresh DB per test."""
    settings = Settings(
        manifest_path=Path("unused"),
        db_path=tmp_path / "test.db",
        tiles_dir=tmp_path / "tiles",
        confidence_threshold=0.8,
        expected_width=64,
        expected_height=64,
        max_upload_bytes=1024 * 1024,
        blank_std_threshold=1.0,
    )
    clients = []

    def _make(probs):
        fake = FakeClassifier(probs)
        client = TestClient(create_app(settings, classifier=fake))
        client.__enter__()  # runs the lifespan (startup): creates the DB
        clients.append(client)
        return client, fake

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def upload(client, data: bytes, name="tile.png"):
    return client.post("/tiles", files={"file": (name, data, "image/png")})


def test_happy_path(make_client, tmp_path):
    client, _ = make_client(one_hot_ish(1, 0.95))  # Forest, confident
    resp = upload(client, png_bytes())
    assert resp.status_code == 200
    body = resp.json()
    pred = body["prediction"]
    assert body["cached"] is False
    assert pred["top1_label"] == "Forest"
    assert pred["status"] == "confident"
    assert pred["top1_conf"] == pytest.approx(0.95, abs=1e-6)
    assert set(pred["probs"]) == set(CLASSES)

    # Raw bytes were stored where the DB says they are.
    stored = Path(client.get(f"/tiles/{body['tile_id']}").json()["path"])
    assert stored.read_bytes() == png_bytes()

    # And it shows up in the analyst query.
    listed = client.get("/predictions", params={"label": "Forest"}).json()
    assert listed["count"] == 1


@pytest.mark.parametrize("data, reason", [
    (b"this is not an image", "Not a readable image"),
    (b"", "Empty file"),
    (png_bytes(size=(32, 32)), "Wrong dimensions"),
    (png_bytes(mode="L"), "expected 3-band RGB"),
    (png_bytes(blank=True), "blank"),
    (png_bytes(fmt="JPEG"), "only PNG"),
])
def test_invalid_file_rejected(make_client, data, reason):
    client, fake = make_client(one_hot_ish(0, 0.9))
    resp = upload(client, data)
    assert resp.status_code == 400
    assert reason in resp.json()["detail"]
    assert fake.calls == 0  # never reached the model


def test_idempotent_reupload(make_client, tmp_path):
    client, fake = make_client(one_hot_ish(2, 0.9))
    first = upload(client, png_bytes(), name="a.png").json()
    second = upload(client, png_bytes(), name="b.png").json()  # same bytes, different name

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["tile_id"] == first["tile_id"]
    assert second["prediction"]["id"] == first["prediction"]["id"]
    assert fake.calls == 1  # model ran only once

    conn = sqlite3.connect(tmp_path / "test.db")
    assert conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1


def test_low_confidence_needs_review(make_client):
    client, _ = make_client(one_hot_ish(5, 0.55))  # River at 0.55 < threshold 0.8
    pred = upload(client, png_bytes()).json()["prediction"]
    assert pred["status"] == "needs_review"
    assert pred["margin"] < 0.55

    review = client.get("/predictions", params={"status": "needs_review"}).json()
    assert review["count"] == 1
    assert client.get("/stats").json()["by_status"] == {"needs_review": 1}


@pytest.mark.parametrize("index", range(len(CLASSES)))
def test_label_index_maps_to_correct_name(make_client, index):
    client, _ = make_client(one_hot_ish(index, 0.9))
    pred = upload(client, png_bytes()).json()["prediction"]
    assert pred["top1_label"] == CLASSES[index]
    assert pred["probs"][CLASSES[index]] == pytest.approx(0.9, abs=1e-6)


def test_manifest_class_order_matches_training_folders():
    """The real model's output index i must mean the i-th class folder (sorted), as in train.py."""
    manifest_path = Path("models/manifest.json")
    data_dir = Path("data/raw/candidate_tiles")
    if not manifest_path.exists() or not data_dir.exists():
        pytest.skip("needs trained model and raw data")
    classes = json.loads(manifest_path.read_text())["classes"]
    assert classes == sorted(p.name for p in data_dir.iterdir() if p.is_dir())


def test_health(make_client):
    client, _ = make_client(one_hot_ish(0, 0.9))
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["db_writable"] is True
