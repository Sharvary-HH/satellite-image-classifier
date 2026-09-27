"""SQLite schema and queries. Plain sqlite3, no ORM.

One short-lived connection per request (see `connection()`), so there is no shared
connection object to worry about across threads.
"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tiles (
    id                INTEGER PRIMARY KEY,
    sha256            TEXT NOT NULL UNIQUE,      -- content hash: same bytes => same tile
    path              TEXT NOT NULL,             -- where the raw bytes are stored on disk
    original_filename TEXT,
    width             INTEGER NOT NULL,
    height            INTEGER NOT NULL,
    bands             INTEGER NOT NULL,
    dtype             TEXT NOT NULL,
    received_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    metadata_json     TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id                 INTEGER PRIMARY KEY,
    tile_id            INTEGER NOT NULL REFERENCES tiles(id),
    model_name         TEXT NOT NULL,
    model_version      TEXT NOT NULL,
    weights_sha        TEXT NOT NULL,
    preprocess_version TEXT NOT NULL,
    top1_label         TEXT NOT NULL,
    top1_conf          REAL NOT NULL,
    margin             REAL NOT NULL,            -- top1_conf - top2_conf
    entropy            REAL NOT NULL,            -- uncertainty over all classes, in nats
    probs_json         TEXT NOT NULL,            -- {"Forest": 0.97, ...} full distribution
    status             TEXT NOT NULL CHECK (status IN ('confident', 'needs_review')),
    latency_ms         REAL NOT NULL,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    -- At most one prediction per tile per model version. Enforced by the DB, so two
    -- concurrent uploads of the same tile can't both insert. Also serves as the lookup index.
    UNIQUE (tile_id, model_version)
);

-- STUB: no endpoint writes here yet. This is where analyst corrections would go.
CREATE TABLE IF NOT EXISTS reviews (
    id              INTEGER PRIMARY KEY,
    prediction_id   INTEGER NOT NULL REFERENCES predictions(id),
    corrected_label TEXT NOT NULL,
    reviewer        TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Indexes for the analyst filters in GET /predictions.
CREATE INDEX IF NOT EXISTS idx_pred_label ON predictions(top1_label);
CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(status);
"""


@contextmanager
def connection(db_path: Path):
    """Open a connection, commit on success, roll back on error, always close."""
    conn = sqlite3.connect(db_path, timeout=5)  # wait up to 5s if another writer holds the lock
    conn.row_factory = sqlite3.Row               # rows behave like dicts: row["top1_label"]
    conn.execute("PRAGMA foreign_keys = ON")     # SQLite doesn't enforce FKs unless asked
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connection(db_path) as conn:
        # WAL: readers don't block the writer and vice versa. Stored in the file, so set once.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)


def check_writable(conn: sqlite3.Connection) -> None:
    """Raises if we can't take the write lock. Writes nothing."""
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("ROLLBACK")


# ---------- tiles ----------

def get_tile_by_sha(conn, sha256: str) -> dict | None:
    row = conn.execute("SELECT * FROM tiles WHERE sha256 = ?", (sha256,)).fetchone()
    return dict(row) if row else None


def insert_tile(conn, tile: dict) -> int:
    """Insert a tile row if its sha256 is new; return the tile id either way.

    INSERT OR IGNORE + SELECT means two concurrent uploads of the same bytes
    end up pointing at the same row instead of one of them crashing.
    """
    conn.execute(
        """INSERT OR IGNORE INTO tiles
           (sha256, path, original_filename, width, height, bands, dtype, metadata_json)
           VALUES (:sha256, :path, :original_filename, :width, :height, :bands, :dtype, :metadata_json)""",
        tile,
    )
    return conn.execute("SELECT id FROM tiles WHERE sha256 = ?", (tile["sha256"],)).fetchone()["id"]


def get_tile(conn, tile_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM tiles WHERE id = ?", (tile_id,)).fetchone()
    return _decode(row) if row else None


# ---------- predictions ----------

def get_prediction_for(conn, tile_id: int, model_version: str) -> dict | None:
    """The existing prediction for this tile by this model version, if any (idempotency)."""
    row = conn.execute(
        "SELECT * FROM predictions WHERE tile_id = ? AND model_version = ?",
        (tile_id, model_version),
    ).fetchone()
    return _decode(row) if row else None


def insert_prediction(conn, pred: dict) -> tuple[dict, bool]:
    """Insert unless this tile already has a prediction from this model version.

    Returns (stored prediction, inserted). inserted=False means another request
    won the race and we return its row instead, so the caller always gets one answer.
    """
    cur = conn.execute(
        """INSERT OR IGNORE INTO predictions
           (tile_id, model_name, model_version, weights_sha, preprocess_version,
            top1_label, top1_conf, margin, entropy, probs_json, status, latency_ms)
           VALUES (:tile_id, :model_name, :model_version, :weights_sha, :preprocess_version,
                   :top1_label, :top1_conf, :margin, :entropy, :probs_json, :status, :latency_ms)""",
        pred,
    )
    inserted = cur.rowcount == 1  # 0 when the UNIQUE constraint made it ignore the row
    return get_prediction_for(conn, pred["tile_id"], pred["model_version"]), inserted


def get_predictions_for_tile(conn, tile_id: int) -> list[dict]:
    rows = conn.execute("SELECT * FROM predictions WHERE tile_id = ? ORDER BY id", (tile_id,))
    return [_decode(r) for r in rows]


def list_predictions(conn, label=None, status=None, min_conf=None, limit=50, offset=0) -> list[dict]:
    """Filtered list for analysts. Only filters that were given are added to the WHERE clause."""
    where, params = [], []
    if label is not None:
        where.append("p.top1_label = ?")
        params.append(label)
    if status is not None:
        where.append("p.status = ?")
        params.append(status)
    if min_conf is not None:
        where.append("p.top1_conf >= ?")
        params.append(min_conf)
    sql = """SELECT p.*, t.sha256, t.original_filename
             FROM predictions p JOIN tiles t ON t.id = p.tile_id"""
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY p.id DESC LIMIT ? OFFSET ?"
    params += [limit, offset]
    return [_decode(r) for r in conn.execute(sql, params)]


def stats(conn, model_version: str) -> dict:
    """Counts and mean confidence for one model version (so re-runs of a tile by a
    newer model don't get double-counted)."""
    by_label = conn.execute(
        "SELECT top1_label, COUNT(*) AS n FROM predictions WHERE model_version = ? GROUP BY top1_label",
        (model_version,),
    )
    by_status = conn.execute(
        "SELECT status, COUNT(*) AS n FROM predictions WHERE model_version = ? GROUP BY status",
        (model_version,),
    )
    overall = conn.execute(
        "SELECT COUNT(*) AS n, AVG(top1_conf) AS mean_conf FROM predictions WHERE model_version = ?",
        (model_version,),
    ).fetchone()
    return {
        "model_version": model_version,
        "total": overall["n"],
        "mean_confidence": round(overall["mean_conf"], 4) if overall["mean_conf"] is not None else None,
        "by_label": {r["top1_label"]: r["n"] for r in by_label},
        "by_status": {r["status"]: r["n"] for r in by_status},
    }


def _decode(row: sqlite3.Row) -> dict:
    """Row -> dict, turning stored JSON text columns back into objects."""
    d = dict(row)
    for key in ("probs_json", "metadata_json"):
        if d.get(key) is not None:
            d[key.removesuffix("_json")] = json.loads(d.pop(key))
    return d
