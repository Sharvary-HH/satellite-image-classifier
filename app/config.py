"""Settings, read once from environment variables (with defaults for local dev).

Settings is frozen: nothing changes it at runtime. To change a value, set the
env var and restart, e.g.  CONFIDENCE_THRESHOLD=0.9 uvicorn app.main:app
"""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    manifest_path: Path   # models/manifest.json (points at the weights file)
    db_path: Path         # SQLite database file
    tiles_dir: Path       # where raw uploaded tile bytes are stored
    confidence_threshold: float  # top1_conf below this -> status "needs_review"
    expected_width: int   # tiles must be exactly this size...
    expected_height: int
    max_upload_bytes: int
    blank_std_threshold: float  # pixel std-dev below this -> tile rejected as blank


def load_settings() -> Settings:
    env = os.environ.get
    return Settings(
        manifest_path=Path(env("MANIFEST_PATH", "models/manifest.json")),
        db_path=Path(env("DB_PATH", "data/results.db")),
        tiles_dir=Path(env("TILES_DIR", "data/tiles")),
        confidence_threshold=float(env("CONFIDENCE_THRESHOLD", "0.8")),
        expected_width=int(env("EXPECTED_WIDTH", "64")),
        expected_height=int(env("EXPECTED_HEIGHT", "64")),
        max_upload_bytes=int(env("MAX_UPLOAD_BYTES", str(1024 * 1024))),
        blank_std_threshold=float(env("BLANK_STD_THRESHOLD", "1.0")),
    )
