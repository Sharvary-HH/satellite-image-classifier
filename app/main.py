"""FastAPI routes. Run with:  uvicorn app.main:app

Flow for POST /tiles:
  validate -> sha256 -> already predicted by this model? return it
  -> save bytes -> preprocess -> infer -> score confidence -> store -> return
"""
import hashlib
import io
import json
import logging
import math
import time
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app import db
from app.config import Settings, load_settings
from app.model import Classifier, ResNetClassifier
from app.preprocess import preprocess

# ---------- structured JSON logging ----------
logger = logging.getLogger("tiles")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_event(event: str, **fields) -> None:
    """One JSON object per line, so logs can be grepped / parsed later without a log server."""
    logger.info(json.dumps({"ts": time.time(), "event": event, **fields}))


# ---------- app setup ----------

def create_app(settings: Settings | None = None, classifier: Classifier | None = None) -> FastAPI:
    """Build the app. Tests pass their own settings and a fake classifier."""
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs once at startup. If the weights are missing or their checksum is wrong,
        # ResNetClassifier raises and the service refuses to start. Fail loudly, not silently.
        app.state.classifier = classifier or ResNetClassifier(settings.manifest_path)
        app.state.settings = settings
        db.init_db(settings.db_path)
        c = app.state.classifier
        log_event("startup", model_version=c.model_version, weights_sha=c.weights_sha,
                  threshold=settings.confidence_threshold)
        yield

    app = FastAPI(title="Tile classifier", lifespan=lifespan)
    _register_routes(app)
    return app


# ---------- helpers ----------

def validate_tile(data: bytes, settings: Settings) -> tuple[Image.Image, np.ndarray]:
    """Reject anything that isn't the kind of tile the model was trained on. 400 with a reason."""
    if len(data) == 0:
        raise HTTPException(400, "Empty file")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(400, f"File too large: {len(data)} bytes > {settings.max_upload_bytes}")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # open() only reads the header; load() decodes pixels and catches truncated files
    except Exception:
        raise HTTPException(400, "Not a readable image")
    if img.format != "PNG":
        raise HTTPException(400, f"Unsupported format {img.format}: only PNG is accepted")
    if img.mode != "RGB":
        raise HTTPException(400, f"Unsupported band layout {img.mode}: expected 3-band RGB")
    if img.size != (settings.expected_width, settings.expected_height):
        raise HTTPException(
            400, f"Wrong dimensions {img.size[0]}x{img.size[1]}: "
                 f"expected {settings.expected_width}x{settings.expected_height}")
    arr = np.asarray(img)
    if arr.std() < settings.blank_std_threshold:
        raise HTTPException(400, "Tile is blank (near-constant pixel values)")
    return img, arr


def score(probs: np.ndarray, classes: list[str]) -> dict:
    """Turn one probability vector into the confidence numbers we store."""
    order = np.argsort(probs)[::-1]  # class indices, most likely first
    top1, top2 = order[0], order[1]
    # Entropy: 0 when all mass is on one class, ln(7)=1.95 when uniform.
    entropy = -sum(float(p) * math.log(float(p)) for p in probs if p > 0)
    return {
        "top1_label": classes[top1],
        "top1_conf": float(probs[top1]),
        "margin": float(probs[top1] - probs[top2]),
        "entropy": entropy,
        "probs_json": json.dumps({c: round(float(p), 6) for c, p in zip(classes, probs)}),
    }


# ---------- routes ----------

def _register_routes(app: FastAPI) -> None:

    # Plain `def` (not `async def`): FastAPI runs it in a worker thread, so slow CPU
    # inference doesn't freeze the event loop for other requests.
    @app.post("/tiles")
    def upload_tile(request: Request, file: UploadFile = File(...)):
        settings: Settings = request.app.state.settings
        clf: Classifier = request.app.state.classifier

        data = file.file.read()
        try:
            img, arr = validate_tile(data, settings)
        except HTTPException as e:
            log_event("tile_rejected", filename=file.filename, reason=e.detail)
            raise
        sha = hashlib.sha256(data).hexdigest()

        # 1) Read-only check. Idempotency: same bytes + same model version => stored result.
        with db.connection(settings.db_path) as conn:
            existing_tile = db.get_tile_by_sha(conn, sha)
            if existing_tile:
                existing = db.get_prediction_for(conn, existing_tile["id"], clf.model_version)
                if existing:
                    log_event("tile_cached", sha256=sha, model_version=clf.model_version,
                              status=existing["status"])
                    return {"tile_id": existing_tile["id"], "sha256": sha, "cached": True,
                            "prediction": existing}

        # 2) Save raw bytes, sharded by the first 2 hex chars so no folder gets huge.
        path = settings.tiles_dir / sha[:2] / f"{sha}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)

        # 3) Inference, outside any DB transaction so we don't hold the write lock meanwhile.
        #    Latency covers preprocess + model, not upload or DB.
        start = time.perf_counter()
        probs = clf.predict(preprocess(img).unsqueeze(0))[0]  # batch of 1 -> first row
        latency_ms = (time.perf_counter() - start) * 1000
        scores = score(probs, clf.classes)
        status = "needs_review" if scores["top1_conf"] < settings.confidence_threshold else "confident"

        # 4) Store tile + prediction together in one transaction.
        with db.connection(settings.db_path) as conn:
            tile_id = db.insert_tile(conn, {
                "sha256": sha,
                "path": str(path),
                "original_filename": file.filename,
                "width": img.width,
                "height": img.height,
                "bands": len(img.getbands()),
                "dtype": str(arr.dtype),
                "metadata_json": json.dumps({"format": img.format, "size_bytes": len(data),
                                             "content_type": file.content_type}),
            })
            pred, inserted = db.insert_prediction(conn, {
                "tile_id": tile_id,
                "model_name": clf.model_name,
                "model_version": clf.model_version,
                "weights_sha": clf.weights_sha,
                "preprocess_version": clf.preprocess_version,
                **scores,
                "status": status,
                "latency_ms": latency_ms,
            })

        # inserted=False: a concurrent upload of the same tile stored first; we return its row.
        log_event("tile_classified" if inserted else "tile_cached", sha256=sha,
                  model_version=clf.model_version, label=pred["top1_label"],
                  conf=round(pred["top1_conf"], 4), status=pred["status"],
                  latency_ms=round(latency_ms, 2))
        return {"tile_id": tile_id, "sha256": sha, "cached": not inserted, "prediction": pred}

    @app.get("/tiles/{tile_id}")
    def get_tile(request: Request, tile_id: int):
        with db.connection(request.app.state.settings.db_path) as conn:
            tile = db.get_tile(conn, tile_id)
            if tile is None:
                raise HTTPException(404, f"Tile {tile_id} not found")
            tile["predictions"] = db.get_predictions_for_tile(conn, tile_id)
        return tile

    @app.get("/predictions")
    def list_predictions(
        request: Request,
        label: str | None = None,
        status: Literal["confident", "needs_review"] | None = None,
        min_conf: float | None = Query(None, ge=0, le=1),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        classes = request.app.state.classifier.classes
        if label is not None and label not in classes:
            raise HTTPException(400, f"Unknown label {label!r}. Valid: {classes}")
        with db.connection(request.app.state.settings.db_path) as conn:
            rows = db.list_predictions(conn, label, status, min_conf, limit, offset)
        return {"count": len(rows), "limit": limit, "offset": offset, "items": rows}

    @app.get("/stats")
    def get_stats(request: Request):
        with db.connection(request.app.state.settings.db_path) as conn:
            return db.stats(conn, request.app.state.classifier.model_version)

    @app.get("/health")
    def health(request: Request):
        clf = request.app.state.classifier
        checks = {
            "model_loaded": clf is not None,
            # ResNetClassifier only exists if the checksum matched at startup.
            "weights_checksum_verified": clf is not None,
            "model_version": clf.model_version,
            "weights_sha": clf.weights_sha,
        }
        try:
            with db.connection(request.app.state.settings.db_path) as conn:
                db.check_writable(conn)
            checks["db_writable"] = True
        except Exception as e:
            checks["db_writable"] = False
            checks["db_error"] = str(e)
        ok = checks["model_loaded"] and checks["db_writable"]
        return JSONResponse({"ok": ok, **checks}, status_code=200 if ok else 503)


app = create_app()
