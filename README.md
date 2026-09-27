# Offline satellite tile classifier: working slice

A small service that takes a satellite image tile, classifies its land use with a
locally-run model on CPU, stores the result in SQLite, and lets an analyst query it.
It makes no network calls at runtime.

The design note (Part 1) and problem-solving answers (Part 3) were submitted separately
through the submission form. This repo is the working slice (Part 2).

## What's in here

```
app/
  config.py       settings from env vars (threshold, paths, validation limits)
  preprocess.py   the ONE preprocessing function + PREPROCESS_VERSION, shared by train/eval/serve
  model.py        Classifier interface, ResNetClassifier (verifies weights SHA-256 on load)
  db.py           SQLite schema + queries (plain sqlite3, WAL mode)
  main.py         FastAPI routes, upload validation, confidence scoring, JSON logging
scripts/
  inspect_data.py report on the raw dataset (formats, sizes, bands, metadata)
  train.py        fine-tune ResNet18 -> models/resnet18_eurosat.pt + models/manifest.json
  evaluate.py     score the held-out eval_set -> reports/eval.md, reports/eval.json
  eval_clip.py    comparison baseline: zero-shot CLIP on the same eval_set (not used by the service)
  ingest_folder.py  POST every tile in a folder to the running service, print a summary
tests/test_api.py pytest suite (uses a fake classifier, no weights needed)
models/           trained weights + manifest (classes, checksum, versions)
Dockerfile        offline image with weights baked in
```

## Setup

Python 3.10. From the project root:

```bash
pip install -r requirements.txt
```

The dataset is not in this repo. Unzip the assignment tiles into `data/raw/` (giving
`data/raw/candidate_tiles/`, `data/raw/eval_set/`, `data/raw/eval_labels.csv`); see
[data/README.md](data/README.md). The service itself only needs `models/`, so it starts
without the dataset; training, evaluation and the ingest examples need it.

## How to run

Trained weights are already in `models/`, so you can start the service directly.

```bash
uvicorn app.main:app                         # serve on http://localhost:8000
python -m scripts.ingest_folder data/raw/eval_set   # in another terminal: classify all 210 eval tiles
python -m pytest -q                          # 18 tests, ~2s
```

To reproduce the model and the eval report (the only step that uses the network:
torchvision downloads the ImageNet ResNet18 weights once, ~45 MB):

```bash
python -m scripts.inspect_data data/raw   # optional: dataset report
python -m scripts.train                    # ~2.5 min on an M3 CPU
python -m scripts.evaluate                 # writes reports/eval.md and reports/eval.json
```

Settings are env vars, read once at startup (see `app/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.8` | top-1 confidence below this is stored as `needs_review` |
| `DB_PATH` | `data/results.db` | SQLite file |
| `TILES_DIR` | `data/tiles` | where raw uploaded bytes are kept |
| `MANIFEST_PATH` | `models/manifest.json` | points at the weights and their checksum |
| `EXPECTED_WIDTH` / `EXPECTED_HEIGHT` | `64` / `64` | uploads of any other size are rejected |
| `MAX_UPLOAD_BYTES` | `1048576` | upload size limit |
| `BLANK_STD_THRESHOLD` | `1.0` | pixel std-dev below this counts as a blank tile |

## API, with example calls

**`POST /tiles`**: upload one tile (multipart form field `file`).

Validation: must be a readable PNG, 3-band RGB, exactly 64x64, under 1 MB, and not blank.
Otherwise the response is `400` with the reason. The tile is identified by the SHA-256
of its bytes: uploading the same bytes again returns the stored prediction
(`"cached": true`) instead of re-running the model.

```bash
curl -F file=@data/raw/eval_set/tile_057.png localhost:8000/tiles
```
```json
{
  "tile_id": 1,
  "sha256": "fb52cbcf7167380741ce8d05e127ebf707c4d50c87c5892decb3e2f7e39ce52e",
  "cached": false,
  "prediction": {
    "id": 1, "tile_id": 1,
    "model_name": "resnet18-eurosat", "model_version": "20260926-113004",
    "weights_sha": "722fcfc2f5a8cfff4f932162d051660346899dc1277c870d237f7ffcd3c26282",
    "preprocess_version": "v1-rgb-resize224-imagenet-norm",
    "top1_label": "AnnualCrop", "top1_conf": 0.9008673429489136,
    "margin": 0.8066298365592957, "entropy": 0.34471563831860774,
    "status": "confident", "latency_ms": 46.87704099342227,
    "created_at": "2026-09-26T11:43:41.740Z",
    "probs": {"AnnualCrop": 0.900867, "Forest": 3.4e-05, "Highway": 0.00441, "Industrial": 0.000307,
              "Residential": 1.9e-05, "River": 0.094237, "SeaLake": 0.000126}
  }
}
```
(This tile is actually **River**. It is the model's most confident mistake on the eval set,
and a confidence threshold alone does not catch it.)

```bash
curl -F file=@requirements.txt localhost:8000/tiles
# 400 {"detail":"Not a readable image"}
# other examples: {"detail":"Wrong dimensions 32x32: expected 64x64"}
#                 {"detail":"Tile is blank (near-constant pixel values)"}
```

**`GET /predictions?label=&status=&min_conf=&limit=&offset=`**: filtered list, newest first.

```bash
curl "localhost:8000/predictions?status=needs_review"
```
```json
{"count": 1, "limit": 50, "offset": 0, "items": [
  {"id": 3, "tile_id": 3, "top1_label": "River", "top1_conf": 0.4856, "status": "needs_review",
   "sha256": "c0a68c46...", "original_filename": "tile_111.png", "probs": {"...": "..."}, "...": "..."}]}
```
An unknown label gets a 400 listing the valid ones.

**`GET /tiles/{id}`**: the tile row (path, size, bands, dtype, metadata) plus every
prediction made for it (one per model version). Unknown id gets a 404.

**`GET /stats`**: counts for the currently loaded model version.

```json
{"model_version": "20260926-113004", "total": 3, "mean_confidence": 0.7954,
 "by_label": {"AnnualCrop": 1, "Forest": 1, "River": 1},
 "by_status": {"confident": 2, "needs_review": 1}}
```

**`GET /health`**: `200` if the model is loaded and the DB is writable, else `503`.

```json
{"ok": true, "model_loaded": true, "weights_checksum_verified": true,
 "model_version": "20260926-113004", "weights_sha": "722fcfc2...", "db_writable": true}
```

Every request is logged as one JSON line, for example:
```json
{"ts": 1790423021.87, "event": "tile_classified", "sha256": "c0a68c46...", "model_version": "20260926-113004", "label": "River", "conf": 0.4856, "status": "needs_review", "latency_ms": 8.56}
{"ts": 1790423021.96, "event": "tile_rejected", "filename": "requirements.txt", "reason": "Not a readable image"}
```

## Proving it runs offline (Docker)

```bash
docker build -t tile-classifier .    # the only step that needs internet (pip install)
docker run -d --name tiles --network none -v "$PWD/data/raw:/srv/raw:ro" tile-classifier
```

`--network none` gives the container only a loopback interface, so there is no route out:

```bash
docker exec tiles python -c "import socket; socket.create_connection(('1.1.1.1', 443), timeout=3)"
# OSError: [Errno 101] Network is unreachable
```

Since the host can't reach it either, run the ingest from inside the container over loopback:

```bash
docker exec tiles python -m scripts.ingest_folder /srv/raw/eval_set
```
```
Folder: /srv/raw/eval_set  files: 210
Outcomes: {'new': 210}
By status: {'confident': 192, 'needs_review': 18}
By label: {'AnnualCrop': 31, 'Forest': 30, 'Highway': 28, 'Industrial': 27, 'Residential': 30, 'River': 35, 'SeaLake': 29}
Inference latency ms: median=24.7 max=39.0
```

Running it a second time gives `Outcomes: {'cached': 210}` (idempotent). The 18 tiles sent
to review and the per-label counts match `reports/eval.md` exactly, so the service and the
offline eval use the same model and preprocessing.

The image is `python:3.10-slim` with CPU-only torch (879 MB). It contains the code and
`models/`, and runs as a non-root user. The DB and stored tiles go to `/srv/data`: mount
a volume there to keep them. No Hugging Face libraries are used, but `HF_HUB_OFFLINE=1`
and `TRANSFORMERS_OFFLINE=1` are set anyway.

## Model and eval results

- **Model:** ResNet18, ImageNet-pretrained. The last residual block and a new 7-class
  head are fine-tuned; everything else is frozen.
- **Input:** 64x64 RGB tiles upscaled to 224, ImageNet normalisation. Random flips are
  used during training only.
- **Data split:** `candidate_tiles` (1,050) is split 80/20 into train and val,
  stratified, seed 42. The best epoch by val accuracy is kept: 96.2% val.
- **Test set:** `eval_set` (210 tiles, 30 per class) is never used for training or
  model selection.

**Held-out eval_set accuracy: 95.24%** (full report in [reports/eval.md](reports/eval.md))

| class | precision | recall |
|---|---|---|
| AnnualCrop | 0.94 | 0.97 |
| Forest | 1.00 | 1.00 |
| Highway | 0.93 | 0.87 |
| Industrial | 1.00 | 0.90 |
| Residential | 1.00 | 1.00 |
| River | 0.83 | 0.97 |
| SeaLake | 1.00 | 0.97 |

| threshold | coverage (auto-accepted) | accuracy on accepted | sent to review |
|---|---|---|---|
| 0.0 | 100.0% | 95.2% | 0 |
| 0.5 | 98.6% | 96.1% | 3 |
| 0.7 | 96.2% | 98.0% | 8 |
| **0.8 (default)** | **91.4%** | **99.5%** | **18** |
| 0.9 | 88.6% | 99.5% | 24 |
| 0.95 | 81.9% | 100.0% | 38 |

Most errors involve the "linear feature" classes: Highway, River and Industrial get
confused with each other. With only 210 test tiles, each error moves accuracy by about 0.5
percentage points, so treat these numbers as rough.

### Comparison: zero-shot CLIP (not used by the service)

`python -m scripts.eval_clip` (needs `pip install -r requirements-clip.txt`) scores the same
210 tiles with `openai/clip-vit-base-patch32` and **no training on our data**, using two
wordings of the class prompts. Full reports: `reports/eval_clip_*.md`.

| model | accuracy | coverage at 0.8 | accuracy on accepted at 0.8 |
|---|---|---|---|
| ResNet18, fine-tuned (served) | **95.2%** | 91.4% | 99.5% |
| CLIP zero-shot, prompts = raw class names ("SeaLake") | 54.8% | 15.7% | 69.7% |
| CLIP zero-shot, prompts = "a centered satellite photo of a sea or lake" | 61.4% | 26.2% | 89.1% |

## Implemented vs stubbed

**Implemented**
- `POST /tiles`: validate, deduplicate by content hash, store raw bytes, classify, store.
- Confidence handling: the full probability vector plus top-1, margin and entropy are
  stored; a configurable threshold sets `needs_review`.
- Provenance on every prediction: model name and version, weights SHA-256,
  preprocess version.
- Analyst queries: `GET /predictions` (filters and pagination), `GET /tiles/{id}`,
  `GET /stats`.
- `GET /health`: the checksum is verified at startup, and the service refuses to start
  on a mismatch.
- Idempotency is enforced by the DB (`UNIQUE(tile_id, model_version)`), so it is safe
  under concurrent uploads.
- Structured JSON logs, a pytest suite, and the offline Docker image.

**Stubbed or skipped**
- **Review workflow:** the `reviews` table exists, but no endpoint writes to it.
- **Async job queue:** inference runs inside the request (about 10–25 ms per tile).
  Batch or backlog ingestion would need a queue and workers.
- **Spatial queries:** the tiles carry no georeferencing (no EXIF, PNG text or GeoTIFF
  tags; see `scripts/inspect_data.py`), so there is nothing to query spatially.
- **Monitoring and golden-set job:** no scheduled re-check of a known labelled set, no
  drift tracking of confidence or label distributions. The stored fields (confidence,
  entropy, model version, timestamps) and the JSON logs are what such a job would read.
- **Other omissions:** auth, rate limiting, retention and cleanup of stored tiles, and
  DB migrations (schema changes currently mean a fresh DB).
