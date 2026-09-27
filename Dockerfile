# Self-contained, offline image: code + trained weights baked in. No downloads at runtime.
#
#   docker build -t tile-classifier .
#   docker run --network none -v "$PWD/data/raw:/srv/raw:ro" tile-classifier
FROM python:3.10-slim

WORKDIR /srv

# CPU-only torch wheels (the default PyPI ones bundle CUDA and are ~2 GB bigger).
# Installed first and on their own so this slow layer stays cached when requirements change.
# --extra-index-url (not --index-url): torch's own dependencies still come from PyPI.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code and weights. The service verifies the weights' SHA-256 against models/manifest.json at startup.
COPY app/ app/
COPY scripts/ scripts/
COPY models/ models/

# No Hugging Face libraries are used; these are belt-and-braces in case one is added later.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DB_PATH=/srv/data/results.db \
    TILES_DIR=/srv/data/tiles

# Run as an unprivileged user; /srv/data is where the DB and stored tiles live (mount a volume to keep them).
RUN useradd --create-home appuser && mkdir -p /srv/data && chown appuser /srv/data
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
