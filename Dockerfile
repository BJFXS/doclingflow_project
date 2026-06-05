FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    DOCLING_ARTIFACTS_PATH=/opt/docling-models \
    DOCLING_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OMP_THREAD_LIMIT=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    TORCH_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    TEST_DOCS_DIR=/data/input \
    OUTPUTS_DIR=/data/output \
    DEFAULT_MEMORY_LIMIT_MB=12288

WORKDIR ${APP_HOME}

# Common runtime packages for document/image parsing in containerized environments.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libreoffice \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p /data/input /data/output ${DOCLING_ARTIFACTS_PATH} \
    && python scripts/preload_docling_models.py

CMD ["python", "-m", "doclingflow", "batch", "/data/input", "-o", "/data/output"]
