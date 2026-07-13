# Vaaani RAG — full app (FastAPI backend + its mounted static frontend) on a free
# HF Docker Space. CPU-only. Talks to the Vaaani engine Space for the LLM.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 USE_TF=0 USE_FLAX=0 \
    HF_HOME=/tmp/hf-cache \
    PORT=7860

# Native deps: OpenBLAS (turbovec), tesseract (OCR), graphviz, fonts (matplotlib).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libopenblas0 tesseract-ocr graphviz \
        libgl1 libglib2.0-0 fonts-dejavu-core curl && \
    rm -rf /var/lib/apt/lists/*

# turbovec's native .so resolves cblas_sgemm via OpenBLAS (same trick as run.sh).
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libopenblas.so.0

WORKDIR /app
COPY requirements.txt .
# CPU torch first (avoids the multi-GB CUDA build pulled in by sentence-transformers).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Option A (ephemeral + seeded): data/ is baked into the image. HF free disk resets on
# restart, so signups/ingests don't persist — fine for the demo; switch to HF persistent
# storage for production. The LLM lives in the separate engine Space (VAAANI_LLM_BASE_URL).
EXPOSE 7860
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
