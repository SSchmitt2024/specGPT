FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer-cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download cross-encoder model weights at build time so the first
# request doesn't stall waiting for an 80MB download.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy source
COPY src/ ./src/

# Railway injects $PORT at runtime; default to 8000 for local docker runs.
ENV PORT=8000
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn src.pipeline.app:app --host $HOST --port $PORT"]
