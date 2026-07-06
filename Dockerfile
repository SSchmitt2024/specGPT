FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer-cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/

# Railway injects $PORT at runtime; default to 8000 for local docker runs.
ENV PORT=8000
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn src.pipeline.app:app --host $HOST --port $PORT"]
