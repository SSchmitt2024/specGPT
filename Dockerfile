FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer-cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/

# Run as a non-root user (security hygiene on a VM). The app reads from
# Supabase at runtime and writes nothing to disk, so /app can stay owned by
# root and simply be readable.
RUN useradd --create-home --uid 10001 appuser
USER appuser

# Railway injects $PORT at runtime; default to 8000 for local docker runs.
ENV PORT=8000
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

EXPOSE $PORT

# --proxy-headers/--forwarded-allow-ips let uvicorn honor X-Forwarded-* when
# fronted by a reverse proxy (TLS termination), so client IP + https scheme are
# correct. Complements the app's TRUST_PROXY_HEADERS handling.
CMD ["sh", "-c", "uvicorn src.pipeline.app:app --host $HOST --port $PORT --proxy-headers --forwarded-allow-ips=*"]
