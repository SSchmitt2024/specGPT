# syntax=docker/dockerfile:1
# Pinned base image digest: a moving tag like `python:3.12-slim` gets
# republished, which invalidates every layer below it (including the pip
# install) and forces a full re-download on an otherwise unchanged build.
# Pinning to a digest keeps the cache stable until we deliberately bump it.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

WORKDIR /app

# Install dependencies first (layer-cached until requirements.txt changes).
# Plain pip install so the build works on the legacy Docker builder too, not
# just BuildKit. Runs as root, before the USER switch below.
# ponytail: dropped the BuildKit --mount=type=cache; re-add it (with
# DOCKER_BUILDKIT=1) only if slow rebuilds actually become a problem.
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
