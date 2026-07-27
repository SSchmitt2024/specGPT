# syntax=docker/dockerfile:1
# Pinned base image digest: a moving tag like `python:3.12-slim` gets
# republished, which invalidates every layer below it (including the pip
# install) and forces a full re-download on an otherwise unchanged build.
# Pinning to a digest keeps the cache stable until we deliberately bump it.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

WORKDIR /app

# Install dependencies first (layer-cached until requirements.txt changes).
#
# The BuildKit cache mount persists pip's downloaded wheels at /root/.cache/pip
# across builds. When requirements.txt changes (or the layer cache is otherwise
# invalidated) pip reuses those cached wheels instead of re-downloading the
# whole dependency tree from PyPI -- the difference between a several-minute and
# a ~15-second rebuild. The cache lives in the mount, not in an image layer, so
# the final image stays slim (this is why we do NOT pass --no-cache-dir, which
# would defeat the mount). Runs as root, before the USER switch below.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

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
