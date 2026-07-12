"""
Password gate for the specGPT MVP.

Single shared password set at deploy time via the `APP_PASSWORD` env var.
On startup the plaintext is run through `scrypt` (memory-hard, slow to
brute-force) and the plaintext is wiped from memory; subsequent verifies
compare against the stored hash in constant time.

Successful logins receive an `HMAC-SHA256` signed session cookie whose
payload is just the expiry timestamp. The HMAC key (`SESSION_SECRET`) is
the only thing that can mint a valid token, so rotating it invalidates
every outstanding session.

No third-party deps — stdlib only (`hashlib.scrypt`, `hmac`, `secrets`).

Threat model in scope:
  - Casual URL leaks (someone shares the link).
  - Drive-by scrapers, search-engine indexers.
  - Online brute-force against the password (mitigated by scrypt cost +
    per-IP exponential backoff on the login endpoint).
  - Forged session tokens without knowledge of SESSION_SECRET.

Out of scope (no shared-password system can defend against these):
  - Server / env compromise (attacker reads APP_PASSWORD directly).
  - User-chosen password is weak enough to fall to a wordlist.
  - TLS not terminated upstream → cookie sniffable on the wire.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time


# ---------------------------------------------------------------------------
# Cookie + token constants

SESSION_COOKIE = "specgpt_session"
SESSION_LIFETIME_SECONDS = 30 * 24 * 60 * 60  # 30 days

# scrypt cost parameters. Chosen for ~100ms verify time on a typical
# Railway-class CPU (2 vCPU). If you change these, every existing hash
# becomes invalid — re-deploy regenerates the in-memory hash from
# APP_PASSWORD on startup, so the only practical effect is per-request CPU.
_SCRYPT_N = 2 ** 15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_LEN = 16


# ---------------------------------------------------------------------------
# Password hashing

def hash_password(plain: str) -> str:
    """
    Hash `plain` with scrypt + a fresh random salt.

    Returns a self-contained string of the form ``<b64-salt>$<b64-hash>``
    so the salt + parameters travel with the digest. `verify_password`
    re-derives using the embedded salt and constant-time-compares.
    """
    if not isinstance(plain, str) or not plain:
        raise ValueError("password must be a non-empty string")
    salt = secrets.token_bytes(_SALT_LEN)
    derived = hashlib.scrypt(
        plain.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
        maxmem=64 * 1024 * 1024,  # cap memory at 64 MiB so we don't OOM
    )
    return (
        base64.urlsafe_b64encode(salt).decode("ascii")
        + "$"
        + base64.urlsafe_b64encode(derived).decode("ascii")
    )


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time verify of `plain` against `hashed` (from `hash_password`)."""
    if not isinstance(plain, str) or not isinstance(hashed, str):
        return False
    try:
        salt_b64, derived_b64 = hashed.split("$", 1)
        salt = base64.urlsafe_b64decode(salt_b64)
        expected = base64.urlsafe_b64decode(derived_b64)
    except (ValueError, base64.binascii.Error):
        return False
    if not salt or not expected:
        return False
    try:
        actual = hashlib.scrypt(
            plain.encode("utf-8"),
            salt=salt,
            n=_SCRYPT_N,
            r=_SCRYPT_R,
            p=_SCRYPT_P,
            dklen=len(expected),
            maxmem=64 * 1024 * 1024,
        )
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(actual, expected)


# ---------------------------------------------------------------------------
# Session token (HMAC-signed expiry)

def _to_bytes(secret: str | bytes) -> bytes:
    if isinstance(secret, bytes):
        return secret
    return secret.encode("utf-8")


def create_session_token(secret: str | bytes, *, lifetime: int = SESSION_LIFETIME_SECONDS) -> str:
    """
    Mint a session token valid for `lifetime` seconds.

    Format: ``<b64-payload>.<b64-signature>`` where payload is the unix
    expiry timestamp (ascii decimal) and signature is HMAC-SHA256 over
    the payload with `secret` as the key.
    """
    if lifetime <= 0:
        raise ValueError("lifetime must be > 0")
    secret_bytes = _to_bytes(secret)
    if len(secret_bytes) < 16:
        raise ValueError("SESSION_SECRET must be at least 16 bytes; got %d" % len(secret_bytes))
    expiry = int(time.time()) + lifetime
    payload = str(expiry).encode("ascii")
    sig = hmac.new(secret_bytes, payload, hashlib.sha256).digest()
    return (
        base64.urlsafe_b64encode(payload).decode("ascii")
        + "."
        + base64.urlsafe_b64encode(sig).decode("ascii")
    )


def verify_session_token(token: str | None, secret: str | bytes) -> bool:
    """
    Constant-time HMAC verify of `token` + expiry check.

    Returns True only if all of: token is well-formed, signature matches,
    expiry is in the future. Returns False (never raises) on any kind of
    malformed input — important so a tampered cookie can't trigger an
    HTTP 500 that distinguishes "we have your token but it failed
    signature check" from "we have no token at all".
    """
    if not token or not isinstance(token, str):
        return False
    secret_bytes = _to_bytes(secret)
    if "." not in token:
        return False
    try:
        payload_b64, sig_b64 = token.split(".", 1)
        payload = base64.urlsafe_b64decode(payload_b64)
        sig = base64.urlsafe_b64decode(sig_b64)
    except (ValueError, base64.binascii.Error):
        return False
    if not payload or not sig:
        return False
    expected_sig = hmac.new(secret_bytes, payload, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected_sig):
        return False
    try:
        expiry = int(payload.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return False
    return time.time() < expiry


# ---------------------------------------------------------------------------
# Per-IP brute-force throttle (in-memory; fine for single-process MVP)

_FAIL_WINDOW_SECONDS = 300
# Sleep applied BEFORE checking the next login attempt, indexed by recent-failure-count
# (0 failures → 0s, 1 → 0s, 2 → 1s, 3 → 2s, ... capped at 60s).
_FAIL_DELAYS = [0.0, 0.0, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]


class RateLimiter:
    """
    Sliding-window request limiter for authenticated endpoints.

    Keyed by session cookie value (falls back to client IP). Same
    single-process / in-memory caveat as LoginThrottle below. Returns the
    seconds to wait before the next request is allowed (0.0 = allowed now);
    an allowed call is recorded immediately.
    """

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = {}

    def retry_after(self, key: str) -> float:
        now = time.monotonic()
        recent = [t for t in self._hits.get(key, []) if t > now - self.window_seconds]
        if len(recent) >= self.max_requests:
            self._hits[key] = recent
            return recent[0] + self.window_seconds - now
        recent.append(now)
        self._hits[key] = recent
        return 0.0


class LoginThrottle:
    """
    Sliding-window per-IP failure counter.

    Single-process / single-worker only — replace with Redis if we ever
    run >1 uvicorn worker. For an MVP deploy on Railway with the default
    `--workers 1`, this is fine.
    """

    def __init__(self) -> None:
        self._fails: dict[str, list[float]] = {}

    def delay_for(self, ip: str) -> float:
        now = time.monotonic()
        recent = [t for t in self._fails.get(ip, []) if t > now - _FAIL_WINDOW_SECONDS]
        self._fails[ip] = recent
        if not recent:
            return 0.0
        idx = min(len(recent), len(_FAIL_DELAYS) - 1)
        return _FAIL_DELAYS[idx]

    def record_failure(self, ip: str) -> None:
        self._fails.setdefault(ip, []).append(time.monotonic())

    def clear(self, ip: str) -> None:
        self._fails.pop(ip, None)
