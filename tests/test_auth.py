"""
Unit tests for src.pipeline.auth.

Covers password hashing, session token signing, malformed input handling,
and the per-IP login throttle. No network, no FastAPI client — pure unit.

Run:  venv/bin/python3 tests/test_auth.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.auth import (  # noqa: E402
    LoginThrottle,
    SESSION_LIFETIME_SECONDS,
    create_session_token,
    hash_password,
    verify_password,
    verify_session_token,
)


_SECRET = "x" * 32  # 32 bytes meets the 16-byte minimum
_PWD = "correct horse battery staple"


# ---------------------------------------------------------------------------
# Password hashing

def test_hash_then_verify_roundtrip():
    h = hash_password(_PWD)
    assert verify_password(_PWD, h) is True


def test_verify_rejects_wrong_password():
    h = hash_password(_PWD)
    assert verify_password(_PWD + "x", h) is False
    assert verify_password("", h) is False


def test_hash_uses_fresh_salt_each_call():
    """Two calls with the same password must produce different stored strings."""
    h1 = hash_password(_PWD)
    h2 = hash_password(_PWD)
    assert h1 != h2
    # Both must still verify
    assert verify_password(_PWD, h1)
    assert verify_password(_PWD, h2)


def test_verify_rejects_malformed_hash():
    for bad in ["", "no-dollar-sign", "$only-suffix", "prefix$", "$", "not-base64$also-not"]:
        assert verify_password(_PWD, bad) is False


def test_verify_rejects_non_string_input():
    h = hash_password(_PWD)
    assert verify_password(None, h) is False  # type: ignore[arg-type]
    assert verify_password(_PWD, None) is False  # type: ignore[arg-type]


def test_hash_password_rejects_empty():
    try:
        hash_password("")
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty password")


# ---------------------------------------------------------------------------
# Session tokens

def test_signed_token_roundtrip():
    tok = create_session_token(_SECRET)
    assert verify_session_token(tok, _SECRET) is True


def test_token_rejected_with_wrong_secret():
    tok = create_session_token(_SECRET)
    assert verify_session_token(tok, _SECRET + "different") is False


def test_token_rejected_when_signature_tampered():
    tok = create_session_token(_SECRET)
    # Flip a character in the signature half (after the '.')
    payload, sig = tok.split(".", 1)
    flipped_sig = ("A" if sig[0] != "A" else "B") + sig[1:]
    tampered = f"{payload}.{flipped_sig}"
    assert verify_session_token(tampered, _SECRET) is False


def test_token_rejected_when_expired():
    tok = create_session_token(_SECRET, lifetime=1)
    time.sleep(1.2)
    assert verify_session_token(tok, _SECRET) is False


def test_token_rejected_when_malformed():
    for bad in [
        None, "", "no-dot-here",
        ".", "only-after-dot", "before-dot-only.",
        "not-base64.also-not", "abc.def",
    ]:
        assert verify_session_token(bad, _SECRET) is False  # type: ignore[arg-type]


def test_create_token_rejects_short_secret():
    try:
        create_session_token("x" * 8)
    except ValueError:
        return
    raise AssertionError("expected ValueError on short secret")


def test_create_token_accepts_bytes_secret():
    tok = create_session_token(_SECRET.encode("utf-8"))
    assert verify_session_token(tok, _SECRET.encode("utf-8")) is True


def test_create_token_rejects_nonpositive_lifetime():
    try:
        create_session_token(_SECRET, lifetime=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError on lifetime=0")


def test_default_lifetime_is_about_thirty_days():
    assert 25 * 24 * 60 * 60 <= SESSION_LIFETIME_SECONDS <= 35 * 24 * 60 * 60


# ---------------------------------------------------------------------------
# LoginThrottle

def test_throttle_zero_delay_on_first_attempt():
    t = LoginThrottle()
    assert t.delay_for("1.2.3.4") == 0.0


def test_throttle_increases_with_repeated_failures():
    t = LoginThrottle()
    ip = "1.2.3.4"
    last = 0.0
    for _ in range(6):
        t.record_failure(ip)
        d = t.delay_for(ip)
        assert d >= last
        last = d
    # After 6 failures the delay should be well above zero.
    assert t.delay_for(ip) > 0


def test_throttle_clear_resets_ip():
    t = LoginThrottle()
    ip = "1.2.3.4"
    for _ in range(5):
        t.record_failure(ip)
    assert t.delay_for(ip) > 0
    t.clear(ip)
    assert t.delay_for(ip) == 0.0


def test_throttle_per_ip_isolation():
    t = LoginThrottle()
    for _ in range(5):
        t.record_failure("1.1.1.1")
    assert t.delay_for("1.1.1.1") > 0
    assert t.delay_for("2.2.2.2") == 0.0


# ---------------------------------------------------------------------------
# Runner

if __name__ == "__main__":
    import inspect

    failures: list[str] = []
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and inspect.isfunction(obj)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failures.append(f"{name}: AssertionError: {e}")
            print(f"FAIL {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: {type(e).__name__}: {e}")
            print(f"ERROR {name}: {type(e).__name__}: {e}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(0 if not failures else 1)
