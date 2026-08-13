"""Session-secret resolution: the cookie signing key is the whole of auth, so a
missing one must never silently become a shared constant."""
import importlib

import pytest

import app.config


def _reload_config(monkeypatch, **env: str):
    for key in ("SESSION_SECRET", "SESSION_COOKIE_SECURE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    # config.py reads os.environ at import time, and load_dotenv() won't
    # override what monkeypatch just set.
    return importlib.reload(app.config)


@pytest.fixture(autouse=True)
def _restore_config():
    yield
    importlib.reload(app.config)


def test_explicit_secret_is_used(monkeypatch):
    secret = "x" * 32
    assert _reload_config(monkeypatch, SESSION_SECRET=secret).SESSION_SECRET == secret


def test_missing_secret_falls_back_to_a_random_key_in_dev(monkeypatch):
    first = _reload_config(monkeypatch).SESSION_SECRET
    second = _reload_config(monkeypatch).SESSION_SECRET
    assert len(first) >= 32
    assert first != second


def test_missing_secret_is_fatal_when_cookies_are_secure(monkeypatch):
    with pytest.raises(RuntimeError, match="SESSION_SECRET"):
        _reload_config(monkeypatch, SESSION_COOKIE_SECURE="true")


def test_short_secret_is_fatal_when_cookies_are_secure(monkeypatch):
    with pytest.raises(RuntimeError, match="SESSION_SECRET"):
        _reload_config(monkeypatch, SESSION_SECRET="short", SESSION_COOKIE_SECURE="true")
