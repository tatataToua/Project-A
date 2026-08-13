"""Requires Postgres running: `docker compose up -d` from the repo root."""
import time

from fastapi.testclient import TestClient

from app import main
from app.auth import require_user
from app.db import SessionLocal
from app.models import EmbeddingChunk, Tenant, create_tables

FAKE_USER = {"email": "chat@example.com", "name": "Chat", "exp": time.time() + 3600}
TENANT_WITH_CONTENT = "test-chat-tenant-with-content"
TENANT_NO_CONTENT = "test-chat-tenant-empty"


def _cleanup():
    session = SessionLocal()
    try:
        for slug in (TENANT_WITH_CONTENT, TENANT_NO_CONTENT):
            tenant = session.query(Tenant).filter_by(slug=slug).one_or_none()
            if tenant is not None:
                session.query(EmbeddingChunk).filter_by(tenant_id=tenant.id).delete()
                session.query(Tenant).filter_by(id=tenant.id).delete()
        session.commit()
    finally:
        session.close()


def _client():
    main.app.dependency_overrides[require_user] = lambda: FAKE_USER
    return TestClient(main.app)


def test_unknown_tenant_returns_404():
    resp = _client().post("/chat/does-not-exist", json={"message": "hi"})
    assert resp.status_code == 404


def test_blank_message_is_rejected():
    resp = _client().post("/chat/does-not-exist", json={"message": "   "})
    assert resp.status_code == 422


def test_oversized_message_is_rejected():
    long_message = "a" * (main.CHAT_MESSAGE_MAX_LENGTH + 1)
    resp = _client().post("/chat/does-not-exist", json={"message": long_message})
    assert resp.status_code == 422


def test_malformed_tenant_slug_is_rejected():
    resp = _client().post("/chat/../../etc/passwd", json={"message": "hi"})
    assert resp.status_code in (404, 422)
    resp = _client().post("/chat/Not_A_Slug", json={"message": "hi"})
    assert resp.status_code == 422


def test_tenant_with_no_content_returns_explicit_message(monkeypatch):
    _cleanup()
    create_tables()
    session = SessionLocal()
    try:
        session.add(Tenant(slug=TENANT_NO_CONTENT, name="Empty", status="active"))
        session.commit()
    finally:
        session.close()

    try:
        resp = _client().post(f"/chat/{TENANT_NO_CONTENT}", json={"message": "hi"})
        assert resp.status_code == 200
        assert "don't have any information" in resp.json()["reply"]
    finally:
        _cleanup()


def test_tenant_with_content_returns_workflow_answer(monkeypatch):
    _cleanup()
    create_tables()
    session = SessionLocal()
    try:
        tenant = Tenant(slug=TENANT_WITH_CONTENT, name="Has Content", status="active")
        session.add(tenant)
        session.flush()
        session.add(
            EmbeddingChunk(
                tenant_id=tenant.id, source_file="bio.md", chunk_index=0,
                chunk_text="chunk", embedding=[0.0] * 768,
            )
        )
        session.commit()
    finally:
        session.close()

    monkeypatch.setattr(main.tracing, "trace_turn", lambda tenant_id, question: ("mocked answer", {}))

    try:
        resp = _client().post(f"/chat/{TENANT_WITH_CONTENT}", json={"message": "hi"})
        assert resp.status_code == 200
        assert resp.json()["reply"] == "mocked answer"
    finally:
        _cleanup()
