"""Shared pytest setup. Requires Postgres running: `docker compose up -d`."""
import pytest

from app.models import create_tables


@pytest.fixture(scope="session", autouse=True)
def _ensure_tables():
    """Create the schema once before any test runs.

    Several test modules query `tenants`/`embeddings` at module or test-setup time,
    and pytest's collection order is alphabetical -- so on a genuinely fresh database
    the first file collected would otherwise fail with `UndefinedTable`. `create_all`
    is idempotent, so this is safe to run alongside the per-module calls.
    """
    create_tables()
