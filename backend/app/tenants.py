"""Tenant lookup by slug -- the one place that turns a `tenant_slug` from a
request, CLI argument, or script into a `Tenant` row (or its id). Every entry
point into the app resolves a slug this way, so the query shape stays in one
place.
"""
from sqlalchemy import select

from app.db import session_scope
from app.models import Tenant


def get_tenant_by_slug(session, slug: str) -> Tenant | None:
    return session.scalar(select(Tenant).where(Tenant.slug == slug))


def lookup_tenant_id(slug: str) -> int | None:
    """Resolve a slug to a tenant id in its own short-lived session, for
    callers that only need the id (scripts, evals, the tracing REPL)."""
    with session_scope() as session:
        tenant = get_tenant_by_slug(session, slug)
        return tenant.id if tenant is not None else None
