from sqlalchemy import create_engine, text

from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def check_connection() -> bool:
    """Confirm Postgres is reachable and pgvector is enabled. Not used for retrieval yet — that's Phase 3."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        return conn.execute(text("SELECT 1")).scalar() == 1
