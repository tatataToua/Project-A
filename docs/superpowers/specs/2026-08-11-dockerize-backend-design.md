# Dockerize the backend

**Status:** Approved
**Phase:** 4 (Scaling) — first sub-project; AWS deploy, widget embed, and Redis caching
are separate specs to follow once this is solid.

## Why

Phase 4 starts with getting the backend running in a container. This is a prerequisite
for any future AWS deploy, but is scoped and designed independently of that decision —
whether/when to actually deploy to AWS (and take on its cost) is deferred to a later
spec. This spec only covers making `docker compose up` bring up Postgres + backend
together, runnable identically to how the backend runs today via `uvicorn --reload`.

## Scope

- **In scope:** `backend/Dockerfile`, `backend/.dockerignore`, adding a `backend`
  service to the root `docker-compose.yml`.
- **Out of scope:** serving the built frontend widget from FastAPI (stays a separate,
  later concern), AWS deployment, Redis caching, Alembic/migrations (not needed — see
  below).

## Architecture

A single `backend/Dockerfile` (python:3.12-slim base) installs `requirements.txt` and
runs `uvicorn app.main:app --host 0.0.0.0 --port 8000`. `docker-compose.yml` gains a
`backend` service alongside the existing `postgres` service, so `docker compose up`
starts both together. The container's `DATABASE_URL` points at the `postgres` service
hostname (not `localhost`, which only resolves inside a container to itself) — every
other setting comes from `backend/.env` via `env_file`. The frontend is untouched:
`npm run dev` keeps proxying to `localhost:8000`, which the container's port mapping
satisfies identically to running uvicorn natively.

Schema creation is already handled at request time via `Base.metadata.create_all()`
(`backend/app/models.py:47`) — there's no Alembic migration step to wire in.

## Dockerfile

```
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`backend/.dockerignore` excludes `.venv`, `__pycache__`, `.env`, `tests/`, `logs/` — so
the image stays lean and secrets can never be baked in even by accident.

## docker-compose.yml changes

Add a `backend` service:

```yaml
  backend:
    build: ./backend
    env_file: backend/.env
    environment:
      DATABASE_URL: postgresql+psycopg://askme:askme@postgres:5432/askme
    depends_on:
      - postgres
    ports:
      - "8000:8000"
```

The explicit `DATABASE_URL` override under `environment:` wins over whatever's in
`backend/.env` (compose applies `environment:` after `env_file:`), which is the point —
`.env`'s `DATABASE_URL` is correct for native `uvicorn --reload` runs (`localhost:5432`)
but wrong inside the compose network, where the Postgres container is reachable at
hostname `postgres`.

## Ingest step

Not baked into the image or auto-run on container start — stays a manual, explicit step
against the running container:

```bash
docker compose exec backend python -m app.ingest two-owls-tavern
```

Documented alongside the existing ingest instructions in `CLAUDE.md`.

## Testing / success criteria

1. `docker compose up -d --build` brings up Postgres + backend with no errors.
2. `curl localhost:8000/health` returns OK.
3. `docker compose exec backend python -m app.ingest two-owls-tavern` populates
   embeddings against the containerized Postgres.
4. The frontend dev server (`npm run dev`, run natively as today) can complete a full
   chat round-trip against the containerized backend — proves the container is a
   drop-in replacement for the native `uvicorn --reload` workflow.
5. The existing native `pytest` suite still passes unchanged (it's not part of the
   image and isn't expected to run inside the container).

## Non-goals (carried forward to later specs)

- Serving `frontend/dist` as static files from FastAPI.
- AWS deployment (ECS/App Runner/EC2) and its cost tradeoffs.
- Redis caching.
- CI build/push of the image.
