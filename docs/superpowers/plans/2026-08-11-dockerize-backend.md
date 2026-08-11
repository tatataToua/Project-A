# Dockerize the Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the FastAPI backend as a Docker image and wire it into
`docker-compose.yml` so `docker compose up` brings up Postgres + backend together,
as a drop-in replacement for the native `uvicorn --reload` dev workflow.

**Architecture:** A `backend/Dockerfile` (python:3.12-slim) installs
`requirements.txt` and runs uvicorn on `0.0.0.0:8000`. `docker-compose.yml` gains a
`backend` service that builds from that Dockerfile, loads `backend/.env` for secrets,
and overrides `DATABASE_URL` to point at the `postgres` service hostname (the
`.env` value of `localhost:5432` is only correct for native runs). No application
code changes — `Base.metadata.create_all()` (already called at startup, see
`backend/app/models.py:47`) handles schema creation with no migration step needed.

**Tech Stack:** Docker, Docker Compose v2, Python 3.12-slim base image.

## Global Constraints

- Backend-only scope: do not add frontend static-file serving, AWS deployment, or
  Redis in this plan — see spec's Non-goals.
- Never bake `backend/.env` or any secret into the image — `.dockerignore` must
  exclude it.
- The container must be behaviorally identical to the native `uvicorn --reload`
  workflow from the app's perspective (same port, same env vars, same DB schema
  behavior) — no shortcuts that only work in one environment.
- `DATABASE_URL` inside compose must point at hostname `postgres`, not `localhost`
  (spec: docker-compose.yml changes section).

---

### Task 1: Backend Dockerfile and .dockerignore

**Files:**
- Create: `backend/Dockerfile`
- Create: `backend/.dockerignore`

**Interfaces:**
- Produces: a buildable image tagged `askme-backend` that runs `uvicorn app.main:app`
  on port 8000 inside the container. Task 2 (compose) builds from this Dockerfile via
  `build: ./backend`.

- [ ] **Step 1: Write `backend/.dockerignore`**

```
.venv/
__pycache__/
*.pyc
.env
tests/
logs/
```

- [ ] **Step 2: Write `backend/Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Build the image and verify it succeeds**

Run (from repo root):
```bash
docker build -t askme-backend ./backend
```
Expected: build completes with `Successfully tagged askme-backend:latest` (or
equivalent final "naming to docker.io/library/askme-backend" line), no errors.

- [ ] **Step 4: Verify secrets are excluded from the build context**

Run:
```bash
docker build -t askme-backend ./backend --no-cache --progress=plain 2>&1 | grep -i "\.env"
```
Expected: no output (empty) — confirms `.env` was never sent to the build context.

- [ ] **Step 5: Commit**

```bash
git add backend/Dockerfile backend/.dockerignore
git commit -m "Add backend Dockerfile"
```

---

### Task 2: Wire backend into docker-compose.yml

**Files:**
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: the `askme-backend` image build context from Task 1
  (`build: ./backend`, using `backend/Dockerfile`).
- Produces: a running `backend` container reachable at `localhost:8000` from the
  host, with Postgres reachable at hostname `postgres:5432` from inside the
  compose network. Task 3 (verification) depends on this being up.

- [ ] **Step 1: Add the `backend` service to `docker-compose.yml`**

Current file:
```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: askme
      POSTGRES_PASSWORD: askme
      POSTGRES_DB: askme
    ports:
      - "5432:5432"
    volumes:
      - askme_pgdata:/var/lib/postgresql/data

volumes:
  askme_pgdata:
```

New file:
```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: askme
      POSTGRES_PASSWORD: askme
      POSTGRES_DB: askme
    ports:
      - "5432:5432"
    volumes:
      - askme_pgdata:/var/lib/postgresql/data

  backend:
    build: ./backend
    env_file: backend/.env
    environment:
      DATABASE_URL: postgresql+psycopg://askme:askme@postgres:5432/askme
    depends_on:
      - postgres
    ports:
      - "8000:8000"

volumes:
  askme_pgdata:
```

- [ ] **Step 2: Bring the stack up and verify both containers start**

Run:
```bash
docker compose up -d --build
docker compose ps
```
Expected: both `postgres` and `backend` services show state `running` (or
`Up`), no restart loops.

- [ ] **Step 3: Verify the backend can reach Postgres**

Run:
```bash
docker compose logs backend --tail 50
```
Expected: no connection-refused / `could not translate host name "postgres"`
errors in the log output — the app started cleanly (uvicorn's startup log line
appears, e.g. "Application startup complete").

- [ ] **Step 4: Verify the health endpoint from the host**

Run:
```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health
```
Expected: `200`

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml
git commit -m "Add backend service to docker-compose.yml"
```

---

### Task 3: Ingest against the container, end-to-end chat verification, and docs

**Files:**
- Modify: `CLAUDE.md` (Infra / Commands section)

**Interfaces:**
- Consumes: the running `backend` + `postgres` containers from Task 2.
- Produces: documented `docker compose` workflow in `CLAUDE.md` alongside the
  existing native-venv instructions; no new code interfaces (this task is
  verification + docs only).

- [ ] **Step 1: Run ingest against the containerized Postgres**

Run:
```bash
docker compose exec backend python -m app.ingest two-owls-tavern
```
Expected: ingest completes without error, prints its usual chunk/embedding
summary output.

- [ ] **Step 2: Verify chunks landed in the containerized DB**

Run:
```bash
docker compose exec postgres psql -U askme -d askme -c "SELECT count(*) FROM embeddings;"
```
Expected: a row count greater than 0.

- [ ] **Step 3: Verify a full chat round-trip through the frontend dev server**

Run (from `frontend/`, in a separate terminal, native — not containerized):
```bash
npm run dev
```
Then open `http://localhost:5173`, sign in with Google, and ask a question about
Two Owls Tavern (e.g. "What are your hours?").
Expected: a grounded answer referencing the tenant content, proving the frontend
(proxying to `localhost:8000`) talks successfully to the containerized backend
exactly as it does to the native one. Stop the dev server after confirming.

- [ ] **Step 4: Run the native test suite unaffected**

Run (from `backend/`, with the venv active):
```bash
.venv\Scripts\python.exe -m pytest
```
Expected: all tests still pass (29/29 or current count) — confirms containerizing
the backend didn't change anything the native dev workflow depends on.

- [ ] **Step 5: Document the Docker workflow in CLAUDE.md**

In `CLAUDE.md`, under the `### Infra` section, replace:
```markdown
### Infra

```bash
docker compose up -d   # Postgres 16 + pgvector, askme/askme/askme, port 5432
```
```
with:
```markdown
### Infra

```bash
docker compose up -d   # Postgres 16 + pgvector, askme/askme/askme, port 5432
```

### Full stack via Docker (backend + Postgres)

```bash
docker compose up -d --build     # Postgres + backend, backend on :8000
docker compose exec backend python -m app.ingest two-owls-tavern
```

Requires `backend/.env` to exist first (see env var table below) — `docker-compose.yml`
loads it via `env_file` and overrides `DATABASE_URL` to point at the `postgres`
service hostname. The frontend dev server (`npm run dev`, run natively) proxies to
this containerized backend exactly as it does to a native `uvicorn --reload` one.
```

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "Document Docker workflow for the backend"
```

---

## Self-Review Notes

- **Spec coverage:** Dockerfile (Task 1), .dockerignore (Task 1), compose changes
  (Task 2), ingest step (Task 3 Step 1), success criteria 1-5 from the spec all map
  to Task 2 Steps 2-4 and Task 3 Steps 1-4 respectively. Non-goals (frontend
  serving, AWS, Redis, CI) are untouched by this plan, consistent with spec scope.
- **No placeholders:** every step has literal commands/file contents, no TBDs.
- **Type/interface consistency:** the `askme-backend` build context path
  (`./backend`) and image behavior (port 8000, env-driven config) are consistent
  between Task 1's Dockerfile and Task 2's compose service definition.
