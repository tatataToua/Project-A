# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

"Ask Me" is a tenant-scoped AI assistant that answers questions grounded in a tenant's
own content — originally a single person's background, now generalized to any entity's
content (the current demo tenant, **Two Owls Tavern**, is a fictional restaurant; see
`docs/content/tenants/two-owls-tavern/`). It's built in 5 phases mapping to the "AI
Engineer pyramid" (see `ROADMAP.md` for the full plan and current phase status). It's
also a portfolio piece — the code is expected to reflect real industry practice (e.g.
the Google OAuth login flow, per-tenant data isolation), not shortcuts, since it's meant
to be shown to employers. It's also being explored, unvalidated, as a multi-tenant SaaS
product — see `BUSINESS.md` and `docs/superpowers/specs/2026-08-10-ask-me-saas-design.md`.
Real, reproducible performance/cost/quality numbers (and resume-ready bullets derived
from them) are tracked in `METRICS.md` — re-run `backend/scripts/bench_chat.py` and add
a ledger row whenever a change is worth quantifying.

Currently implemented: Phase 1 (SALT Foundation), Phase 2 (Controlled Intelligence,
superseded), Phase 3 (RAG + LangGraph — tenant-scoped retrieval over pgvector behind a
`POST /chat/{tenant_slug}` endpoint), and Google OAuth login gating the whole app.

## Commands

### Backend (FastAPI, `backend/`)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate                          # Windows
pip install -r requirements.txt
cp .env.example .env                             # then fill in secrets, see below
uvicorn app.main:app --reload --port 8000
```

Run tests: `.venv\Scripts\python.exe -m pytest` (from `backend/`). Run a single test:
`.venv\Scripts\python.exe -m pytest tests/test_auth.py::test_require_user_rejects_missing_session -v`.

### Frontend (Vite + React, `frontend/`)

```bash
cd frontend
npm install
npm run dev        # dev server on :5173, proxies /auth, /chat, /health to :8000
npm run build       # production build to frontend/dist
```

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
service hostname. It also bind-mounts `docs/content` read-only into the container at
`/docs/content` (where `config.py`'s `CONTENT_DIR` resolves inside the container, since
the image only copies `backend/app/`), so ingest sees the same tenant content without
baking it into the image or rebuilding after a content edit. The frontend dev server
(`npm run dev`, run natively) proxies to this containerized backend exactly as it does
to a native `uvicorn --reload` one.

If `GEMINI_BASE_URL` points at a local Ollama server (`http://localhost:11434/v1`),
note that inside the container `localhost` means the container itself, not the host —
override it for containerized calls with `http://host.docker.internal:11434/v1`, e.g.
`docker compose exec -e GEMINI_BASE_URL=http://host.docker.internal:11434/v1 backend python -m app.ingest two-owls-tavern`.

Always access the app via `http://localhost:5173` (not `127.0.0.1`) — the session
cookie's same-origin behavior depends on frontend and backend both using the
`localhost` hostname consistently.

## Architecture

**Backend is a small, flat FastAPI app — not a package-per-feature layout.**
`backend/app/main.py` wires everything together (middleware, routers, the two
non-auth endpoints `/health` and `/chat/{tenant_slug}`); `config.py` is the single
source of env-driven constants (loaded via `python-dotenv` from `backend/.env`);
`db.py` holds the SQLAlchemy engine (Postgres/pgvector, which now stores the
tenant-scoped embedding chunks retrieval runs against); `auth.py` holds the Google
OAuth login flow.
New backend concerns should generally follow this pattern: a focused module in
`app/`, wired into `main.py`, with its config constants added to `config.py`
alongside the existing ones rather than scattered as raw `os.environ` reads.
Cross-cutting helpers already factored out and worth reusing instead of
re-rolling: `db.session_scope()` (session that's always closed),
`tenants.get_tenant_by_slug`/`lookup_tenant_id` (slug -> tenant), and
`trace_log.LOG_PATH`/`load_records` (the chat trace log's location and JSONL
reader, shared by `tracing.py`, `trace_chat.py`, and `trace_report.py`).

**Auth: session-cookie based, not token-based.** Google OAuth 2.0 Authorization
Code flow (via Authlib) gates the entire app — every route except `/health` and
`/auth/*` requires a valid session. The session is a single Starlette
`SessionMiddleware` signed cookie that does double duty: it holds OAuth
`state`/`nonce`/PKCE data during the login handshake, and `{"user": {email,
name, exp}}` afterward. There is deliberately no server-side session store or
revocation list yet (logout just clears the cookie) — this is a known, called-out
gap, not an oversight, and should stay that way unless the roadmap changes.
`app/auth.py`'s `require_user` FastAPI dependency is how routes opt into
requiring a logged-in user (see its use on `/chat/{tenant_slug}` in `main.py`).

**Frontend and backend are made to look same-origin in dev on purpose.**
`vite.config.js` proxies `/auth`, `/chat`, `/health` from `:5173` to `:8000` so
the browser never makes a genuine cross-port request — this avoids CORS and
`SameSite=None` cookie complications entirely. All frontend `fetch` calls to
the backend use **relative paths** (`/chat`, `/auth/me`, not an absolute
`API_URL`) for this reason; don't reintroduce an absolute backend URL in
frontend fetches without also reintroducing CORS handling in `main.py`. In
production the plan is to serve the built frontend from FastAPI itself (see
`ROADMAP.md` Phase 4), which keeps this same-origin property without the dev
proxy.

**`AuthGate.jsx` wraps `ChatWidget.jsx`** and is what `main.jsx` actually
renders. It checks `/auth/me` on mount and shows a "Sign in with Google" screen
or the chat widget accordingly — `ChatWidget.jsx` itself has no auth awareness
beyond reloading the page if a `/chat` call comes back `401` (session expired
mid-conversation).

**Answers come from retrieval, not from a whole file stuffed into the prompt.**
Tenant content lives at `docs/content/tenants/<slug>/*.md` (every `*.md` in that
directory is content — filenames vary per tenant). It's split on markdown heading
boundaries (`chunking.py`), embedded (`embeddings.py`), and stored per-tenant in
pgvector (`models.py`) by an offline ingest step: `python -m app.ingest <slug>`
(`ingest.py`), which must be re-run after editing content. At request time
`POST /chat/{tenant_slug}` in `main.py` resolves the tenant and hands off to
`workflow.py`, a LangGraph classify→retrieve→generate→self-critique graph; the
retrieve step (`retrieval.py`) does hybrid retrieval — pgvector semantic search
fused with Postgres full-text search via reciprocal rank fusion, then reranked
by a local cross-encoder (`reranking.py`) down to the final top-K — always
filtered by `tenant_id` (the tenant isolation boundary), and a failed
self-critique triggers exactly one retry. An optional local `backend/instructions.txt`
(gitignored) is appended to the generate node's system prompt for operator tuning, read
fresh on every request.

**LLM provider is swappable by design.** The backend calls Gemini's free tier
through its OpenAI-compatible endpoint using the `openai` SDK — same code shape
as real OpenAI, different `base_url`/model/key (`GEMINI_BASE_URL`,
`GEMINI_MODEL`, `GEMINI_API_KEY` in `config.py`). There is exactly one client,
constructed in `app/llm.py` and imported by both `embeddings.py` and
`workflow.py`, so switching providers stays a change to `llm.py`/`config.py`
rather than a hunt through call sites. Any OpenAI-compatible endpoint works,
including a local Ollama server (`GEMINI_BASE_URL=http://localhost:11434/v1`).

## Required environment variables (`backend/.env`, not committed)

| Var | Purpose |
|---|---|
| `GEMINI_API_KEY` | Free key from https://aistudio.google.com/apikey |
| `GEMINI_MODEL` | Defaults to `gemini-flash-latest`; `gemini-3.1-flash-lite` is a good pin if the aliased model's free-tier quota is too tight |
| `GEMINI_BASE_URL` | Defaults to Gemini's OpenAI-compatible endpoint; point it at any OpenAI-compatible server (e.g. `http://localhost:11434/v1` for local Ollama) |
| `GEMINI_EMBEDDING_MODEL` | Embedding model, defaults to `gemini-embedding-001` (`nomic-embed-text` for Ollama) |
| `EMBEDDING_DIMENSIONS` | Vector width, defaults to `768`. Must match the existing `embeddings` table — changing it later needs `DROP TABLE embeddings;` and a re-ingest |
| `RETRIEVAL_TOP_K` | Chunks retrieved per question, defaults to `5` |
| `RETRIEVAL_OVERFETCH_K` | Candidates fetched per branch (vector + full-text) before fusion/reranking, defaults to `15` |
| `RERANK_MODEL` | Cross-encoder model for reranking, defaults to `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `DATABASE_URL` | Defaults to the docker-compose Postgres |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | From a Google Cloud Console OAuth client (Web application), redirect URI `http://localhost:8000/auth/callback` |
| `SESSION_SECRET` | Signs the session cookie — generate with `python -c "import secrets; print(secrets.token_hex(32))"`. Must be ≥ 32 chars; if unset/too short the app uses a random per-process key in dev and refuses to start when `SESSION_COOKIE_SECURE=true` |
| `CHAT_MESSAGE_MAX_LENGTH` | Longest accepted `/chat` message, defaults to `2000` characters |
| `FRONTEND_URL` | Where `/auth/callback` redirects after login; defaults to `http://localhost:5173` |
| `SESSION_COOKIE_SECURE` | `false` for local http dev, `true` once served over https |
