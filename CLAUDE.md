# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

"Ask Me" is an AI assistant that answers questions about the site owner (background,
skills, projects), grounded in their own content. It's built in 5 phases mapping to
the "AI Engineer pyramid" (see `ROADMAP.md` for the full plan and current phase status).
It's also a portfolio piece — the code is expected to reflect real industry practice
(e.g. the Google OAuth login flow), not shortcuts, since it's meant to be shown to
employers.

Currently implemented: Phase 1 (SALT Foundation), first slice of Phase 2 (Controlled
Intelligence — a `/chat` endpoint with no retrieval yet), and Google OAuth login
gating the whole app.

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

Always access the app via `http://localhost:5173` (not `127.0.0.1`) — the session
cookie's same-origin behavior depends on frontend and backend both using the
`localhost` hostname consistently.

## Architecture

**Backend is a small, flat FastAPI app — not a package-per-feature layout.**
`backend/app/main.py` wires everything together (middleware, routers, the two
non-auth endpoints `/health` and `/chat`); `config.py` is the single source of
env-driven constants (loaded via `python-dotenv` from `backend/.env`); `db.py`
holds the SQLAlchemy engine (Postgres/pgvector — provisioned but not yet used
for retrieval, that's Phase 3); `auth.py` holds the Google OAuth login flow.
New backend concerns should generally follow this pattern: a focused module in
`app/`, wired into `main.py`, with its config constants added to `config.py`
alongside the existing ones rather than scattered as raw `os.environ` reads.

**Auth: session-cookie based, not token-based.** Google OAuth 2.0 Authorization
Code flow (via Authlib) gates the entire app — every route except `/health` and
`/auth/*` requires a valid session. The session is a single Starlette
`SessionMiddleware` signed cookie that does double duty: it holds OAuth
`state`/`nonce`/PKCE data during the login handshake, and `{"user": {email,
name, exp}}` afterward. There is deliberately no server-side session store or
revocation list yet (logout just clears the cookie) — this is a known, called-out
gap, not an oversight, and should stay that way unless the roadmap changes.
`app/auth.py`'s `require_user` FastAPI dependency is how routes opt into
requiring a logged-in user (see its use on `/chat` in `main.py`).

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

**The system prompt is built from `docs/content/bio.md` at request time**
(`build_system_prompt()` in `main.py`), read fresh on every `/chat` call, not
cached or embedded — that's the point where Phase 3's RAG (chunk + embed
`docs/content/*.md` into pgvector) will eventually plug in.

**LLM provider is swappable by design.** The backend calls Gemini's free tier
through its OpenAI-compatible endpoint using the `openai` SDK — same code shape
as real OpenAI, different `base_url`/model/key (`GEMINI_BASE_URL`,
`GEMINI_MODEL`, `GEMINI_API_KEY` in `config.py`). Switching providers is meant
to stay a one-line change in `main.py`.

## Required environment variables (`backend/.env`, not committed)

| Var | Purpose |
|---|---|
| `GEMINI_API_KEY` | Free key from https://aistudio.google.com/apikey |
| `GEMINI_MODEL` | Defaults to `gemini-2.5-flash` |
| `DATABASE_URL` | Defaults to the docker-compose Postgres |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | From a Google Cloud Console OAuth client (Web application), redirect URI `http://localhost:8000/auth/callback` |
| `SESSION_SECRET` | Signs the session cookie — generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `FRONTEND_URL` | Where `/auth/callback` redirects after login; defaults to `http://localhost:5173` |
| `SESSION_COOKIE_SECURE` | `false` for local http dev, `true` once served over https |
