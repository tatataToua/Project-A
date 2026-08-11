# Ask Me

An AI assistant that answers questions about you — background, skills, projects — grounded
in your own content. Built in phases that map to the 5-level AI Engineer pyramid from
[How to Become an AI Engineer FAST (2026)](https://www.youtube.com/watch?v=aAItDrJ8-rE).
See `ROADMAP.md` for the full phase-by-phase plan.

Currently implemented: Phase 1 (SALT Foundation) + first slice of Phase 2 (Controlled
Intelligence) — a working local chat loop with no retrieval yet.

This is also being explored as a potential multi-tenant SaaS product, not just a
personal instance — see `BUSINESS.md` for the (unvalidated) business gameplan and
`docs/superpowers/specs/2026-08-10-ask-me-saas-design.md` for the technical design.

## Setup

### 1. Fill in your content

Edit the markdown files in `docs/content/tenants/<slug>/` — for the demo tenant, that's
`docs/content/tenants/two-owls-tavern/{about,menu,faq}.md`. Filenames are up to you: every
`*.md` file in the tenant's directory is ingested. This is the only source of truth for
what the assistant knows.

### 2. Start Postgres

```bash
docker compose up -d
```

### 3. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # then fill in GEMINI_API_KEY
uvicorn app.main:app --reload
```

Get a free `GEMINI_API_KEY` (no credit card) at https://aistudio.google.com/apikey.
The backend calls it through Gemini's OpenAI-compatible endpoint, so the code is the
same shape it'd be for real OpenAI — swapping providers later is a one-line change
in `backend/app/main.py`.

Verify:

```bash
curl http://localhost:8000/health
```

Chat itself can't be tested with curl — `/chat/{tenant_slug}` requires a signed-in Google
session cookie, so try it in the browser after Step 5.

### 4. Ingest your content

```bash
cd backend
.venv\Scripts\python.exe -m app.ingest two-owls-tavern
```

Re-run this any time you edit `docs/content/tenants/two-owls-tavern/*.md`.

### 5. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the printed local URL and chat with the widget — it talks to the backend on
`localhost:8000`.

## Project structure

```
backend/    FastAPI app: /health, /chat/{tenant_slug}
frontend/   Vite + React embeddable chat widget
docs/content/tenants/<slug>/  Your tenant's markdown content — see docs/content/tenants/two-owls-tavern/ for the demo tenant
docker-compose.yml  Local Postgres + pgvector
ROADMAP.md  Full 5-phase plan (this is phase 1-2)
```
