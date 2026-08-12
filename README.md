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

Optionally, `cp instructions.txt.example instructions.txt` and edit it to tune the
assistant's tone/scope — edits take effect on the next chat message, no restart needed.

Get a free `GEMINI_API_KEY` (no credit card) at https://aistudio.google.com/apikey.
The backend calls it through Gemini's OpenAI-compatible endpoint, so the code is the
same shape it'd be for real OpenAI — swapping providers later is a one-line change
in `backend/app/llm.py` (any OpenAI-compatible endpoint works, including a local
Ollama server).

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

## Debugging the chat workflow

Every real chat turn — including ones sent from the browser widget — auto-logs to
`backend/logs/chat_trace.log` (gitignored) with per-node timing, token counts, and the
critique verdict for the classify → retrieve → generate → critique graph (wired up in
`app/tracing.py`, used by both `/chat/{tenant_slug}` and the tools below). No extra
setup needed — just use the app normally and the log fills in.

To watch it live in a second terminal while you chat in the browser:

```bash
cd backend
.venv\Scripts\python.exe -m app.trace_chat --watch
```

```
You: hello
  [classify ]  0.27s |   41 in /   2 out tok | category=general
  [retrieve ]  0.04s |    6 in /   0 out tok | 5 chunks retrieved
  [generate ]  0.67s |  319 in /  36 out tok | answer="Hello! Welcome to Two Owls Tavern. How can I assist you today? We're a..."
  [critique ]  0.20s |  349 in /   2 out tok | pass

  answer: Hello! Welcome to Two Owls Tavern. How can I assist you today? We're always happy
  to help with reservations or answer any questions about our menu and seating arrangements.

  total: 1.17s | 715 in / 40 out tok | first-pass | critique: pass
```

To review after the fact — first-pass rate, average latency per node, average tokens
per turn, across every turn ever logged:

```bash
.venv\Scripts\python.exe -m app.trace_chat --stats
```

To browse the full history in Excel/Sheets — one row per graph node per turn, full
untruncated answer text, no 70-character preview cutoff:

```bash
.venv\Scripts\python.exe -m app.trace_chat --export-csv
```

See `backend/app/tracing.py` (the tracer) and `backend/app/trace_chat.py` (the CLI) for
details.

## Project structure

```
backend/    FastAPI app: /health, /chat/{tenant_slug}
frontend/   Vite + React embeddable chat widget
docs/content/tenants/<slug>/  Your tenant's markdown content — see docs/content/tenants/two-owls-tavern/ for the demo tenant
docker-compose.yml  Local Postgres + pgvector
ROADMAP.md  Full 5-phase plan (this is phase 1-2)
```
