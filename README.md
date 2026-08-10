# Ask Me

An AI assistant that answers questions about you — background, skills, projects — grounded
in your own content. Built in phases that map to the 5-level AI Engineer pyramid from
[How to Become an AI Engineer FAST (2026)](https://www.youtube.com/watch?v=aAItDrJ8-rE).
See `ROADMAP.md` for the full phase-by-phase plan.

Currently implemented: Phase 1 (SALT Foundation) + first slice of Phase 2 (Controlled
Intelligence) — a working local chat loop with no retrieval yet.

## Setup

### 1. Fill in your content

Edit `docs/content/bio.md` (used today), plus `projects.md` and `resume.md` (used starting
Phase 3). This is the only source of truth for what the assistant knows about you.

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
cp .env.example .env          # then fill in OPENAI_API_KEY
uvicorn app.main:app --reload
```

Verify:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"What is your background?\"}"
```

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the printed local URL and chat with the widget — it talks to the backend on
`localhost:8000`.

## Project structure

```
backend/    FastAPI app: /health, /chat
frontend/   Vite + React embeddable chat widget
docs/content/  Your bio/projects/resume — source material for the assistant
docker-compose.yml  Local Postgres + pgvector
ROADMAP.md  Full 5-phase plan (this is phase 1-2)
```
