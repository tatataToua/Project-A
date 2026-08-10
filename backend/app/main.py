from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

from app.config import CONTENT_DIR, OPENAI_API_KEY

app = FastAPI(title="Ask Me")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


def build_system_prompt() -> str:
    bio_path = CONTENT_DIR / "bio.md"
    bio = bio_path.read_text(encoding="utf-8") if bio_path.exists() else ""
    return (
        "You are an AI assistant speaking on behalf of the person described below. "
        "Answer questions about their background, skills, and experience in first person, "
        "as if you were their professional voice. Stay grounded in the provided background "
        "and say when something isn't covered by it.\n\n"
        f"--- BACKGROUND ---\n{bio}"
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": req.message},
        ],
    )
    return ChatResponse(reply=completion.choices[0].message.content or "")
