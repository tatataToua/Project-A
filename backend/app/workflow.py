from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from app.config import GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL
from app.embeddings import embed_texts
from app.retrieval import retrieve_chunks

_client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)


class ChatState(TypedDict):
    tenant_id: int
    question: str
    query: str
    category: str
    chunks: list[str]
    answer: str
    retry_used: bool
    needs_retry: bool


def _classify_node(state: ChatState) -> dict:
    completion = _client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user's question into exactly one category: "
                    "'background', 'project', or 'general'. Respond with only that word."
                ),
            },
            {"role": "user", "content": state["question"]},
        ],
    )
    category = (completion.choices[0].message.content or "general").strip().lower()
    if category not in ("background", "project", "general"):
        category = "general"
    return {"category": category}


def _retrieve_node(state: ChatState) -> dict:
    [query_vector] = embed_texts([state["query"]])
    chunks = retrieve_chunks(state["tenant_id"], query_vector)
    return {"chunks": chunks}


def _generate_node(state: ChatState) -> dict:
    context = "\n\n".join(state["chunks"]) or "(no matching background information found)"
    completion = _client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant speaking on behalf of the person described "
                    "below. Answer in first person, grounded only in the provided "
                    "background, and say when something isn't covered by it.\n\n"
                    f"--- BACKGROUND ---\n{context}"
                ),
            },
            {"role": "user", "content": state["question"]},
        ],
    )
    return {"answer": completion.choices[0].message.content or ""}


def _critique_node(state: ChatState) -> dict:
    if state["retry_used"]:
        return {"needs_retry": False}

    completion = _client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Judge whether the ANSWER is grounded in the CONTEXT and actually "
                    "addresses the QUESTION. Respond with only 'pass' or 'fail'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION: {state['question']}\n\n"
                    f"CONTEXT: {chr(10).join(state['chunks'])}\n\n"
                    f"ANSWER: {state['answer']}"
                ),
            },
        ],
    )
    verdict = (completion.choices[0].message.content or "pass").strip().lower()
    if verdict.startswith("fail"):
        return {
            "needs_retry": True,
            "retry_used": True,
            "query": f"{state['question']} (be more specific and grounded)",
        }
    return {"needs_retry": False}


def _route_after_critique(state: ChatState) -> str:
    return "retrieve" if state["needs_retry"] else END


def _build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("classify", _classify_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("generate", _generate_node)
    graph.add_node("critique", _critique_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges("critique", _route_after_critique, {"retrieve": "retrieve", END: END})
    return graph.compile()


_graph = _build_graph()


def run_chat_workflow(tenant_id: int, question: str) -> str:
    result = _graph.invoke(
        {
            "tenant_id": tenant_id,
            "question": question,
            "query": question,
            "category": "",
            "chunks": [],
            "answer": "",
            "retry_used": False,
            "needs_retry": False,
        }
    )
    return result["answer"]
