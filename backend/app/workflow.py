from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAIError

from app.config import GEMINI_MODEL
from app.embeddings import embed_texts
from app.instructions import get_custom_instructions
from app.llm import client as _client
from app.retrieval import retrieve_chunks


class ChatState(TypedDict):
    tenant_id: int
    question: str
    query: str
    category: str
    search_text: str
    chunks: list[str]
    answer: str
    retry_used: bool
    needs_retry: bool


def _classify_node(state: ChatState) -> dict:
    try:
        completion = _client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user's question into exactly one category:\n"
                        "'menu' - dishes, prices, ingredients, allergens, drinks.\n"
                        "'hours_location' - hours, address, parking, reservations.\n"
                        "'policies' - dress code, gratuity/split-check, dietary "
                        "accommodation, pets, private events, gift cards, holiday closures.\n"
                        "'general' - anything else.\n"
                        "Respond with only that word."
                    ),
                },
                {"role": "user", "content": state["question"]},
            ],
        )
    except OpenAIError:
        # Classification only biases the retrieval query -- nothing depends on it,
        # so degrade to the same "general" fallback used for an unusable answer.
        return {"category": "general"}
    category = (completion.choices[0].message.content or "general").strip().lower()
    if category not in ("menu", "hours_location", "policies", "general"):
        category = "general"
    return {"category": category}


def _retrieve_node(state: ChatState) -> dict:
    search_text = state["query"]
    if state["category"]:
        search_text = f"[{state['category']}] {search_text}"
    [query_vector] = embed_texts([search_text])
    chunks = retrieve_chunks(state["tenant_id"], state["question"], query_vector)
    return {"chunks": chunks, "search_text": search_text}


def _generate_node(state: ChatState) -> dict:
    context = "\n\n".join(state["chunks"]) or "(no matching restaurant information found)"
    system_prompt = (
        "You are the AI assistant for a restaurant, answering as the "
        "restaurant itself (first person plural -- 'we' / 'our'). Answer "
        "grounded only in the provided context, and say when something "
        "isn't covered by it. Never assert with certainty that a dish is "
        "safe for a given allergy -- defer to asking staff directly.\n\n"
        f"--- CONTEXT ---\n{context}"
    )
    custom_instructions = get_custom_instructions()
    if custom_instructions:
        system_prompt += (
            f"\n\n--- ADDITIONAL OPERATOR INSTRUCTIONS ---\n{custom_instructions}"
        )
    completion = _client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["question"]},
        ],
    )
    return {"answer": completion.choices[0].message.content or ""}


def _critique_node(state: ChatState) -> dict:
    if state["retry_used"]:
        return {"needs_retry": False}

    try:
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
    except OpenAIError:
        # We already have a usable answer -- skip the retry rather than fail the request.
        return {"needs_retry": False}
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
            "search_text": "",
            "chunks": [],
            "answer": "",
            "retry_used": False,
            "needs_retry": False,
        }
    )
    return result["answer"]
