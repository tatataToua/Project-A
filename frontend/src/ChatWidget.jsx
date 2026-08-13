import { useEffect, useRef, useState } from "react";
import { SendIcon } from "./icons.jsx";
import { TENANT_EXAMPLE_QUESTIONS, TENANT_NAME, TENANT_SLUG } from "./tenant.js";

const STAGE_LABELS = [
  `Searching ${TENANT_NAME}'s notes…`,
  "Drafting an answer…",
  "Double-checking…",
];

function StageStatus() {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setStage((s) => (s + 1) % STAGE_LABELS.length);
    }, 1100);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="msg-assistant" aria-hidden="true">
      <div className="msg-assistant__meta">Ask Me · {TENANT_NAME}</div>
      <div className="status-cycle__label" key={stage}>
        {STAGE_LABELS[stage]}
      </div>
    </div>
  );
}

export default function ChatWidget() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const textareaRef = useRef(null);
  const transcriptEndRef = useRef(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ block: "end" });
  }, [messages, loading]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  }, [input]);

  async function sendMessage(overrideText) {
    const text = (overrideText ?? input).trim();
    if (!text || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`/chat/${TENANT_SLUG}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ message: text }),
      });
      if (res.status === 401) {
        window.location.reload();
        return;
      }
      // An error response from a proxy or crashed worker may not be JSON at all.
      const data = await res.json().catch((err) => {
        console.error(`Non-JSON response from /chat (status ${res.status})`, err);
        return null;
      });
      if (!res.ok) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data?.detail || `Something went wrong (error ${res.status}).`,
            error: true,
          },
        ]);
        return;
      }
      if (!data || typeof data.reply !== "string") {
        throw new Error("Malformed /chat response");
      }
      setMessages((prev) => [...prev, { role: "assistant", content: data.reply }]);
    } catch (err) {
      console.error("Chat request failed", err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Something went wrong reaching the backend.",
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="chat-column">
      <div className="transcript" aria-live="polite">
      <div className="transcript-inner">
        {messages.length === 0 && !loading && (
          <div className="empty-state">
            <p className="empty-state__title">Ask {TENANT_NAME} anything.</p>
            <div className="empty-state__chips">
              {TENANT_EXAMPLE_QUESTIONS.map((q) => (
                <button key={q} className="chip" onClick={() => sendMessage(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) =>
          m.role === "user" ? (
            <div className="msg-row--user" key={i}>
              <div className="msg-user">{m.content}</div>
            </div>
          ) : (
            <div className="msg-row--assistant" key={i}>
              <div className={`msg-assistant${m.error ? " msg-assistant--error" : ""}`}>
                <div className="msg-assistant__meta">Ask Me · {TENANT_NAME}</div>
                <div className="msg-assistant__body">{m.content}</div>
              </div>
            </div>
          )
        )}

        {loading && (
          <div className="msg-row--assistant">
            <StageStatus />
          </div>
        )}

        <div ref={transcriptEndRef} />
      </div>
      </div>

      <div className="composer">
        <div className="composer__form">
          <textarea
            ref={textareaRef}
            className="composer__input"
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask something..."
            aria-label="Ask a question"
          />
          <button
            className="send-btn"
            onClick={() => sendMessage()}
            disabled={loading || !input.trim()}
            aria-label="Send"
          >
            <SendIcon />
          </button>
        </div>
      </div>
    </div>
  );
}
