import { useEffect, useState } from "react";
import ChatWidget from "./ChatWidget.jsx";

export default function AuthGate() {
  const [status, setStatus] = useState("loading"); // loading | signed-out | signed-in
  const [user, setUser] = useState(null);
  const [loginError, setLoginError] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("error") === "login_failed") {
      setLoginError(true);
      window.history.replaceState({}, "", window.location.pathname);
    }

    fetch("/auth/me", { credentials: "include" })
      .then((res) => {
        if (!res.ok) throw new Error("not authenticated");
        return res.json();
      })
      .then((data) => {
        setUser(data);
        setStatus("signed-in");
      })
      .catch(() => setStatus("signed-out"));
  }, []);

  async function logout() {
    await fetch("/auth/logout", { method: "POST", credentials: "include" });
    window.location.reload();
  }

  if (status === "loading") return null;

  if (status === "signed-out") {
    return (
      <div style={{ maxWidth: 480, margin: "40px auto", fontFamily: "sans-serif", textAlign: "center" }}>
        <h3>Ask Me</h3>
        {loginError && <p style={{ color: "crimson" }}>Login failed, try again.</p>}
        <a href="/auth/login">Sign in with Google</a>
      </div>
    );
  }

  return (
    <div>
      <div style={{ maxWidth: 480, margin: "0 auto", padding: "8px 0", textAlign: "right", fontFamily: "sans-serif" }}>
        <span>{user?.name || user?.email}</span>{" "}
        <button onClick={logout}>Log out</button>
      </div>
      <ChatWidget />
    </div>
  );
}
