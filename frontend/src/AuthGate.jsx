import { useEffect, useState } from "react";
import ChatWidget from "./ChatWidget.jsx";
import TavernInfoRail from "./TavernInfoRail.jsx";
import { ClockIcon, GoogleLogo, LogoutIcon, OwlMark } from "./icons.jsx";
import { TENANT_NAME, TENANT_TAGLINE, formatHoursLabel, getTodayRow, isOpenNow } from "./tenant.js";

function initials(name, email) {
  const source = name || email || "?";
  return source.trim().charAt(0).toUpperCase();
}

function TopbarStatus() {
  const open = isOpenNow();
  const today = getTodayRow();
  return (
    <span className={`status-pill ${open ? "status-pill--open" : "status-pill--closed"}`}>
      <ClockIcon size={13} />
      {open ? "Open now" : "Closed now"}
      {today ? ` · ${formatHoursLabel(today)}` : ""}
    </span>
  );
}

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
      <div className="signin">
        <div className="signin__card">
          <OwlMark size={40} className="signin__mark" />
          <p className="signin__wordmark">Ask Me</p>
          <p className="signin__desc">{TENANT_TAGLINE}</p>
          <TavernInfoRail variant="hero" />
          {loginError && <div className="alert-banner">Login failed — try again.</div>}
          <a className="google-btn" href="/auth/login">
            <GoogleLogo />
            Sign in with Google
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <div className="topbar">
        <div className="topbar__brand">
          <OwlMark size={20} className="topbar__mark" />
          <p className="wordmark">Ask Me</p>
          <span className="tenant-badge">{TENANT_NAME}</span>
          <TopbarStatus />
        </div>
        <div className="topbar__user">
          <span className="avatar">{initials(user?.name, user?.email)}</span>
          <span className="user-name">{user?.name || user?.email}</span>
          <button className="icon-btn" onClick={logout} aria-label="Log out">
            <LogoutIcon />
          </button>
        </div>
      </div>
      <div className="app-shell__body">
        <TavernInfoRail variant="rail" />
        <ChatWidget />
      </div>
    </div>
  );
}
