"use client";
import { useEffect, useState } from "react";

// Top bar with the site's tabs and theme switch; shared by the app (/) and the evaluation page (/eval).
export type Tab = "home" | "coach" | "upload" | "models" | "eval" | "about";
const TABS: [Tab, string, string][] = [["home", "Overview", "/"], ["coach", "Coach", "/?tab=coach"], ["upload", "Upload a video", "/?tab=upload"], ["models", "Models & frameworks", "/?tab=models"],
  ["eval", "Evaluation", "/eval"], ["about", "About", "/?tab=about"]];
type Theme = "system" | "light" | "dark";
const ICON: Record<Theme, React.ReactNode> = {
  system: <><rect x="3" y="4" width="18" height="12" rx="2" /><path d="M8 20h8M12 16v4" /></>,
  light: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
  dark: <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />,
};

function ThemeSwitch() {
  const [t, setT] = useState<Theme>("system");
  useEffect(() => { try { const s = localStorage.getItem("theme"); if (s === "light" || s === "dark") setT(s); } catch {} }, []);
  function next() {
    const n: Theme = t === "system" ? "light" : t === "light" ? "dark" : "system";
    setT(n);
    if (n === "system") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = n;
    try { n === "system" ? localStorage.removeItem("theme") : localStorage.setItem("theme", n); } catch {}
  }
  const label = { system: "Theme: system", light: "Theme: light", dark: "Theme: dark" }[t];
  return (
    <button className="iconbtn" onClick={next} title={`${label} (click to change)`} aria-label={`${label}, click to change`}>
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{ICON[t]}</svg>
    </button>
  );
}

export default function Nav({ tab, onTab }: { tab: Tab; onTab?: (t: Tab) => void }) {
  return (
    <header className="topbar">
      <a href="/" className="brand" aria-label="AI Tennis Coach home">
        <span className="logo"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="12" cy="12" r="9" /><path d="M5.6 5.6c3.5 3.5 3.5 9.3 0 12.8M18.4 5.6c-3.5 3.5-3.5 9.3 0 12.8" /></svg></span>
        <span className="name">AI Tennis Coach</span>
      </a>
      <nav className="tabs" aria-label="Sections">
        {TABS.map(([t, label, href]) => (
          <a key={t} href={href} className={"tab" + (tab === t ? " on" : "")} aria-current={tab === t ? "page" : undefined}
            onClick={(e) => { if (onTab && t !== "eval") { e.preventDefault(); onTab(t); } }}>{label}</a>))}
      </nav>
      <ThemeSwitch />
    </header>
  );
}
