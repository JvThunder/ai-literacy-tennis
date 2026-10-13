// Maps a free-text prompt to the closest supported question (bag-of-words cosine with light stemming).
const STOP = new Set("a an the my i me is are do does how why what to of in on it and or for can with be so when".split(" "));
const stem = (w: string) => w.replace(/(ing|ed|es|s)$/, "");
const toks = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]/g, " ").split(/\s+/).filter((w) => w && !STOP.has(w)).map(stem);
const SYN: Record<string, string> = { wobbly: "unstable", shaky: "unstable", inconsistent: "inconsistent", erratic: "inconsistent", delayed: "late", slow: "late", strength: "power", hard: "power", fall: "balance", lose: "balance", court: "nocourt", home: "nocourt", practise: "week", practice: "week", drill: "week", priority: "first", start: "first", finish: "followthrough", follow: "followthrough", net: "net" };

function vec(s: string) {
  const v = new Map<string, number>();
  for (const t of toks(s)) { v.set(t, (v.get(t) ?? 0) + 1); if (SYN[t]) v.set(SYN[t], (v.get(SYN[t]) ?? 0) + 1); }
  return v;
}
function cos(a: Map<string, number>, b: Map<string, number>) {
  let d = 0, na = 0, nb = 0;
  a.forEach((x, k) => { na += x * x; const y = b.get(k); if (y) d += x * y; });
  b.forEach((y) => (nb += y * y));
  return na && nb ? d / Math.sqrt(na * nb) : 0;
}
export function matchQuestion(text: string, bank: { id: string; text: string }[]) {
  const q = vec(text);
  let best = { id: bank[0]?.id, text: bank[0]?.text, sim: 0 };
  for (const b of bank) {
    const extra = vec(b.id.replace("q_", ""));
    const s = Math.max(cos(q, vec(b.text)), cos(q, extra) * 0.9);
    if (s > best.sim) best = { id: b.id, text: b.text, sim: s };
  }
  return best;
}
