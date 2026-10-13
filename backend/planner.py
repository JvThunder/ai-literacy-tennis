"""Weekly practice plan + adaptation from session feedback."""
from __future__ import annotations
from .config import DRILLS, DRILL_BY_ID, ISSUES

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _pick(issue, level, court, partner):
    """Highest-level drill <= level that the player's setup allows (falls back down the levels)."""
    for lv in range(min(3, max(1, level)), 0, -1):
        for d in DRILLS:
            if d["issue"] == issue and d["level"] == lv and (court or not d["court"]) and (partner or not d["partner"]):
                return d
    return next(d for d in DRILLS if d["issue"] == issue and d["level"] == 1)


def adapt(prev_plan: dict | None, feedback: dict | None, new_issue_ids: list[str]):
    """Return (focus_ids, levels, adjustments[]) given last plan, its feedback and the new video's issues."""
    levels = dict((prev_plan or {}).get("levels", {}))
    adjustments = []
    if prev_plan and feedback:
        focus = prev_plan["focus"]
        for entry in feedback.get("drills", []):
            d = DRILL_BY_ID.get(entry["id"])
            if not d:
                continue
            iss = d["issue"]; cur = levels.get(iss, d["level"])
            if not entry.get("completed"):
                adjustments.append(f"Repeating '{d['name']}': it wasn't completed last week.")
            elif entry.get("difficulty") == "too_hard":
                levels[iss] = max(1, cur - 1)
                adjustments.append(f"Simplifying '{d['name']}': you found it too hard.")
            elif entry.get("difficulty") == "too_easy":
                levels[iss] = min(3, cur + 1)
                adjustments.append(f"Progressing '{d['name']}': it felt too easy.")
            else:
                adjustments.append(f"Keeping '{d['name']}' at the same level: difficulty felt right.")
        if focus and new_issue_ids and focus[0] not in new_issue_ids:
            adjustments.append(f"'{ISSUES[focus[0]]['title']}' no longer shows up in your new video: moving on to the next issue.")
    return levels, adjustments


def build(player: dict, issue_ids: list[str], prev_plan=None, feedback=None):
    levels, adjustments = adapt(prev_plan, feedback, issue_ids)
    if prev_plan and feedback and prev_plan.get("focus"):
        old = prev_plan["focus"][0]
        resolved = issue_ids and old not in issue_ids
        # keep the previous focus unless the new video shows it resolved
        order = ([old] if not resolved else []) + [i for i in issue_ids if i != old]
        issue_ids = order or issue_ids
    focus = issue_ids[:2] or ["contact"]
    n = int(player.get("sessions_per_week") or 3)
    minutes = int(player.get("minutes") or 20)
    court_access = player.get("court_access", "sometimes")      # none | sometimes | regular
    partner = bool(player.get("partner"))
    court_sessions = {"none": 0, "sometimes": min(1, n), "regular": min(2, n)}.get(court_access, 1)
    sessions = []
    for i in range(n):
        at_court = i < court_sessions
        issue = focus[0] if (i % 3 != 2 or len(focus) == 1) else focus[1]
        lvl = levels.get(issue, 1 if not at_court else 2)
        drills, left = [], minutes
        d = _pick(issue, lvl if at_court else min(lvl, 1), at_court, partner and at_court)
        drills.append(d); left -= d["minutes"]
        if left >= 5:
            iss2 = focus[1] if len(focus) > 1 and issue == focus[0] else focus[0]
            d2 = _pick(iss2, 1, False, False)
            if d2["id"] != d["id"]:
                drills.append(d2); left -= d2["minutes"]
        sessions.append(dict(day=DAYS[[0, 2, 5, 1, 3, 4, 6][i % 7]], at_court=at_court, minutes=minutes,
                             drills=[dict(**x) for x in drills]))
    sessions.sort(key=lambda s: DAYS.index(s["day"]))
    for s in sessions:
        s["label"] = "Court session" if s["at_court"] else "Home session (no court needed)"
    levels = {**levels, **{i: levels.get(i, 2 if court_sessions else 1) for i in focus}}
    rationale = (f"Focus: {ISSUES[focus[0]]['title'].lower()}"
                 + (f", then {ISSUES[focus[1]]['title'].lower()}" if len(focus) > 1 else "")
                 + f". Built for {n} sessions/week of ~{minutes} min, court access: {court_access}, "
                   f"partner: {'yes' if partner else 'no'}.")
    return dict(focus=focus, levels=levels, sessions=sessions, rationale=rationale, adjustments=adjustments)
