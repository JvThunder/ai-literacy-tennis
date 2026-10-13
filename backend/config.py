"""Shared constants: reference ranges, issue metadata and drill library.

NOTE: thresholds are starting values for the prototype. They must be validated /
tuned with the coach review described in Section 5 of the proposal.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KNOWLEDGE_DIR = ROOT / "knowledge"
MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"

# Landmark indices (MediaPipe BlazePose, 33 points)
LM = dict(nose=0, l_sh=11, r_sh=12, l_el=13, r_el=14, l_wr=15, r_wr=16,
          l_hip=23, r_hip=24, l_kn=25, r_kn=26, l_an=27, r_an=28)

# ---------------------------------------------------------------------------
# Issues (the bounded fault set for adult beginners < NTRP 3.0, forehand only)
# ---------------------------------------------------------------------------
# relevance = coaching relevance weight for beginners (used in prioritisation)
ISSUES = {
    "turn": dict(
        title="Limited shoulder rotation",
        relevance=0.95, phase="preparation",
        summary="Your shoulders barely rotate through the swing, so the arm has to do all the work.",
        cue="Turn so your non-racket hand points at the side fence, then let the racket sweep through the ball toward the target.",
        tags="shoulder turn unit turn rotation preparation",
    ),
    "contact": dict(
        title="Contact too close to / behind the body",
        relevance=1.0, phase="contact",
        summary="You meet the ball beside or behind your hip rather than out in front, which makes the shot late and inconsistent.",
        cue="Meet the ball out in front, about a racket's length ahead of your front hip.",
        tags="contact point in front late timing",
    ),
    "follow": dict(
        title="Short follow-through",
        relevance=0.8, phase="follow-through",
        summary="The swing stops soon after contact instead of finishing across and up.",
        cue="Send the ball over the net by a metre or more and let your racket finish over your opposite shoulder.",
        tags="follow-through finish swing path",
    ),
    "elbow": dict(
        title="Arm cramped or locked at contact",
        relevance=0.75, phase="contact",
        summary="Your hitting arm is either jammed in tight or fully locked straight at contact.",
        cue="Give the ball room: contact it with a relaxed, slightly bent arm at a comfortable arm's length.",
        tags="elbow arm extension contact spacing",
    ),
    "knee": dict(
        title="Not enough knee bend",
        relevance=0.6, phase="preparation",
        summary="You stay tall through the swing, which limits balance and the ability to adjust to ball height.",
        cue="Sit slightly into your legs so you can push up through the ball.",
        tags="knee bend balance legs stability",
    ),
    "lean": dict(
        title="Excess trunk lean",
        relevance=0.6, phase="contact",
        summary="Your upper body tips noticeably away from vertical at contact, which hurts balance and consistency.",
        cue="Keep your head still and over your base while the racket swings past it.",
        tags="trunk lean balance posture head stable",
    ),
}

# Reference ranges (degrees unless stated). 'fault_*' triggers an issue.
REF = dict(
    turn_min_deg=25.0,          # shoulder yaw change prep -> contact
    elbow_lo=90.0, elbow_hi=172.0,
    knee_max_deg=165.0,         # knee angle (180 = straight)
    lean_max_deg=20.0,
    contact_fwd_min=0.25,       # wrist ahead of hip, in torso lengths
    follow_min=0.9,             # wrist travel after contact, torso lengths
)

# ---------------------------------------------------------------------------
# Drill library: 3 levels per issue (1 = simplest). requires: court / partner
# ---------------------------------------------------------------------------
def D(id, issue, level, name, minutes, court, partner, steps, goal):
    return dict(id=id, issue=issue, level=level, name=name, minutes=minutes,
                court=court, partner=partner, steps=steps, goal=goal)

DRILLS = [
    D("turn1", "turn", 1, "Shadow turn and point", 5, False, False,
      "Stand side-on to a mirror or wall. Turn your shoulders until your non-racket hand points at the side fence. Hold 2 s, return. 3 sets of 10.",
      "Chest faces the side fence on 8 of 10 turns."),
    D("turn2", "turn", 2, "Drop-feed turn and hit", 10, True, True,
      "Partner drop-feeds a bounce from the service line. Turn early (before the bounce), hit crosscourt. 3 sets of 8.",
      "Shoulders turned before the ball bounces on 6 of 8 balls."),
    D("turn3", "turn", 3, "Turn on the move", 10, True, True,
      "Partner feeds alternate sides; recover to the middle and turn each time. 4 sets of 6.",
      "Visible shoulder turn on every ball."),
    D("contact1", "contact", 1, "Shadow swing past a marker", 5, False, False,
      "Put a cone or water bottle ~1 racket length in front of your front hip. Swing and brush it with the strings at contact. 3 sets of 10.",
      "Strings pass the marker at the same point on 8 of 10 swings."),
    D("contact2", "contact", 2, "Self-drop forehands to a hoop", 10, True, False,
      "Drop the ball in front of your front hip, hit it into the cross-court service box. 3 sets of 8.",
      "Ball contacted in front of the hip on 6 of 8."),
    D("contact3", "contact", 3, "Fed forehands, call 'now'", 10, True, True,
      "Partner feeds; say 'now' out loud at contact and have the partner judge if it was in front. 4 sets of 6.",
      "Partner agrees contact was in front on 4 of 6."),
    D("follow1", "follow", 1, "Finish and freeze", 5, False, False,
      "Slow-motion forehand swings. Finish with the racket over the opposite shoulder and freeze 2 s. 3 sets of 10.",
      "Held finish above shoulder height on 8 of 10."),
    D("follow2", "follow", 2, "Over-the-net-by-a-metre rally", 10, True, False,
      "Hit ten forehands (self-feed) aiming ~1 m above the net, over a rope or line. 3 sets.",
      "7 of 10 balls clear the target height and land in."),
    D("follow3", "follow", 3, "Finish-high rally", 10, True, True,
      "Cooperative cross-court rally, finish over the shoulder every shot. 5 min x 2.",
      "Rally of 6+ shots with a full finish."),
    D("elbow1", "elbow", 1, "Arm-length spacing check", 5, False, False,
      "Extend the racket arm out to the side, then relax the elbow slightly. Swing slowly keeping that spacing. 3 sets of 10.",
      "Same relaxed arm shape at contact on 8 of 10."),
    D("elbow2", "elbow", 2, "Wall or fence spacing", 10, True, False,
      "Stand a racket-length plus an arm from a fence; swing so the racket brushes without jamming. 3 sets of 8.",
      "No jammed contacts on 6 of 8."),
    D("elbow3", "elbow", 3, "Fed balls at varied depth", 10, True, True,
      "Partner varies depth; adjust your feet so contact is at comfortable arm's length. 4 sets of 6.",
      "Comfortable spacing on 4 of 6."),
    D("knee1", "knee", 1, "Sit and swing", 5, False, False,
      "Stand with feet shoulder width. Bend knees ~20 degrees, shadow swing while staying low, rise through contact. 3 sets of 10.",
      "Stay bent until contact on 8 of 10."),
    D("knee2", "knee", 2, "Split-step and hit", 10, True, False,
      "Self-feed, split-step as the ball bounces, hit. 3 sets of 8.",
      "Balanced, knees bent at contact on 6 of 8."),
    D("knee3", "knee", 3, "Low-ball focus", 10, True, True,
      "Partner feeds low balls; bend legs to get under the ball. 4 sets of 6.",
      "Stable, not bending at the waist, on 4 of 6."),
    D("lean1", "lean", 1, "Head-still shadow swing", 5, False, False,
      "Place a small sticker on a mirror at head height; keep your eyes on it while swinging. 3 sets of 10.",
      "Head stays on the sticker on 8 of 10 swings."),
    D("lean2", "lean", 2, "Balanced finish hold", 10, True, False,
      "Self-feed forehands and hold your finish for 2 s without stepping. 3 sets of 8.",
      "Hold finish balanced on 6 of 8."),
    D("lean3", "lean", 3, "Fed balls, upright posture", 10, True, True,
      "Partner feeds wider balls; step to the ball, keep upright on contact. 4 sets of 6.",
      "Upright at contact on 4 of 6."),
]
DRILL_BY_ID = {d["id"]: d for d in DRILLS}
