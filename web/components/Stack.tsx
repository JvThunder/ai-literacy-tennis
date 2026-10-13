export const STACK = [
  { stage: "Video + pose estimation", model: "MediaPipe Pose Landmarker (BlazePose, lite, TFLite)", detail: "33 body keypoints in image and metric 3D (world) coordinates per frame; OpenCV for decoding and frame handling." },
  { stage: "Keypoint cleaning + features", model: "NumPy · pandas", detail: "Low-confidence flagging, gap interpolation, moving-average smoothing; torso-length normalisation; wrist-speed phase detection; joint angles." },
  { stage: "Stroke check", model: "Rule-based pose-sequence classifier", detail: "Hitting-wrist path and reach decide forehand / backhand / serve and return a confidence score." },
  { stage: "Expert benchmark", model: "THETIS expert forehands (53 clips)", detail: "Feature distribution of expert players is the reference; each measurement is expressed as an expert percentile." },
  { stage: "Retrieval (RAG)", model: "all-MiniLM-L6-v2 bi-encoder + FAISS (IndexFlatIP)", detail: "Dense retrieval over passage-level chunks of the coaching documents: top-8 candidates by cosine similarity." },
  { stage: "Reranking", model: "cross-encoder/ms-marco-MiniLM-L-6-v2", detail: "Cross-encoder rescoring of the 8 candidates; the top-3 passages become the evidence for the answer." },
  { stage: "Answer generation", model: "Qwen2.5-3B-Instruct (Hugging Face transformers, bf16)", detail: "Grounded generation: question + measurements + retrieved passages → answer, reasoning, improvement steps, with [n] citations." },
  { stage: "Practice planning", model: "Rule-based adaptive planner", detail: "Drill library × time × court access × partner; adjusts levels from completion, difficulty and new video observations." },
  { stage: "Application", model: "Next.js 15 (App Router, TypeScript) + Python 3.13", detail: "API routes serve results and run the Python pipeline for new uploads." },
];

export default function StackPanel() {
  return (
    <details className="card" style={{ marginTop: 10 }}>
      <summary style={{ cursor: "pointer", fontWeight: 600 }}>Models &amp; frameworks</summary>
      {STACK.map((s) => (
        <div key={s.stage} style={{ marginTop: 10, fontSize: 13 }}>
          <div style={{ color: "var(--mute)", fontSize: 11, textTransform: "uppercase", letterSpacing: ".05em" }}>{s.stage}</div>
          <b>{s.model}</b>
          <div style={{ color: "var(--mute)" }}>{s.detail}</div>
        </div>
      ))}
    </details>
  );
}
