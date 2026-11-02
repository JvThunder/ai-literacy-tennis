export const STACK = [
  { stage: "Video + pose estimation", model: "MediaPipe Pose Landmarker (BlazePose full, TFLite)", detail: "33 body keypoints in image and metric 3D (world) coordinates per frame; OpenCV for decoding and frame handling." },
  { stage: "Keypoint cleaning + features", model: "NumPy · pandas", detail: "Low-confidence flagging, gap interpolation, moving-average smoothing; torso-length normalisation; wrist-speed phase detection; joint angles." },
  { stage: "Stroke check", model: "Learned pose-sequence classifier (logistic regression)", detail: "Nine trajectory features (wrist path across the body, reach, hand gap, swing timing) decide forehand / backhand / serve with a probability; trained on THETIS, 89% accurate on unseen players." },
  { stage: "Racket detection", model: "MediaPipe Object Detector (EfficientDet-Lite0, COCO \"tennis racket\")", detail: "Runs on a full-resolution crop around the player for each swing frame; marks the racket on the frames and measures its low-to-high path. Experimental: shown, not yet used for ranking." },
  { stage: "Smartwatch (optional)", model: "Sensor Logger WristMotion CSV · NumPy", detail: "Uploaded with the clip: the watch is aligned to the video by matching its rotation speed to the video's wrist speed; the impact jolt marks the ball hit, a check on the video's contact moment. Shown, not used for ranking." },
  { stage: "Expert benchmark", model: "THETIS expert forehands (130 clips)", detail: "Feature distribution of expert players is the reference; each measurement is expressed as an expert percentile." },
  { stage: "Retrieval (RAG)", model: "all-MiniLM-L6-v2 bi-encoder (ONNX on CPU; FAISS in GPU mode)", detail: "Dense retrieval over passage-level chunks of the coaching documents: top-8 candidates by cosine similarity." },
  { stage: "Reranking", model: "cross-encoder/ms-marco-MiniLM-L-6-v2 (ONNX on CPU)", detail: "Cross-encoder rescoring of the 8 candidates; the top-3 passages become the evidence for the answer." },
  { stage: "Answer generation", model: "Gemini 2.5 Flash-Lite via OpenRouter (GPU mode: Qwen2.5-3B-Instruct)", detail: "Grounded generation: question + measurements + retrieved passages → answer, reasoning, improvement steps, with [n] citations; chosen from 11 models on the evaluation page." },
  { stage: "Practice planning", model: "Rule-based adaptive planner", detail: "Drill library × time × court access × partner; adjusts levels from completion, difficulty and new video observations." },
  { stage: "Live uploads", model: "CPU server: MiniLM + cross-encoder via ONNX, hosted LLM (OpenRouter)", detail: "Precomputed sessions and new uploads use the same stack: pose and the two retrieval models run on the CPU, answers come from the API (uploads: on demand per question). A GPU deployment can run Qwen locally instead." },
  { stage: "Application", model: "Next.js 15 (App Router, TypeScript) + Python 3.13", detail: "API routes serve results and run the Python pipeline for new uploads." },
];

export default function ModelsPage() {
  return (
    <div className="page">
      <h2>Models &amp; frameworks</h2>
      <p className="lead">What runs at each step, from your video to the practice plan. How well each step does is on the <a href="/eval">Evaluation</a> page.</p>
      <div className="mgrid">
        {STACK.map((s, i) => (
          <div key={s.stage} className="mcard">
            <div className="small">{i + 1} · {s.stage}</div>
            <b>{s.model}</b>
            <p>{s.detail}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
