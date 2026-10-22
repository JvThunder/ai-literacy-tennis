export type Passage = { id: string; doc: string; doc_title: string; section: string; text: string; dense: number; rerank: number; rank_dense: number; rank_final: number };
export type Retrieval = { query: string; stage1: boolean; kind?: string; candidates: Passage[]; final: Passage[] };
export type Issue = {
  id: string; title: string; phase: string; value: number; unit: string; reference: string; deviation: number;
  confidence: number; consistency: number; relevance: number; score: number; time_s: number; frame: number;
  explanation: string; explanation_source: string; cue: string; references: Passage[]; retrieval?: Retrieval;
  reasoning?: { step: string; text: string }[];
  expert_median?: number; expert_p10?: number; expert_p90?: number; percentile?: number; z?: number;
};
export type Img = { label: string; file: string; time_s?: number; idx?: number };
export type Stage = { name: string; tool: string; status: string; detail: string; ms: number };
export type QA = { question: string; retrieval?: Retrieval; answer?: string; why?: string; improve?: string; model: string };
export type Feature = { value: number; unit: string; confidence: number; reference?: string; expert_median?: number; expert_p10?: number; expert_p90?: number; percentile?: number; z?: number };
export type Report = {
  status: "ok" | "not_forehand" | "rejected"; reason?: string;
  stroke?: { label: string; confidence: number; scores: Record<string, number> };
  issues: Issue[]; other_candidates?: Partial<Issue>[];
  features?: Record<string, Feature>;
  phases?: { preparation_s: number[]; contact_s: number; follow_through_s: number[] };
  wrist_speed?: number[]; t?: number[]; contact_idx?: number;
  frames: Img[]; strip?: Img[]; scrub?: Img[]; warnings: string[]; capture_tips?: string[]; stages: Stage[];
  qa?: Record<string, QA>;
};
export type Player = { level?: string; dominant?: string; goal?: string; question?: string; sessions_per_week?: number; minutes?: number; court_access?: string; partner?: boolean };
export type CaseData = { id?: string; title: string; blurb: string; player: Player; report: Report; plan: any; base: string; live?: boolean };
export type CaseItem = { id: string; title: string; top: string[]; n_issues: number };
export type Question = { id: string; text: string };
export type RagEval = { n: number; passages: number; dense: { r1: number; r3: number; mrr: number }; reranked: { r1: number; r3: number; mrr: number } };
export type Reference = { n_clips: number; clips: { id: string; video: string; title: string; contact_s: number; features: Record<string, number> }[]; stats: Record<string, { p10: number; p25: number; p50: number; p75: number; p90: number }> };
