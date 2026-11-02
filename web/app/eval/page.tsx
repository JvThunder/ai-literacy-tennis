import type { Metadata } from "next";
import Eval from "@/components/Eval";

export const metadata: Metadata = { title: "Evaluation · AI Tennis Coach", description: "How the AI Tennis Coach measures up against human labels and coach review" };
export default function Page() {
  return <Eval />;
}
