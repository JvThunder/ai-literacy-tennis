import { NextResponse } from "next/server";
import { PRE, readJson } from "@/lib/data";

export const dynamic = "force-dynamic";
export async function GET() {
  return NextResponse.json(readJson(PRE, "questions.json") ?? []);
}
