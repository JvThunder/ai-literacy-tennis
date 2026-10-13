import { NextResponse } from "next/server";
import { PRE, readJson, safe } from "@/lib/data";

export const dynamic = "force-dynamic";
export async function GET(_: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const c = safe(id) ? readJson(PRE, id, "case.json") : null;
  return c ? NextResponse.json(c) : NextResponse.json({ error: "not found" }, { status: 404 });
}
