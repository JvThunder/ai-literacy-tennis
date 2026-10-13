import fs from "node:fs";
import path from "node:path";
import { PRE, LIVE, safe } from "@/lib/data";

export const dynamic = "force-dynamic";
export async function GET(_: Request, { params }: { params: Promise<{ id: string; file: string }> }) {
  const { id, file } = await params;
  if (!safe(id) || !safe(file) || !file.endsWith(".jpg")) return new Response("bad request", { status: 400 });
  for (const base of [PRE, LIVE]) {
    const p = path.join(base, id, file);
    if (fs.existsSync(p)) return new Response(fs.readFileSync(p), { headers: { "Content-Type": "image/jpeg", "Cache-Control": "public, max-age=3600" } });
  }
  return new Response("not found", { status: 404 });
}
