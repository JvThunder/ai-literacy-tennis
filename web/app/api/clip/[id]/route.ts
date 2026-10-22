import path from "node:path";
import { PRE, safe } from "@/lib/data";
import { serveVideo } from "@/lib/video";

export const dynamic = "force-dynamic";
// Original footage of a precomputed case (H.264 copy), used by the coach review.
export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!safe(id)) return new Response("bad request", { status: 400 });
  return serveVideo(req, path.join(PRE, id, "clip.mp4"));
}
