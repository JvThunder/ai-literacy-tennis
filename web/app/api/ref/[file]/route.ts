import path from "node:path";
import { ROOT, safe } from "@/lib/data";
import { serveVideo } from "@/lib/video";

export const dynamic = "force-dynamic";
// Expert reference videos.
export async function GET(req: Request, { params }: { params: Promise<{ file: string }> }) {
  const { file } = await params;
  if (!safe(file) || !file.endsWith(".mp4")) return new Response("bad request", { status: 400 });
  return serveVideo(req, path.join(ROOT, "data", "reference", file));
}
