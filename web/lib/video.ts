import fs from "node:fs";

// Serves an mp4 with HTTP Range support so browsers can seek.
export function serveVideo(req: Request, p: string) {
  if (!fs.existsSync(p)) return new Response("not found", { status: 404 });
  const size = fs.statSync(p).size;
  const range = req.headers.get("range");
  const base = { "Content-Type": "video/mp4", "Accept-Ranges": "bytes" };
  if (range) {
    const m = /bytes=(\d*)-(\d*)/.exec(range);
    const start = m && m[1] ? parseInt(m[1]) : 0;
    const end = m && m[2] ? Math.min(parseInt(m[2]), size - 1) : size - 1;
    const buf = Buffer.alloc(end - start + 1);
    const fd = fs.openSync(p, "r"); fs.readSync(fd, buf, 0, buf.length, start); fs.closeSync(fd);
    return new Response(buf, { status: 206, headers: { ...base, "Content-Range": `bytes ${start}-${end}/${size}`, "Content-Length": String(buf.length) } });
  }
  return new Response(fs.readFileSync(p), { headers: { ...base, "Content-Length": String(size) } });
}
