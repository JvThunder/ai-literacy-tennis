"""Coaching knowledge base: chunk, embed, retrieve (dense) and rerank (cross-encoder).

TENNIS_RETRIEVAL picks the backend (same two models in "dense" and "onnx"):
  dense (default when TENNIS_MODE=gpu): MiniLM bi-encoder + FAISS, then cross-encoder rerank (torch).
  onnx  (default when TENNIS_MODE=api): the same MiniLM + ms-marco cross-encoder via fastembed/ONNX on CPU, no torch.
  tfidf: TF-IDF cosine ranking; also the fallback when fastembed is not installed.
"""
from __future__ import annotations
import json, os, re, subprocess, zipfile
from pathlib import Path
import numpy as np
from .config import KNOWLEDGE_DIR, ROOT, DATA_DIR, MODE

BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ONNX_CROSS = "Xenova/ms-marco-MiniLM-L-6-v2"          # ONNX export of CROSS_ENCODER
ONNX_CACHE = os.environ.get("TENNIS_FASTEMBED_CACHE", str(ROOT / ".cache" / "fastembed"))
_idx: dict = {}


def kind_wanted() -> str:
    if os.environ.get("TENNIS_EMBED") == "1":             # legacy switch used by precompute.py
        return "dense"
    return os.environ.get("TENNIS_RETRIEVAL", "dense" if MODE == "gpu" else "onnx")


def use(kind: str):
    """Switch backend in-process (evaluation)."""
    os.environ["TENNIS_RETRIEVAL"] = kind; os.environ.pop("TENNIS_EMBED", None); _idx.clear()


def _read(path: Path) -> str:
    if path.suffix == ".docx":
        xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf8", "ignore")
        return re.sub(r"</w:p>", "\n\n", re.sub(r"<(?!/w:p)[^>]+>", "", xml))
    return path.read_text(errors="ignore")


def _pdf_chunks():
    """Short excerpts of reference PDFs listed in knowledge/sources.json, cited by printed page number.
    Extracted once and cached so the deployed server does not need the PDF."""
    cache = DATA_DIR / "cache" / "pdf_chunks.json"
    if cache.exists():
        return json.loads(cache.read_text())
    man = KNOWLEDGE_DIR / "sources.json"
    out = []
    if man.exists():
        for src in json.loads(man.read_text()):
            pdf = ROOT / src["pdf"]
            if not pdf.exists():
                continue
            for pg in src["pages"]:
                txt = subprocess.run(["pdftotext", "-f", str(pg), "-l", str(pg), "-layout", str(pdf), "-"],
                                     capture_output=True, text=True).stdout
                m = re.search(r"Page\s+(\d+)", txt)
                printed = m.group(1) if m else str(pg)
                txt = re.sub(r"Page\s+\d+", " ", txt, count=1)
                txt = txt.replace("\u00ad", "-").replace("\f", " ")
                paras = [re.sub(r"\s+", " ", q).strip() for q in re.split(r"\n\s*\n", txt)]
                buf, n = "", 0
                for q in paras:
                    if len(q.split()) < 6:
                        continue
                    buf = (buf + " " + q).strip()
                    if len(buf.split()) >= 70:
                        n += 1
                        out.append(dict(doc=src["doc"], doc_title=src["title"], section=f"p. {printed} ({n})", text=buf,
                                        id=f"{Path(src['doc']).stem}#p{printed}-{n}"))
                        buf = ""
                if len(buf.split()) >= 30:
                    n += 1
                    out.append(dict(doc=src["doc"], doc_title=src["title"], section=f"p. {printed} ({n})", text=buf,
                                    id=f"{Path(src['doc']).stem}#p{printed}-{n}"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def load_chunks():
    """One chunk per '## ' section of the sample documents, plus excerpt chunks from reference PDFs."""
    chunks = _pdf_chunks()
    for p in sorted(KNOWLEDGE_DIR.glob("*")):
        if p.suffix not in (".md", ".txt", ".docx"):
            continue
        text = _read(p)
        m = re.match(r"#\s+(.+)", text)
        doc_title = m.group(1).strip() if m else p.stem
        for sec in re.split(r"\n(?=## )", text)[1:] if "\n## " in text else [text]:
            head, _, body = sec.partition("\n")
            title = head.lstrip("# ").strip()
            body = re.sub(r"\s+", " ", body).strip()
            if len(body) > 40:
                chunks.append(dict(doc=p.name, doc_title=doc_title, section=title, text=body,
                                   id=f"{p.stem}#{re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')}"))
    return chunks


def _build():
    if _idx:
        return
    chunks = load_chunks()
    _idx["chunks"] = chunks
    texts = [f"{c['section']}. {c['text']}" for c in chunks]
    kind = kind_wanted()
    if kind == "onnx":
        try:
            from fastembed import TextEmbedding
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError:
            kind = "tfidf"
        else:
            bi = TextEmbedding(BI_ENCODER, cache_dir=ONNX_CACHE)
            emb = np.asarray(list(bi.embed(texts)), dtype="float32")
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            _idx.update(kind="onnx", bi=bi, emb=emb, cross=TextCrossEncoder(ONNX_CROSS, cache_dir=ONNX_CACHE), texts=texts)
            return
    if kind == "dense":
        import faiss
        from sentence_transformers import SentenceTransformer, CrossEncoder
        dev = os.environ.get("TENNIS_DEVICE", "cuda" if MODE == "gpu" else "cpu")
        bi = SentenceTransformer(BI_ENCODER, device=dev)
        emb = np.asarray(bi.encode(texts, normalize_embeddings=True), dtype="float32")
        ix = faiss.IndexFlatIP(emb.shape[1]); ix.add(emb)
        _idx.update(kind="dense", bi=bi, faiss=ix, cross=CrossEncoder(CROSS_ENCODER, device=dev), texts=texts)
    if kind not in ("dense", "onnx"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        _idx.update(kind="tfidf", vec=vec, X=vec.fit_transform(texts), texts=texts)


def backend_name() -> str:
    _build()
    if _idx["kind"] == "dense":
        return f"MiniLM-L6 bi-encoder + FAISS → ms-marco MiniLM cross-encoder rerank ({len(_idx['chunks'])} passages)"
    if _idx["kind"] == "onnx":
        return f"MiniLM-L6 bi-encoder → ms-marco MiniLM cross-encoder rerank, ONNX on CPU ({len(_idx['chunks'])} passages)"
    return f"TF-IDF cosine ranking ({len(_idx['chunks'])} passages)"


def retrieve(query: str, k_dense: int = 8, k_final: int = 3) -> dict:
    """Two-stage retrieval. Returns query, candidates (stage 1, with ranks) and final (stage 2)."""
    _build()
    ch = _idx["chunks"]
    if _idx["kind"] == "dense":
        q = np.asarray(_idx["bi"].encode([query], normalize_embeddings=True), dtype="float32")
        sc, ix = _idx["faiss"].search(q, k_dense)
        cand = [dict(i=int(i), dense=float(s)) for i, s in zip(ix[0], sc[0])]
        ce = _idx["cross"].predict([(query, _idx["texts"][c["i"]]) for c in cand])
        for c, s in zip(cand, ce):
            c["rerank"] = float(s)
        order = sorted(cand, key=lambda c: -c["rerank"])
    elif _idx["kind"] == "onnx":
        q = np.asarray(list(_idx["bi"].embed([query])), dtype="float32")[0]
        sims = _idx["emb"] @ (q / np.linalg.norm(q))
        cand = [dict(i=int(i), dense=float(sims[i])) for i in np.argsort(-sims)[:k_dense]]
        ce = list(_idx["cross"].rerank(query, [_idx["texts"][c["i"]] for c in cand]))
        for c, s in zip(cand, ce):
            c["rerank"] = float(s)
        order = sorted(cand, key=lambda c: -c["rerank"])
    else:
        s = (_idx["X"] @ _idx["vec"].transform([query]).T).toarray().ravel()
        cand = [dict(i=int(i), dense=float(s[i]), rerank=float(s[i])) for i in np.argsort(-s)[:k_dense]]
        order = cand
    for r, c in enumerate(cand):
        c["rank_dense"] = r + 1
    for r, c in enumerate(order):
        c["rank_final"] = r + 1

    def pack(c):
        x = ch[c["i"]]
        return dict(id=x["id"], doc=x["doc"], doc_title=x["doc_title"], section=x["section"], text=x["text"],
                    dense=round(c["dense"], 3), rerank=round(c["rerank"], 2),
                    rank_dense=c["rank_dense"], rank_final=c["rank_final"])
    return dict(query=query, stage1=_idx["kind"] != "tfidf", kind=_idx["kind"], candidates=[pack(c) for c in cand],
                final=[pack(c) for c in order[:k_final]])


def n_chunks() -> int:
    _build(); return len(_idx["chunks"])
