"""L09.1 — Build a dense retrieval index over rag-mini-wikipedia with a small embedding model.

Output directory contains `index.faiss` (inner-product over normalised vectors = cosine),
`passages.jsonl` (id, passage) and `meta.json` (model, dim, count).

Run:  python applications/rag/index.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from llm_systems_zoo import Run  # noqa: E402
from llm_systems_zoo.io import dump_json, read_jsonl, write_jsonl  # noqa: E402
from llm_systems_zoo.registry import DATASETS, MODELS  # noqa: E402

DEFAULT_INDEX_DIR = Path(__file__).resolve().parent / "data" / "index"


def load_passages(limit: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    spec = DATASETS["rag_corpus"]
    ds = load_dataset(spec.hf_id, spec.config, split=spec.split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [{"id": int(r["id"]), "passage": str(r["passage"])} for r in ds]


def embed_texts(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    return np.asarray(model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def build_index(embeddings: np.ndarray):
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index, passages: list[dict[str, Any]], meta: dict[str, Any], out_dir: Path) -> None:
    import faiss

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    write_jsonl(passages, out_dir / "passages.jsonl")
    dump_json(meta, out_dir / "meta.json")


def load_index(index_dir: Path):
    import faiss

    index = faiss.read_index(str(index_dir / "index.faiss"))
    passages = list(read_jsonl(index_dir / "passages.jsonl"))
    meta = json.loads((index_dir / "meta.json").read_text())
    return index, passages, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed-model", default=MODELS["embed"].hf_id)
    ap.add_argument("--out", type=Path, default=DEFAULT_INDEX_DIR)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.limit = 200

    from sentence_transformers import SentenceTransformer

    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with Run("L09_app_to_infra", "build_index", cfg, model={"embed": args.embed_model}) as run:
        passages = load_passages(args.limit)
        t0 = time.perf_counter()
        model = SentenceTransformer(args.embed_model)
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        emb = embed_texts(model, [p["passage"] for p in passages], args.batch_size)
        t_embed = time.perf_counter() - t0
        index = build_index(emb)
        meta = {"embed_model": args.embed_model, "dim": int(emb.shape[1]), "n_passages": len(passages), "dataset": DATASETS["rag_corpus"].hf_id, "device": str(model.device)}
        save_index(index, passages, meta, args.out)
        summary = {**meta, "load_s": t_load, "embed_s": t_embed, "passages_per_s": len(passages) / t_embed, "index_dir": str(args.out)}
        run.save_summary(summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
