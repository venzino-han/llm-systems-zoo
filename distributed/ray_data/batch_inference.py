"""L07.6 — Ray Data batch inference: streaming execution and the input bottleneck.

Pipeline:  from_items(prompts) -> map(preprocess, CPU) -> map_batches(Generator actor, GPU/CPU)
           -> write_parquet.  `ds.stats()` shows per-stage wall/CPU time, so an artificially slow
preprocess stage (`--slow-preprocess-ms`) is visible as the stage starving the generator.

Run:  python distributed/ray_data/batch_inference.py --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import ray  # noqa: E402

from distributed.ray_core._common import cuda_available, env_summary, init_ray  # noqa: E402
from llm_systems_zoo import Run  # noqa: E402
from llm_systems_zoo.plotting import plot_bars  # noqa: E402
from llm_systems_zoo.registry import DATASETS, MODELS  # noqa: E402


def load_prompts(source: str, limit: int) -> list[str]:
    if source == "synthetic":
        return [f"Write one sentence about the number {i}." for i in range(limit)]
    from datasets import load_dataset

    if source == "rag_qa":
        spec = DATASETS["rag_qa"]
        ds = load_dataset(spec.hf_id, spec.config, split=spec.split)
        return [r["question"] for r in ds.select(range(min(limit, len(ds))))]
    if source == "ultrachat":
        spec = DATASETS["chat_prompts"]
        ds = load_dataset(spec.hf_id, split=spec.split, streaming=True)
        return [r["prompt"] for _, r in zip(range(limit), ds)]
    raise ValueError(source)


class Generator:
    """Stateful callable used by `map_batches`: the model is loaded once per actor."""

    def __init__(self, model_id: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if cuda_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        import torch

        enc = self.tok(list(batch["text"]), return_tensors="pt", padding=True).to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
        gen = self.tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        dt = time.perf_counter() - t0
        return {"id": batch["id"], "output": np.array(gen, dtype=object), "gen_s": np.full(len(gen), dt / len(gen))}


def make_preprocess(slow_ms: float):
    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        if slow_ms > 0:
            time.sleep(slow_ms / 1e3)  # stands in for tokenisation / parsing / feature extraction
        row["text"] = f"<|im_start|>user\n{row['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        return row

    return preprocess


def run_pipeline(prompts: list[str], model_id: str, batch_size: int, concurrency: int, slow_ms: float, max_new_tokens: int, out_dir: Path, gpu_frac: float) -> dict[str, Any]:
    ds = ray.data.from_items([{"id": i, "prompt": p} for i, p in enumerate(prompts)])
    ds = ds.map(make_preprocess(slow_ms), compute=ray.data.TaskPoolStrategy(size=2))
    res: dict[str, Any] = {"num_cpus": 2}
    if cuda_available():
        res["num_gpus"] = gpu_frac
    ds = ds.map_batches(Generator, fn_constructor_args=(model_id, max_new_tokens), batch_size=batch_size, compute=ray.data.ActorPoolStrategy(size=concurrency), **res)
    t0 = time.perf_counter()
    ds.write_parquet(str(out_dir))
    wall = time.perf_counter() - t0
    stats = ds.stats()
    n = len(prompts)
    return {"batch_size": batch_size, "concurrency": concurrency, "slow_preprocess_ms": slow_ms, "n_rows": n, "wall_s": wall, "rows_per_s": n / wall, "stats": stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODELS["small"].hf_id)
    ap.add_argument("--source", choices=["synthetic", "rag_qa", "ultrachat"], default="rag_qa")
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--slow-preprocess-ms", type=float, nargs="+", default=[0, 50])
    ap.add_argument("--gpu-frac", type=float, default=0.25)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.model, args.source, args.limit, args.max_new_tokens = MODELS["tiny"].hf_id, "synthetic", 16, 8
        args.batch_sizes, args.concurrency, args.slow_preprocess_ms = [4], [1], [0, 100]

    prompts = load_prompts(args.source, args.limit)
    cfg = vars(args)
    with Run("L07_ray_systems", "ray_data_batch_inference", cfg, model={"id": args.model}, workload={"source": args.source, "n": len(prompts)}) as run:
        init_ray(num_cpus=8, num_gpus=1 if cuda_available() else 0)
        run.update(software={"ray_env": env_summary()})
        rows = []
        try:
            for slow in args.slow_preprocess_ms:
                for conc in args.concurrency:
                    for bs in args.batch_sizes:
                        out = run.path(f"out_bs{bs}_c{conc}_s{int(slow)}")
                        r = run_pipeline(prompts, args.model, bs, conc, slow, args.max_new_tokens, out, args.gpu_frac)
                        (run.path(f"stats_bs{bs}_c{conc}_s{int(slow)}.txt")).write_text(r.pop("stats"))
                        rows.append(r)
                        run.log_metric(**r)
                        print(f"bs={bs:<3} conc={conc} slow={slow:>4}ms wall={r['wall_s']:.2f}s rows/s={r['rows_per_s']:.2f}", flush=True)
        finally:
            ray.shutdown()
        run.log_records(rows)
        labels = [f"bs{r['batch_size']}/c{r['concurrency']}/s{int(r['slow_preprocess_ms'])}" for r in rows]
        plot_bars(labels, [r["rows_per_s"] for r in rows], ylabel="rows / s", title=f"{args.model}, {len(prompts)} prompts", path=run.path("throughput.png"))
        run.save_summary({"rows": rows, "device": "cuda" if cuda_available() else "cpu"})
        print("run dir:", run.dir)


if __name__ == "__main__":
    main()
