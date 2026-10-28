"""Shared building blocks for the L05 training loops (single GPU, DDP, FSDP).

Keeping data, model construction and the step function in one place is what makes the
comparisons fair: every configuration runs *the same* loop and only the switch under test
changes.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import torch

from llm_systems_zoo import PhaseTimer
from llm_systems_zoo.registry import MODELS
from training.packing import (
    PackedBatch,
    build_packed_batch,
    build_padded_batch,
    first_fit,
    truncate,
)

TINY_RANDOM = "random-tiny"  # a randomly initialised ~1M-param Llama for CPU tests, no download

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    model: str = "small"
    dataset: str = "synthetic"  # synthetic | smoltalk | wikitext
    precision: str = "bf16_autocast"  # fp32 | bf16_autocast | fp16_autocast | bf16_weights
    grad_checkpointing: bool = False
    grad_accum: int = 1
    packing: bool = False
    lora: int = 0  # LoRA rank, 0 = full fine-tuning
    qlora: bool = False
    batch_size: int = 4
    seq_len: int = 512
    steps: int = 40
    warmup_steps: int = 5
    lr: float = 1e-4
    seed: int = 0
    n_samples: int = 2048
    device: str = "cuda"
    attn: str = "sdpa"

    @property
    def global_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_model_id(name: str) -> str:
    return MODELS[name].hf_id if name in MODELS else name


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def build_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_samples(cfg: TrainConfig, tokenizer=None, vocab_size: int = 256) -> list[list[int]]:
    """Token-id samples. `synthetic` needs no tokenizer/network and mimics a skewed length
    distribution (many short, some long) so padding vs packing differences are visible."""
    rng = random.Random(cfg.seed)
    if cfg.dataset == "synthetic":
        samples = []
        for _ in range(cfg.n_samples):
            # log-normal-ish lengths clipped to seq_len
            n = int(min(cfg.seq_len, max(8, rng.lognormvariate(math.log(cfg.seq_len / 4), 0.8))))
            samples.append([rng.randrange(vocab_size) for _ in range(n)])
        return samples
    from datasets import load_dataset

    if cfg.dataset == "smoltalk":
        ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train")
        ds = ds.shuffle(seed=cfg.seed).select(range(min(cfg.n_samples, len(ds))))
        texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in ds["messages"]]
    elif cfg.dataset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 200][: cfg.n_samples]
    else:
        raise ValueError(cfg.dataset)
    return truncate(tokenizer(texts, add_special_tokens=False)["input_ids"], cfg.seq_len)


def iter_batches(samples: list[list[int]], cfg: TrainConfig, pad_id: int) -> Iterator[PackedBatch]:
    """Infinite, seeded batch iterator. Packed mode: first-fit bins of `seq_len`, `batch_size`
    rows per step. Padded mode: `batch_size` samples per step, dynamic padding to the longest."""
    rng = random.Random(cfg.seed + 1)
    order = list(range(len(samples)))
    while True:
        rng.shuffle(order)
        if cfg.packing:
            bins = first_fit([len(samples[i]) for i in order], cfg.seq_len)
            bins = [[order[j] for j in b] for b in bins]
            for k in range(0, len(bins) - cfg.batch_size + 1, cfg.batch_size):
                yield build_packed_batch(samples, bins[k : k + cfg.batch_size], cfg.seq_len, pad_id, "block")
        else:
            for k in range(0, len(order) - cfg.batch_size + 1, cfg.batch_size):
                yield build_padded_batch([samples[i] for i in order[k : k + cfg.batch_size]], pad_id)


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------
def tiny_random_model(vocab_size: int = 256, seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)
    return LlamaForCausalLM(cfg)


def weight_dtype(precision: str) -> torch.dtype:
    return torch.bfloat16 if precision == "bf16_weights" else torch.float32


def autocast_dtype(precision: str) -> torch.dtype | None:
    return {"bf16_autocast": torch.bfloat16, "fp16_autocast": torch.float16}.get(precision)


def build_model(cfg: TrainConfig) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Construct the model for one configuration and return it with a parameter summary."""
    from transformers import AutoModelForCausalLM

    model_id = resolve_model_id(cfg.model)
    if cfg.model == TINY_RANDOM:
        model = tiny_random_model(seed=cfg.seed).to(weight_dtype(cfg.precision))
    elif cfg.qlora:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map={"": 0},
                                                     attn_implementation=cfg.attn)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=weight_dtype(cfg.precision),
                                                     attn_implementation=cfg.attn)
    model.config.use_cache = False

    if cfg.qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.grad_checkpointing)
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if cfg.lora or cfg.qlora:
        from peft import LoraConfig, get_peft_model

        r = cfg.lora or 16
        model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0,
                                                 target_modules=ALL_LINEAR, task_type="CAUSAL_LM"))
        if cfg.grad_checkpointing:
            model.enable_input_require_grads()
    if not cfg.qlora:
        model.to(cfg.device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = {
        "model_id": model_id,
        "n_params": n_total,
        "n_trainable": n_train,
        "trainable_frac": n_train / n_total if n_total else 0.0,
        "weight_dtype": str(weight_dtype(cfg.precision)) if not cfg.qlora else "nf4+bf16",
    }
    info.update(theoretical_memory(model, cfg))
    return model, info


def theoretical_memory(model: torch.nn.Module, cfg: TrainConfig) -> dict[str, float]:
    """Static memory estimate (weights + grads + AdamW states) to compare with the measured peak.
    Activations are deliberately excluded: they are what the measurement reveals."""
    w_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    train_params = [p for p in model.parameters() if p.requires_grad]
    g_bytes = sum(p.numel() * p.element_size() for p in train_params)
    # AdamW keeps exp_avg and exp_avg_sq in the parameter dtype (fp32 master weights for AMP)
    o_bytes = sum(2 * p.numel() * p.element_size() for p in train_params)
    gib = 2**30
    return {
        "est_weights_gib": w_bytes / gib,
        "est_grads_gib": g_bytes / gib,
        "est_optimizer_gib": o_bytes / gib,
        "est_static_gib": (w_bytes + g_bytes + o_bytes) / gib,
    }


# --------------------------------------------------------------------------------------------
# Step
# --------------------------------------------------------------------------------------------
def memory_stats(device: str) -> dict[str, float]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {"peak_allocated_gib": 0.0, "peak_reserved_gib": 0.0}
    return {
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def reset_peak_memory(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def make_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)


def optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[PackedBatch],
    cfg: TrainConfig,
    timer: PhaseTimer,
    scaler: torch.amp.GradScaler | None = None,
    no_sync_ctx=None,
) -> dict[str, float]:
    """One optimizer step = `grad_accum` micro-batches. Returns loss and token counts.

    Loss is averaged over micro-batches (each micro-batch already averages over its tokens),
    which is the standard approximation; exact token-weighted averaging would need the token
    count of all micro-batches up front.
    """
    ac = autocast_dtype(cfg.precision)
    total_loss = 0.0
    n_real = n_total = 0
    for i, batch in enumerate(batches):
        batch.to(cfg.device)
        kw = batch.model_kwargs()
        is_last = i == len(batches) - 1
        # DDP: skip the all-reduce on all but the last micro-batch
        ctx = no_sync_ctx() if (no_sync_ctx is not None and not is_last) else _nullcontext()
        with ctx:
            with timer.phase("forward"):
                with torch.autocast(device_type="cuda" if str(cfg.device).startswith("cuda") else "cpu",
                                    dtype=ac, enabled=ac is not None):
                    loss = model(**kw).loss / len(batches)
            with timer.phase("backward"):
                (scaler.scale(loss) if scaler else loss).backward()
        total_loss += float(loss.detach())
        n_real += batch.n_real_tokens
        n_total += batch.input_ids.numel()
    with timer.phase("optimizer"):
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"loss": total_loss, "real_tokens": n_real, "total_tokens": n_total}


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def train_loop(
    model: torch.nn.Module,
    cfg: TrainConfig,
    samples: list[list[int]],
    pad_id: int,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    log=None,
    start_step: int = 0,
    batch_iter: Iterator[PackedBatch] | None = None,
    no_sync_ctx=None,
) -> dict[str, Any]:
    """Run `cfg.steps` optimizer steps and return the summary used by every L05 script."""
    optimizer = optimizer or make_optimizer(model, cfg.lr)
    scaler = torch.amp.GradScaler("cuda") if cfg.precision == "fp16_autocast" else None
    batch_iter = batch_iter or iter_batches(samples, cfg, pad_id)
    timer = PhaseTimer(sync_cuda=str(cfg.device).startswith("cuda"))
    losses: list[float] = []
    step_times: list[float] = []
    real_tokens = total_tokens = 0
    reset_peak_memory(cfg.device)
    model.train()
    t_measure0 = None
    for step in range(start_step, cfg.steps):
        batches = [next(batch_iter) for _ in range(cfg.grad_accum)]
        t0 = time.perf_counter()
        out = optimizer_step(model, optimizer, batches, cfg, timer, scaler, no_sync_ctx)
        if timer.sync_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        losses.append(out["loss"])
        if step >= cfg.warmup_steps:
            if t_measure0 is None:
                t_measure0 = t0
                timer.reset()
            step_times.append(dt)
            real_tokens += out["real_tokens"]
            total_tokens += out["total_tokens"]
        if log:
            log(step=step, loss=out["loss"], step_time_s=dt, real_tokens=out["real_tokens"],
                total_tokens=out["total_tokens"], **memory_stats(cfg.device))
    measured_s = sum(step_times)
    phases = timer.summary()
    return {
        "losses": losses,
        "final_loss": losses[-1] if losses else math.nan,
        "mean_loss_last5": sum(losses[-5:]) / max(1, len(losses[-5:])),
        "measured_steps": len(step_times),
        "mean_step_s": measured_s / len(step_times) if step_times else math.nan,
        "step_s_std": (sum((t - measured_s / len(step_times)) ** 2 for t in step_times) / max(1, len(step_times) - 1)) ** 0.5 if len(step_times) > 1 else 0.0,
        "real_tokens_per_s": real_tokens / measured_s if measured_s else math.nan,
        "total_tokens_per_s": total_tokens / measured_s if measured_s else math.nan,
        "padding_fraction": 1 - real_tokens / total_tokens if total_tokens else math.nan,
        "phase_share": {k: v["share_of_wall"] for k, v in phases.items() if not k.startswith("_")},
        "phase_mean_s": {k: v["mean_s"] for k, v in phases.items() if not k.startswith("_")},
        **memory_stats(cfg.device),
    }
