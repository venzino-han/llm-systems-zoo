"""L01 — analytic memory estimator for transformer training and inference.

Pure Python (no torch) so it runs anywhere, including CI. All sizes are in bytes.

The estimator splits memory into the categories the report discusses separately:

    parameters, gradients, optimizer states, activations, KV cache, logits, runtime overhead

It deliberately models *what the allocator must hold*, not what `nvidia-smi` shows: the CUDA
context (~300-600 MB), cuBLAS/attention workspaces, and allocator fragmentation are reported
as "runtime overhead" and measured, not estimated. See `measure_memory.py` for the comparison.

Equations (b = batch, s = sequence length, h = d_model, L = layers, a = heads, a_kv = KV heads,
d = head_dim, f = d_ff, V = vocab, p = bytes per element):

    params            N (counted exactly from the architecture)
    KV cache / token  2 * L * a_kv * d * p                     (K and V, every layer)
    activations/layer ~ b*s*p*(8h + 4f + 2*a_kv*d) [+ 2*a*s*s*b*p if attention scores
                       are materialised, i.e. no flash/SDPA kernel]
    logits            b*s*V*(p + 4)    (fp32 upcast for the loss + grad of logits)
    training total    weights + grads + optimizer + activations + logits
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

DTYPE_BYTES: dict[str, float] = {
    "fp32": 4, "float32": 4,
    "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2,
    "int8": 1, "fp8": 1,
    "nf4": 0.5, "int4": 0.5,
}

GIB = 2**30


@dataclass(frozen=True)
class ModelConfig:
    """Minimal description of a decoder-only transformer (Llama/Qwen style, SwiGLU MLP)."""

    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab_size: int
    tie_embeddings: bool = True
    qkv_bias: bool = False       # Qwen2/2.5 use bias on q/k/v projections
    gated_mlp: bool = True       # SwiGLU: gate, up, down
    norm_params_per_layer: int | None = None  # defaults to 2 RMSNorm vectors of size d_model

    # ---- parameter counting ------------------------------------------------------------
    def param_breakdown(self) -> dict[str, int]:
        h, d, a, akv, f = self.d_model, self.head_dim, self.n_heads, self.n_kv_heads, self.d_ff
        q = h * (a * d) + (a * d if self.qkv_bias else 0)
        k = h * (akv * d) + (akv * d if self.qkv_bias else 0)
        v = k
        o = (a * d) * h
        attn = q + k + v + o
        mlp = (3 if self.gated_mlp else 2) * h * f
        norms = self.norm_params_per_layer if self.norm_params_per_layer is not None else 2 * h
        per_layer = attn + mlp + norms
        embed = self.vocab_size * h
        lm_head = 0 if self.tie_embeddings else self.vocab_size * h
        final_norm = h
        return {
            "embeddings": embed,
            "attention": attn * self.n_layers,
            "mlp": mlp * self.n_layers,
            "norms": norms * self.n_layers + final_norm,
            "lm_head": lm_head,
            "per_layer": per_layer,
            "total": embed + per_layer * self.n_layers + final_norm + lm_head,
        }

    @property
    def n_params(self) -> int:
        return self.param_breakdown()["total"]

    def kv_bytes_per_token(self, dtype: str = "bf16") -> float:
        """2 (K and V) × layers × kv_heads × head_dim × bytes."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * DTYPE_BYTES[dtype]

    def lora_trainable_params(self, rank: int, targets: tuple[str, ...] = ("q", "k", "v", "o", "gate", "up", "down")) -> int:
        """LoRA adds r*(d_in + d_out) per adapted linear layer."""
        h, d, a, akv, f = self.d_model, self.head_dim, self.n_heads, self.n_kv_heads, self.d_ff
        dims = {
            "q": (h, a * d), "k": (h, akv * d), "v": (h, akv * d), "o": (a * d, h),
            "gate": (h, f), "up": (h, f), "down": (f, h),
        }
        return self.n_layers * sum(rank * (dims[t][0] + dims[t][1]) for t in targets)

    # ---- construction ------------------------------------------------------------------
    @classmethod
    def from_hf_config(cls, cfg: dict[str, Any] | Any, name: str | None = None) -> ModelConfig:
        """Build from a HF `config.json` dict or a `transformers.PretrainedConfig`."""
        get = (lambda k, default=None: cfg.get(k, default)) if isinstance(cfg, dict) else (lambda k, default=None: getattr(cfg, k, default))
        h = get("hidden_size")
        a = get("num_attention_heads")
        akv = get("num_key_value_heads", a) or a
        d = get("head_dim") or h // a
        model_type = get("model_type", "")
        return cls(
            name=name or get("_name_or_path", model_type) or model_type,
            n_layers=get("num_hidden_layers"),
            d_model=h,
            n_heads=a,
            n_kv_heads=akv,
            head_dim=d,
            d_ff=get("intermediate_size"),
            vocab_size=get("vocab_size"),
            tie_embeddings=bool(get("tie_word_embeddings", False)),
            qkv_bias=bool(get("attention_bias", False)) or model_type in ("qwen2", "qwen2_moe"),
            gated_mlp=True,
        )

    @classmethod
    def from_pretrained(cls, model_id: str) -> ModelConfig:
        """Load through transformers (uses the local HF cache; downloads config.json only)."""
        from transformers import AutoConfig  # optional dependency

        return cls.from_hf_config(AutoConfig.from_pretrained(model_id), name=model_id)


# Known configurations so the calculator (and the tests) work offline.
KNOWN_CONFIGS: dict[str, ModelConfig] = {
    "Qwen/Qwen2.5-0.5B-Instruct": ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", 24, 896, 14, 2, 64, 4864, 151936, True, True),
    "Qwen/Qwen2.5-1.5B-Instruct": ModelConfig("Qwen/Qwen2.5-1.5B-Instruct", 28, 1536, 12, 2, 128, 8960, 151936, True, True),
    "Qwen/Qwen2.5-7B-Instruct": ModelConfig("Qwen/Qwen2.5-7B-Instruct", 28, 3584, 28, 4, 128, 18944, 152064, False, True),
    "meta-llama/Llama-3.1-8B": ModelConfig("meta-llama/Llama-3.1-8B", 32, 4096, 32, 8, 128, 14336, 128256, False, False),
    "meta-llama/Llama-2-7b": ModelConfig("meta-llama/Llama-2-7b", 32, 4096, 32, 32, 128, 11008, 32000, False, False),
    "HuggingFaceTB/SmolLM2-135M-Instruct": ModelConfig("HuggingFaceTB/SmolLM2-135M-Instruct", 30, 576, 9, 3, 64, 1536, 49152, True, False),
}


def get_config(name_or_id: str) -> ModelConfig:
    if name_or_id in KNOWN_CONFIGS:
        return KNOWN_CONFIGS[name_or_id]
    return ModelConfig.from_pretrained(name_or_id)


# ---- activations -----------------------------------------------------------------------------
def activation_bytes_per_layer(
    cfg: ModelConfig,
    batch: int,
    seq_len: int,
    act_dtype: str = "bf16",
    flash_attention: bool = True,
) -> float:
    """Tensors a layer must keep for backward (Korthikanti et al. 2022 style estimate, adapted
    to GQA + SwiGLU). With a fused attention kernel the s×s score matrix is never stored."""
    p = DTYPE_BYTES[act_dtype]
    b, s, h, f = batch, seq_len, cfg.d_model, cfg.d_ff
    kv = cfg.n_kv_heads * cfg.head_dim
    # attention block: norm in, qkv in, q, k, v, attn out, o_proj in, residual
    attn = b * s * p * (h + h + h + 2 * kv + h + h) + b * s * cfg.n_heads * 4  # + logsumexp (fp32)
    if not flash_attention:
        attn += 2 * b * cfg.n_heads * s * s * p  # scores and softmax output
    # mlp block: norm in, mlp in, gate out, up out, silu out, product (down in)
    mlp = b * s * p * (h + h + f + f + f + f)
    return attn + mlp


def activation_bytes(
    cfg: ModelConfig,
    batch: int,
    seq_len: int,
    act_dtype: str = "bf16",
    gradient_checkpointing: bool = False,
    flash_attention: bool = True,
) -> dict[str, float]:
    per_layer = activation_bytes_per_layer(cfg, batch, seq_len, act_dtype, flash_attention)
    p = DTYPE_BYTES[act_dtype]
    layer_input = batch * seq_len * cfg.d_model * p
    if gradient_checkpointing:
        # keep only each layer's input; recompute one layer at a time during backward
        stored = layer_input * cfg.n_layers
        peak = stored + per_layer
    else:
        stored = per_layer * cfg.n_layers
        peak = stored
    logits = batch * seq_len * cfg.vocab_size * (p + 4)  # bf16 logits + fp32 copy for the loss
    return {"per_layer": per_layer, "stored": stored, "peak": peak, "logits": logits}


# ---- optimizer / parameter memory --------------------------------------------------------------
OPTIMIZER_STATES = {"adamw": 2, "adam": 2, "sgd": 0, "sgd_momentum": 1, "adamw_8bit": 2, "adafactor": 0.5}


def optimizer_bytes_per_param(optimizer: str, param_dtype: str, master_weights_fp32: bool) -> float:
    """torch optimizers keep state in the parameter dtype; 8-bit optimizers keep 1 byte/state."""
    n_states = OPTIMIZER_STATES[optimizer]
    if optimizer == "adamw_8bit":
        return n_states * 1.0
    state_bytes = 4 if master_weights_fp32 else DTYPE_BYTES[param_dtype]
    return n_states * state_bytes


@dataclass
class MemoryEstimate:
    scenario: str
    model: str
    config: dict[str, Any]
    components: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(self.components.values())

    def as_gib(self) -> dict[str, float]:
        out = {k: v / GIB for k, v in self.components.items()}
        out["total"] = self.total / GIB
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"scenario": self.scenario, "model": self.model, "config": self.config,
                "components_bytes": self.components, "total_bytes": self.total, "components_gib": self.as_gib()}


def estimate_training(
    cfg: ModelConfig,
    batch: int,
    seq_len: int,
    *,
    param_dtype: str = "fp32",
    optimizer: str = "adamw",
    mixed_precision: bool = False,
    master_weights_fp32: bool = False,
    gradient_checkpointing: bool = False,
    flash_attention: bool = True,
    lora_rank: int | None = None,
    base_weight_dtype: str | None = None,
) -> MemoryEstimate:
    """Steady-state peak memory of one training step.

    mixed_precision=True means autocast(bf16): parameters stay in `param_dtype` (fp32 in the
    classic AMP recipe) and activations are bf16. master_weights_fp32=True models a
    DeepSpeed/FSDP style setup where bf16 parameters have an fp32 master copy + fp32 states.
    lora_rank freezes the base model (no grads / optimizer states for it) and adds adapters.
    """
    n = cfg.n_params
    act_dtype = "bf16" if mixed_precision or DTYPE_BYTES[param_dtype] == 2 else param_dtype
    base_dtype = base_weight_dtype or param_dtype
    if lora_rank:
        trainable = cfg.lora_trainable_params(lora_rank)
        weights = n * DTYPE_BYTES[base_dtype] + trainable * 4  # adapters are kept in fp32 by PEFT
        grads = trainable * 4
        opt = trainable * optimizer_bytes_per_param(optimizer, "fp32", False)
        master = 0.0
    else:
        trainable = n
        weights = n * DTYPE_BYTES[param_dtype]
        grads = n * DTYPE_BYTES[param_dtype]
        opt = n * optimizer_bytes_per_param(optimizer, param_dtype, master_weights_fp32)
        master = n * 4 if master_weights_fp32 and DTYPE_BYTES[param_dtype] < 4 else 0.0
    acts = activation_bytes(cfg, batch, seq_len, act_dtype, gradient_checkpointing, flash_attention)
    comps = {
        "weights": weights,
        "master_weights": master,
        "gradients": grads,
        "optimizer_states": opt,
        "activations": acts["peak"],
        "logits": acts["logits"],
    }
    if mixed_precision and not lora_rank:
        comps["autocast_weight_copies"] = n * 2  # bf16 casts of fp32 weights cached inside autocast
    return MemoryEstimate(
        scenario="training",
        model=cfg.name,
        config={
            "batch": batch, "seq_len": seq_len, "param_dtype": param_dtype, "optimizer": optimizer,
            "mixed_precision": mixed_precision, "master_weights_fp32": master_weights_fp32,
            "gradient_checkpointing": gradient_checkpointing, "flash_attention": flash_attention,
            "lora_rank": lora_rank, "trainable_params": trainable, "n_params": n,
        },
        components=comps,
    )


def estimate_inference(
    cfg: ModelConfig,
    batch: int,
    prompt_len: int,
    gen_len: int = 0,
    *,
    weight_dtype: str = "bf16",
    kv_dtype: str | None = None,
    flash_attention: bool = True,
    full_prefill_logits: bool = False,
) -> MemoryEstimate:
    """Peak memory for serving `batch` sequences of prompt_len + gen_len tokens.

    Prefill keeps only one layer's working set alive at a time (no_grad), so the transient
    activation term is one layer, not L layers. HF `generate` computes logits only for the last
    position by default; set full_prefill_logits for a plain forward over the whole prompt."""
    kv_dtype = kv_dtype or weight_dtype
    p = DTYPE_BYTES[weight_dtype]
    tokens = batch * (prompt_len + gen_len)
    kv = cfg.kv_bytes_per_token(kv_dtype) * tokens
    transient = activation_bytes_per_layer(cfg, batch, prompt_len, weight_dtype, flash_attention)
    logits_positions = prompt_len if full_prefill_logits else 1
    logits = batch * logits_positions * cfg.vocab_size * p  # no loss, so no fp32 upcast
    comps = {"weights": cfg.n_params * p, "kv_cache": kv, "prefill_activations": transient, "logits": logits}
    return MemoryEstimate(
        scenario="inference", model=cfg.name,
        config={"batch": batch, "prompt_len": prompt_len, "gen_len": gen_len, "weight_dtype": weight_dtype,
                "kv_dtype": kv_dtype, "flash_attention": flash_attention, "n_params": cfg.n_params},
        components=comps,
    )


def format_table(est: MemoryEstimate) -> str:
    rows = [f"{'component':<26}{'GiB':>10}{'share':>8}"]
    total = est.total
    for k, v in est.components.items():
        if v:
            rows.append(f"{k:<26}{v / GIB:>10.3f}{v / total:>8.1%}")
    rows.append(f"{'total':<26}{total / GIB:>10.3f}{'':>8}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Estimate LLM memory for training or inference.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF id (known configs work offline)")
    ap.add_argument("--scenario", choices=["training", "inference"], default="training")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=512, help="training sequence length / inference prompt length")
    ap.add_argument("--gen-len", type=int, default=0)
    ap.add_argument("--dtype", default=None, help="param/weight dtype (default fp32 for training, bf16 for inference)")
    ap.add_argument("--optimizer", default="adamw", choices=sorted(OPTIMIZER_STATES))
    ap.add_argument("--amp", action="store_true", help="bf16 autocast mixed precision")
    ap.add_argument("--master-fp32", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--no-flash", action="store_true", help="assume attention scores are materialised")
    ap.add_argument("--lora-rank", type=int, default=None)
    ap.add_argument("--base-dtype", default=None, help="base weight dtype for (Q)LoRA, e.g. nf4")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cfg = get_config(args.model)
    if args.scenario == "training":
        est = estimate_training(
            cfg, args.batch, args.seq_len, param_dtype=args.dtype or "fp32", optimizer=args.optimizer,
            mixed_precision=args.amp, master_weights_fp32=args.master_fp32,
            gradient_checkpointing=args.grad_ckpt, flash_attention=not args.no_flash,
            lora_rank=args.lora_rank, base_weight_dtype=args.base_dtype,
        )
    else:
        est = estimate_inference(cfg, args.batch, args.seq_len, args.gen_len,
                                 weight_dtype=args.dtype or "bf16", flash_attention=not args.no_flash)
    if args.json:
        print(json.dumps(est.to_dict(), indent=2))
    else:
        pb = cfg.param_breakdown()
        print(f"{cfg.name}: {pb['total'] / 1e6:.1f}M params "
              f"(embeddings {pb['embeddings'] / 1e6:.1f}M, attention {pb['attention'] / 1e6:.1f}M, mlp {pb['mlp'] / 1e6:.1f}M)")
        print(f"KV cache: {cfg.kv_bytes_per_token('bf16') / 1024:.1f} KiB/token (bf16)")
        print(f"scenario={est.scenario} {json.dumps(est.config)}")
        print(format_table(est))


if __name__ == "__main__":
    main()

