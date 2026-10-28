"""L01 tests: analytic memory / FLOPs / roofline estimators (CPU, offline)."""

from __future__ import annotations

import math

import pytest

from fundamentals.flops_calculator import (
    KNOWN_GPUS,
    decode_roofline,
    decode_step_bytes,
    decode_step_flops,
    forward_flops_per_token,
    prefill_flops,
    prefill_roofline,
    ridge_point,
    roofline_time_s,
    training_flops,
)
from fundamentals.memory_calculator import (
    DTYPE_BYTES,
    GIB,
    KNOWN_CONFIGS,
    ModelConfig,
    activation_bytes,
    estimate_inference,
    estimate_training,
    optimizer_bytes_per_param,
)

QWEN05 = KNOWN_CONFIGS["Qwen/Qwen2.5-0.5B-Instruct"]
QWEN7 = KNOWN_CONFIGS["Qwen/Qwen2.5-7B-Instruct"]
LLAMA8 = KNOWN_CONFIGS["meta-llama/Llama-3.1-8B"]


def test_param_count_matches_published_sizes():
    assert QWEN05.n_params == 494_032_768          # exact: Qwen2.5-0.5B has 494M parameters
    assert abs(QWEN7.n_params / 1e9 - 7.62) < 0.02  # 7.61B published
    assert abs(LLAMA8.n_params / 1e9 - 8.03) < 0.02  # 8.03B published


def test_param_breakdown_sums():
    b = QWEN05.param_breakdown()
    assert b["embeddings"] + b["attention"] + b["mlp"] + b["norms"] + b["lm_head"] == b["total"]
    assert b["lm_head"] == 0  # tied embeddings
    assert QWEN7.param_breakdown()["lm_head"] == QWEN7.vocab_size * QWEN7.d_model


def test_kv_bytes_per_token_formula():
    # 2 (K,V) × 24 layers × 2 kv heads × 64 head_dim × 2 bytes = 12 KiB
    assert QWEN05.kv_bytes_per_token("bf16") == 2 * 24 * 2 * 64 * 2 == 12 * 1024
    assert QWEN05.kv_bytes_per_token("fp8") == 6 * 1024
    # GQA: 7 query heads per KV head → KV cache is 7× smaller than the MHA equivalent
    mha = ModelConfig(**{**QWEN05.__dict__, "n_kv_heads": QWEN05.n_heads})
    assert mha.kv_bytes_per_token() == 7 * QWEN05.kv_bytes_per_token()


def test_from_hf_config_dict():
    cfg = ModelConfig.from_hf_config({
        "model_type": "qwen2", "hidden_size": 896, "num_attention_heads": 14, "num_key_value_heads": 2,
        "num_hidden_layers": 24, "intermediate_size": 4864, "vocab_size": 151936, "tie_word_embeddings": True,
    }, name="x")
    assert cfg.n_params == QWEN05.n_params and cfg.head_dim == 64 and cfg.qkv_bias


def test_checkpointing_reduces_activation_estimate():
    full = activation_bytes(QWEN05, 2, 512, gradient_checkpointing=False)
    ckpt = activation_bytes(QWEN05, 2, 512, gradient_checkpointing=True)
    assert ckpt["peak"] < full["peak"] / 5
    assert ckpt["peak"] > ckpt["stored"]  # one recomputed layer on top of the checkpoints


def test_attention_materialisation_is_quadratic():
    flash = activation_bytes(QWEN05, 1, 4096, flash_attention=True)["per_layer"]
    naive = activation_bytes(QWEN05, 1, 4096, flash_attention=False)["per_layer"]
    assert naive - flash == 2 * 1 * QWEN05.n_heads * 4096 * 4096 * 2


def test_training_components_and_dtype_rules():
    est = estimate_training(QWEN05, 1, 256, param_dtype="fp32", optimizer="adamw")
    n = QWEN05.n_params
    assert est.components["weights"] == 4 * n and est.components["gradients"] == 4 * n
    assert est.components["optimizer_states"] == 8 * n       # two fp32 states
    bf16 = estimate_training(QWEN05, 1, 256, param_dtype="bf16", optimizer="adamw")
    assert bf16.components["optimizer_states"] == 4 * n      # torch keeps states in param dtype
    master = estimate_training(QWEN05, 1, 256, param_dtype="bf16", master_weights_fp32=True)
    assert master.components["master_weights"] == 4 * n and master.components["optimizer_states"] == 8 * n
    assert optimizer_bytes_per_param("adamw_8bit", "fp32", False) == 2
    assert optimizer_bytes_per_param("sgd", "fp32", False) == 0


def test_lora_shrinks_gradient_and_optimizer_memory():
    full = estimate_training(QWEN7, 1, 1024, param_dtype="bf16")
    lora = estimate_training(QWEN7, 1, 1024, param_dtype="bf16", lora_rank=16)
    qlora = estimate_training(QWEN7, 1, 1024, param_dtype="bf16", lora_rank=16, base_weight_dtype="nf4")
    assert lora.config["trainable_params"] == QWEN7.lora_trainable_params(16) < 0.01 * QWEN7.n_params
    assert lora.components["gradients"] < full.components["gradients"] / 50
    assert lora.components["optimizer_states"] < full.components["optimizer_states"] / 50
    assert qlora.components["weights"] < lora.components["weights"] / 3
    assert lora.components["activations"] == full.components["activations"]  # LoRA does not save activations


def test_inference_estimate_scales_with_tokens():
    a = estimate_inference(QWEN05, 1, 1024, 0)
    b = estimate_inference(QWEN05, 4, 1024, 1024)
    assert math.isclose(b.components["kv_cache"], 8 * a.components["kv_cache"])
    assert a.components["weights"] == QWEN05.n_params * DTYPE_BYTES["bf16"]
    assert a.total / GIB < 1.5


# ---- FLOPs / roofline -----------------------------------------------------------------------------
def test_training_flops_close_to_6nd():
    tokens, s = 1_000_000, 1
    assert math.isclose(training_flops(QWEN05, tokens, s), 6 * QWEN05.n_params * tokens + 12 * QWEN05.n_layers * QWEN05.d_model * tokens)


def test_prefill_and_decode_flops():
    assert math.isclose(prefill_flops(QWEN05, 1) , forward_flops_per_token(QWEN05, 1) - 2 * QWEN05.n_layers * QWEN05.d_model)
    assert decode_step_flops(QWEN05, 100, batch=4) == 4 * forward_flops_per_token(QWEN05, 100)
    assert decode_step_bytes(QWEN05, 100, batch=4) == QWEN05.n_params * 2 + 4 * 100 * QWEN05.kv_bytes_per_token()


def test_decode_is_memory_bound_and_prefill_compute_bound():
    gpu = KNOWN_GPUS["RTX A6000"]
    assert decode_roofline(QWEN7, 1, 1024, gpu)["bound"] == "memory"
    assert prefill_roofline(QWEN7, 1, 1024, gpu)["bound"] == "compute"
    # batching decode raises intensity roughly linearly until the KV term dominates
    i1 = decode_roofline(QWEN7, 1, 128, gpu)["intensity"]
    i8 = decode_roofline(QWEN7, 8, 128, gpu)["intensity"]
    assert 7 < i8 / i1 <= 8


def test_roofline_time_is_max_of_bounds():
    gpu = KNOWN_GPUS["H100 SXM"]
    r = roofline_time_s(1e12, 1e9, gpu)
    assert r["t_roofline_s"] == max(r["t_compute_s"], r["t_memory_s"])
    assert math.isclose(ridge_point(gpu), 989e12 / 3350e9)


@pytest.mark.parametrize("gpu", list(KNOWN_GPUS))
def test_known_gpus_have_positive_specs(gpu):
    g = KNOWN_GPUS[gpu]
    assert g.peak_tflops_bf16 > 0 and g.bandwidth_gb_s > 0 and g.source
