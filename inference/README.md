# Inference track

Labs on the model-runtime layer: how a request turns into GPU work, and what each optimisation
buys. Each sub-directory is a self-contained lab with its own README (technique, when to use it,
trade-offs, how to run, measured results).

| Lab | Directory | Question |
|---|---|---|
| L02 KV cache | [`kv_cache/`](kv_cache/) | What limits do context length and concurrency put on the KV cache, and when do paging and prefix caching help? |
| L08a Quantization | [`quantization/`](quantization/) | Memory and latency saved vs perplexity cost per scheme, and which kernels exist on this GPU |
| L08b Speculative decoding | [`speculative_decoding/`](speculative_decoding/) | When does draft-and-verify lower latency without changing outputs? |
| L08c Kernels | [`kernels/`](kernels/) | Attention backends and fused Triton ops across shapes |
| L08d Compilation | [`compilation/`](compilation/) | What `torch.compile` and CUDA graphs recover from launch overhead |
| L04 Multi-GPU | [`multi_gpu/`](multi_gpu/) | TP vs replicas for a given model and workload |

The L08 labs share helpers in `quantization/common.py` (prompt loading, wikitext tokenisation,
vLLM log capture, shared-GPU guards) and write to `runs/L08_inference_optimization/`; the curated
numbers are in `reports/L08_inference_optimization/`.
