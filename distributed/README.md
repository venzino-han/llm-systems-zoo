# L07. Ray Systems Lab

**Question.** When does distributed execution pay off, and how do task size, data placement, actor
placement and failure handling change that answer?
**Status.** Implemented; CPU experiments *Measured* on one node; GPU experiments *Measured (pilot, shared GPU)*.
See section 6 for the exact status of each sub-experiment.
**Hardware needed.** CPU-only for 1, 3, 4 (CPU part), 5, 7; one GPU with a few GB free for 2, 4 (GPU part), 6, 8.
Multi-node behaviour is **not** validated here (see section 7).

## 1. What the technique is

Ray is a distributed execution layer: it schedules *tasks* (stateless functions) and *actors*
(stateful processes) across a cluster, moves data through a shared-memory *object store*, and
restarts work after failures. The LLM-specific layers on top of it (Ray Data, Ray Train, Ray Serve)
reuse those primitives. The responsibilities that Ray does **not** take are as important as the
ones it does, so each experiment in this lab isolates one mechanism and its cost model.

| Mechanism | What Ray does | What it costs | What it does *not* do |
|---|---|---|---|
| Tasks (`@ray.remote`) | Schedules a function call on any worker with free resources | ~0.1–1 ms of scheduling + argument serialisation per task | Make tiny tasks fast: below the break-even granularity a task is slower than a local call |
| Actors | Keeps a process (and its model weights) alive between calls | Start-up time once; a mailbox queue per actor | Share state across actors: each actor is an independent process |
| Object store (Plasma) | Zero-copy shared memory per node; `ray.put` once, read many | Serialisation on put; disk *spilling* when the store is full | Free memory while a `ray.get` result (a zero-copy view) is still referenced |
| Resources (`num_cpus`, `num_gpus`) | Logical accounting used for placement | Nothing physical | Isolate memory or compute: a `num_gpus=0.25` actor can allocate the whole GPU |
| Placement groups | Reserve bundles PACKed on one node or SPREAD across nodes | Reservation wait; stranded capacity | Guarantee NVLink locality unless bundles are defined per node |
| Fault tolerance | Re-executes failed tasks, restarts dead actors (`max_restarts`, `max_task_retries`) | Re-execution of side effects: **at-least-once** semantics | Exactly-once results: the application needs idempotency keys |
| Ray Data | Streaming `map`/`map_batches` with stateful actors and backpressure | Per-block overhead; actor start-up | Make the model faster: the input stage can starve the GPU |
| Ray Train | Starts N workers, sets up `torch.distributed`, collects metrics/checkpoints | Process start-up and a driver | The gradient math: `prepare_model` wraps the model in plain DDP |
| Ray Serve | HTTP ingress, replica routing, per-replica queue limits, autoscaling, optional `@serve.batch` | A proxy hop (~ms) and replica actors | Engine-internal continuous batching or KV-cache management (vLLM/SGLang do that) |

**Cost model for task granularity.** With `n` units of work of duration `g` on `w` workers and a
per-task overhead `o`, the parallel wall time is roughly `n·g/w + n·o/w + c` where `c` is a fixed
start-up cost. Distribution helps only when `g ≫ o`; chunking `k` units per task turns the
overhead term into `n·o/(w·k)`.

**Cost model for the object store.** Passing an argument of `S` bytes by value to `K` tasks
serialises it `K` times; `ray.put` once and passing the reference serialises it once and the
workers map it zero-copy. When the sum of live objects exceeds the store size, Ray spills to local
disk and `ray.get` of a spilled object pays a disk read.

**Exactly-once.** Ray retries a task after a worker crash. If the task performed a side effect
before crashing (wrote a row, sent a request), the retry repeats it. Deduplication therefore has to
live in the sink: a keyed write (`sink[item_id] = value`) is idempotent, an append is not.

## 2. When to use it

* **Use Ray tasks** for embarrassingly parallel work whose unit is at least milliseconds (per the
  measured break-even in section 6), or chunk smaller units.
* **Use actors** when the per-call set-up (loading model weights, opening connections) dominates:
  one actor per GPU with batched calls is the usual shape for inference.
* **Use `ray.put` + references** whenever the same large object (weights, a corpus, an index) is
  consumed by many tasks; bound the in-flight work (`ray.wait`) when producers out-run consumers.
* **Use placement groups** to co-locate the workers of one job (PACK) or to spread replicas for
  availability (SPREAD); treat `num_gpus` as a scheduling hint and set memory limits inside the
  process.
* **Use Ray Data** when preprocessing and inference should overlap in a streaming pipeline and the
  dataset does not fit in memory; **Ray Train** when you want the launcher, checkpoints and metrics
  without owning `torchrun`; **Ray Serve** for multi-replica HTTP serving, composition and
  autoscaling in front of an engine that does its own batching.
* **Do not use Ray** for a single-GPU, single-process job: the extra layer adds start-up time,
  object-store copies and debugging surface without a scaling benefit.

## 3. Trade-offs

| Advantages | Disadvantages |
|---|---|
| One API for tasks, actors and data movement, from a laptop to a cluster | Per-task overhead makes fine-grained work slower than local execution |
| Stateful actors amortise model loading and keep GPU context warm | Actors are processes: memory is duplicated, not shared, and state dies with them |
| Zero-copy object store on a node; automatic spilling instead of OOM | Spilling silently turns memory into disk I/O; zero-copy views pin the store |
| Logical resources make heterogeneous scheduling easy | Logical resources give no isolation: co-located actors can still OOM each other |
| Built-in retries and actor restarts | At-least-once semantics: duplicated side effects unless the application is idempotent |
| Ray Data / Train / Serve remove boilerplate for pipelines, launchers and HTTP serving | Another layer to version, observe and debug; it does not replace the engine's own optimisations |

What you pay: start-up latency, copies, operational complexity. What you gain: scale-out,
overlap of CPU and GPU stages, and recovery. The experiments below quantify both sides.

## 4. What is in this directory

| File | Purpose | Runs on |
|---|---|---|
| `ray_core/_common.py` | Local helpers: bounded `ray.init`, spilled-MiB parser, nvidia-smi query, busy-wait work | CPU |
| `ray_core/task_overhead.py` | Sequential vs `multiprocessing` vs Ray tasks vs chunked Ray tasks across task granularities | CPU |
| `ray_core/actor_vs_task.py` | Model inference as task-per-call vs actor pool vs batched actors | CPU or GPU |
| `ray_core/object_store.py` | By-value vs `ray.put`; spilling under a small store; unbounded vs bounded submission | CPU |
| `ray_core/placement.py` | PACK/SPREAD placement groups; fractional-GPU actors vs real memory in nvidia-smi | CPU (+GPU) |
| `ray_core/failure_injection.py` | naive / at-least-once / exactly-once under worker crashes and transient errors | CPU |
| `ray_data/batch_inference.py` | Streaming batch inference: preprocess → stateful generator → parquet, with `ds.stats()` | CPU or GPU |
| `ray_train/torch_trainer.py` | `TorchTrainer` + DDP on a tiny causal LM; checks parameter digests across workers | CPU or GPU |
| `ray_serve/deployment.py` | Serve deployment: replicas, `max_ongoing_requests`, autoscaling, `@serve.batch`, proxy mode | CPU or GPU |
| `ray_serve/bench.py` | Closed-loop client: replicas × Serve-level batching × concurrency | CPU or GPU |
| `../tests/test_l07_*.py` | Pure helpers + tiny local-Ray tests (CPU); Serve/Train smokes marked `slow` | CPU |

## 5. How to run

All commands were executed from the repository root with `.venv/bin/python` (Ray 2.53, torch 2.11).
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` selected the GPU for the GPU pilots;
`CUDA_VISIBLE_DEVICES=""` forces CPU.

```bash
# CPU-only (what CI can run)
python -m pytest tests/test_l07_ray_core.py -q
python -m pytest tests/test_l07_ray_serve_train.py -q --slow        # Serve + Train smokes, ~1 min

# Experiments (add --smoke for a < 2 min version of each)
python distributed/ray_core/task_overhead.py
python distributed/ray_core/object_store.py
python distributed/ray_core/failure_injection.py
python distributed/ray_core/placement.py            # GPU part needs a CUDA device
python distributed/ray_train/torch_trainer.py       # 2 CPU workers; --use-gpu for fractional-GPU workers
python distributed/ray_core/actor_vs_task.py --model HuggingFaceTB/SmolLM2-135M-Instruct
python distributed/ray_data/batch_inference.py --model HuggingFaceTB/SmolLM2-135M-Instruct
python distributed/ray_serve/bench.py --fake                         # scheduling only
python distributed/ray_serve/bench.py --model HuggingFaceTB/SmolLM2-135M-Instruct
python distributed/ray_serve/deployment.py --proxy http://127.0.0.1:8000/ --port 8001   # route to a vLLM server
```

Every run writes `runs/L07_ray_systems/<timestamp>_<name>_<cfghash>/` with `manifest.json`,
`summary.json`, `records.jsonl`, and figures. Curated copies are in `reports/L07_ray_systems/results/`.

## 6. Results

All numbers below are from single pilot runs on a shared host (see section 7); run directories
are named in `reports/L07_ray_systems/results/RESULTS.md`, which holds the full tables, and each
`results/<experiment>/` folder has the manifest, summary and figures. Status per experiment:
**Measured** (CPU: 1, 3, 5, 7, and the CPU pilots of 2, 6, 8) and **Measured, pilot on shared GPU** (4).

**6.1 Task granularity vs overhead** (8 workers, busy-wait units, best of 2 repeats).
Single-item Ray tasks cost ~0.07–0.12 ms each. At 10 µs per unit, 4 000 Ray tasks ran **7× slower**
than the plain loop (0.14× speed-up; `multiprocessing` 0.78×), while chunking 64 units per task gave
2.8×. Single-item tasks break even at ~100 µs (1.06×), reach 4.95× at 1 ms and ≥7.3× (of a maximum 8)
from 10 ms upward. Rule of thumb from this machine: distribute units of ≥10 ms as-is, chunk anything
below ~1 ms. Figures: `results/task_overhead/speedup_vs_granularity.png`, `overhead_vs_granularity.png`.

**6.2 Object store.** Passing a 100 MiB array by value to 16 tasks took 0.262 s; `ray.put` once
plus references took 0.021 s (**12.4×**). Putting 600 MiB into a 256 MiB store spilled 500 MiB to
disk: `put` rose to 724 ms max and `get` of spilled objects to 540 ms max. Producing 120 × 10 MiB
results with unbounded submission took 2.79 s and spilled 1 200 MiB; bounding in-flight work to 8
with `ray.wait` took 0.17 s and spilled nothing (**16×**). A first version of the bounded loop kept
the zero-copy `ray.get` views alive and was *slower* than unbounded: the consumer must release
results, otherwise the store cannot evict.

**6.3 Placement and logical GPUs.** PACK and SPREAD groups were ready in <1 s and, on one node,
land on the same node. Three actors each requesting `num_gpus=0.25` were all given GPU `1` and each
allocated 512 MiB without objection; nvidia-smi grew by 3 202 MiB, i.e. ~555 MiB per actor of CUDA
context on top of the tensors. Ray's GPU share is a scheduling hint; memory limits must be set in
the process (or by running one actor per GPU).

**6.4 Failure injection** (200 items, 4 workers, crashes after the side effect on items 7 and 101,
transient errors on 3 items). `naive`: 77 task references failed and **75 of 200 results were lost**
(everything queued on the two dead actors). `at_least_once`: nothing lost, **2 duplicates** (one per
crash, the write before the crash was repeated on retry), actor restart recovery 0.26 s.
`exactly_once` (keyed sink + check-before-work): **200/200, no duplicates**, same recovery time.

**6.5 Ray Train.** Two CPU workers trained the tiny LM for 3 epochs in 13.9 s at ~31.8 k tokens/s
per worker, wrote a checkpoint, and both ranks ended with the **identical parameter digest**
(all-gathered), which is the DDP invariant Ray Train relies on. The loss (5.546 ≈ ln 256) is the
entropy of the random token stream: the run validates orchestration, not learning.

**6.6 Actor vs task** (CPU, SmolLM2-135M, 64 inputs × 32 tokens, 2 actors / ≤4 parallel tasks).
Loading the model in every task (64 loads × 1.77 s) gave 2.09 inputs/s; a 2-actor pool 6.63 inputs/s
(**3.2×**); the same pool with batches of 8 gave 9.35 inputs/s (**4.5×**). Time to first result was
4.5–5.3 s in every mode: actors must load before they can answer, so pools win on throughput, not
on cold latency.

**6.7 Ray Data** (CPU, 64 rows × 16 tokens). Batch size 1 → 8 → 32 gave 2.26 → 5.46 → 7.27 rows/s
with one actor; two CPU actors did not help (same cores). The slow-preprocess condition (50 ms per
row) ran 2.4× slower, but `ds.stats()` attributes the loss to the generator's own task time
(0.38 → 1.01 s mean, 8.5 s max) while the preprocess UDF took exactly 50.8 ms per row: the host was
under external CPU load during those conditions, so **the input-bottleneck claim is not supported
by this run** and must be repeated on an idle host. The per-operator statistics are the tool that
told us so.

**6.8 Ray Serve.** With a fake 50 ms model, one replica saturates at 19.6 req/s (= 1/50 ms) and p50
grows from 0.056 s to 0.81 s as concurrency rises (pure queueing); two replicas give 36.9 req/s
(1.9×). With a 10 ms Serve-level batching window, one replica reaches 90 req/s and two 150 req/s,
because the fake model's cost per *batch* is flat. With the real 135M HF model on CPU, one replica
is flat at ~4.1–4.3 req/s whatever the concurrency, two replicas give ~6.0–6.4 (1.5×, cores are
shared), one replica **with batching** gives 13.3 req/s (3.1×, mean batch 8), and two replicas with
batching 10.7 req/s: splitting load across replicas shrinks the batches, so replicas and batching
are not additive on a compute-bound host. This is exactly the division of labour to respect in
front of an engine: let vLLM batch, let Serve route.

## 7. Limitations and next experiments

* **Single node only.** Placement groups, spilling and recovery were exercised on one machine;
  network transfer, cross-node object fetches and node loss are not measured. The next step is a
  two-node KubeRay cluster with the same scripts and a node-kill scenario.
* **Shared host.** Other experiments ran concurrently on the same CPU and GPU, so absolute
  throughputs are pilot numbers; the comparisons within one run share the same noise.
* **Tiny models on purpose.** The GPU budget available to this lab was a few GB, so the actor and
  Serve experiments use a 135M-parameter model. The *mechanisms* (load amortisation, batching,
  routing) are the same for larger models; the ratios are not.
* **Ray Serve vs engine batching** is shown with HF `generate`. With a vLLM backend the right
  configuration is `--proxy`: Serve routes, the engine batches; measuring that needs a free GPU.
* Not covered yet: Ray Train with FSDP and real checkpoint resume on GPU, Ray Data with vLLM
  (`ray.data.llm`), autoscaling under an open-loop load (see L03's load generator).

## 8. References

* Ray documentation: Core (tasks, actors, objects, placement groups, fault tolerance), Data, Train, Serve — https://docs.ray.io
* Moritz et al., *Ray: A Distributed Framework for Emerging AI Applications*, OSDI 2018.
* Ray object-spilling design and memory management: https://docs.ray.io/en/latest/ray-core/objects/object-spilling.html
* vLLM + Ray Serve integration guidance: https://docs.ray.io/en/latest/serve/llm/serving-llms.html
