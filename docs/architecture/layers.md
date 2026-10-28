# System architecture: layers and responsibilities

The labs are organised around the layers of a real LLM system. Each layer has one job and a
clear boundary with its neighbours, and every lab names the layer whose bottleneck it studies.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Application        agent / RAG workflow, user-level quality        (L09)      │
│                    generates the workload, calls the model API                │
├──────────────────────────────────────────────────────────────────────────────┤
│ Serving            admission, queueing, routing, replicas          (L03, L07) │
│                    hands requests to a model runtime                          │
├──────────────────────────────────────────────────────────────────────────────┤
│ Model runtime      batching, attention kernels, KV cache,          (L02, L04, │
│                    quantisation, speculation, TP/PP                 L08)      │
│                    controls GPU compute and memory                            │
├──────────────────────────────────────────────────────────────────────────────┤
│ Distributed exec.  task/actor placement, resource reservation,     (L05, L06, │
│                    object transfer, retries, recovery               L07)      │
├──────────────────────────────────────────────────────────────────────────────┤
│ Cluster infra      containers, nodes, GPU lifecycle, scheduling    (docker/)  │
├──────────────────────────────────────────────────────────────────────────────┤
│ Evaluation &       quality, performance, cost; one run id ties      (harness, │
│ observability      every layer together                             all labs) │
└──────────────────────────────────────────────────────────────────────────────┘
```

| Layer | Responsibility | Boundary with neighbours |
|---|---|---|
| Application | Agent/RAG workflows and user-level evaluation | Produces the workload and calls the model API; never touches GPU memory directly |
| Serving | Accept requests, queue, route, manage replicas | Forwards requests to the model runtime; owns admission control and SLOs |
| Model runtime | Batching, attention execution, KV-cache management, model parallelism | Owns GPU compute and memory; exposes an engine API (vLLM, SGLang, HF) |
| Distributed execution | Place tasks/actors, reserve resources, move data, recover from failures | Independent of the model's internal algorithms (Ray, torchrun) |
| Cluster infrastructure | Container, node and GPU lifecycle | Below the process level (Docker, Kubernetes, KubeRay) |
| Evaluation / observability | Measure quality, performance and cost | Cross-cutting; the run manifest links every layer with one run id |

## Where Ray fits (and where it does not)

Ray is a distributed execution layer, not a model runtime. In this repository:

* **Ray Core** runs tasks and stateful actors, requests CPU/GPU resources and places work (L07).
* **Ray Data** streams read → preprocess → batch-inference pipelines and exposes stage statistics (L07).
* **Ray Train** orchestrates distributed training processes and checkpoints; the maths is PyTorch DDP/FSDP (L05, L07).
* **Ray Serve** manages replicas, request routing and autoscaling above an engine (L07). The engine (vLLM) still
  does the batching and KV management underneath.

Two rules the labs enforce in their write-ups:

1. A Ray GPU *resource request* (`num_gpus=0.25`) is a scheduling hint. It does not isolate or limit GPU memory.
2. Tensor parallelism and KV-cache optimisation belong to the inference engine. Ray does not make a single
   request faster; it decides where requests and models run.
