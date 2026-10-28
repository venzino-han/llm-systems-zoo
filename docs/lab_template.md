# Lab README template

Every lab directory has a `README.md` that follows this structure so a reader can jump from
a claimed competency to the technique, its trade-offs, the code, and the evidence.

```markdown
# Lxx. <Lab title>

**Question.** One sentence: the question this lab answers.
**Status.** Planned | Implemented | Measured | Reproduced   (per sub-experiment if needed)
**Hardware needed.** CPU-only / 1 GPU (xx GB) / 2+ GPUs / multi-node

## 1. What the technique is
Concept, the cost model (equations or a diagram), what the minimal implementation does and
how it differs from the production engine's implementation.

## 2. When to use it
Workload / hardware conditions under which it helps, and the conditions under which it does not.

## 3. Trade-offs
| Advantages | Disadvantages |
|---|---|
| ... | ... |

What you pay (memory, latency, quality, complexity, operational risk) for what you gain.

## 4. What is in this directory
| File | Purpose | Runs on |
|---|---|---|

## 5. How to run
CPU smoke test command, then the GPU command(s). Only commands that were actually executed.

## 6. Results
Environment summary (link to the run manifest), the table / figure, and the interpretation.
Negative or null results are reported with the same care as positive ones.

## 7. Limitations and next experiments

## 8. References
```

Rules:

* Explain → Implement → Benchmark → Operate: say which depth the lab reaches.
* Distinguish the educational minimal implementation from the production engine.
* Never present a target or an example number as a measured result.
