# Reports

One directory per lab: `reports/<Lxx_name>/REPORT.md` plus `results/` with curated `summary.json`
files and figures copied from the raw run directories (`runs/`, git-ignored).

Every report follows the same outline:

1. Question
2. Environment (link to the run manifest)
3. Hypothesis
4. Baseline
5. Change
6. Method (workload, controls, repeats)
7. Results (table / figure, with the measurement grade: smoke / pilot / measurement)
8. Bottleneck analysis
9. Limits
10. Reproduction commands
11. Next experiments

Aggregate the summaries of all runs with:

```bash
python benchmarks/analysis/aggregate_runs.py            # table of every run under runs/
python benchmarks/analysis/aggregate_runs.py --lab L03  # one lab
```
