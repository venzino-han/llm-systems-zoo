### Index build (`runs/L09_app_to_infra/20261029_020110_build_index_07a35a2258d4`)

3200 passages, dim 384, BAAI/bge-small-en-v1.5 on cuda:0: model load 4.8s, embedding 0.93s (3426 passages/s).

### RAG evaluation (`runs/L09_app_to_infra/20261029_020818_rag_eval_fceb555f6ef4`, model=Qwen/Qwen2.5-0.5B-Instruct, k=10, top_n=3, device=cuda)

| variant | n | EM | EM 95% CI | F1 | recall@10 | recall in context | prompt tokens | wall (ms) | retrieval (ms) | rerank (ms) | model (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no_rerank | 100 | 0.57 | [0.48, 0.66] | 0.632 | 0.51 | 0.44 | 409 | 44.6 | 6.39 | 0 | 38.2 |
| rerank | 100 | 0.48 | [0.39, 0.58] | 0.541 | 0.51 | 0.44 | 424 | 54.9 | 4.79 | 15.8 | 34.2 |

### Agent evaluation (`runs/L09_app_to_infra/20261029_022209_agent_eval_636b48ec8a74`, model=Qwen/Qwen2.5-0.5B-Instruct, max_steps=3, device=cuda)

| verify | n | success | 95% CI | success QA | success arith | F1 QA | model calls | tool calls | prompt tokens | unsupported claims | parse-fail rate | wall (s) | model (s) | tool (s) | verify (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 50 | 0.24 | [0.12, 0.36] | 0.1 | 0.8 | 0.0487 | 2.96 | 1.96 | 1.38e+03 | 0 | 0.26 | 1.34 | 1.33 | 0.00503 | 0 |
| flag | 50 | 0.24 | [0.12, 0.36] | 0.1 | 0.8 | 0.0484 | 2.96 | 1.96 | 1.38e+03 | 0.74 | 0.26 | 1.36 | 1.35 | 0.00412 | 0.0122 |
| remove | 50 | 0.24 | [0.12, 0.36] | 0.1 | 0.8 | 0.0453 | 2.96 | 1.96 | 1.38e+03 | 0.74 | 0.26 | 1.36 | 1.35 | 0.00416 | 0.0115 |
