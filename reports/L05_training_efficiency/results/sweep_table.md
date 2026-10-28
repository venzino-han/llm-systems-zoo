| condition | precision | ckpt | accum | packing | lora | trainable % | real tok/s | pad frac | peak alloc GiB | peak reserved GiB | est. static GiB | step ms | fwd/bwd/opt share | loss (last 5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | fp32 | False | 1 | False | - | 100.00 | 3048 | 0.15 | 10.28 | 15.63 | 7.36 | 274 | 26/50/23 | 1.380 |
| bf16_autocast | bf16_autocast | False | 1 | False | - | 100.00 | 5503 | 0.15 | 10.13 | 15.63 | 7.36 | 152 | 21/37/41 | 1.478 |
| bf16_weights | bf16_weights | False | 1 | False | - | 100.00 | 8265 | 0.15 | 6.21 | 12.04 | 3.68 | 101 | 22/44/32 | 1.250 |
| bf16+ckpt | bf16_autocast | True | 1 | False | - | 100.00 | 4779 | 0.15 | 9.25 | 15.72 | 7.36 | 174 | 21/42/36 | 1.478 |
| bf16+accum4 | bf16_autocast | False | 4 | False | - | 100.00 | 3497 | 0.00 | 9.26 | 10.45 | 7.36 | 238 | 23/50/26 | 1.466 |
| bf16+packing | bf16_autocast | False | 1 | True | - | 100.00 | 8323 | 0.10 | 13.26 | 14.58 | 7.36 | 220 | 24/47/28 | 1.237 |
| bf16+lora16 | bf16_autocast | False | 1 | False | 16 | 1.75 | 8563 | 0.15 | 6.40 | 12.24 | 1.97 | 97 | 42/51/5 | 0.870 |
| qlora16 | bf16_autocast | False | 1 | False | 16(q) | 2.72 | 6648 | 0.15 | 5.21 | 10.97 | 0.81 | 125 | 41/49/8 | 0.911 |
