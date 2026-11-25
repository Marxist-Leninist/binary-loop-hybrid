# Fully-Binary Variants

Follow-up to the binary looped transformer–Mamba hybrid: the whole model
family made **fully binary**, and a matched comparison of the three
architectures at (nearly) identical parameter counts.

## The three variants (`model2.py`)

All block projection matmuls use `BinaryLinear` — forward weights are
`alpha * sign(W)` with a per-row scale `alpha = mean(|W|)`, trained
through a straight-through estimator on latent full-precision weights.

| Variant | Architecture | Params | Binary-latent |
|---|---|---:|---:|
| A. Binary looped transformer | 4 transformer blocks, weights shared, each applied 3× (looped); no SSM | 3,442,944 | 3,309,568 |
| B. Binary looped Mamba | 7 selective-SSM blocks with **binary projections** (`in_proj`, `x_proj`, `dt_proj`, `out_proj`), each block weight-shared and applied 3×; no attention | 3,437,176 | 3,244,416 |
| C. Fully binary hybrid | 3 binary SSM blocks (single pass) alternating with 3 binary looped transformer blocks — the original hybrid with the SSM side made binary too | 3,450,240 | 3,299,328 |

Parameter spread across variants: ≤0.4 %. What stays full precision in
all variants: the tied embedding / LM head, and the SSM internals that
are not projections (depthwise conv, `A_log`, `D`).

## Matched training (`train2.py`)

Identical settings for every run: TinyStories streamed from HuggingFace
(`datasets`, `streaming=True`), byte-level, batch 16, seq 256, AdamW
lr 3e-3, warmup 100, cosine decay, seed 1234, 1,500 steps, fp16 autocast,
on a GTX 1660 SUPER.

### Repetition A/B (training-side fix)

The binary transformer tends to degenerate into repeats ("and and
and"). `train2.py --unlikelihood 1` adds an unlikelihood term
(Welleck et al. 2019): at each position, tokens that would complete a
4-gram already seen earlier in the same sequence are penalised with
`-log(1 - p)`. Variant A was trained with it OFF (`runs/tf-base`) and
ON (`runs/tf-ul`); the reported loss/EMA is pure cross-entropy in both
so the pair compares fairly. Generation in all samples is raw greedy —
the separate decoder-side repeat guard is not used.

## State footprint

Attention variants grow a KV cache linearly with context
(A: 12,288 B/token; C: 9,216 B/token, fp16, counting every loop
application's own K/V). SSM state is constant regardless of context
length (B: 524,832 B total — 7 blocks × 3 loop applications ×
(d_inner·d_state + conv window) in fp32; C: 50,688 B).

## Bit-packed inference checkpoints (`pack_inference.py`)

`pack_inference.py runs/<name>` writes `pack_<name>.npz`: every binary
weight stored as **packed sign bits + a per-row fp16 scale**, everything
else fp16. It also measures the same state dict saved as fp32 and as
fp16, and verifies parity: the pack is reloaded, dequantised, and its
logits / greedy generations compared against the training-format model.
The only parity error source is the fp16 rounding of the per-row scales.

Storage reality check: the bit pack shrinks the **inference** weights.
Training checkpoints still carry latent fp32 weights, and the optimiser
state (AdamW moments, 2× fp32 per parameter) does not shrink at all.

## Files

- `model.py` — original hybrid code (BinaryLinear, SSM, looped transformer)
- `model2.py` — the fully-binary family + state-footprint accounting
- `train2.py` — matched trainer with the unlikelihood option
- `eval2.py` — final comparison (loss, samples, repeat-4-gram counts, state)
- `pack_inference.py` — bit-packing + size comparison + parity check
- `dashboard2.py` — multi-run live dashboard (stdlib HTTP, auto-refresh)
- `results.json` / `pack_*.json` — measured outcomes of the runs above
