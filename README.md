# Binary Looped (Internal-Reasoning) Transformer–Mamba Hybrid

A small, live-trained language model built as a demonstrator in the spirit of
Scott's AGILLM experiments: a **binary-weight, weight-shared looped
transformer** hybridised with **Mamba-style selective state-space blocks**,
trained from scratch on text **streamed live from HuggingFace** (no corpus is
downloaded).

- **3,450,240 parameters** total, of which **2,359,296 are binary-latent**
  (the linear layers inside the looped transformer blocks).
- Byte-level LM: vocabulary = 256 raw UTF-8 bytes, no trained tokenizer.
- `d_model` 256, sequence length 256, batch 16, AdamW lr 3e-3 (cosine).

## Architecture (`model.py`)

The stack alternates two block types, 3 of each:

- **SSMBlock** — a simplified Mamba-1-style *selective* SSM written in pure
  PyTorch (no `mamba-ssm` package, no custom CUDA kernels). Δ, B and C are
  input-dependent ("selective"); A is a learned diagonal matrix (S4D-style
  init). The recurrence `h_t = a_t · h_{t-1} + u_t` is evaluated **exactly**
  with a parallel associative (prefix) scan. Depthwise causal conv + SiLU
  gating, as in Mamba.
- **LoopedTransformerBlock** — ONE transformer block (8 heads, MLP ratio 4)
  whose weights are shared and applied **R = 3 times** in a row per stack
  position — the "internal reasoning" loop. Every linear layer inside it is a
  **BinaryLinear**: the forward pass uses `W_eff = α · sign(W)` with per-row
  scale `α = mean(|W|)` (genuinely 1-bit weights plus a scale), while
  gradients reach the latent full-precision weights through a
  straight-through estimator (STE).

Embeddings (tied with the LM head), the SSM projections and the output head
stay full precision.

## Files

| File | What it is |
|---|---|
| `model.py` | The hybrid model: `BinaryLinear`, `SSMBlock`, `LoopedTransformerBlock`, `BinaryLoopHybridLM` |
| `train.py` | Training loop: HF streaming data, AdamW, fp16 autocast, atomic `state.json` + `history.jsonl` logging, checkpoints every 1,000 steps to `ckpt_last.pt`, clean exit when a `STOP` file appears |
| `dashboard.py` | The GUI: a stdlib-only web monitor (live loss curve, step / loss / EMA / tokens-per-second / GPU memory, latest greedy sample). Endpoints `/`, `/state`, `/history` |
| `ckpt_step3000.pt` | Checkpoint snapshot from the live run (see below) |

A second GUI variant — a Tkinter desktop monitor that polls the same
`state.json` — was used when the model first ran on Scott's laptop (MSI GP65
Leopard); the web dashboard in this repo is the canonical monitor for the
current run.

## Train it

```bash
pip install torch datasets
python3 train.py
```

Data is streamed with `datasets` `streaming=True`, trying in order:
`roneneldan/TinyStories`, then `Salesforce/wikitext` (103, then 2) as
fallbacks. Training runs on CUDA if available, else CPU. Create a file named
`STOP` in the folder to stop cleanly (a final checkpoint is saved).

## Run the monitor (GUI)

```bash
python3 dashboard.py 8080    # then open http://localhost:8080/
```

The dashboard reads `state.json` / `history.jsonl` written by the trainer
and auto-refreshes every 2 seconds.

## Checkpoint

`ckpt_step3000.pt` — a snapshot of `ckpt_last.pt` taken from the live run
without interrupting it (verified: it loads and its SHA-256 matches the
source file on the training box).

- **Step:** 3,000
- **Loss at step 3,000:** 0.8828 (EMA loss 1.0109)
- **Size:** 13,819,708 bytes
- **SHA-256:** `a28a3e1ca81a01f97e8a6f74e37220490dd060f7b5fdef4e0d8a35d26a447c0b`
- **Contents:** `{"step": 3000, "model": <state_dict>, "cfg": <training config>}`
- **Run it came from:** live training on an RTX 2080 Ti (Vast.ai instance
  61918457), streaming `roneneldan/TinyStories`, ~23,500 tokens/s. The run
  continued past this snapshot.

Load it:

```python
import torch
from model import BinaryLoopHybridLM

ckpt = torch.load("ckpt_step3000.pt", map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
model = BinaryLoopHybridLM(d_model=cfg["d_model"], seq_len=cfg["seq_len"],
                           n_ssm=cfg["n_ssm"], n_loop_blocks=cfg["n_loop_blocks"],
                           loops=cfg["loops"], n_head=cfg["n_head"],
                           d_inner=cfg["d_inner"], d_state=cfg["d_state"])
model.load_state_dict(ckpt["model"])
```

## Honest scope

This is a 3.45M-parameter demonstrator, not a frontier model: it shows the
binary + looped + SSM-hybrid mechanics training stably end-to-end on streamed
data, with samples improving from word salad toward simple TinyStories-style
sentences within a few thousand steps.
