"""
Matched training for the fully-binary variants (model2.py).

Identical settings for every run: TinyStories streamed from HuggingFace
(byte-level), batch 16, seq 256, AdamW lr 3e-3, warmup 100, cosine to
10% over --steps, seed 1234, fp16 autocast. The ONLY differences between
runs are the architecture (--variant) and, for the A/B pair, the
--unlikelihood flag.

Unlikelihood (Welleck et al. 2019), byte-level with n=4: at each position
the negative candidates are the tokens that would complete a 4-gram that
already occurred earlier in the same sequence. Loss adds
alpha * mean(-log(1 - p(negative))). It is a training-side fix for the
degenerate "and and and" repetition; the separate decoder-side repeat
guard is NOT used here - generation in the samples is raw greedy.

Usage: python3 train2.py --variant tf|mamba|hybrid [--unlikelihood 0|1]
       [--alpha 1.0] --steps 1500 --out runs/tf-base
"""

import argparse
import json
import math
import os
import time
import traceback

import torch
import torch.nn.functional as F

from model2 import VARIANTS, count_params, binary_latent_count, state_footprint

DATASETS = [
    ("roneneldan/TinyStories", None),
    ("Salesforce/wikitext", "wikitext-103-raw-v1"),
    ("Salesforce/wikitext", "wikitext-2-raw-v1"),
]


def make_stream():
    from datasets import load_dataset
    last_err = None
    for name, cfg in DATASETS:
        try:
            ds = load_dataset(name, cfg, split="train", streaming=True)
            it = iter(ds)
            buf = bytearray()
            while len(buf) < 1 << 16:
                row = next(it)
                buf += row["text"].encode("utf-8", "ignore") + b"\n"
            return name, it, buf
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"no dataset could be streamed: {last_err}")


def refill(it, buf, need):
    while len(buf) < need:
        try:
            row = next(it)
        except StopIteration:
            raise RuntimeError("stream ended")
        buf += row["text"].encode("utf-8", "ignore") + b"\n"
    return buf


def get_batch(it, buf, bs, seq, device):
    need = bs * (seq + 1) * 2
    buf = refill(it, buf, need)
    chunk = bytes(buf[: bs * (seq + 1)])
    del buf[: bs * (seq + 1) // 2]
    arr = torch.frombuffer(bytearray(chunk), dtype=torch.uint8)
    arr = arr.view(bs, seq + 1).long()
    x = arr[:, :-1].to(device, non_blocking=True)
    y = arr[:, 1:].to(device, non_blocking=True)
    return x, y, buf


def repeat4_count(ids):
    """Number of positions whose 4-gram already occurred earlier in the
    sequence (every repeat occurrence counts). Same metric for all runs."""
    seen = set()
    reps = 0
    for i in range(len(ids) - 3):
        g = bytes(ids[i:i + 4])
        if g in seen:
            reps += 1
        else:
            seen.add(g)
    return reps


def unlikelihood_loss(logits, x, alpha):
    """logits: (B,T,V) predicting x[t+1] at position t. x: (B,T) inputs.
    Negatives at t: tokens completing a 4-gram already fully inside the
    prefix x[0..t]. Returns (loss_term, n_pairs)."""
    B, T, V = logits.shape
    device = logits.device
    # trigram ids for starts s = 0..T-3
    tri = (x[:, :-2] * 65536 + x[:, 1:-1] * 256 + x[:, 2:])      # (B, T-2)
    neg_mask = torch.zeros(B, T, V, dtype=torch.bool, device=device)
    for t in range(2, T - 1):
        # earlier 4-grams end at <= t  =>  starts s <= t-3
        matches = (tri[:, :t - 2] == tri[:, t - 2].unsqueeze(1))  # (B, t-2)
        if not bool(matches.any()):
            continue
        b_idx, s_idx = matches.nonzero(as_tuple=True)
        cand = x[b_idx, s_idx + 3]
        neg_mask[b_idx, t, cand] = True
    n_pairs = int(neg_mask.sum().item())
    if n_pairs == 0:
        return logits.new_zeros(()), 0
    p = torch.softmax(logits.float(), dim=-1)
    ul = -torch.log(1.0 - p.clamp(max=1 - 1e-4))
    return alpha * ul[neg_mask].mean(), n_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--unlikelihood", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    state_path = os.path.join(args.out, "state.json")
    hist_path = os.path.join(args.out, "history.jsonl")
    ckpt_path = os.path.join(args.out, "ckpt_last.pt")
    stop_path = os.path.join(args.out, "STOP")

    cfg = dict(batch_size=16, seq_len=256, lr=3e-3, warmup=100,
               max_steps=args.steps)
    model = VARIANTS[args.variant]().to(device)
    n_params = count_params(model)
    n_binary = binary_latent_count(model)
    fp = state_footprint(model)
    meta = dict(variant=args.variant, unlikelihood=bool(args.unlikelihood),
                alpha=args.alpha, steps=args.steps, seed=args.seed,
                params=n_params, binary_latent_params=n_binary,
                footprint=fp, cfg=cfg)
    with open(os.path.join(args.out, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"run meta: {json.dumps(meta)}", flush=True)

    ds_name, it, buf = make_stream()
    print(f"streaming dataset: {ds_name}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            betas=(0.9, 0.95), weight_decay=0.0)

    def lr_at(step):
        if step < cfg["warmup"]:
            return cfg["lr"] * (step + 1) / cfg["warmup"]
        t = (step - cfg["warmup"]) / max(1, cfg["max_steps"] - cfg["warmup"])
        return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1))))

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    model.train()
    step = 0
    ema_loss = None
    t0 = time.time()
    tok_window = 0
    t_window = time.time()
    tok_s = 0.0
    sample_text = ""
    sample_reps = None
    hist = open(hist_path, "a", encoding="utf-8")

    while step < cfg["max_steps"]:
        if step % 20 == 0 and os.path.exists(stop_path):
            print("STOP file seen, exiting cleanly", flush=True)
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y, buf = get_batch(it, buf, cfg["batch_size"], cfg["seq_len"],
                              device)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=(device == "cuda")):
            logits = model(x)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss = ce
            if args.unlikelihood:
                ul, _npairs = unlikelihood_loss(logits, x, args.alpha)
                loss = ce + ul
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        lv = ce.item()   # EMA tracks pure CE so UL on/off runs compare fairly
        ema_loss = lv if ema_loss is None else 0.98 * ema_loss + 0.02 * lv
        tok_window += cfg["batch_size"] * cfg["seq_len"]
        step += 1

        if step % 250 == 0:
            try:
                ids = model.generate(b"The ", 200, device)
                sample_text = bytes(ids).decode("utf-8", "replace")
                sample_reps = repeat4_count(ids)
            except Exception:  # noqa: BLE001
                sample_text = "(sample failed)"

        if step % 5 == 0 or step == 1:
            now = time.time()
            dt = now - t_window
            if dt > 0:
                tok_s = tok_window / dt
            tok_window = 0
            t_window = now
            gpu_mb = (torch.cuda.max_memory_allocated() / 1e6
                      if device == "cuda" else 0.0)
            state = dict(
                run=os.path.basename(args.out), variant=args.variant,
                unlikelihood=bool(args.unlikelihood),
                step=step, loss=round(lv, 4),
                ema_loss=round(ema_loss, 4) if ema_loss is not None else None,
                tok_s=round(tok_s, 1), lr=lr_at(step),
                elapsed_s=round(now - t0, 1), dataset=ds_name,
                params=n_params, binary_latent_params=n_binary,
                device=device, gpu_mem_mb=round(gpu_mb, 1),
                sample=sample_text, sample_repeat4=sample_reps,
                pid=os.getpid())
            tmp = state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f)
            os.replace(tmp, state_path)
            try:
                hist.write(json.dumps(state) + "\n")
                hist.flush()
            except OSError:
                pass
            print(f"step {step} loss {lv:.4f} ema {ema_loss:.4f} "
                  f"tok/s {tok_s:.0f}", flush=True)

        if step % 500 == 0:
            try:
                torch.save({"step": step, "model": model.state_dict(),
                            "meta": meta}, ckpt_path)
            except OSError:
                pass

    torch.save({"step": step, "model": model.state_dict(), "meta": meta},
               ckpt_path)
    with open(os.path.join(args.out, "DONE"), "w") as f:
        f.write(str(step))
    print("training finished", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
