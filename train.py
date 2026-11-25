"""
Training loop for the Binary Looped Transformer-Mamba Hybrid LM.

Data: streamed live from HuggingFace with datasets streaming=True
(byte-level; no corpus is downloaded). Falls back through a list of
datasets until one streams successfully.

Logs step/loss/tokens-per-second to state.json (atomic rewrite) and
history.jsonl for the GUI monitor. Checkpoints to ckpt_last.pt.
Exits cleanly when a file named STOP appears in this folder.
"""

import json
import math
import os
import time
import traceback

import torch
import torch.nn.functional as F

from model import BinaryLoopHybridLM, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "state.json")
HISTORY = os.path.join(HERE, "history.jsonl")
CKPT = os.path.join(HERE, "ckpt_last.pt")
STOP = os.path.join(HERE, "STOP")

# ---------------------------------------------------------------------------
CFG = dict(
    d_model=256, seq_len=256, batch_size=16,
    n_ssm=3, n_loop_blocks=3, loops=3, n_head=8,
    d_inner=384, d_state=8,
    lr=3e-3, warmup=100, max_steps=200000,
    ckpt_every=1000, sample_every=500,
)
# ---------------------------------------------------------------------------

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
    del buf[: bs * (seq + 1) // 2]          # slide by half a batch
    arr = torch.frombuffer(bytearray(chunk), dtype=torch.uint8)
    arr = arr.view(bs, seq + 1).long()
    x = arr[:, :-1].to(device, non_blocking=True)
    y = arr[:, 1:].to(device, non_blocking=True)
    return x, y, buf


def write_state(state):
    """Atomic state write. On Windows, os.replace can fail with WinError 5
    while the GUI monitor has state.json open for reading; retry briefly
    and never let a state-write failure kill training."""
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    for attempt in range(20):
        try:
            os.replace(tmp, STATE)
            return
        except PermissionError:
            time.sleep(0.05)
        except OSError:
            return


def main():
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CFG
    model = BinaryLoopHybridLM(
        d_model=cfg["d_model"], seq_len=cfg["seq_len"],
        n_ssm=cfg["n_ssm"], n_loop_blocks=cfg["n_loop_blocks"],
        loops=cfg["loops"], n_head=cfg["n_head"],
        d_inner=cfg["d_inner"], d_state=cfg["d_state"]).to(device)
    n_params = count_params(model)
    n_binary = sum(p.numel() for m in model.modules()
                   if m.__class__.__name__ == "BinaryLinear"
                   for p in m.parameters(recurse=False)
                   if p.dim() == 2)
    print(f"params total={n_params:,} binary-latent={n_binary:,}", flush=True)

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
    hist = open(HISTORY, "a", encoding="utf-8")

    while step < cfg["max_steps"]:
        if step % 20 == 0 and os.path.exists(STOP):
            print("STOP file seen, exiting cleanly", flush=True)
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y, buf = get_batch(it, buf, cfg["batch_size"], cfg["seq_len"],
                              device)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=(device == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        lv = loss.item()
        ema_loss = lv if ema_loss is None else 0.98 * ema_loss + 0.02 * lv
        tok_window += cfg["batch_size"] * cfg["seq_len"]
        step += 1

        if step % 500 == 0:
            try:
                ids = model.generate(b"The ", 48, device)
                sample_text = bytes(ids).decode("utf-8", "replace")
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
                step=step, loss=round(lv, 4), ema_loss=round(ema_loss, 4),
                tok_s=round(tok_s, 1), lr=lr_at(step),
                elapsed_s=round(now - t0, 1), dataset=ds_name,
                params=n_params, binary_latent_params=n_binary,
                loops=cfg["loops"], d_model=cfg["d_model"],
                batch_size=cfg["batch_size"], seq_len=cfg["seq_len"],
                device=device, gpu_mem_mb=round(gpu_mb, 1),
                sample=sample_text, pid=os.getpid(),
            )
            write_state(state)
            try:
                hist.write(json.dumps(state) + "\n")
                hist.flush()
            except OSError:
                pass
            print(f"step {step} loss {lv:.4f} ema {ema_loss:.4f} "
                  f"tok/s {tok_s:.0f}", flush=True)

        if step % cfg["ckpt_every"] == 0:
            try:
                torch.save({"step": step, "model": model.state_dict(),
                            "cfg": cfg}, CKPT)
            except OSError:
                pass

    torch.save({"step": step, "model": model.state_dict(), "cfg": cfg}, CKPT)
    print("training finished", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
