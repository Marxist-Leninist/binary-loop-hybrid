"""
Evaluate the matched variant runs: final CE loss / EMA (from each run's
history), greedy samples from fixed prompts, repeat-4-gram counts
(measured identically for every run, raw greedy - no decoder guard),
and the inference state footprint. Writes results.json next to the runs.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model2 import VARIANTS, state_footprint, binary_latent_count, count_params
from train2 import repeat4_count

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
PROMPTS = [b"Once upon a time", b"The "]


def eval_run(name, device):
    out = os.path.join(RUNS, name)
    with open(os.path.join(out, "run_meta.json")) as f:
        meta = json.load(f)
    hist = [json.loads(l) for l in open(os.path.join(out, "history.jsonl"))]
    last = hist[-1]
    model = VARIANTS[meta["variant"]]()
    ck = torch.load(os.path.join(out, "ckpt_last.pt"),
                    map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device)
    samples = {}
    for p in PROMPTS:
        ids = model.generate(p, 512, device)
        gen = ids[len(p):]
        samples[p.decode()] = dict(
            text=bytes(ids).decode("utf-8", "replace"),
            repeat4=repeat4_count(gen),
            n_grams=max(0, len(gen) - 3))
    fp = state_footprint(model)
    return dict(
        run=name, variant=meta["variant"],
        unlikelihood=meta["unlikelihood"],
        params=count_params(model),
        binary_latent_params=binary_latent_count(model),
        steps=last["step"], final_loss=last["loss"],
        final_ema_loss=last["ema_loss"], tok_s=last["tok_s"],
        kv_bytes_per_token=fp["kv_bytes_per_token"],
        ssm_state_bytes=fp["ssm_state_bytes"],
        samples=samples)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = sys.argv[1:] or sorted(
        d for d in os.listdir(RUNS)
        if os.path.exists(os.path.join(RUNS, d, "DONE")))
    results = [eval_run(n, device) for n in names]
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    for r in results:
        s = r["samples"]["Once upon a time"]
        print(f"{r['run']:14s} var={r['variant']:7s} ul={int(r['unlikelihood'])} "
              f"params={r['params']:,} steps={r['steps']} "
              f"loss={r['final_loss']:.4f} ema={r['final_ema_loss']:.4f} "
              f"tok/s={r['tok_s']:.0f} kv/tok={r['kv_bytes_per_token']}B "
              f"ssm={r['ssm_state_bytes']}B rep4={s['repeat4']}/{s['n_grams']}")
        print(f"   sample: {s['text'][:200]!r}")


if __name__ == "__main__":
    main()
