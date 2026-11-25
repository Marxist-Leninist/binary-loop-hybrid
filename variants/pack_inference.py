"""
Bit-packed inference checkpoint for the fully-binary variants.

For every BinaryLinear weight matrix, the effective inference weight is
W_eff = alpha * sign(W) with per-row alpha = mean(|W|). The pack stores:
  * the sign bits, bit-packed with numpy.packbits (1 bit per weight)
  * alpha per row, fp16
  * every OTHER tensor (embeddings, norms, SSM conv / A_log / D, biases)
    stored fp16 - these are NOT binary and do not shrink to bits.

It then reloads the pack, dequantises, loads the weights into a fresh
model and checks output parity against the training-format model:
max |logit diff| on fixed prompts and greedy-generation agreement.

Also writes the same state_dict in plain fp32 and fp16 torch formats so
the three on-disk sizes can be compared like for like.

Usage: python3 pack_inference.py runs/hybrid [--out pack_hybrid.npz]
"""

import argparse
import io
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import BinaryLinear
from model2 import VARIANTS

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = [b"Once upon a time", b"The little girl", b"He said that"]


def is_binary_weight(model, name):
    """True if `name` is the .weight of a BinaryLinear module."""
    parts = name.split(".")
    if parts[-1] != "weight":
        return False
    mod = model
    for p in parts[:-1]:
        mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
    return isinstance(mod, BinaryLinear)


def pack_state(model, sd):
    arrays = {}
    manifest = {}
    seen = {}
    for name, w in sd.items():
        key = (w.data_ptr(), tuple(w.shape))
        if key in seen:   # tied/shared tensor (e.g. lm_head == tok_emb)
            manifest[name] = dict(kind="alias", alias_of=seen[key],
                                  shape=list(w.shape))
            continue
        seen[key] = name
        arr = w.detach().cpu().numpy()
        if is_binary_weight(model, name) and arr.ndim == 2:
            alpha = np.abs(arr).mean(axis=1)                    # (rows,)
            bits = np.packbits((arr >= 0).astype(np.uint8), axis=1)
            arrays[name + ".bits"] = bits
            arrays[name + ".alpha"] = alpha.astype(np.float16)
            manifest[name] = dict(kind="binary", shape=list(arr.shape))
        else:
            arrays[name] = arr.astype(np.float16)
            manifest[name] = dict(kind="fp16", shape=list(arr.shape))
    return arrays, manifest


def unpack_state(arrays, manifest):
    sd = {}
    for name, info in manifest.items():
        if info["kind"] == "alias":
            sd[name] = sd[info["alias_of"]]
        elif info["kind"] == "binary":
            rows, cols = info["shape"]
            bits = np.unpackbits(arrays[name + ".bits"], axis=1)[:, :cols]
            alpha = arrays[name + ".alpha"].astype(np.float32)
            signs = np.where(bits > 0, 1.0, -1.0).astype(np.float32)
            sd[name] = torch.from_numpy(signs * alpha[:, None])
        else:
            sd[name] = torch.from_numpy(
                arrays[name].astype(np.float32))
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = args.run if os.path.isabs(args.run) else os.path.join(HERE, args.run)
    name = os.path.basename(run_dir.rstrip("/"))
    out_npz = args.out or os.path.join(HERE, f"pack_{name}.npz")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.load(open(os.path.join(run_dir, "run_meta.json")))
    ck = torch.load(os.path.join(run_dir, "ckpt_last.pt"),
                    map_location="cpu", weights_only=False)
    sd = ck["model"]

    model = VARIANTS[meta["variant"]]()
    model.load_state_dict(sd)
    model.to(device).eval()

    # --- reference outputs from the training-format model ---------------
    with torch.no_grad():
        ref_logits = []
        for p in PROMPTS:
            ids = torch.tensor([list(p)], dtype=torch.long, device=device)
            ref_logits.append(model(ids)[0, -1].float().cpu())
        ref_gen = [model.generate(p, 64, device) for p in PROMPTS]

    # --- the three formats, sizes measured on identical content ---------
    buf = io.BytesIO()
    torch.save(sd, buf)
    size_fp32 = buf.tell()
    sd16 = {k: v.half() for k, v in sd.items()}
    buf = io.BytesIO()
    torch.save(sd16, buf)
    size_fp16 = buf.tell()

    arrays, manifest = pack_state(model, sd)
    np.savez(out_npz, **arrays)
    size_packed = os.path.getsize(out_npz)
    n_binary = sum(int(np.prod(v["shape"])) for v in manifest.values()
                   if v["kind"] == "binary")
    n_other = sum(int(np.prod(v["shape"])) for v in manifest.values()
                  if v["kind"] == "fp16")  # aliases excluded: stored once

    # --- reload from the pack and check parity ---------------------------
    z = np.load(out_npz)
    arrays2 = {k: z[k] for k in z.files}
    sd2 = unpack_state(arrays2, manifest)
    model2 = VARIANTS[meta["variant"]]()
    model2.load_state_dict(sd2)
    model2.to(device).eval()
    max_diff = 0.0
    gen_match = []
    with torch.no_grad():
        for i, p in enumerate(PROMPTS):
            ids = torch.tensor([list(p)], dtype=torch.long, device=device)
            lg = model2(ids)[0, -1].float().cpu()
            max_diff = max(max_diff, float((lg - ref_logits[i]).abs().max()))
        for i, p in enumerate(PROMPTS):
            g = model2.generate(p, 64, device)
            gen_match.append(sum(1 for a, b in zip(g[len(p):],
                                                   ref_gen[i][len(p):])
                                 if a == b))
    report = dict(
        run=name, variant=meta["variant"], ckpt_step=ck["step"],
        size_fp32_bytes=size_fp32, size_fp16_bytes=size_fp16,
        size_bitpacked_bytes=size_packed,
        binary_params=n_binary, nonbinary_params=n_other,
        max_logit_diff=max_diff,
        greedy_tokens_matched=gen_match, greedy_tokens_total=64)
    with open(os.path.join(HERE, f"pack_{name}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
