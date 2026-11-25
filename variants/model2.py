"""
Fully-binary model family (follow-up to binary-loop-hybrid).

Three parameter-matched variants, ALL of whose block projection matmuls
are binary (sign weights + per-row mean-|W| scale, straight-through
estimator - the same BinaryLinear as the original project):

  A. BinaryLoopedTransformerLM - only LoopedTransformerBlocks.
  B. BinaryLoopedMambaLM       - only SSM blocks, with BINARY projections
     (in_proj / x_proj / dt_proj / out_proj are BinaryLinear) and each
     distinct block's weights shared and applied R times (a "looped
     Mamba"); no attention anywhere.
  C. FullyBinaryHybridLM       - alternating BinarySSMBlock (single pass)
     + LoopedTransformerBlock: the original hybrid with the SSM side
     made binary too.

Full-precision holdouts (same as the original design): the tied
embedding / LM head, and the SSM internals that are not projections
(depthwise conv weights, A_log, D).
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import BinaryLinear, LoopedTransformerBlock, count_params  # noqa: F401


# --------------------------------------------------------------------------
# SSM block with binary projections
# --------------------------------------------------------------------------
class BinarySSMBlock(nn.Module):
    """Mamba-style selective SSM (parallel associative scan, pure PyTorch)
    whose four projection linears are BinaryLinear."""

    def __init__(self, d_model, d_inner=384, d_state=8, d_conv=4, dt_rank=16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.norm = nn.RMSNorm(d_model)
        self.in_proj = BinaryLinear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1,
                                groups=d_inner, bias=True)
        self.x_proj = BinaryLinear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = BinaryLinear(dt_rank, d_inner, bias=True)
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = BinaryLinear(d_inner, d_model, bias=False)

    def forward(self, x):
        from model import associative_scan
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)                       # (B, T, 2*d_inner)
        x_, z = xz.chunk(2, dim=-1)
        x_ = self.conv1d(x_.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
        x_ = F.silu(x_)
        dtbc = self.x_proj(x_)
        dt_r, Bv, Cv = torch.split(
            dtbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_r.float()))
        A = -torch.exp(self.A_log.float())
        if os.environ.get("BLH_FAST_SCAN") == "1":
            from scan_fast import fused_selective_scan
            y = fused_selective_scan(
                dt, Bv, Cv, x_, A,
                chunk=int(os.environ.get("BLH_SCAN_CHUNK", "32")))
        else:
            a = torch.exp(dt.unsqueeze(-1) * A)
            u = (dt.unsqueeze(-1) * Bv.float().unsqueeze(2)) * x_.float().unsqueeze(-1)
            h = associative_scan(a, u)
            y = (h * Cv.float().unsqueeze(2)).sum(-1)
        y = y + self.D.float() * x_.float()
        y = (y * F.silu(z.float())).to(residual.dtype)
        return residual + self.out_proj(y)


class LoopedBinarySSMBlock(nn.Module):
    """One BinarySSMBlock whose weights are shared and applied `loops`
    times in a row - the SSM analogue of the looped transformer block."""

    def __init__(self, d_model, loops=3, **ssm_kw):
        super().__init__()
        self.loops = loops
        self.block = BinarySSMBlock(d_model, **ssm_kw)

    def forward(self, x):
        for _ in range(self.loops):
            if os.environ.get("BLH_NO_CKPT") == "1" \
                    or os.environ.get("BLH_FAST_SCAN") == "1":
                x = self.block(x)
            else:
                x = torch.utils.checkpoint.checkpoint(
                    self.block, x, use_reentrant=False)
        return x


# --------------------------------------------------------------------------
# Shared LM shell
# --------------------------------------------------------------------------
class _LMShell(nn.Module):
    def __init__(self, vocab_size=256, d_model=256, seq_len=256):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList()
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos).unsqueeze(0)
        for blk in self.blocks:
            if isinstance(blk, BinarySSMBlock) and self.training \
                    and os.environ.get("BLH_NO_CKPT") != "1" \
                    and os.environ.get("BLH_FAST_SCAN") != "1":
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids, n_new, device):
        self.eval()
        ids = list(prompt_ids)
        for _ in range(n_new):
            ctx = torch.tensor([ids[-self.seq_len:]], dtype=torch.long,
                               device=device)
            logits = self(ctx)[0, -1]
            nxt = int(torch.argmax(logits).item())
            ids.append(nxt)
        self.train()
        return ids


class BinaryLoopedTransformerLM(_LMShell):
    """Variant A: binary looped transformer only."""

    def __init__(self, vocab_size=256, d_model=256, seq_len=256,
                 n_blocks=4, loops=3, n_head=8, mlp_hidden=1104):
        super().__init__(vocab_size, d_model, seq_len)
        self.variant = "tf"
        self.loops = loops
        blocks = []
        for _ in range(n_blocks):
            blk = LoopedTransformerBlock(d_model, n_head=n_head, loops=loops)
            # widen MLP to the requested hidden size (param matching)
            blk.fc1 = BinaryLinear(d_model, mlp_hidden, bias=False)
            blk.fc2 = BinaryLinear(mlp_hidden, d_model, bias=False)
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)


class BinaryLoopedMambaLM(_LMShell):
    """Variant B: binary looped Mamba only (binary SSM projections,
    each block weight-shared and applied `loops` times)."""

    def __init__(self, vocab_size=256, d_model=256, seq_len=256,
                 n_blocks=7, loops=3, d_inner=568, d_state=8):
        super().__init__(vocab_size, d_model, seq_len)
        self.variant = "mamba"
        self.loops = loops
        self.blocks = nn.ModuleList([
            LoopedBinarySSMBlock(d_model, loops=loops, d_inner=d_inner,
                                 d_state=d_state)
            for _ in range(n_blocks)])


class FullyBinaryHybridLM(_LMShell):
    """Variant C: alternating binary SSM blocks (single pass) and binary
    looped transformer blocks - the original hybrid, fully binary."""

    def __init__(self, vocab_size=256, d_model=256, seq_len=256,
                 n_ssm=3, n_loop_blocks=3, loops=3, n_head=8,
                 d_inner=384, d_state=8):
        super().__init__(vocab_size, d_model, seq_len)
        self.variant = "hybrid"
        self.loops = loops
        blocks = []
        for i in range(max(n_ssm, n_loop_blocks)):
            if i < n_ssm:
                blocks.append(BinarySSMBlock(d_model, d_inner=d_inner,
                                             d_state=d_state))
            if i < n_loop_blocks:
                blocks.append(LoopedTransformerBlock(
                    d_model, n_head=n_head, loops=loops))
        self.blocks = nn.ModuleList(blocks)


VARIANTS = {
    "tf": BinaryLoopedTransformerLM,
    "mamba": BinaryLoopedMambaLM,
    "hybrid": FullyBinaryHybridLM,
}


# --------------------------------------------------------------------------
# Inference state footprint (bytes retained per generated token)
# --------------------------------------------------------------------------
def state_footprint(model):
    """Returns dict with kv_bytes_per_token (grows linearly with context)
    and ssm_state_bytes (constant), computed from the actual modules.
    Assumes fp16 KV cache and fp32 SSM state (as the scan runs in fp32)."""
    kv_per_token = 0
    ssm_bytes = 0
    d_model = model.d_model

    def ssm_block_bytes(b):
        # recurrent state h: (d_inner, d_state) fp32, plus conv window
        return b.d_inner * b.d_state * 4 + b.d_inner * (4 - 1) * 4

    for blk in model.blocks:
        if isinstance(blk, LoopedTransformerBlock):
            # each loop application keeps its own K and V for every token
            kv_per_token += blk.loops * 2 * d_model * 2
        elif isinstance(blk, LoopedBinarySSMBlock):
            # each loop application maintains its own recurrent state
            ssm_bytes += blk.loops * ssm_block_bytes(blk.block)
        elif isinstance(blk, BinarySSMBlock):
            ssm_bytes += ssm_block_bytes(blk)
    return dict(kv_bytes_per_token=kv_per_token, ssm_state_bytes=ssm_bytes)


def binary_latent_count(model):
    return sum(p.numel() for m in model.modules()
               if m.__class__.__name__ == "BinaryLinear"
               for p in m.parameters(recurse=False) if p.dim() == 2)
