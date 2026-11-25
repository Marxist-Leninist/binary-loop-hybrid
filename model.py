"""
Binary Looped Transformer-Mamba Hybrid LM (small demonstrator).

Design, in the spirit of Scott's AGILLM experiments but tiny enough to train
live on a laptop RTX 2060:

  * Hybrid stack alternating two block types:
      - SSMBlock: a Mamba-style *selective* state-space block written in pure
        PyTorch (no mamba-ssm / CUDA kernels). Delta, B and C are
        input-dependent ("selective"); A is a learned diagonal matrix.
        The recurrence h_t = a_t * h_{t-1} + u_t is evaluated exactly with a
        parallel associative (prefix) scan, O(log T) sequential steps.
      - LoopedTransformerBlock: ONE transformer block whose weights are
        shared and applied R times in a row (the "internal reasoning" loop).
        Every linear layer inside it is a BinaryLinear.
  * BinaryLinear: forward uses W_eff = alpha * sign(W) with per-row scale
    alpha = mean(|W|); gradients reach the latent full-precision W through a
    straight-through estimator (STE). So the looped transformer's matmuls are
    genuinely 1-bit (plus a scale), while training keeps latent fp weights.
  * Byte-level LM: vocab = 256 raw UTF-8 bytes, no tokenizer to train.

Everything else (embeddings, SSM projections, LM head) stays full precision.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Binary linear (sign weights + STE)
# --------------------------------------------------------------------------
class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight
        alpha = w.abs().mean(dim=1, keepdim=True)          # per-row scale
        w_bin = alpha * torch.sign(w)                       # 1-bit weights
        w_eff = w + (w_bin - w).detach()                    # STE
        return F.linear(x, w_eff, self.bias)


# --------------------------------------------------------------------------
# Selective SSM (Mamba-style) with exact parallel associative scan
# --------------------------------------------------------------------------
def associative_scan(a, u):
    """h_t = a_t * h_{t-1} + u_t with h_{-1} = 0.
    a, u: (B, T, D, N). Returns h: (B, T, D, N). Exact, via prefix doubling."""
    A, U = a, u
    T = a.size(1)
    shift = 1
    while shift < T:
        A_new = A[:, shift:] * A[:, :-shift]
        U_new = A[:, shift:] * U[:, :-shift] + U[:, shift:]
        A = torch.cat([A[:, :shift], A_new], dim=1)
        U = torch.cat([U[:, :shift], U_new], dim=1)
        shift *= 2
    return U


class SSMBlock(nn.Module):
    """Simplified Mamba-1 block: gated selective SSM, pure PyTorch."""

    def __init__(self, d_model, d_inner=384, d_state=8, d_conv=4, dt_rank=16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1,
                                groups=d_inner, bias=True)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        # dt bias init: softplus(bias) spread over ~[0.001, 0.1]
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # A: diagonal, negative, S4D-style init A_n = n+1 stored in log space
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)                       # (B, T, 2*d_inner)
        x_, z = xz.chunk(2, dim=-1)
        # depthwise causal conv over time
        x_ = self.conv1d(x_.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
        x_ = F.silu(x_)
        # selective parameters (computed in fp32 for scan stability)
        dtbc = self.x_proj(x_)
        dt_r, Bv, Cv = torch.split(
            dtbc, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_r.float()))            # (B,T,d_inner)
        A = -torch.exp(self.A_log.float())                     # (d_inner, N)
        a = torch.exp(dt.unsqueeze(-1) * A)                    # (B,T,d_inner,N)
        u = (dt.unsqueeze(-1) * Bv.float().unsqueeze(2)) * x_.float().unsqueeze(-1)
        h = associative_scan(a, u)                             # (B,T,d_inner,N)
        y = (h * Cv.float().unsqueeze(2)).sum(-1)              # (B,T,d_inner)
        y = y + self.D.float() * x_.float()
        y = (y * F.silu(z.float())).to(residual.dtype)
        return residual + self.out_proj(y)


# --------------------------------------------------------------------------
# Looped (weight-shared) transformer block with binary linears
# --------------------------------------------------------------------------
class LoopedTransformerBlock(nn.Module):
    """A single transformer block applied `loops` times with shared weights.
    Each application is one 'internal reasoning' iteration. All of its linear
    layers are BinaryLinear (1-bit forward weights, STE training)."""

    def __init__(self, d_model, n_head=8, mlp_ratio=4, loops=3):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.loops = loops
        self.norm1 = nn.RMSNorm(d_model)
        self.qkv = BinaryLinear(d_model, 3 * d_model, bias=False)
        self.attn_out = BinaryLinear(d_model, d_model, bias=False)
        self.norm2 = nn.RMSNorm(d_model)
        self.fc1 = BinaryLinear(d_model, mlp_ratio * d_model, bias=False)
        self.fc2 = BinaryLinear(mlp_ratio * d_model, d_model, bias=False)

    def _apply_once(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).reshape(B, T, D)
        x = x + self.attn_out(att)
        h = self.norm2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x

    def forward(self, x):
        for _ in range(self.loops):
            x = self._apply_once(x)
        return x


# --------------------------------------------------------------------------
# Full hybrid LM
# --------------------------------------------------------------------------
class BinaryLoopHybridLM(nn.Module):
    def __init__(self, vocab_size=256, d_model=256, seq_len=256,
                 n_ssm=3, n_loop_blocks=3, loops=3, n_head=8,
                 d_inner=384, d_state=8):
        super().__init__()
        self.seq_len = seq_len
        self.loops = loops
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        blocks = []
        for i in range(max(n_ssm, n_loop_blocks)):
            if i < n_ssm:
                blocks.append(SSMBlock(d_model, d_inner=d_inner,
                                       d_state=d_state))
            if i < n_loop_blocks:
                blocks.append(LoopedTransformerBlock(
                    d_model, n_head=n_head, loops=loops))
        self.blocks = nn.ModuleList(blocks)
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # tied embeddings

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos).unsqueeze(0)
        for blk in self.blocks:
            if isinstance(blk, SSMBlock):
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


def count_params(model):
    return sum(p.numel() for p in model.parameters())
