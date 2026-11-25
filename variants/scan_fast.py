"""Fast selective scan for the binary looped Mamba (training).

The original scan (model.py `associative_scan`) materialises the full
(B, T, D, N) a/u/h tensors, doubles them with torch.cat, and - via
gradient checkpointing around the block - recomputes the whole forward
in the backward pass. That is why the mamba variant trained at ~1.06k
tok/s. This module keeps the maths identical in exact arithmetic (same
parameterisation; only floating-point summation order differs) while
cutting the memory traffic:

* The scan runs in chunks along T. Only chunk-boundary states are
  stored for the backward; within-chunk states are recomputed in the
  backward pass. That removes the need for blanket gradient
  checkpointing of the block.
* Two chunk kernels, both exact for the same recurrence
  h_t = a_t * h_{t-1} + u_t with a = exp(dt * A), u = dt * x * B:
  - chunkwise cumsum form (fast path): with g = dt * A (<= 0), G the
    global cumsum of g and m = G at the chunk end,
        P_t = exp(m - G_t),  Shat = cumsum(P * u),
        Q = exp(m - G_prev), R_t = exp(G_t - m),
        h_t = R_t * (Shat_t + Q * h_prev).
    Every exponential formed is <= 1, and its hand-derived exact
    adjoint pairs each exp(+decay) intermediate with the matching
    exp(-decay) factor, so it stays well-conditioned. It is safe
    only while the chunk decay (G_start - G_end) stays below ~89,
    where the backward's exp(+decay) intermediates overflow fp32;
    DECAY_LIMIT = 60 keeps a wide margin.
  - prefix-doubling form (fallback): the original doubling
    recurrence restricted to one chunk, with the boundary state
    folded in via cumprod(a). Unconditionally stable for any decay;
    its backward recomputes the chunk forward under autograd.
* Branching is PER CHANNEL-PAIR (d, n), not per chunk: a pair is
  "hot" for a chunk when its decay (max over the batch) exceeds
  DECAY_LIMIT. In the trained model only ~4% of pairs are hot at
  chunk 64 (a handful of channels learn large dt * |A|); a per-chunk
  branch would let those few channels force the whole chunk onto
  the slow kernel. Cold pairs in a chunk take the cumsum kernel,
  hot pairs the doubling kernel, via flattened-channel gathers;
  uniformly cold/hot chunks skip the gather entirely. Branch
  decisions are recomputed in the backward from the saved boundary
  cumsums, so forward and backward always agree.
* BLH_SCAN_IMPL=doubling forces the doubling kernel everywhere;
  anything else uses the cumsum form with the doubling fallback.
  The original slow path in model.py / model2.py stays untouched
  and remains the default when BLH_FAST_SCAN is unset.
* Branch statistics: with BLH_SCAN_STATS_FILE=<path> set, cumulative
  counts (calls, split calls, hot-pair fraction) are written there
  every 21 calls.
"""

import os

import torch

DECAY_LIMIT = 60.0
SCAN_STATS = {"calls": 0, "split": 0, "hot_pairs": 0, "total_pairs": 0}


def _note_stats(split, hot, total):
    SCAN_STATS["calls"] += 1
    SCAN_STATS["split"] += int(split)
    SCAN_STATS["hot_pairs"] += int(hot)
    SCAN_STATS["total_pairs"] += int(total)
    path = os.environ.get("BLH_SCAN_STATS_FILE")
    if path and SCAN_STATS["calls"] % 21 == 0:
        try:
            with open(path, "w") as f:
                f.write(str(SCAN_STATS))
        except OSError:
            pass


def _doubling(a, u):
    """Original parallel-scan doubling: (a, u) -> h along dim 1.
    Works for (B, C, D, N) and flattened (B, C, S) layouts."""
    A, U = a, u
    T = a.size(1)
    shift = 1
    while shift < T:
        A_new = torch.cat([A[:, :shift], A[:, shift:] * A[:, :-shift]], dim=1)
        U_new = torch.cat([U[:, :shift],
                           A[:, shift:] * U[:, :-shift] + U[:, shift:]], dim=1)
        A, U = A_new, U_new
        shift *= 2
    return U


def _revcumsum(v, dim=1):
    return torch.cumsum(v.flip(dim), dim=dim).flip(dim)


# ---------------------------------------------------------------------
# Grid-layout chunk kernels: tensors (B, C, D, N), A (D, N), dt (B, C, D)
# ---------------------------------------------------------------------

def _grid_cumsum_fwd(G_c, m, Gprev, dtc, Bc, xc, h_prev):
    u = (dtc * xc).unsqueeze(-1) * Bc.unsqueeze(2)
    P = torch.exp(m.unsqueeze(1) - G_c)
    Shat = torch.cumsum(P * u, dim=1)
    Q = torch.exp(m - Gprev)
    R = torch.exp(G_c - m.unsqueeze(1))
    h = R * (Shat + Q.unsqueeze(1) * h_prev.unsqueeze(1))
    return h, h[:, -1]


def _grid_doubling_fwd(g_c, dtc, Bc, xc, h_prev):
    a_c = torch.exp(g_c)
    u_c = (dtc * xc).unsqueeze(-1) * Bc.unsqueeze(2)
    cp = torch.cumprod(a_c, dim=1)
    h = _doubling(a_c, u_c) + cp * h_prev.unsqueeze(1)
    return h, h[:, -1]


# ---------------------------------------------------------------------
# Flat-layout chunk kernels: channels flattened to S pairs.
# dt_S/Bv_S/Cv_S/xs_S: (B, C, S); A_S: (S,); G_S: (B, C, S);
# boundary states: (B, S).
# ---------------------------------------------------------------------

def _flat_cumsum_fwd(G_c, m, Gprev, u_c, h_prev):
    P = torch.exp(m.unsqueeze(1) - G_c)
    Shat = torch.cumsum(P * u_c, dim=1)
    Q = torch.exp(m - Gprev)
    R = torch.exp(G_c - m.unsqueeze(1))
    h = R * (Shat + Q.unsqueeze(1) * h_prev.unsqueeze(1))
    return h, h[:, -1]


def _flat_doubling_fwd(g_c, u_c, h_prev):
    a_c = torch.exp(g_c)
    cp = torch.cumprod(a_c, dim=1)
    h = _doubling(a_c, u_c) + cp * h_prev.unsqueeze(1)
    return h, h[:, -1]


def _flat_cumsum_bwd(G_c, m, Gprev, u_c, h_prev, dh, psi_G):
    """Exact adjoint of _flat_cumsum_fwd. dh: (B, C, S) adjoint wrt h
    (already includes the boundary psi_h at the last position);
    psi_G: (B, S) incoming adjoint wrt m = G at this chunk's end.
    Returns (du, dg, dh_prev, dGprev, h): grads wrt u and g, the
    adjoints wrt the incoming boundary state and boundary cumsum,
    and the recomputed h (for the Cv gradient)."""
    P = torch.exp(m.unsqueeze(1) - G_c)
    Shat = torch.cumsum(P * u_c, dim=1)
    Q = torch.exp(m - Gprev)
    R = torch.exp(G_c - m.unsqueeze(1))
    inner = Shat + Q.unsqueeze(1) * h_prev.unsqueeze(1)
    h = R * inner
    dShat = dh * R
    dR = dh * inner
    dhR = (dh * R).sum(dim=1)
    dQ = dhR * h_prev
    dh_prev = Q * dhR
    dG = dR * R
    dm = -(dR * R).sum(dim=1) + psi_G
    dPu = _revcumsum(dShat)
    dP = dPu * u_c
    du = dPu * P
    dG = dG - dP * P
    dm = dm + (dP * P).sum(dim=1)
    dm = dm + dQ * Q
    dG[:, -1] += dm
    dGprev = -dQ * Q + dG.sum(dim=1)
    dg = _revcumsum(dG)
    return du, dg, dh_prev, dGprev, h


class FusedChunkwiseScan(torch.autograd.Function):
    """Chunked selective scan with per-channel-pair branching;
    see module docstring."""

    @staticmethod
    def _hot_pairs(Gprev, m, force_dbl):
        """(D, N) bool: pairs whose chunk decay (max over batch)
        exceeds DECAY_LIMIT."""
        if force_dbl:
            return torch.ones_like(Gprev[0], dtype=torch.bool)
        return ((Gprev - m) > DECAY_LIMIT).any(dim=0)

    @staticmethod
    def forward(ctx, dt, Bv, Cv, xs, A, chunk):
        force_dbl = os.environ.get("BLH_SCAN_IMPL") == "doubling"
        B, T, D = dt.shape
        N = A.size(1)
        g = dt.unsqueeze(-1) * A
        G = torch.cumsum(g, dim=1)
        nch = (T + chunk - 1) // chunk
        bnd_G = [dt.new_zeros(B, D, N)]
        for ci in range(nch):
            bnd_G.append(G[:, min((ci + 1) * chunk, T) - 1])
        bounds_h = [dt.new_zeros(B, D, N)]
        h_prev = bounds_h[0]
        ys = []
        for ci in range(nch):
            s = ci * chunk
            e = min(s + chunk, T)
            dtc = dt[:, s:e]
            Bc = Bv[:, s:e]
            Cc = Cv[:, s:e]
            xc = xs[:, s:e]
            G_c = G[:, s:e]
            g_c = g[:, s:e]
            m = bnd_G[ci + 1]
            Gprev = bnd_G[ci]
            hot = FusedChunkwiseScan._hot_pairs(Gprev, m, force_dbl)
            n_hot = int(hot.sum().item())
            _note_stats(0 < n_hot < D * N, n_hot, D * N)
            if n_hot == 0:
                h, h_last = _grid_cumsum_fwd(G_c, m, Gprev, dtc, Bc, xc,
                                             h_prev)
            elif n_hot == D * N:
                h, h_last = _grid_doubling_fwd(g_c, dtc, Bc, xc, h_prev)
            else:
                hot_idx = hot.flatten().nonzero(as_tuple=True)[0]
                cold_idx = (~hot.flatten()).nonzero(as_tuple=True)[0]
                h_flat = dt.new_zeros(B, e - s, D * N)
                for idx, is_hot in ((cold_idx, False), (hot_idx, True)):
                    d_idx = idx // N
                    n_idx = idx % N
                    dt_S = dtc.index_select(2, d_idx)
                    xs_S = xc.index_select(2, d_idx)
                    Bv_S = Bc.index_select(2, n_idx)
                    u_S = (dt_S * xs_S) * Bv_S
                    G_S = G_c.flatten(2).index_select(2, idx)
                    m_S = m.flatten(1).index_select(1, idx)
                    Gp_S = Gprev.flatten(1).index_select(1, idx)
                    hp_S = h_prev.flatten(1).index_select(1, idx)
                    if is_hot:
                        g_S = g_c.flatten(2).index_select(2, idx)
                        h_S, _ = _flat_doubling_fwd(g_S, u_S, hp_S)
                    else:
                        h_S, _ = _flat_cumsum_fwd(G_S, m_S, Gp_S, u_S, hp_S)
                    h_flat.index_copy_(2, idx, h_S)
                h = h_flat.view(B, e - s, D, N)
                h_last = h[:, -1]
            ys.append((h * Cc.unsqueeze(2)).sum(-1))
            h_prev = h_last
            bounds_h.append(h_prev)
        ctx.save_for_backward(dt, Bv, Cv, xs, A,
                              torch.stack(bounds_h, dim=1),
                              torch.stack(bnd_G, dim=1))
        ctx.chunk = chunk
        ctx.force_dbl = force_dbl
        return torch.cat(ys, dim=1)

    @staticmethod
    def backward(ctx, dy):
        dt, Bv, Cv, xs, A, bounds_h, bounds_G = ctx.saved_tensors
        chunk = ctx.chunk
        B, T, D = dt.shape
        N = A.size(1)
        DN = D * N
        g = dt.unsqueeze(-1) * A
        G = torch.cumsum(g, dim=1)
        d_dt = torch.zeros_like(dt)
        d_xs = torch.zeros_like(xs)
        d_Bv = torch.zeros_like(Bv)
        d_Cv = torch.zeros_like(Cv)
        d_A = torch.zeros_like(A)
        psi_h = dt.new_zeros(B, DN)   # adjoint wrt chunk-end h, flat
        psi_G = dt.new_zeros(B, DN)   # adjoint wrt chunk-end m, flat
        for ci in range(bounds_h.size(1) - 2, -1, -1):
            s = ci * chunk
            e = min(s + chunk, T)
            dtc = dt[:, s:e]
            Bc = Bv[:, s:e]
            Cc = Cv[:, s:e]
            xc = xs[:, s:e]
            h_prev = bounds_h[:, ci]
            Gprev = bounds_G[:, ci]
            G_c = G[:, s:e]
            g_c = g[:, s:e]
            m = G_c[:, -1]
            hot = FusedChunkwiseScan._hot_pairs(Gprev, m, ctx.force_dbl)
            hot_idx = hot.flatten().nonzero(as_tuple=True)[0]
            cold_idx = (~hot.flatten()).nonzero(as_tuple=True)[0]
            # flat views of this chunk's scan tensors
            Gf = G_c.flatten(2)
            gf = g_c.flatten(2)
            uf = ((dtc * xc).unsqueeze(-1) *
                  Bc.unsqueeze(2)).flatten(2)
            hp_f = h_prev.flatten(1)
            m_f = m.flatten(1)
            Gp_f = Gprev.flatten(1)
            dh_f = (dy[:, s:e].unsqueeze(-1) *
                    Cc.unsqueeze(2)).flatten(2)
            dh_f[:, -1] += psi_h
            du_f = torch.zeros_like(uf)
            dg_f = torch.zeros_like(gf)
            dh_prev_f = torch.zeros_like(hp_f)
            dGprev_f = torch.zeros_like(Gp_f)
            h_flat = torch.zeros_like(uf)
            if cold_idx.numel() > 0:
                du_c, dg_c, dhp_c, dGp_c, h_c = _flat_cumsum_bwd(
                    Gf.index_select(2, cold_idx),
                    m_f.index_select(1, cold_idx),
                    Gp_f.index_select(1, cold_idx),
                    uf.index_select(2, cold_idx),
                    hp_f.index_select(1, cold_idx),
                    dh_f.index_select(2, cold_idx),
                    psi_G.index_select(1, cold_idx))
                du_f.index_copy_(2, cold_idx, du_c)
                dg_f.index_copy_(2, cold_idx, dg_c)
                dh_prev_f.index_copy_(1, cold_idx, dhp_c)
                dGprev_f.index_copy_(1, cold_idx, dGp_c)
                h_flat.index_copy_(2, cold_idx, h_c)
            if hot_idx.numel() > 0:
                g_h = gf.index_select(2, hot_idx)
                u_h = uf.index_select(2, hot_idx)
                hp_h = hp_f.index_select(1, hot_idx)
                with torch.enable_grad():
                    g_l = g_h.detach().clone().requires_grad_(True)
                    u_l = u_h.detach().clone().requires_grad_(True)
                    hp_l = hp_h.detach().clone().requires_grad_(True)
                    h_h, _ = _flat_doubling_fwd(g_l, u_l, hp_l)
                    torch.autograd.backward(
                        [h_h], [dh_f.index_select(2, hot_idx)])
                du_f.index_copy_(2, hot_idx, u_l.grad)
                dg_h = g_l.grad + psi_G.index_select(1, hot_idx).unsqueeze(1)
                dg_f.index_copy_(2, hot_idx, dg_h)
                dh_prev_f.index_copy_(1, hot_idx, hp_l.grad)
                dGprev_f.index_copy_(1, hot_idx,
                                    psi_G.index_select(1, hot_idx))
                h_flat.index_copy_(2, hot_idx, h_h.detach())
            h = h_flat.view(B, e - s, D, N)
            d_Cv[:, s:e] = (dy[:, s:e].unsqueeze(-1) * h).sum(2)
            du_g = du_f.view(B, e - s, D, N)
            dg_g = dg_f.view(B, e - s, D, N)
            w = (du_g * Bc.unsqueeze(2)).sum(-1)
            d_dt[:, s:e] = w * xc + (dg_g * A).sum(-1)
            d_xs[:, s:e] = w * dtc
            d_Bv[:, s:e] = (du_g * (dtc * xc).unsqueeze(-1)).sum(2)
            d_A += (dg_g * dtc.unsqueeze(-1)).sum(dim=(0, 1))
            psi_h = dh_prev_f
            psi_G = dGprev_f
        return d_dt, d_Bv, d_Cv, d_xs, d_A, None


# With BLH_SCAN_IMPL=doubling every pair is "hot", i.e. the v1
# algorithm (chunked doubling, boundary states stored, within-chunk
# recompute in the backward).
FusedSelectiveScan = FusedChunkwiseScan


def fused_selective_scan(dt, Bv, Cv, xs, A, chunk=32):
    """fp32 in, fp32 out; see FusedChunkwiseScan."""
    return FusedChunkwiseScan.apply(
        dt.float(), Bv.float(), Cv.float(), xs.float(), A.float(), chunk)
