"""Inversion model v3 (plan Phase 4): the set-valued, marker-permutation-invariant network that
recovers theta v3 (flexible panels + lineage cell types) from a multiplexed tile.

Why it looks like this (short form; see docs/INVERSION_V3_PLAN.md Appendix D for the long form):

- The panel is a SET of markers whose order and count both vary per sample and carry no meaning,
  so the only channel-identity signal is a fresh random tag per marker (sec 4.2). Everything that
  crosses markers is permutation-equivariant (masked DeepSets mixing, attention with no marker
  positional encoding), and padded markers are masked everywhere, so adding or reordering channels
  cannot move a global or type output.
- Absolute brightness is a target (expression level), so the stem carries NO normalisation layers
  that would erase it; the input is log-scaled instead (sec 4.1).
- Cell types are an unordered set of unknown size, so they are recovered as SLOTS (slot attention,
  sec 4.8) seeded by farthest-point sampling in a learned embedding, matched to the truth by the
  Phase 5 loss rather than by a fixed index.

The forward returns a dict of posteriors and dense maps; the losses, matching and training loop
are Phase 5/6. This module is selected by `model.arch = "v3"` and leaves v2 untouched.

Shape tags use B batch, M markers, H/W pixels, D token dim, N4 = (H/4)(W/4) stride-4 locations,
K type slots (+1 background). H and W must be divisible by 32.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

import config as cfg

LOGVAR_MIN, LOGVAR_MAX = cfg.LOGVAR_MIN, cfg.LOGVAR_MAX


def _clamp_logvar(raw):
    """Squash a raw logvar head into [LOGVAR_MIN, LOGVAR_MAX] the same way v2 does, so the
    posterior sigma can never collapse to 0 or explode."""
    mid = (LOGVAR_MAX + LOGVAR_MIN) / 2
    half = (LOGVAR_MAX - LOGVAR_MIN) / 2
    return mid + half * torch.tanh(raw / half)


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return self.net(x)


def sincos_2d(h, w, d, device=None):
    """Standard 2D sinusoidal positional encoding, [h*w, d]. Half the channels encode the row, half
    the column, so a token knows where it sits without any learned grid-size-specific parameter."""
    assert d % 4 == 0, "2D sincos needs d divisible by 4"
    dq = d // 2
    omega = torch.arange(dq // 2, device=device) / (dq // 2)
    omega = 1.0 / (10000 ** omega)
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                            indexing="ij")
    out = []
    for p in (yy.flatten(), xx.flatten()):
        a = p[:, None].float() * omega[None]
        out.append(torch.cat([a.sin(), a.cos()], dim=1))
    return torch.cat(out, dim=1)                                    # [h*w, d]


# --- A.1 pixel-level cross-marker mixing ------------------------------------------------------
class PixelMix(nn.Module):
    """DeepSets mixing across markers at one resolution. The tag enters as a per-marker bias, then
    a masked mean and max over the present markers form a coexpression code broadcast back to every
    marker. Permutation-invariant over markers; padded markers are masked out of both pools."""

    def __init__(self, c, c_mix, d_tag=32):
        super().__init__()
        self.inp = nn.Conv2d(c, c_mix, 1)
        self.tag = nn.Linear(d_tag, c_mix, bias=False)
        self.out = nn.Conv2d(2 * c_mix, c, 1)

    def forward(self, h, tags, present):
        B, M, c, H, W = h.shape
        a = self.inp(h.flatten(0, 1)).view(B, M, -1, H, W)
        a = F.gelu(a + self.tag(tags)[..., None, None])
        m = present[:, :, None, None, None].float()
        mean = (a * m).sum(1) / m.sum(1).clamp(min=1)
        mx = a.masked_fill(m == 0, float("-inf")).amax(1)
        s = torch.cat([mean, mx], dim=1)                           # [B, 2*c_mix, H, W]
        h = (h + self.out(s)[:, None]) * m
        return h, s


# --- A.2 axial block: Swin window attention per marker, then marker attention per location -----
class WindowAttention(nn.Module):
    """Multi-head self-attention within non-overlapping windows (Swin), with optional cyclic shift
    and the standard shifted-window attention mask. Weights are shared across whatever is folded
    into the leading axis (here: markers), so a marker's spatial reasoning is identical regardless
    of which marker it is."""

    def __init__(self, dim, heads, window):
        super().__init__()
        self.dim, self.heads, self.window = dim, heads, window
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def _mask(self, Hp, Wp, H, W, ws, s, device):
        """Additive attention mask [n_win, L, L]: -inf where a key is in a different shift region
        than the query, or is a PADDED cell (so the grid can be any size, not just a multiple of
        ws). None when there is neither a shift nor padding."""
        if s == 0 and Hp == H and Wp == W:
            return None
        nh, nw = Hp // ws, Wp // ws
        r = torch.zeros(1, Hp, Wp, 1, device=device)
        if s:
            cnt = 0
            for hsl in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                for wsl in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                    r[:, hsl, wsl, :] = cnt
                    cnt += 1
        valid = torch.zeros(1, Hp, Wp, 1, device=device)
        valid[:, :H, :W, :] = 1.0
        if s:
            r = torch.roll(r, (-s, -s), dims=(1, 2))
            valid = torch.roll(valid, (-s, -s), dims=(1, 2))

        def part(a):
            return a.view(1, nh, ws, nw, ws, 1).permute(0, 1, 3, 2, 4, 5).reshape(nh * nw, ws * ws)

        rw, vw = part(r), part(valid)
        allow = (rw[:, :, None] == rw[:, None, :]) & (vw[:, None, :] > 0.5)
        mask = torch.where(allow, torch.zeros_like(allow, dtype=torch.float32),
                           torch.full_like(allow, float("-inf"), dtype=torch.float32))
        dead = torch.isinf(mask).all(-1)                          # padded-query rows -> avoid NaN
        mask[dead] = 0.0
        return mask

    def forward(self, x, shift=False):
        n, H, W, C = x.shape
        ws = min(self.window, H, W)
        ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws           # pad grid up to a multiple of ws
        if ph or pw:
            x = F.pad(x, (0, 0, 0, pw, 0, ph))
        Hp, Wp = H + ph, W + pw
        s = ws // 2 if (shift and ws < Hp and ws < Wp) else 0
        if s:
            x = torch.roll(x, (-s, -s), dims=(1, 2))
        nh, nw = Hp // ws, Wp // ws
        xw = x.view(n, nh, ws, nw, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        hd = C // self.heads
        qkv = self.qkv(xw).reshape(-1, ws * ws, 3, self.heads, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                           # [B_, heads, L, hd]
        attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
        mask = self._mask(Hp, Wp, H, W, ws, s, x.device)
        if mask is not None:
            attn = attn.view(n, nh * nw, self.heads, ws * ws, ws * ws) + mask[None, :, None]
            attn = attn.view(-1, self.heads, ws * ws, ws * ws)
        o = (attn.softmax(-1) @ v).transpose(1, 2).reshape(-1, ws * ws, C)
        o = self.proj(o)
        o = o.view(n, nh, nw, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(n, Hp, Wp, C)
        if s:
            o = torch.roll(o, (s, s), dims=(1, 2))
        return o[:, :H, :W, :] if (ph or pw) else o


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, device=x.device, dtype=x.dtype)
        return x / keep * mask.floor()


class AxialBlock(nn.Module):
    """One block of stage D (Appendix A.2): a spatial step (Swin window attention within each
    marker, shared weights) and a marker step (self-attention over the present markers at each
    location, no marker positional encoding). Padded markers are key-masked and zeroed."""

    def __init__(self, d, heads, window, shift, mlp_ratio=4.0, drop_path=0.1):
        super().__init__()
        self.shift = shift
        self.n1 = nn.LayerNorm(d)
        self.win = WindowAttention(d, heads, window)
        self.n2 = nn.LayerNorm(d)
        self.mlp1 = MLP(d, int(d * mlp_ratio), d)
        self.n3 = nn.LayerNorm(d)
        self.mk = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n4 = nn.LayerNorm(d)
        self.mlp2 = MLP(d, int(d * mlp_ratio), d)
        self.drop = DropPath(drop_path)

    def forward(self, x, present):                                 # x: [B, M, Hg, Wg, D]
        B, M, Hg, Wg, D = x.shape
        y = x.reshape(B * M, Hg, Wg, D)
        y = y + self.drop(self.win(self.n1(y), shift=self.shift))
        y = y + self.drop(self.mlp1(self.n2(y)))
        x = y.view(B, M, Hg, Wg, D)
        z = x.permute(0, 2, 3, 1, 4).reshape(B * Hg * Wg, M, D)
        kpm = (~present)[:, None, None, :].expand(B, Hg, Wg, M).reshape(B * Hg * Wg, M)
        q = self.n3(z)
        a, _ = self.mk(q, q, q, key_padding_mask=kpm, need_weights=False)
        z = z + self.drop(a)
        z = z + self.drop(self.mlp2(self.n4(z)))
        x = z.view(B, Hg, Wg, M, D).permute(0, 3, 1, 2, 4)
        return x * present[:, :, None, None, None]


# --- stem (stages A + B) ----------------------------------------------------------------------
def _conv(c_in, c_out, k, stride=1):
    return nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2)


class Stem(nn.Module):
    """Per-marker convolutional stem with cross-marker PixelMix at strides 1, 2, 4 (no
    normalisation layers, so absolute brightness survives). Returns s1/s2/s4 coexpression codes,
    the stride-4 per-marker features h4 (= u, for the expression readout), and logI4, the
    4x4-pooled per-marker log-intensity."""

    def __init__(self, d_tag=32):
        super().__init__()
        # 2 input channels: the (per-marker normalised) intensity + the per-marker expression-strength
        # plane, so the stem sees each channel's relative brightness alongside its texture.
        self.c1 = nn.ModuleList([_conv(2, 16, 3), _conv(16, 16, 3), _conv(16, 16, 5)])
        self.mix1 = PixelMix(16, 16, d_tag)
        self.c2 = nn.ModuleList([_conv(16, 32, 3, 2), _conv(32, 32, 3)])
        self.mix2 = PixelMix(32, 32, d_tag)
        self.c4 = nn.ModuleList([_conv(32, 48, 3, 2), _conv(48, 48, 3)])
        self.mix4 = PixelMix(48, 48, d_tag)

    def _run_convs(self, convs, h):                               # h: [B, M, c, H, W]
        B, M = h.shape[:2]
        y = h.flatten(0, 1)
        for conv in convs:
            y = F.gelu(conv(y))
        return y.view(B, M, *y.shape[1:])

    def forward(self, tile, tags, present, strength):            # tile: [B, M, H, W]; strength: [B, M]
        logI4 = F.avg_pool2d(tile.flatten(0, 1), 4).view(*tile.shape[:2], tile.shape[2] // 4, -1)
        B, M, H, W = tile.shape
        sp = strength[:, :, None, None, None].expand(B, M, 1, H, W)   # strength broadcast as a plane
        h = torch.cat([tile[:, :, None], sp], dim=2)              # [B, M, 2, H, W]
        h = self._run_convs(self.c1, h)
        h, s1 = self.mix1(h, tags, present)
        h = self._run_convs(self.c2, h)
        h, s2 = self.mix2(h, tags, present)
        h = self._run_convs(self.c4, h)
        h, s4 = self.mix4(h, tags, present)
        return dict(s1=s1, s2=s2, s4=s4, h4=h, logI4=logI4)


# --- tokens (stage C) + axial trunk (stage D) -------------------------------------------------
class Trunk(nn.Module):
    def __init__(self, d=128, d_tag=32, n_blocks=4, heads=4, window=8):
        super().__init__()
        self.tokenize = nn.Conv2d(48, d, 4, stride=4)
        self.tag_proj = nn.Linear(d_tag, d)
        self.strength_proj = nn.Linear(1, d)                     # per-marker expression strength
        self.ln = nn.LayerNorm(d)
        self.blocks = nn.ModuleList(
            AxialBlock(d, heads, window, shift=(i % 2 == 1)) for i in range(n_blocks))
        self.d = d

    def forward(self, h4, tags, present, strength):              # h4: [B, M, 48, Hf, Wf]
        B, M, _, Hf, Wf = h4.shape
        t = self.tokenize(h4.flatten(0, 1))                      # [B*M, d, Hg, Wg]
        Hg, Wg = t.shape[-2:]
        t = t.view(B, M, self.d, Hg, Wg).permute(0, 1, 3, 4, 2)  # [B, M, Hg, Wg, d]
        t = (t + self.tag_proj(tags)[:, :, None, None, :]
             + self.strength_proj(strength[:, :, None])[:, :, None, None, :])
        pe = sincos_2d(Hg, Wg, self.d, t.device).view(1, 1, Hg, Wg, self.d)
        t = self.ln(t + pe)
        for blk in self.blocks:
            t = blk(t, present)
        return t                                                 # [B, M, Hg, Wg, d]


# --- fusion and tissue context (stage E) ------------------------------------------------------
class _GlobalBlock(nn.Module):
    def __init__(self, d, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = MLP(d, int(d * mlp_ratio), d)

    def forward(self, x):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class Fusion(nn.Module):
    """Stage E: attention-pool the present markers at each stride-16 location into g16, merge 2x2
    to g32, run global tissue attention, and fold the tissue context back into g16'."""

    def __init__(self, d=128, g16=192, g32=256, heads=4, tissue_heads=8, n_tissue=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.to_g16 = nn.Linear(d, g16)
        self.merge = nn.Linear(4 * g16, g32)
        self.tissue = nn.ModuleList(_GlobalBlock(g32, tissue_heads) for _ in range(n_tissue))
        self.ctx_back = nn.Linear(g32, g16)
        self.g16, self.g32 = g16, g32

    def forward(self, t, present):                               # t: [B, M, Hg, Wg, d]
        B, M, Hg, Wg, d = t.shape
        z = t.permute(0, 2, 3, 1, 4).reshape(B * Hg * Wg, M, d)
        kpm = (~present)[:, None, None, :].expand(B, Hg, Wg, M).reshape(B * Hg * Wg, M)
        q = self.query.expand(B * Hg * Wg, -1, -1)
        pooled, _ = self.pool(q, z, z, key_padding_mask=kpm, need_weights=False)
        g16 = self.to_g16(pooled.squeeze(1)).view(B, Hg, Wg, self.g16)

        merged = g16.view(B, Hg // 2, 2, Wg // 2, 2, self.g16).permute(0, 1, 3, 2, 4, 5)
        merged = merged.reshape(B, (Hg // 2) * (Wg // 2), 4 * self.g16)
        g32 = self.merge(merged)
        g32 = g32 + sincos_2d(Hg // 2, Wg // 2, self.g32, g32.device)[None]
        for blk in self.tissue:
            g32 = blk(g32)

        up = self.ctx_back(g32).view(B, Hg // 2, Wg // 2, self.g16).permute(0, 3, 1, 2)
        up = F.interpolate(up, scale_factor=2, mode="bilinear", align_corners=False)
        g16p = g16 + up.permute(0, 2, 3, 1)
        return g16p, g32                                         # [B,Hg,Wg,g16], [B,Ng32,g32]


# --- dense decoder (stage F) ------------------------------------------------------------------
def _up(c_in, c_out):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                         _conv(c_in, c_out, 3), nn.GELU())


class DenseDecoder(nn.Module):
    """Stage F: from g16' (+ the single s4 skip) decode nucleus/cell/flow at full resolution, the
    structure logits at stride 16, the stride-4 context features ctx4, and dec4 (the stride-4
    decoder features reused by the geometry head). s4 is the ONLY skip, so the decoder cannot
    bypass the trunk."""

    def __init__(self, g16=192, n_context=3):
        super().__init__()
        self.up8 = _up(g16, 128)
        self.up4 = _up(128, 96)
        self.fuse4 = nn.Sequential(_conv(96 + 96, 96, 3), nn.GELU())
        self.up2 = _up(96, 64)
        self.up1 = _up(64, 32)
        self.nucleus = _conv(32, 1, 1)
        self.cell = _conv(32, 1, 1)
        self.flow = _conv(32, 2, 1)
        self.nuc_flow = _conv(32, 2, 1)             # Omnipose flows for nuclei (separate touching ones)
        self.structure = _conv(g16, n_context, 1)
        self.ctx4 = _conv(g16, 64, 1)

    def forward(self, g16p, s4):                                 # g16p: [B,Hg,Wg,g16]
        x = g16p.permute(0, 3, 1, 2)                             # [B, g16, Hg, Wg] (stride 16)
        d8 = self.up8(x)
        d4 = self.up4(d8)
        dec4 = self.fuse4(torch.cat([d4, s4], dim=1))           # [B, 96, H/4, W/4]
        d2 = self.up2(dec4)
        d1 = self.up1(d2)
        out = dict(nucleus=self.nucleus(d1), cell=self.cell(d1), flow=torch.tanh(self.flow(d1)),
                   nuc_flow=torch.tanh(self.nuc_flow(d1)),
                   structure=self.structure(x), dec4=dec4,
                   ctx4=F.interpolate(self.ctx4(x), scale_factor=4, mode="bilinear",
                                      align_corners=False))
        return out


# --- A.3 seeds --------------------------------------------------------------------------------
@torch.no_grad()
def select_seeds(emb, cand, K, init=()):
    """Farthest-point sampling in embedding space over the candidate stride-4 locations, started
    from the most typical candidate (or from the given teacher `init` seeds). Returns up to K
    indices into N."""
    chosen = list(init)
    idx = cand.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return chosen[:K]
    E = emb[idx]
    if chosen:
        d = torch.cdist(E, emb[torch.as_tensor(chosen, device=emb.device)]).min(1).values
    else:
        j0 = int(((E - E.mean(0)) ** 2).sum(1).argmin())
        chosen.append(int(idx[j0]))
        d = (E - E[j0]).norm(dim=1)
    while len(chosen) < K and float(d.max()) > 0:
        j = int(d.argmax())
        chosen.append(int(idx[j]))
        d = torch.minimum(d, (E - E[j]).norm(dim=1))
    return chosen[:K]


def teacher_seeds(type4_pure, present_types, gen):
    """One teacher seed per present type at a random pure stride-4 location of that type (Phase 5
    uses this with probability p_teacher). type4_pure: [N] type index at pure locations, -1 else."""
    seeds = []
    for j in present_types:
        loc = (type4_pure == j).nonzero(as_tuple=True)[0]
        if loc.numel():
            seeds.append(int(loc[torch.randint(loc.numel(), (1,), generator=gen)]))
    return seeds


def _smooth4(emb, Hf, Wf):
    """3x3 mean filter over the stride-4 embedding grid (A.3), kept as [N, d]."""
    B = 1
    d = emb.shape[-1]
    g = emb.view(1, Hf, Wf, d).permute(0, 3, 1, 2)
    g = F.avg_pool2d(g, 3, stride=1, padding=1)
    return g.permute(0, 2, 3, 1).reshape(Hf * Wf, d)


# --- A.4 slot attention -----------------------------------------------------------------------
class TypeSlots(nn.Module):
    def __init__(self, d=128, K=30, T=3, heads=4):
        super().__init__()
        self.K, self.T, self.d = K, T, d
        self.norm_in, self.norm_s = nn.LayerNorm(d), nn.LayerNorm(d)
        self.to_k, self.to_v, self.to_q = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.gru = nn.GRUCell(d, d)
        self.norm_m, self.mlp = nn.LayerNorm(d), MLP(d, 2 * d, d)
        self.norm_sa = nn.LayerNorm(d)
        self.slot_sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.seed_mlp = MLP(d, d, d)
        self.bg = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.extra = nn.Parameter(torch.randn(1, K, d) * 0.02)

    def attn(self, slots, k):
        q = self.to_q(self.norm_s(slots))
        return (torch.einsum("bkd,bnd->bkn", q, k) / self.d ** 0.5).softmax(dim=1) + 1e-8

    def forward(self, F4, seeds, n_seeds):                       # F4: [B, N, d]; seeds: [B, K]
        B, N, d = F4.shape
        x = self.norm_in(F4)
        k, v = self.to_k(x), self.to_v(x)
        init = self.seed_mlp(torch.gather(F4, 1, seeds[..., None].expand(-1, -1, d)))
        valid = torch.arange(self.K, device=F4.device)[None] < n_seeds[:, None]
        slots = torch.where(valid[..., None], init, self.extra.expand(B, -1, -1))
        slots = torch.cat([slots, self.bg.expand(B, 1, d)], dim=1)   # [B, K+1, d]
        A, states, maps = self.attn(slots, k), [], []
        for _ in range(self.T):
            W = A / A.sum(-1, keepdim=True)
            upd = torch.einsum("bkn,bnd->bkd", W, v)
            slots = self.gru(upd.reshape(-1, d), slots.reshape(-1, d)).view(B, -1, d)
            slots = slots + self.mlp(self.norm_m(slots))
            s = self.norm_sa(slots)
            slots = slots + self.slot_sa(s, s, s, need_weights=False)[0]
            A = self.attn(slots, k)
            states.append(slots)
            maps.append(A)
        return states, maps


# --- A.5 slot x marker readout ----------------------------------------------------------------
def slot_marker_readout(A, u, logI, dec4):
    """A: [B, K+1, N]  u: [B, M, c, N]  logI: [B, M, N]  dec4: [B, c_dec, N]."""
    Ak = A[:, :-1]
    W = Ak / Ak.sum(-1, keepdim=True).clamp(min=1e-6)
    u_bar = torch.einsum("bkn,bmcn->bkmc", W, u)
    i_bar = torch.einsum("bkn,bmn->bkm", W, logI)
    g_bar = torch.einsum("bkn,bcn->bkc", W, dec4)
    return u_bar, i_bar, g_bar


# --- heads ------------------------------------------------------------------------------------
class MarkerHead(nn.Module):
    """Per-marker head (4.6). Two attention pools per marker, both learned to WEIGHT EVIDENCE BY
    RELIABILITY rather than averaging every cell equally (a fixed average lets PSF-washed small cells
    dilute the crisp signal from large ones):
      - over the stride-16 trunk tokens (processed texture / context), and
      - over stride-4 LOCATION tokens that each carry their sub-cellular region membership
        (nucleus / cytoplasm / flow-derived membrane) + intensity + stem features, so the query can
        focus on the informative, well-resolved cells.
    Reads out the noise posterior, the localisation `mu` posterior (a REGRESSION -- mean + logvar in
    the normalised-logit space, replacing the old collapsing 3-way classifier) and the channel
    offset. Equivariant: output m depends only on channel m; padded markers are unused downstream."""

    def __init__(self, d=128, n_noise=4, loc_in=52, n_queries=4, predict_offset=True):
        super().__init__()
        self.q_tok = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool_tok = nn.MultiheadAttention(d, 4, batch_first=True)
        self.loc_proj = nn.Linear(loc_in, d)
        self.ln_loc = nn.LayerNorm(d)
        self.q_loc = nn.Parameter(torch.randn(1, n_queries, d) * 0.02)
        self.pool_loc = nn.MultiheadAttention(d, 4, batch_first=True)
        self.n_noise, self.n_queries = n_noise, n_queries
        self.predict_offset = predict_offset
        out = n_noise + 1 + (2 if predict_offset else 0)        # noise + mu + offset (point estimates)
        self.mlp = MLP(d + n_queries * d, 256, out)

    def forward(self, t, loc_feat):                             # t:[B,M,Hg,Wg,d] loc:[B,M,N4,loc_in]
        B, M, Hg, Wg, d = t.shape
        zt = t.reshape(B * M, Hg * Wg, d)
        p_tok = self.pool_tok(self.q_tok.expand(B * M, -1, -1), zt, zt, need_weights=False)[0]
        zl = self.ln_loc(self.loc_proj(loc_feat.reshape(B * M, loc_feat.shape[2], -1)))
        p_loc = self.pool_loc(self.q_loc.expand(B * M, -1, -1), zl, zl, need_weights=False)[0]
        o = self.mlp(torch.cat([p_tok.squeeze(1), p_loc.reshape(B * M, -1)], dim=-1)).view(B, M, -1)
        out = dict(noise_mu=o[..., :self.n_noise], mu=o[..., self.n_noise])
        off = self.n_noise + 1
        if self.predict_offset:
            out["offset"] = o[..., off:off + 2]
        return out


class GlobalHead(nn.Module):
    """Global head (4.10): learned queries cross-attend the g32 tissue tokens through 2 decoder
    layers, then read out a POINT ESTIMATE per (padded-to-n_global) global theta entry. (Learned
    variance removed -- a heteroscedastic NLL let the heads under-fit the mean; uncertainty is
    deferred to a later calibration pass.)"""

    def __init__(self, g32=256, n_global=16, n_queries=8, heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, n_queries, g32) * 0.02)
        self.layers = nn.ModuleList(nn.ModuleDict(dict(
            n1=nn.LayerNorm(g32), ca=nn.MultiheadAttention(g32, heads, batch_first=True),
            n2=nn.LayerNorm(g32), sa=nn.MultiheadAttention(g32, heads, batch_first=True),
            n3=nn.LayerNorm(g32), mlp=MLP(g32, 4 * g32, g32))) for _ in range(2))
        self.head = MLP(n_queries * g32, g32, n_global)
        self.n_global = n_global

    def forward(self, g32):
        q = self.q.expand(g32.shape[0], -1, -1)
        for L in self.layers:
            h = L["n1"](q)
            q = q + L["ca"](h, g32, g32, need_weights=False)[0]
            h = L["n2"](q)
            q = q + L["sa"](h, h, h, need_weights=False)[0]
            q = q + L["mlp"](L["n3"](q))
        return self.head(q.flatten(1)).view(-1, self.n_global)


class SlotHeads(nn.Module):
    """Per-slot heads (4.9), applied at every slot-attention iteration for deep supervision:
    existence, per-marker expression, geometry, per-context composition alpha and the token
    composition map. All equivariant in slot k and marker m, so they work for any K, M."""

    def __init__(self, d=128, u_c=48, dec_c=96, n_geom=5, n_context=3, ctx_emb=16):
        super().__init__()
        self.exist = MLP(d, d, 1)
        self.expr = MLP(d + u_c + 1, d, 2)                       # [slot; u_bar; i_bar] -> pos, level
        self.geom = MLP(d + dec_c, d, n_geom)                    # point estimate (no variance)
        self.ctx_emb = nn.Embedding(n_context, ctx_emb)
        self.alpha = MLP(d + ctx_emb + 2, d, 1)
        self.n_geom, self.n_context = n_geom, n_context

    def forward(self, slots, maps, u, logI, dec4, P4):
        # slots: [B, K+1, d]; maps: [B, K+1, N]; P4: [B, C, N]
        type_slots = slots[:, :-1]                              # drop background for per-type heads
        B, K, d = type_slots.shape
        u_bar, i_bar, g_bar = slot_marker_readout(maps, u, logI, dec4)
        M = u_bar.shape[2]

        exist = self.exist(type_slots).squeeze(-1)             # [B, K]

        se = type_slots[:, :, None].expand(-1, -1, M, -1)       # [B, K, M, d]
        expr_in = torch.cat([se, u_bar, i_bar[..., None]], dim=-1)
        e = self.expr(expr_in)                                 # [B, K, M, 2]
        expr = dict(positive=e[..., 0], level_mu=e[..., 1])

        geom = dict(mu=self.geom(torch.cat([type_slots, g_bar], dim=-1)).view(B, K, self.n_geom))

        Ak = maps[:, :-1]                                       # [B, K, N]
        n = torch.einsum("bkn,bcn->bkc", Ak, P4)               # soft pixel counts [B, K, C]
        C = self.n_context
        ce = self.ctx_emb(torch.arange(C, device=slots.device))[None, None].expand(B, K, -1, -1)
        sc = type_slots[:, :, None].expand(-1, -1, C, -1)
        n_tot = n.sum(1, keepdim=True).expand(-1, K, -1)
        a_in = torch.cat([sc, ce, torch.log1p(n)[..., None], torch.log1p(n_tot)[..., None]], dim=-1)
        alpha = F.softplus(self.alpha(a_in).squeeze(-1)) + 1e-3  # [B, K, C]

        return dict(existence=exist, expression=expr, geometry=geom, alpha=alpha)


# --- orchestrator -----------------------------------------------------------------------------
class ModelV3(nn.Module):
    """The full v3 inversion network. `forward` takes a tile [B, M, H, W] (already log-scaled by
    model.to_input) and returns a dict of posteriors and dense maps; see the module docstring.

    `tags` and `seeds` may be passed in to make the permutation/padding/seed invariances (plan
    4.11) testable; otherwise tags are drawn fresh each forward (4.2) and seeds come from farthest-
    point sampling over the predicted cell interior.
    """

    N_REGIONS = 4        # nucleus, cytoplasm, membrane, background
    U_C = 48             # stem stride-4 per-marker feature channels (h4)

    def __init__(self, n_global=16, n_marker_noise=4, n_context=None,
                 K=30, T=3, d=128, d_tag=32, predict_offset=True):
        super().__init__()
        n_context = (1 + len(cfg.CONTEXT_KINDS)) if n_context is None else n_context
        self.K, self.T, self.d, self.d_tag = K, T, d, d_tag
        self.stem = Stem(d_tag)
        self.trunk = Trunk(d, d_tag)
        self.fusion = Fusion(d)
        self.decoder = DenseDecoder(self.fusion.g16, n_context)
        self.f4 = nn.Sequential(nn.Linear(96 + 64, d))
        self.f4_ln = nn.LayerNorm(d)
        self.emb = nn.Linear(d, 16)
        self.slots = TypeSlots(d, K, T)
        loc_in = self.U_C + 1 + 3          # stem feats + log-intensity + (nucleus, cyto, membrane) probs
        self.marker_head = MarkerHead(d, n_marker_noise, loc_in, predict_offset=predict_offset)
        self.global_head = GlobalHead(self.fusion.g32, n_global)
        self.slot_heads = SlotHeads(d, u_c=48, dec_c=96, n_geom=5, n_context=n_context)
        self.n_context = n_context

    def _seeds(self, emb, cell_prob4, Hf, Wf, seeds, n_seeds, device):
        """Seeds for the type slots. `seeds=None` runs farthest-point sampling for every sample
        (inference). During training, pass teacher seeds for some samples and mark the rest for FPS
        with `n_seeds[b] < 0` -- those rows are recomputed by FPS here (plan 5.4)."""
        B, N, _ = emb.shape
        if seeds is None:
            seeds = torch.zeros(B, self.K, dtype=torch.long, device=device)
            n_seeds = torch.full((B,), -1, dtype=torch.long, device=device)
        else:
            seeds, n_seeds = seeds.to(device).clone(), n_seeds.to(device).clone()
        for b in range(B):
            if n_seeds[b] >= 0:
                continue                                      # teacher seeds already provided
            cand = cell_prob4[b] > 0.5
            chosen = select_seeds(_smooth4(emb[b], Hf, Wf), cand, self.K)
            seeds[b] = 0
            if chosen:
                seeds[b, :len(chosen)] = torch.as_tensor(chosen, device=device)
            n_seeds[b] = len(chosen)
        return seeds, n_seeds

    def forward(self, tile, present=None, strength=None, tags=None, seeds=None, n_seeds=None):
        B, M, H, W = tile.shape
        device = tile.device
        if present is None:
            present = torch.ones(B, M, dtype=torch.bool, device=device)
        if strength is None:
            strength = torch.zeros(B, M, device=device)
        strength = strength.to(device)
        if tags is None:
            tags = torch.randn(B, M, self.d_tag, device=device) / math.sqrt(self.d_tag)

        feats = self.stem(tile, tags, present, strength)
        t = self.trunk(feats["h4"], tags, present, strength)     # [B, M, Hg, Wg, d]
        g16p, g32 = self.fusion(t, present)
        dense = self.decoder(g16p, feats["s4"])

        Hf, Wf = feats["s4"].shape[-2:]                          # stride-4 grid
        N4 = Hf * Wf
        s4_flat = feats["s4"].permute(0, 2, 3, 1).reshape(B, N4, 96)
        ctx4_flat = dense["ctx4"].permute(0, 2, 3, 1).reshape(B, N4, 64)
        F4 = self.f4_ln(self.f4(torch.cat([s4_flat, ctx4_flat], dim=-1)))
        emb = self.emb(F4)

        cell_prob4 = F.avg_pool2d(torch.sigmoid(dense["cell"]), 4).reshape(B, N4)
        seeds, n_seeds = self._seeds(emb, cell_prob4, Hf, Wf, seeds, n_seeds, device)
        states, maps = self.slots(F4, seeds, n_seeds)

        u = feats["h4"].reshape(B, M, 48, N4)
        logI = feats["logI4"].reshape(B, M, N4)
        dec4 = dense["dec4"].reshape(B, 96, N4)
        P4 = F.softmax(dense["structure"], dim=1)
        P4 = F.interpolate(P4, size=(Hf, Wf), mode="bilinear", align_corners=False)
        P4 = P4.reshape(B, self.n_context, N4)

        slot_out = []
        for ti in range(self.T):
            o = self.slot_heads(states[ti], maps[ti], u, logI, dec4, P4)
            o["token_comp"] = F.avg_pool2d(maps[ti].reshape(B, self.K + 1, Hf, Wf), 4).reshape(
                B, self.K + 1, -1)
            o["maps"] = maps[ti]
            slot_out.append(o)

        loc_feat = self._loc_features(dense, logI, u, Hf, Wf)      # [B, M, N4, loc_in]
        return dict(
            global_mu=self.global_head(g32),
            marker=self.marker_head(t, loc_feat),
            dense={k: dense[k] for k in ("nucleus", "cell", "flow", "nuc_flow", "structure")},
            slots=slot_out, seeds=seeds, n_seeds=n_seeds, emb=emb)

    def _region_probs(self, dense, N4):
        """Per-location sub-cellular region membership at stride 4: (nucleus, cytoplasm, membrane),
        each [B, 1, N4]. The membrane is relu(divergence of the predicted FLOW field): the flows
        point to each cell's centre, so they diverge at cell-cell boundaries -- the only cue that
        resolves INTERNAL membranes in packed tissue (a union cell mask cannot)."""
        nuc4 = F.avg_pool2d(torch.sigmoid(dense["nucleus"]), 4).reshape(dense["nucleus"].shape[0], 1, N4)
        cell2d = F.avg_pool2d(torch.sigmoid(dense["cell"]), 4)
        fy, fx = dense["flow"][:, 0:1], dense["flow"][:, 1:2]
        div = (F.pad(fx[..., :, 1:] - fx[..., :, :-1], (0, 1))
               + F.pad(fy[..., 1:, :] - fy[..., :-1, :], (0, 0, 0, 1)))
        mem4 = (F.avg_pool2d(F.relu(div), 4) * cell2d).reshape(cell2d.shape[0], 1, N4)
        cell4 = cell2d.reshape(cell2d.shape[0], 1, N4)
        cyto4 = (cell4 - nuc4 - mem4).clamp(min=0)
        return nuc4, cyto4, mem4

    def _loc_features(self, dense, logI, u, Hf, Wf):
        """Stride-4 per-marker, per-location tokens for the marker head's attention pool: the stem
        features `u`, the marker's log-intensity, and the (nucleus, cyto, membrane) region memberships
        broadcast to every marker. The attention query then learns which locations/cells to trust.
        Padding-safe (regions from present-only dense maps) and per-marker (permutation-equivariant).
        """
        B, M, N4 = logI.shape
        nuc4, cyto4, mem4 = self._region_probs(dense, N4)        # each [B, 1, N4]
        reg = torch.cat([nuc4, cyto4, mem4], dim=1).transpose(1, 2)[:, None]   # [B, 1, N4, 3]
        reg = reg.expand(B, M, N4, 3)
        feat = u.permute(0, 1, 3, 2)                             # [B, M, N4, 48]
        return torch.cat([feat, logI[..., None], reg], dim=-1)   # [B, M, N4, 48+1+3]
