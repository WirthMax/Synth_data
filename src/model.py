import argparse, math, re, os, time, json, sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from torchinfo import summary

import config as cfg

ADU_MAX = cfg.ADU_MAX
LOGVAR_MIN, LOGVAR_MAX = cfg.LOGVAR_MIN, cfg.LOGVAR_MAX


class Block(nn.Module):
    """Residual conv-norm-act pair. Dilation grows the receptive field without
    extra downsampling; GroupNorm because batches are small (gcd keeps the
    group count valid for any channel width)."""

    def __init__(self, cin, cout, groups=8, dilation=1):
        super().__init__()
        g = math.gcd(groups, cout)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation)
        self.n1 = nn.GroupNorm(g, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=dilation, dilation=dilation)
        self.n2 = nn.GroupNorm(g, cout)
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.gelu(h + self.skip(x))


def _stats(x, eps=1e-5):
    """Per-channel spatial statistics: mean, (biased) std, max."""
    m = x.mean(dim=(2, 3))
    s = (x.var(dim=(2, 3), unbiased=False) + eps).sqrt()
    q = x.amax(dim=(2, 3))
    return torch.cat([m, s, q], dim=1)


class ParamNet(nn.Module):
    """Image -> logit-normal posterior over parameters in [0, 1]."""

    def __init__(self, in_ch, n_out, width=32, depth=4, hidden=256, groups=8):
        super().__init__()
        chans = [in_ch] + [width * 2 ** min(i, 3) for i in range(depth)]
        dils = [2 ** max(0, i - depth // 2) for i in range(depth)]
        self.blocks = nn.ModuleList(
            Block(chans[i], chans[i + 1], groups, dils[i]) for i in range(depth)
        )
        # Three statistics per channel:
        # a multi-scale texture descriptor
        feat = 3 * sum(chans[1:])
        self.head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Linear(feat, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * n_out),
        )
        self.n_out = n_out

    # single clamp constant so train/eval logits agree exactly
    EPS = 1e-4

    def forward(self, x):
        taps = []
        for i, b in enumerate(self.blocks):
            x = b(x)
            taps.append(_stats(x))
            # last map stays unpooled: 4x more
            if i < len(self.blocks) - 1:
                # spatial samples for its std/max
                x = F.avg_pool2d(x, 2)
        h = self.head(torch.cat(taps, dim=1))
        mu, raw = h[:, : self.n_out], h[:, self.n_out :]
        mid = (LOGVAR_MAX + LOGVAR_MIN) / 2
        half = (LOGVAR_MAX - LOGVAR_MIN) / 2
        return mu, mid + half * torch.tanh(raw / half)

    @staticmethod
    def to_logit(y, eps=EPS):
        """Map [0, 1] targets to logit space, clamped away from exact 0/1."""
        y = y.clamp(eps, 1 - eps)
        return torch.log(y) - torch.log1p(-y)

    @staticmethod
    def point_estimate(mu):
        """Posterior median in [0, 1] units (sigmoid preserves quantiles)."""
        return torch.sigmoid(mu)

    def nll(self, mu, logvar, y01):
        """Element-wise Gaussian negative log-likelihood against logit-transformed targets."""
        z = self.to_logit(y01)
        return (0.5 * (logvar + (z - mu) ** 2 * torch.exp(-logvar))).mean()


class _MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out)
        )

    def forward(self, x):
        return self.net(x)


class _TransformerBlock(nn.Module):
    """Pre-LN self-attention + MLP, the standard ViT/Transformer encoder layer. Used both inside
    the per-marker encoder (over spatial patches, WITH a positional embedding) and across markers
    (over the marker axis, deliberately WITHOUT one -- see `CrossMarkerExchange`)."""

    def __init__(self, d_model, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = _MLP(d_model, int(d_model * mlp_ratio), d_model)

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class MarkerEncoder(nn.Module):
    """Per-marker encoder: a small conv stem (keeps a local inductive bias for the
    high-frequency noise texture that a large-stride patch embedding would blur past) followed by
    a ViT-Tiny-scale transformer. Runs on `(N, 1, H, W)` where `N = B * n_markers` -- the marker
    axis is folded into the batch dimension before this module ever sees the data, so it
    structurally cannot condition on marker order or count. Every marker channel of every
    training image is processed by these same weights, which is what makes a ViT-scale encoder
    reasonable here despite ViTs' usual data hunger: one rendered frame yields `n_markers`
    independent training examples for this module alone, and synthetic data is unlimited.
    """

    def __init__(self, img_size, patch_size=16, d_model=128, depth=4, n_heads=4,
                stem_channels=(16, 32)):
        super().__init__()
        c_in = 1
        stem = []
        for c_out in stem_channels:
            stem += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
                    nn.GroupNorm(math.gcd(8, c_out), c_out), nn.GELU()]
            c_in = c_out
        self.stem = nn.Sequential(*stem)
        stem_stride = 2 ** len(stem_channels)
        stem_size = img_size // stem_stride
        pstride = max(1, patch_size // stem_stride)
        self.patch = nn.Conv2d(c_in, d_model, kernel_size=pstride, stride=pstride)
        n_side = stem_size // pstride
        n_patches = n_side * n_side
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList(_TransformerBlock(d_model, n_heads) for _ in range(depth))
        self.ln_out = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x):
        # x: (N, 1, H, W), N = B * n_markers
        h = self.patch(self.stem(x))            # (N, d_model, n_side, n_side)
        h = h.flatten(2).transpose(1, 2)         # (N, n_patches, d_model)
        h = h + self.pos_embed
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], dim=1)
        for blk in self.blocks:
            h = blk(h)
        return self.ln_out(h)[:, 0]              # (N, d_model), the CLS summary token


class CrossMarkerExchange(nn.Module):
    """Self-attention over the marker axis, with NO positional encoding -- the one
    non-negotiable design choice that makes this equivariant to marker order: permuting the input
    token set permutes the output token set identically, since attention weights are recomputed
    from token content alone. `n_markers` is small (10-40 in practice) so plain O(n^2) attention
    is cheap; no inducing-point approximation is needed."""

    def __init__(self, d_model, n_heads, depth=3, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            _TransformerBlock(d_model, n_heads, mlp_ratio) for _ in range(depth)
        )

    def forward(self, tok):
        for blk in self.blocks:
            tok = blk(tok)
        return tok


class PMAPool(nn.Module):
    """Pooling by Multihead Attention (Set Transformer, Lee et al. 2019): `n_queries` learned
    query vectors -- independent of marker count/order -- cross-attend over the (unordered)
    marker tokens as keys/values, giving `n_queries` pooled summary vectors that can each
    specialise on a different tissue-level aspect. This is the richer, still permutation-invariant
    replacement for mean pooling: no positional encoding is used here either, so the output is
    identical regardless of the order the marker tokens arrive in."""

    def __init__(self, d_model, n_heads, n_queries=8):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, n_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.n_queries = n_queries

    def forward(self, tok):
        q = self.queries.expand(tok.shape[0], -1, -1)
        pooled, _ = self.attn(q, tok, tok, need_weights=False)
        return self.ln(pooled)  # (B, n_queries, d_model)


class SetParamNet(nn.Module):
    """Permutation-invariant/equivariant replacement for `ParamNet`: a weight-shared per-marker
    ViT encoder (`MarkerEncoder`), a cross-marker self-attention exchange with no positional
    encoding (`CrossMarkerExchange`), a shared per-marker MLP head, and a multi-query
    attention-pooling global head (`PMAPool`). See `rounds/tests/test_set_paramnet.py
    ::test_shuffle_invariance` for the executable proof of the invariance claim.

    `marker_scatter`: `(n_markers_total, n_field)` int array/tensor, giving the absolute
    ParamSpace path index for each (canonical marker slot, per-marker field) pair, or -1 where
    that field doesn't apply to that marker. `global_path_idx`: `(n_global,)`, the absolute path
    index for each non-marker-specific parameter. Both are precomputed once from a `ParamSpace`
    via `ParamSpace.marker_field_map` (see `rounds/param_space.py`) -- this module only ever sees
    plain index tensors, not `ParamSpace` itself, so `model.py` stays independent of `rounds/`.

    `forward(x, marker_slot=None)`: `marker_slot[c]` is the canonical marker-slot id that input
    channel `c` actually represents (default: identity, i.e. channel order already matches the
    canonical/training order). This is what lets the SAME model be probed with a shuffled channel
    order for the invariance test -- reassembly and the per-marker identity embedding both route
    through `marker_slot`, not through raw channel position.
    """

    EPS = ParamNet.EPS
    to_logit = staticmethod(ParamNet.to_logit)
    point_estimate = staticmethod(ParamNet.point_estimate)
    nll = ParamNet.nll

    def __init__(self, in_ch, n_out, marker_scatter, global_path_idx, img_size=160,
                patch_size=16, d_model=128, enc_depth=4, enc_heads=4, cross_depth=3,
                cross_heads=4, n_pma_queries=8):
        super().__init__()
        marker_scatter = torch.as_tensor(marker_scatter, dtype=torch.long)
        global_path_idx = torch.as_tensor(global_path_idx, dtype=torch.long)
        n_markers_total, n_field = marker_scatter.shape
        n_global = global_path_idx.numel()
        if in_ch != n_markers_total:
            raise ValueError(
                f"in_ch={in_ch} but marker_scatter has {n_markers_total} marker slots -- "
                f"SetParamNet must be built for the same n_markers the ParamSpace/composition uses"
            )
        n_marker_paths = int((marker_scatter >= 0).sum())
        if n_global + n_marker_paths != n_out:
            raise ValueError(
                f"marker_scatter/global_path_idx account for {n_global + n_marker_paths} of "
                f"{n_out} output paths -- they must exactly partition the full ParamSpace "
                f"(every path is either global or PANEL.Markers.<name>.*, never both/neither)"
            )
        self.register_buffer("marker_scatter", marker_scatter)
        self.register_buffer("global_path_idx", global_path_idx)
        self.n_out, self.n_field, self.n_global = n_out, n_field, n_global

        self.marker_id_embed = nn.Embedding(n_markers_total, d_model)
        self.encoder = MarkerEncoder(img_size, patch_size=patch_size, d_model=d_model,
                                     depth=enc_depth, n_heads=enc_heads)
        self.cross_marker = CrossMarkerExchange(d_model, cross_heads, depth=cross_depth)
        self.marker_head = _MLP(d_model, d_model, 2 * n_field)
        self.pma = PMAPool(d_model, cross_heads, n_queries=n_pma_queries)
        self.global_head = _MLP(n_pma_queries * d_model, d_model, 2 * n_global)

    def forward(self, x, marker_slot=None):
        B, C, H, W = x.shape
        device = x.device
        marker_slot = (torch.arange(C, device=device) if marker_slot is None
                      else marker_slot.to(device=device, dtype=torch.long))

        cls = self.encoder(x.reshape(B * C, 1, H, W)).reshape(B, C, -1)
        cls = cls + self.marker_id_embed(marker_slot).unsqueeze(0)  # content-based conditioning

        tok = self.cross_marker(cls)  # (B, C, d_model)

        n_field = self.n_field
        marker_pred = self.marker_head(tok).reshape(B, C, n_field, 2)
        pooled = self.pma(tok).reshape(B, -1)
        global_pred = self.global_head(pooled).reshape(B, self.n_global, 2)

        scatter_c = self.marker_scatter[marker_slot]  # (C, n_field) -> path idx or -1
        valid = scatter_c >= 0
        combined = torch.cat(
            [global_pred, marker_pred.reshape(B, C * n_field, 2)], dim=1
        )
        gather_idx = torch.empty(self.n_out, dtype=torch.long, device=device)
        gather_idx[self.global_path_idx] = torch.arange(self.n_global, device=device)
        flat_pos = self.n_global + torch.arange(C * n_field, device=device).reshape(C, n_field)
        gather_idx[scatter_c[valid]] = flat_pos[valid]

        out = combined[:, gather_idx, :]
        mu, raw = out[..., 0], out[..., 1]
        mid = (LOGVAR_MAX + LOGVAR_MIN) / 2
        half = (LOGVAR_MAX - LOGVAR_MIN) / 2
        return mu, mid + half * torch.tanh(raw / half)


def build_model(arch, in_ch, n_out, **kwargs):
    """Dispatch to the requested architecture. `kwargs` holds architecture-specific hyperparams
    (see checkpoints' `arch_kwargs`); unrecognised keys for the chosen arch surface as a plain
    TypeError from the constructor, not a silent no-op."""
    if arch == "cnn":
        return ParamNet(in_ch, n_out, **kwargs)
    if arch == "set_transformer":
        return SetParamNet(in_ch, n_out, **kwargs)
    raise ValueError(f"unknown model arch {arch!r} (expected 'cnn' or 'set_transformer')")


def to_input(adu):
    """uint16 ADU -> network input.
    float32 explicitly: dividing by a numpy float64 scalar would silently upcast, and torch
    then refuses the float64 tensor against float32 weights.
    """
    x = np.asarray(adu, np.float32)
    return (np.log1p(np.clip(x, 0, ADU_MAX)) / np.float32(np.log1p(ADU_MAX))).astype(
        np.float32
    )


class MMAPView(torch.utils.data.Dataset):
    """A view onto the dataset. `idx` selects this split without copying any pixels -- the
    images may be a memmap, and fancy-indexing it would pull the whole split into RAM."""

    def __init__(self, images, theta, idx, augment=False, augment_seed=0):
        # (N, H, W, C) uint16, possibly a memmap
        self.x = images
        # (N, d) in [0, 1]
        self.y = theta.astype(np.float32)
        self.idx = np.asarray(idx)
        self.augment = augment
        # a fresh RNG seeded from (augment_seed, sample index) per __getitem__ call, not one
        # RNG advanced call-to-call -- so augmentation is a pure function of which sample is
        # asked for, independent of shuffle order, epoch, or DataLoader worker count.
        self.augment_seed = augment_seed

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        a = np.asarray(self.x[i])
        if self.augment:
            rng = np.random.default_rng(self.augment_seed + i)
            if rng.random() < 0.5:
                a = a[::-1]
            if rng.random() < 0.5:
                a = a[:, ::-1]
            r = int(rng.integers(4))
            if r:
                a = np.rot90(a, r, axes=(0, 1))
            a = np.ascontiguousarray(a)
        return torch.from_numpy(to_input(a)).permute(2, 0, 1), torch.from_numpy(
            self.y[i]
        )


def load(path, val_frac=0.2, seed=0):
    """Read a dataset written by gen_dataset as a MEMMAP to save RAM"""
    stem = re.sub(r"\.(npy|meta\.npz|npz)$", "", path)
    meta_path, img_path = stem + ".meta.npz", stem + ".npy"
    if os.path.exists(meta_path) and os.path.exists(img_path):
        d = np.load(meta_path, allow_pickle=True)
        images = np.load(img_path, mmap_mode="r")
    else:
        raise FileNotFoundError(img_path)
    theta, paths = d["theta"], list(d["paths"])
    lo, hi = d["lo"], d["hi"]
    if len(images) != len(theta):
        raise ValueError(f"{len(images)} images but {len(theta)} theta rows")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(images))
    n_val = max(1, int(round(val_frac * len(images))))
    va, tr = idx[:n_val], idx[n_val:]
    # index lists, not fancy-indexed copies
    return (
        MMAPView(images, theta, tr, augment=True, augment_seed=seed),
        MMAPView(images, theta, va, augment=False),
        paths,
        lo,
        hi,
        images.shape[-1],
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    MU, LV, Y = [], [], []
    for x, y in loader:
        mu, lv = model(x.to(device))
        MU.append(mu.cpu().numpy())
        LV.append(lv.cpu().numpy())
        Y.append(y.numpy())
    mu = np.concatenate(MU)
    lv = np.concatenate(LV)
    y = np.concatenate(Y)

    # sigmoid: back to [0, 1]
    p = 1.0 / (1.0 + np.exp(-mu))
    ss_res = ((p - y) ** 2).mean(0)
    var = y.var(0)
    # constant columns: R2 undefined,
    ok = var > 1e-8
    # report NaN instead of +-1e12
    r2 = np.full(var.shape, np.nan)
    r2[ok] = 1.0 - ss_res[ok] / var[ok]

    # sd in logit units
    sig_z = np.exp(0.5 * lv)
    # delta method -> [0, 1] units
    sig01 = (sig_z * p * (1.0 - p)).mean(0)

    eps = ParamNet.EPS
    yc = np.clip(y, eps, 1 - eps)
    z = ((np.log(yc) - np.log1p(-yc)) - mu) / sig_z  # ~N(0,1) iff calibrated
    return dict(
        r2=r2,
        rmse=np.sqrt(ss_res),
        sigma=sig01,
        z_mean=z.mean(0),
        z_std=z.std(0),
        mu=mu,
        y=y,
        lv=lv,
    )


def _load_ckpt(path, map_location):
    """torch.load across versions: `weights_only` appeared in 2.0 and defaults to True in 2.6,
    but the checkpoint holds a path list and numpy bounds, so it has to be False."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def train(args):
    tr, va, paths, lo, hi, in_ch = load(args.data, args.val_frac, args.seed)
    torch.manual_seed(args.seed)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(
        f"device {device} | train {len(tr)} val {len(va)} | {len(paths)} parameters "
        f"| {in_ch} channels"
    )
    dl_tr = torch.utils.data.DataLoader(
        tr, batch_size=args.batch, shuffle=True, drop_last=False
    )
    dl_va = torch.utils.data.DataLoader(va, batch_size=args.batch)
    model = ParamNet(in_ch, len(paths), width=args.width, depth=args.depth).to(device)
    summary(model, [1, in_ch, 160, 160])
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(dl_tr))
    )
    os.makedirs(args.out, exist_ok=True)
    best, t0 = np.inf, time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = k = 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            mu, lv = model(x)
            # warm up on plain MSE: with a random init the model can otherwise drive logvar up
            # and "explain" its error as noise instead of learning anything
            loss = (
                F.mse_loss(mu, model.to_logit(y))
                if ep <= args.warmup
                else model.nll(mu, lv, y)
            )
            if not torch.isfinite(loss):
                print(
                    f"non-finite loss, ep {ep}: y_ok={torch.isfinite(y).all().item()} "
                    f"mu_ok={torch.isfinite(mu).all().item()} lv_ok={torch.isfinite(lv).all().item()} "
                    f"y range [{y.min():.4f}, {y.max():.4f}]"
                )
                raise SystemExit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(x)
            k += len(x)
        if ep % args.every == 0 or ep == args.epochs:
            ev = evaluate(model, dl_va, device)
            med = float(np.nanmedian(ev["r2"]))
            # Only relevant for NLL
            zs = float(np.median(ev["z_std"]))
            print(
                f"ep {ep:4d}  loss {tot / max(k, 1):8.4f}  val median R2 {med:6.3f}  "
                f"mean RMSE {ev['rmse'].mean():.4f}  z-std {zs:5.2f}  "
                f"{time.time() - t0:5.0f}s",
                flush=True,
            )
            score = float(ev["rmse"].mean())
            if score < best:
                best = score
                torch.save(
                    dict(
                        model=model.state_dict(),
                        paths=paths,
                        lo=lo,
                        hi=hi,
                        in_ch=in_ch,
                        width=args.width,
                        depth=args.depth,
                    ),
                    os.path.join(args.out, "model.pt"),
                )

    ck = _load_ckpt(os.path.join(args.out, "model.pt"), device)
    model.load_state_dict(ck["model"])
    ev = evaluate(model, dl_va, device)
    span = np.asarray(hi, float) - np.asarray(lo, float)  # [0,1] -> raw units
    print(
        f"\n{'R2':>7} {'RMSE(u)':>9} {'pred sd':>9} {'RMSE(raw)':>11} {'z-std':>6}  parameter"
    )
    report = {}
    for j, name in enumerate(paths):
        n = str(name)
        print(
            f"{ev['r2'][j]:7.3f} {ev['rmse'][j]:9.4f} {ev['sigma'][j]:9.4f} "
            f"{ev['rmse'][j] * span[j]:11.4g} {ev['z_std'][j]:6.2f}  {n}"
        )
        report[n] = dict(
            r2=float(ev["r2"][j]),
            rmse01=float(ev["rmse"][j]),
            rmse_raw=float(ev["rmse"][j] * span[j]),
            pred_sd01=float(ev["sigma"][j]),
            z_mean=float(ev["z_mean"][j]),
            z_std=float(ev["z_std"][j]),
        )
    with open(os.path.join(args.out, "val_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {args.out}/model.pt and val_report.json")


def read_image(path):
    if path.endswith(".npy"):
        return np.load(path)
    try:
        import tifffile
    except ImportError:
        sys.exit("reading .tif needs tifffile:  pip install tifffile")
    a = tifffile.imread(path)
    # (C,H,W) -> (H,W,C)
    if a.ndim == 3 and a.shape[0] <= 8 and a.shape[0] < a.shape[-1]:
        a = np.moveaxis(a, 0, -1)
    return a


@torch.no_grad()
def predict(args):
    ck = _load_ckpt(args.ckpt, "cpu")
    paths, lo, hi = list(ck["paths"]), ck["lo"], ck["hi"]
    model = ParamNet(ck["in_ch"], len(paths), width=ck["width"], depth=ck["depth"])
    model.load_state_dict(ck["model"])
    model.eval()

    img = read_image(args.image)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] != ck["in_ch"]:
        sys.exit(f"image has {img.shape[-1]} channels, model expects {ck['in_ch']}")
    x = torch.from_numpy(to_input(img)).permute(2, 0, 1)[None]
    mu, lv = model(x)
    u = mu[0].numpy()
    sd = np.exp(0.5 * lv[0].numpy())

    print(f"{args.image}  {img.shape}  {img.dtype}\n")
    print(f"{'value':>12} {'+/-':>10}  {'u':>6}  parameter")
    print("-" * 92)
    for j, p in enumerate(paths):
        span = hi[j] - lo[j]
        v = lo[j] + np.clip(u[j], 0, 1) * span
        # a sigma anywhere near the prior sd (1/sqrt(12) = 0.289) means the image did not
        # constrain this parameter and the value below is essentially the prior mean
        flag = "   <- unconstrained" if sd[j] > 0.20 else ""
        print(f"{v:12.4g} {sd[j] * span:10.4g}  {u[j]:6.3f}  {p}{flag}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    p: dict(
                        value=float(lo[j] + np.clip(u[j], 0, 1) * (hi[j] - lo[j])),
                        sd=float(sd[j] * (hi[j] - lo[j])),
                        u=float(u[j]),
                    )
                    for j, p in enumerate(paths)
                },
                f,
                indent=1,
            )
        print(f"\nwrote {args.json}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", default="runs")
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--width", type=int, default=32)
    t.add_argument("--depth", type=int, default=4)
    t.add_argument(
        "--warmup", type=int, default=20, help="epochs of MSE before switching to NLL"
    )
    t.add_argument("--val-frac", type=float, default=0.2, dest="val_frac")
    t.add_argument("--every", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("predict")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--json", default=None)

    a = ap.parse_args()
    (train if a.cmd == "train" else predict)(a)


if __name__ == "__main__":
    main()
