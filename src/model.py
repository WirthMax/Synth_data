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

    def __init__(self, images, theta, idx, augment=False):
        # (N, H, W, C) uint16, possibly a memmap
        self.x = images
        # (N, d) in [0, 1]
        self.y = theta.astype(np.float32)
        self.idx = np.asarray(idx)
        self.augment = augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        a = np.asarray(self.x[i])
        if self.augment:
            # ToDo: Update with more advanced Augmentation
            if np.random.rand() < 0.5:
                a = a[::-1]
            if np.random.rand() < 0.5:
                a = a[:, ::-1]
            k = np.random.randint(4)
            if k:
                a = np.rot90(a, k, axes=(0, 1))
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
        MMAPView(images, theta, tr, augment=True),
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
