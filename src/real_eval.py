"""Annotation-free real-data proxy for the inversion model: how it segments REAL staged tiles.

The synthetic eval set says how well the model inverts the generator; it cannot say whether a change
helps on real tissue, and it is replaced every time the generator changes. This is the one yardstick
that stays put across generator versions. It is a MEASURING STICK ONLY: real tiles never enter a
loss, and nothing here may drive a learning-rate, "best checkpoint" or prior decision automatically
(Max, 2026-11-12) -- it is logged next to the synthetic score so a human can read both.

The set (`build`) is a fixed list of tiles from several runs spanning both panel families (Slide1-5
with a sharp DAPI_cycle5 re-stain; Slide6-10, Controls and Test2 with cycle-1 DAPI only), read at the
training pixel size by `real_tiles.read_tile` and cached as npz. Nuclei are the only thing real tiles
show unambiguously, so the pseudo-truth is classical nucleus segmentation of the counterstain
(top-hat, Otsu, distance-transform watershed). It is good enough to COUNT nuclei and place their
centres; it is not a boundary-exact reference, so the headline numbers are centroid-based:

- nuc_rec_c: share of pseudo-nuclei whose centre lies inside a predicted nucleus;
- nuc_prec_c: share of predicted nuclei that contain a pseudo-nucleus centre or sit >= 30% on the
  pseudo-nucleus mask -- a predicted nucleus with no DAPI under it is a hallucination;
- nuc_count_lr: log((pred + 1) / (pseudo + 1)), signed (> 0 = over-segmentation);
- cells_per_nuc: predicted cells per pseudo-nucleus -- reported, never a target, because anucleate
  cell profiles (a long cell cut away from its nucleus) are real;
- flip_f1: instance F1 of the decode against the decode of the flipped + rotated tile, mapped back
  -- a model that is sure of a cell finds it either way;
- p_exist_mid: share of queries with p_exist in (0.2, 0.8) -- confident heads are bimodal;
- at_cap: share of tiles where the decode uses > 80% of the queries.

    python real_eval.py build --out ../data/real_eval
    python real_eval.py eval --data ../data/real_eval --ckpt ../runs/run13/model_best.pt
"""
import argparse
import glob
import json
import math
import os

import numpy as np
from scipy import ndimage as ndi

import config as cfg

# runs of the fixed set: both panel families, several slides, healthy controls and the Test2 panel
DEFAULT_RUNS = ("Slide1_260129_361046_R1_A1_ROI1", "Slide2_260216_420907_R1_A1_ROI2",
                "Slide4_260504_461338_R1_A1_ROI3", "Slide6_260624_510806_R1_B1_ROI1",
                "Slide8_260506_161510_R1_B1_ROI1", "Slide9_Slide10_260514_311307_R1_B1_ROI1",
                "Control1-3_260507_461536_R1_B1_ROI1", "Test2_260312_431629_R1_A1_ROI1")


def pseudo_nuclei(dapi, um_per_px=cfg.UM_PER_VOX):
    """Classical nuclei on one counterstain tile -> int32 labels. A white top-hat with a 6 um disc
    removes the slowly varying autofluorescence a cycle-1 DAPI carries (off/on ~0.35), a TRIANGLE
    threshold picks the stain, and a distance-transform watershed with seeds >= 2 um apart splits
    touching nuclei. Objects under 3 um^2 are dropped (a real nucleus is 10-60 um^2). Triangle, not
    Otsu: nuclear brightness spans ~10x within one tile, and Otsu split the bright nuclei from the
    dim ones instead of nuclei from background (it missed every dim nucleus on Slide1/2)."""
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_triangle
    from skimage.morphology import disk, white_tophat
    from skimage.segmentation import watershed
    g = ndi.gaussian_filter(np.asarray(dapi, np.float32), 1.0)
    th = white_tophat(g, disk(max(int(round(6.0 / um_per_px)), 3)))
    if th.max() <= 0:
        return np.zeros(g.shape, np.int32)
    fg = th > threshold_triangle(th)
    fg = ndi.binary_opening(fg, iterations=1)
    lab, n = ndi.label(fg)
    if n == 0:
        return lab.astype(np.int32)
    area = np.bincount(lab.ravel())
    small = area < 3.0 / um_per_px ** 2
    small[0] = False
    fg &= ~small[lab]
    dist = ndi.distance_transform_edt(fg)
    pk = peak_local_max(dist, min_distance=max(int(round(2.0 / um_per_px)), 2), labels=ndi.label(fg)[0],
                        exclude_border=False)
    markers = np.zeros(fg.shape, np.int32)
    markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    return watershed(-dist, markers, mask=fg).astype(np.int32)


def build(out, runs=DEFAULT_RUNS, n_per_run=16, seed=0, um_per_px=None, mode="area", root=None):
    """Read and cache the fixed real set. Each tile -> tile_XXXX.npz (image uint16 [T, T, M],
    names, nuclear_idx, pseudo labels, provenance)."""
    import real_tiles as rt   # tifffile / zarr: generation env only; scoring needs neither
    os.makedirs(out, exist_ok=True)
    paths = rt.find_slides(root) if root else rt.find_slides()
    k = 0
    for run in runs:
        hit = [p for p in paths if run in p]
        if not hit:
            raise FileNotFoundError(f"no staged backsub image matches {run!r} -- pick runs from "
                                    f"real_tiles.find_slides()")
        sl = rt.open_slide(hit[0])
        pos = rt.tissue_positions(sl, n_per_run, um_per_px=um_per_px, seed=seed)
        nuc = rt.nuclear_channel(sl, [(r, c) for r, c, _ in pos[:4]])
        for r0, c0, frac in pos:
            t = rt.read_tile(sl, r0, c0, nuc, um_per_px=um_per_px, mode=mode)
            ps = pseudo_nuclei(t["image"][..., t["nuclear_idx"]], um_per_px=t["um_per_px"])
            np.savez_compressed(os.path.join(out, f"tile_{k:04d}.npz"), image=t["image"],
                                names=np.array(t["names"]), nuclear_idx=t["nuclear_idx"], pseudo=ps,
                                run=run, path=hit[0], r0=r0, c0=c0, S=t["S"], um_per_px=t["um_per_px"],
                                tissue_frac=frac, mode=mode,
                                nuclear_name=sl["names"][nuc])
            k += 1
        print(f"{run}: {len(pos)} tiles, nuclear channel {sl['names'][nuc]}, "
              f"{len(t['names'])} model channels", flush=True)
    return k


def relabel(d):
    """Recompute the cached pseudo-nuclei in place (after a change to `pseudo_nuclei`)."""
    fs = sorted(glob.glob(os.path.join(d, "tile_*.npz")))
    for f in fs:
        z = dict(np.load(f, allow_pickle=False))
        z["pseudo"] = pseudo_nuclei(z["image"][..., int(z["nuclear_idx"])], float(z["um_per_px"]))
        np.savez_compressed(f, **z)
    return len(fs)


def build_regions(out, src, T=512, um_per_px=None):
    """The fixed real set again, but each tile read as the CENTRE of a T x T px region (512 px =
    166 um at the training pixel size), for a model trained on regions: it then sees the tissue
    around the tile, and is scored on the instance window over the tile. The window corner sits
    on the stride-4 grid (`win_yx`, a multiple of 4; the region is shifted inward at a slide
    edge), and the pseudo-nuclei are recomputed on exactly that window of the region read, so
    labels and pixels agree to the pixel. The nuclear channel is the one `build` chose (same
    positions, same rule)."""
    import real_tiles as rt
    from model_v3 import INST_WIN
    um_per_px = float(cfg.UM_PER_VOX if um_per_px is None else um_per_px)
    os.makedirs(out, exist_ok=True)
    tiles = load_set(src)
    slides, k = {}, 0
    for t in tiles:
        path = str(t["path"])
        if path not in slides:
            sl = rt.open_slide(path)
            n_run = sum(str(x["path"]) == path for x in tiles)
            pos = rt.tissue_positions(sl, n_run, um_per_px=um_per_px, seed=0)
            nuc = rt.nuclear_channel(sl, [(r, c) for r, c, _ in pos[:4]])
            if sl["names"][nuc] != str(t["nuclear_name"]):
                raise ValueError(f"{path}: re-derived nuclear channel {sl['names'][nuc]!r} is not the "
                                 f"cached {t['nuclear_name']!r} -- rebuild the tile set first")
            slides[path] = (sl, nuc)
        sl, nuc = slides[path]
        H0, W0 = sl["levels"][0].shape[1:]
        S_R = int(round(T * um_per_px / sl["um_raw"]))
        r0, c0, S = int(t["r0"]), int(t["c0"]), int(t["S"])
        rr = int(np.clip(round(r0 + S / 2 - S_R / 2), 0, H0 - S_R))
        cc = int(np.clip(round(c0 + S / 2 - S_R / 2), 0, W0 - S_R))
        reg = rt.read_tile(sl, rr, cc, nuc, um_per_px=um_per_px, T=T)
        f = T / S_R
        oy = int(np.clip(round((r0 - rr) * f / 4.0) * 4, 0, T - INST_WIN))
        ox = int(np.clip(round((c0 - cc) * f / 4.0) * 4, 0, T - INST_WIN))
        img = reg["image"]
        ps = pseudo_nuclei(img[oy:oy + INST_WIN, ox:ox + INST_WIN, reg["nuclear_idx"]],
                           um_per_px=reg["um_per_px"])
        np.savez_compressed(os.path.join(out, f"tile_{k:04d}.npz"), image=img,
                            names=np.array(reg["names"]), nuclear_idx=reg["nuclear_idx"], pseudo=ps,
                            win_yx=np.array([oy, ox]), run=str(t["run"]), path=path, r0=rr, c0=cc,
                            S=S_R, um_per_px=reg["um_per_px"], tissue_frac=float(t["tissue_frac"]),
                            mode="area", nuclear_name=str(t["nuclear_name"]), src_tile=k)
        k += 1
    print(f"{k} regions ({T} px) -> {out}", flush=True)
    return k


def load_set(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "tile_*.npz"))):
        z = np.load(f, allow_pickle=False)
        out.append({k: z[k] for k in z.files})
    if not out:
        raise FileNotFoundError(f"no tile_*.npz under {d} -- run `python real_eval.py build --out {d}`")
    return out


def load_model(ckpt, device):
    import torch
    from model_v3 import ModelV3
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    c = ck["cfg"]
    model = ModelV3(n_global=c["n_global"], n_marker_noise=4, K=c["k_slots"], T=3,
                    n_queries=c.get("n_queries", 0), use_flow=c.get("use_flow", True),
                    **({"use_ref": c["use_ref"]} if "use_ref" in c else {})).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, ck


def fixed_tags(M, d_tag, seed, device, B=1):
    """Seeded marker tags: the model draws fresh random tags every forward, which makes the same
    weights score differently twice. Evaluation fixes them."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    return (torch.randn(B, M, d_tag, generator=g) / math.sqrt(d_tag)).to(device)


def predict(model, image, device, nuclear_idx=None, tag_seed=0, decode=None, win_yx=None):
    """One real tile [H, W, M] uint16 -> (cell labels, nucleus labels, props, p_exist). For a
    region (`win_yx` given) the model sees the whole region -- the instance head on its covering
    grid plus the scored window at win_yx -- and the labels are that window's."""
    import torch
    from model import normalize_markers
    from model_v3 import decode_queries
    norm, strn = normalize_markers(image)
    M = norm.shape[-1]
    kw = {}
    if getattr(model, "use_ref", False):
        ref = torch.zeros(1, M, device=device)
        if nuclear_idx is not None and nuclear_idx >= 0:
            ref[0, int(nuclear_idx)] = 1.0
        kw["ref"] = ref
    wi = 0
    if win_yx is not None:
        from model_v3 import window_grid
        grid = [(0, y, x) for y, x in window_grid(*image.shape[:2])]
        kw["inst_windows"] = torch.tensor(grid + [(0, int(win_yx[0]), int(win_yx[1]))],
                                          dtype=torch.long, device=device)
        wi = len(grid)
    with torch.no_grad():
        o = model(torch.from_numpy(norm).permute(2, 0, 1)[None].to(device),
                  present=torch.ones(1, M, dtype=torch.bool, device=device),
                  strength=torch.from_numpy(strn)[None].to(device),
                  tags=fixed_tags(M, model.d_tag, tag_seed, device), **kw)
    qc, qn, props = decode_queries(o["inst"], wi, **(decode or {}))
    pe = torch.sigmoid(o["inst"]["exist"][wi]).cpu().numpy()
    return qc, qn, props, pe


def _centroids(lab):
    n = int(lab.max())
    if n == 0:
        return np.zeros((0, 2), int)
    c = ndi.center_of_mass(np.ones_like(lab), lab, np.arange(1, n + 1))
    return np.rint(np.asarray(c)).astype(int).reshape(-1, 2)


def score_tile(qc, qn, pe, pseudo, n_queries, qc_flip=None):
    from flow_decode import match, relabel
    ps = relabel(pseudo)
    pn = relabel(qn)
    n_ps, n_pn, n_pc = int(ps.max()), int(pn.max()), int(relabel(qc).max())
    cen = _centroids(ps)
    hit_nuc = pn[cen[:, 0], cen[:, 1]] if len(cen) else np.zeros(0, int)
    rec = float((hit_nuc > 0).mean()) if n_ps else float("nan")
    if n_pn:
        has_c = np.zeros(n_pn + 1, bool)
        has_c[hit_nuc[hit_nuc > 0]] = True
        on = np.bincount(pn.ravel(), weights=(ps > 0).ravel().astype(float), minlength=n_pn + 1)
        area = np.bincount(pn.ravel(), minlength=n_pn + 1).astype(float)
        sup = has_c | (on / np.maximum(area, 1) >= 0.3)
        prec = float(sup[1:].mean())
    else:
        prec = float("nan")
    out = dict(n_pseudo=n_ps, n_pred_nuc=n_pn, n_pred_cell=n_pc,
               nuc_rec_c=rec, nuc_prec_c=prec,
               nuc_f1_iou=match(ps, pn)["f1"] if (n_ps or n_pn) else 1.0,
               nuc_count_lr=float(np.log((n_pn + 1) / (n_ps + 1))),
               cells_per_nuc=float(n_pc / max(n_ps, 1)),
               p_exist_mid=float(((pe > 0.2) & (pe < 0.8)).mean()),
               at_cap=float(n_pc > 0.8 * n_queries))
    if np.isfinite(rec) and np.isfinite(prec) and rec + prec > 0:
        out["nuc_f1_c"] = 2 * rec * prec / (rec + prec)
    elif n_ps == 0 and n_pn == 0:
        out["nuc_f1_c"] = 1.0
    else:
        out["nuc_f1_c"] = 0.0
    if qc_flip is not None:
        out["flip_f1"] = match(qc, qc_flip)["f1"] if (qc.max() or qc_flip.max()) else 1.0
    return out


def evaluate(model, tiles, device, decode=None, flip=True, tag_seed=0):
    """Score `model` on the cached real tiles. Returns (aggregate dict, per-tile list)."""
    rows = []
    for i, t in enumerate(tiles):
        img = t["image"]
        wyx = t.get("win_yx")                      # a region: the scored window's corner
        qc, qn, props, pe = predict(model, img, device, nuclear_idx=int(t["nuclear_idx"]),
                                    tag_seed=tag_seed + i, decode=decode, win_yx=wyx)
        qf = None
        if flip:
            fimg = np.ascontiguousarray(np.rot90(img[:, ::-1], 1, axes=(0, 1)))
            fwyx = None
            if wyx is not None:                    # where the same window lands in the flipped region
                m = np.zeros(img.shape[:2], bool)
                m[int(wyx[0]):int(wyx[0]) + qc.shape[0], int(wyx[1]):int(wyx[1]) + qc.shape[1]] = True
                ys, xs = np.nonzero(np.rot90(m[:, ::-1], 1))
                fwyx = (int(ys.min()), int(xs.min()))
            qcf, _, _, _ = predict(model, fimg, device, nuclear_idx=int(t["nuclear_idx"]),
                                   tag_seed=tag_seed + i, decode=decode, win_yx=fwyx)
            qf = np.ascontiguousarray(np.rot90(qcf, -1, axes=(0, 1))[:, ::-1])
        r = score_tile(qc, qn, pe, t["pseudo"], model.n_queries, qc_flip=qf)
        r["run"] = str(t["run"])
        rows.append(r)
    keys = [k for k in rows[0] if k != "run"]
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)
    agg["by_run"] = {run: {k: float(np.nanmean([x[k] for x in rs])) for k in
                           ("nuc_f1_c", "nuc_count_lr", "n_pred_cell", "n_pseudo")}
                     for run, rs in by_run.items()}
    return agg, rows


def _cli():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--n-per-run", type=int, default=16)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--um-per-px", type=float, default=None,
                   help="pixel size of the cached tiles (default: the training cfg.UM_PER_VOX)")
    b.add_argument("--mode", default="area", choices=("area", "legacy"))
    b.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    br = sub.add_parser("build-regions", help="the fixed set re-read as the centres of larger regions")
    br.add_argument("--src", required=True, help="the cached tile set (build) to re-read")
    br.add_argument("--out", required=True)
    br.add_argument("--size", type=int, default=512)
    rl = sub.add_parser("relabel")
    rl.add_argument("--data", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--data", required=True)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--device", default="cuda")
    e.add_argument("--no-filter", action="store_true", help="the old, unfiltered query decode")
    e.add_argument("--exist-thr", type=float, default=0.5)
    e.add_argument("--no-flip", action="store_true")
    e.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "build":
        n = build(a.out, runs=tuple(r for r in a.runs.split(",") if r), n_per_run=a.n_per_run,
                  seed=a.seed, um_per_px=a.um_per_px, mode=a.mode)
        print(f"{n} tiles -> {a.out}")
        return
    if a.cmd == "build-regions":
        build_regions(a.out, a.src, T=a.size)
        return
    if a.cmd == "relabel":
        print(f"{relabel(a.data)} tiles relabelled in {a.data}")
        return
    import torch
    dev = a.device if torch.cuda.is_available() else "cpu"
    model, ck = load_model(a.ckpt, dev)
    dec = dict(exist_thr=a.exist_thr)
    if a.no_filter:
        dec.update(min_area=0, overlap_thr=0.0)
    agg, rows = evaluate(model, load_set(a.data), dev, decode=dec, flip=not a.no_flip)
    head = {k: v for k, v in agg.items() if k != "by_run"}
    print(f"{a.ckpt} (step {ck.get('step')}) on {a.data}, decode {dec}")
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in head.items()))
    for run, v in agg["by_run"].items():
        print(f"  {run:42s} " + "  ".join(f"{k}={x:.2f}" for k, x in v.items()))
    if a.json:
        json.dump(dict(ckpt=a.ckpt, step=ck.get("step"), decode=dec, agg=agg, rows=rows),
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    _cli()
