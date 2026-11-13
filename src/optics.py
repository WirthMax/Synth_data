import psfmodels as psfm
import numpy as np
from scipy.signal import fftconvolve
from scipy import ndimage as ndi

_KERNEL_CACHE = {}
# Verify
FLUOROPHORES = {"DAPI": 0.461, "FITC": 0.519, "PE": 0.578, "APC": 0.660}
AUTOFLUOR = {"DAPI": 0.20, "FITC": 0.60, "PE": 0.30, "APC": 0.10}


def plane_heights(nz, um_per_pz):
    """Axial coordinate of each slice of a centred volume, in microns."""
    return (np.arange(nz) - (nz - 1) / 2.0) * um_per_pz

def psf_support_px(opt, thickness_um, pad=6):
    """Odd kernel width that contains the most defocused PSF for a section of `thickness_um`."""
    r = 0.5 * thickness_um * opt.tan_theta / opt.um_per_px
    return int(2 * (int(np.ceil(r)) + pad) + 1)

def quad_weights(nz):
    """Trapezoid weights for the depth integral -- half weight on the two cut faces."""
    w = np.ones(nz, float)
    w[0] = w[-1] = 0.5
    return w

def psf_kernels(opt, nxy, wavelength_um, z_um):
    
    z_um = np.asarray(z_um, float)
    # depth below the coverslip, per plane
    depth = opt.sample_depth_um + z_um
    zf = opt.sample_depth_um + opt.focal_um

    # psfmodels requires pz >= 0. Reachable in normal use: kryostat(centre_um=c) puts the
    # slab at depth in [c, c + section_um], so ANY negative centre_um trips this.
    if depth.min() < 0:
        need = float(opt.sample_depth_um - depth.min())
        raise ValueError(
            f"plane at depth {depth.min():+.3f} um sits above the coverslip; psfmodels needs "
            f"pz >= 0. Use Optics(depth_um={need:.2f}) or more (currently "
            f"{opt.sample_depth_um:.2f}), or section closer to the centre.")
    
    # Cache the kernel per fluorophore
    key = (nxy, round(float(wavelength_um), 6), round(float(zf), 6),
           opt.um_per_px, opt.na, opt.n_immersion, opt.n_sample, z_um.tobytes())
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    
    base = dict(nx=int(nxy), dxy=opt.um_per_px, NA=opt.na, wvl=float(wavelength_um),
                    ni=opt.n_immersion, ni0=opt.n_immersion, ns=opt.n_sample)

    ks = [np.asarray(psfm.make_psf(z=[float(zf)], pz=float(d), model="scalar", **base)[0],
                            np.float32) for d in depth]
    
    ks = [k / (k.sum() + 1e-30) for k in ks]
    _KERNEL_CACHE[key] = ks
    return ks


def psf_project(slice_vol, z_um, optics, fluorophore = None, weights = None):
    nz = slice_vol.shape[0]
    # the dye is a property of the marker, so the caller hands it over directly
    wavelength_um = optics.wavelength_um if fluorophore is None else FLUOROPHORES[fluorophore]
    w = quad_weights(nz) if weights is None else np.asarray(weights, float)
    nxy = psf_support_px(opt = optics, thickness_um = float(np.ptp(z_um)) + optics.um_per_pz)
    ks = psf_kernels(opt = optics, nxy = nxy, wavelength_um = wavelength_um, z_um = z_um)
    out = np.zeros(slice_vol.shape[1:], np.float32)
    for i in range(nz):
        if w[i] == 0:
            continue
        out += np.float32(w[i]) * fftconvolve(slice_vol[i].astype(np.float32), ks[i], mode="same")
    return np.maximum(out, 0.0) / np.float32(w.sum())

def kryostat(vol, opt, centre_um = 0.0):
    """Cut a `section_um`-thick slab. Returns (subvolume, its plane heights in um)."""
    z = plane_heights(vol.shape[0], opt.um_per_pz)
    keep = np.abs(z - centre_um) <= opt.section_um / 2
    return vol[keep], z[keep]

def _cov_thresh(cov, mask_pct):
            flat = np.sort(cov.ravel())[::-1]
            csum = np.cumsum(flat)
            if csum[-1] <= 0:
                return np.zeros(cov.shape, bool)
            k = int(np.searchsorted(csum, mask_pct * csum[-1])) 
            return flat[min(k, flat.size - 1)]
        
def mask_collapse(mask, z_um, opt=None, mask_pct=0.95, keep_largest = True):
    """The honest 2D footprint of a 3D object seen through the exact optics.
    """
    mask = np.asarray(mask)
    # empty tile: nothing is labelled. Return an honest empty footprint -- without this the
    # `max() <= 1` branch below thresholds an all-zero coverage and declares the WHOLE frame one
    # instance (a frame-filling phantom cell on a tile that should have none).
    if mask.max() <= 0:
        z = np.zeros(mask.shape[1:], np.float32)
        return np.zeros(mask.shape[1:], bool), z

    # single mask case
    if mask.dtype == bool or mask.max() <= 1:
        cov = psf_project(slice_vol=mask.astype(np.float32), z_um=z_um, optics=opt)
        thr = _cov_thresh(cov, mask_pct)
        return (cov >= thr), cov
    
    # Tissue case
    ids = np.unique(mask)
    ids = ids[ids > 0]
    if ids.size == 0:
        z = np.zeros(mask.shape[1:], np.float32)
        return z.astype(np.int32), z

    # project every label, then decide ownership once
    covs = np.stack([psf_project(slice_vol=(mask == n).astype(np.float32), z_um=z_um, optics=opt)
                     for n in ids])
    thr = np.array([_cov_thresh(c, mask_pct) for c in covs], np.float32)
    scored = np.where(covs >= thr[:, None, None], covs, -1.0)
    out = np.where(scored.max(0) > 0, ids[scored.argmax(0)], 0).astype(np.int32)

    if keep_largest:
        for n in ids:
            m = out == n
            if not m.any():
                continue
            cc, k = ndi.label(m)
            if k > 1:
                main = 1 + int(np.argmax(np.bincount(cc.ravel())[1:]))
                out[m & (cc != main)] = 0
    return out, covs.sum(0)


### DETECTOR

def _smooth_unit(w, sigma_px):
    """Circular Gaussian low-pass of a frozen field, rescaled to zero mean and unit variance.
    """
    ny, nx = w.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.rfftfreq(nx)[None, :]
    H = np.exp(-2 * np.pi ** 2 * float(sigma_px) ** 2 * (fy ** 2 + fx ** 2))
    z = np.fft.irfftn(np.fft.rfftn(w) * H, s=(ny, nx), axes=(0, 1))
    z = z - z.mean()
    return (z / (z.std() + 1e-12)).astype(np.float32)


def _corr_noise(z, sigma_px):
    """Give a frozen white-noise field a short correlation length, variance-preserving. A real
    sensor's read noise is not perfectly per-pixel white (demosaic, on-chip readout, the resampling
    the registration applies downstream), so the empty-tile floor is slightly blobby. sigma_px 0
    returns the field untouched (pure white); a small sigma renormalises back to unit SD so the
    read_e scaling downstream is unchanged."""
    if sigma_px <= 0:
        return z
    zz = ndi.gaussian_filter(np.asarray(z, np.float32), float(sigma_px), mode="reflect")
    return (zz / (zz.std() + 1e-12)).astype(np.float32)


def illumination(shape, illum_cv, vignette, tape):
    """Multiplicative flat-field: a smooth random component times a radial falloff."""
    ny, nx = shape
    rnd = 1.0 + illum_cv * _smooth_unit(tape["w_ill"], 0.25 * max(ny, nx))
    yy = (np.arange(ny) - (ny - 1) / 2.0)[:, None] / max(ny / 2.0, 1)
    xx = (np.arange(nx) - (nx - 1) / 2.0)[None, :] / max(nx / 2.0, 1)
    rr2 = np.clip((yy ** 2 + xx ** 2) / 2.0, 0.0, 1.0)     # 0 at centre, 1 at the corners
    return np.maximum(rnd * (1.0 - vignette * rr2), 0.0).astype(np.float32)


def tissue_background(tissue2d, levels, det, opt, tape):
    """The light every channel picks up from tissue whether or not its antibody is there: per
    channel k, levels[k] x the PSF-projected tissue occupancy (in [0, 1], one map per dye in
    `tissue2d`) x one shared texture. The texture is the tissue's own structure, so every channel
    gets the SAME mottling at its own brightness -- which is what makes an unexpressed real channel
    look like faint tissue instead of a black frame. Added to the clean projection BEFORE the camera,
    so it carries shot noise and survives backsub like any signal. Returns (H, W, C), clean units."""
    tissue2d = [np.asarray(t, np.float32) for t in tissue2d]
    ny, nx = tissue2d[0].shape
    if getattr(tape, "w_tbg", None) is None or tape.w_tbg.shape != (ny, nx):
        raise ValueError(
            f"tissue background needs tape.w_tbg of shape {(ny, nx)}. "
            f"Call tape.drawTissueBg(shape=({ny}, {nx})) before the detector.")
    sig_px = max(float(det.TBG_TEX_UM.v) / opt.um_per_px, 0.5)
    tex = np.maximum(1.0 + float(det.TBG_TEX_CV.v) * _smooth_unit(tape.w_tbg, sig_px), 0.0)
    return np.stack([float(l) * t * tex for l, t in zip(levels, tissue2d)], -1).astype(np.float32)


def apply_cycle_misregistration(adu, cycle_of, offsets, mode="reflect"):
    """Shift each channel by its imaging cycle's residual (dy, dx), in pixels.

    adu is the per-channel float frame (H, W, C); cycle_of[c] is the cycle index of channel c;
    offsets[j] is cycle j's (dy, dx) in pixels. +dy moves content DOWN (toward larger y), +dx
    RIGHT (toward larger x). All channels of one cycle move together; the reference cycle (index 0)
    has offset 0 and is left untouched, as is any channel whose offset is exactly zero -- so a zero
    shift is bit-identical, with no spline round-trip. mode is passed to ndi.shift: "constant"
    (cval 0) leaves a black band at the trailing edges, the honest result for a tile cropped from
    the slide border where the shifted cycle has no data; "reflect" fills the edge, standing in for
    the neighbouring scan of an interior crop. A cubic shift can ring slightly negative against a
    black edge; the caller clips to >= 0 afterwards."""
    out = adu.copy()
    for c in range(adu.shape[-1]):
        dy, dx = offsets[cycle_of[c]]
        if dy == 0.0 and dx == 0.0:
            continue
        out[..., c] = ndi.shift(adu[..., c], (float(dy), float(dx)), order=3, mode=mode, cval=0.0)
    return out


def detector(img_psf, profile, det, opt, tape, markers, quantise=False,
             autofluor = AUTOFLUOR, cycle_of=None, cycle_offsets=None, cycle_mode="reflect"):
    """Turn a clean PSF projection (H, W, C) into a detector frame.

    If cycle_of/cycle_offsets are given, cycle misregistration (AR.Cycle) is applied to the noisy
    float frame AFTER the camera model and BEFORE clipping/quantisation -- the physical order, since
    real registration resamples images that have already been acquired. Clipping and rounding are
    deferred to a single pass after the shift; with no cycle args this is elementwise-identical to
    the old per-channel clip+round.
    """
    e_per_unit  = det.E_PER_UNIT.v
    read_e      = det.READ_E.v
    dark_e      = det.DARK_E.v
    adu_per_e   = det.ADU_PER_E.v
    offset_adu  = det.OFFSET_ADU.v
    af_scale_um = det.AF_SCALE_UM.v
    af_cv       = det.AF_CV.v
    illum_cv    = det.ILLUM_CV.v
    vignette    = det.VIGNETTE.v
    bit_depth   = det.BIT_DEPTH
    # MACSIMA background-subtraction + sensor-floor knobs (absent on old Detector instances -> off)
    bg_subtract = bool(det.BG_SUBTRACT.v) if hasattr(det, "BG_SUBTRACT") else False
    bg_drift    = det.BG_DRIFT.v      if hasattr(det, "BG_DRIFT")      else 0.0
    bg_shift_px = det.BG_SHIFT_PX.v   if hasattr(det, "BG_SHIFT_PX")   else 0.0
    bg_margin   = det.BG_MARGIN_ADU.v if hasattr(det, "BG_MARGIN_ADU") else 0.0
    hot_rate    = det.HOT_RATE.v      if hasattr(det, "HOT_RATE")      else 0.0
    hot_e       = det.HOT_E.v         if hasattr(det, "HOT_E")         else 0.0
    read_smooth = det.READ_SMOOTH_PX.v if hasattr(det, "READ_SMOOTH_PX") else 0.0

    img_psf = np.asarray(img_psf, np.float32)
    ny, nx, nc = img_psf.shape
    if tape.z_shot.shape != (nc, ny, nx):
        raise ValueError(
            f"sensor tape is {tape.z_shot.shape} but the image is {(nc, ny, nx)}. "
            f"Re-run tape.drawSensor(shape=({ny}, {nx}, {nc})) -- and set IMG to match.")
    if bg_subtract and (not hasattr(tape, "u_hot") or tape.u_hot.shape != (nc, ny, nx)):
        raise ValueError(
            f"BG_SUBTRACT is on but the backsub tape fields are missing or {getattr(tape, 'u_hot', None) is not None and tape.u_hot.shape}. "
            f"Re-run tape.drawSensor(shape=({ny}, {nx}, {nc})) so z_shot_bg/z_read_bg/u_hot/af_shift/"
            f"z_bg_drift are drawn -- or set DET.BG_SUBTRACT=0 for the raw-pedestal camera.")

    ill = illumination((ny, nx), illum_cv, vignette, tape)
    sig_px = max(af_scale_um / opt.um_per_px, 0.5)

    adu = np.empty_like(img_psf)
    for k, m in enumerate(markers):
        fl = profile.Markers[m].fluorophore
        af_level = float(autofluor.get(fl, 0.0))
        # autofluorescence is emitted BY the tissue, so it is illuminated like everything else
        af_prof = np.maximum(1.0 + af_cv * _smooth_unit(tape["w_af"][k], sig_px), 0.0)
        af = af_level * af_prof

        # marker photoelectrons
        sig_e = e_per_unit * img_psf[..., k] * ill
        # autofluorescence photoelectrons
        bg_e = e_per_unit * af * ill
        e = np.maximum(sig_e + bg_e, 0.0)

        if not bg_subtract:
            # raw-pedestal camera, bit-for-bit the old behaviour
            # shot noise scales as sqrt(signal): bright pixels are noisier
            noisy = e + np.sqrt(e) * tape["z_shot"][k] + read_e * tape["z_read"][k] + dark_e
            # unclipped, unrounded ADU -- clip/round happen once, after the optional cycle shift
            adu[..., k] = noisy * adu_per_e + offset_adu
            continue

        # staged-pipeline domain: acquire the marker frame (with hot pixels), then an independent
        # blank/bleach reference of the SAME autofluorescence, and subtract. DAPI is the
        # registration target -- it has no blank, so it keeps its pedestal and is not subtracted.
        z_read_m = _corr_noise(tape["z_read"][k], read_smooth)
        hot = hot_e * (tape["u_hot"][k] < hot_rate)
        marker = e + np.sqrt(e) * tape["z_shot"][k] + read_e * z_read_m + dark_e + hot
        marker_adu = marker * adu_per_e + offset_adu

        if fl == "DAPI":
            # the staged DAPI is offset-corrected to ~10 ADU on glass, not the camera pedestal
            dapi_off = det.DAPI_OFFSET_ADU.v if hasattr(det, "DAPI_OFFSET_ADU") else offset_adu
            adu[..., k] = marker_adu - offset_adu + dapi_off
            continue

        # the blank images the same AF, bleach-drifted (brighter by bg_drift, so AF over-subtracts)
        # and misregistered by bg_shift_px in a per-channel random direction, with its own noise
        sh = tape["af_shift"][k]
        nrm = float(np.hypot(sh[0], sh[1])) + 1e-6
        dy, dx = bg_shift_px * sh[0] / nrm, bg_shift_px * sh[1] / nrm
        drift = max(0.0, 1.0 + bg_drift * (1.0 + 0.5 * float(tape["z_bg_drift"][k])))
        af_b = drift * af_level * np.maximum(
            ndi.shift(af_prof, (dy, dx), order=1, mode="reflect"), 0.0)
        bg_e_b = np.maximum(e_per_unit * af_b * ill, 0.0)
        z_read_b = _corr_noise(tape["z_read_bg"][k], read_smooth)
        blank = bg_e_b + np.sqrt(bg_e_b) * tape["z_shot_bg"][k] + read_e * z_read_b + dark_e
        blank_adu = blank * adu_per_e + offset_adu
        # subtract + a small over-subtraction margin; the final np.clip(., 0, .) rectifies at 0,
        # leaving the near-zero floor with sparse positive speckle the staged images show
        adu[..., k] = marker_adu - blank_adu - bg_margin

    if cycle_of is not None:
        adu = apply_cycle_misregistration(adu, cycle_of, cycle_offsets, cycle_mode)

    np.clip(adu, 0.0, 2 ** bit_depth - 1, out=adu)
    if quantise:
        np.rint(adu, out=adu)
    return adu