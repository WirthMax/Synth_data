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


def psf_project(slice_vol, z_um, optics, fluorophore = None):
    nz = slice_vol.shape[0]
    # the dye is a property of the marker, so the caller hands it over directly
    wavelength_um = optics.wavelength_um if fluorophore is None else FLUOROPHORES[fluorophore]
    w = quad_weights(nz)
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
    # single mask case
    mask = np.asarray(mask)
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


def illumination(shape, illum_cv, vignette, tape):
    """Multiplicative flat-field: a smooth random component times a radial falloff."""
    ny, nx = shape
    rnd = 1.0 + illum_cv * _smooth_unit(tape["w_ill"], 0.25 * max(ny, nx))
    yy = (np.arange(ny) - (ny - 1) / 2.0)[:, None] / max(ny / 2.0, 1)
    xx = (np.arange(nx) - (nx - 1) / 2.0)[None, :] / max(nx / 2.0, 1)
    rr2 = np.clip((yy ** 2 + xx ** 2) / 2.0, 0.0, 1.0)     # 0 at centre, 1 at the corners
    return np.maximum(rnd * (1.0 - vignette * rr2), 0.0).astype(np.float32)


def detector(img_psf, profile, det, opt, tape, markers, quantise=False,
             autofluor = AUTOFLUOR):
    """Turn a clean PSF projection (H, W, C) into a detector frame.
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
    
    img_psf = np.asarray(img_psf, np.float32)
    ny, nx, nc = img_psf.shape
    if tape.z_shot.shape != (nc, ny, nx):
        raise ValueError(
            f"sensor tape is {tape.z_shot.shape} but the image is {(nc, ny, nx)}. "
            f"Re-run tape.drawSensor(shape=({ny}, {nx}, {nc})) -- and set IMG to match.")
    
    ill = illumination((ny, nx), illum_cv, vignette, tape)
    sig_px = max(af_scale_um / opt.um_per_px, 0.5)

    adu = np.empty_like(img_psf)
    for k, m in enumerate(markers):
        fl = profile.Markers[m].fluorophore
        af_level = float(autofluor.get(fl, 0.0))
        # autofluorescence is emitted BY the tissue, so it is illuminated like everything else
        af = af_level * np.maximum(1.0 + af_cv * _smooth_unit(tape["w_af"][k], sig_px), 0.0)

        # marker photoelectrons
        sig_e = e_per_unit * img_psf[..., k] * ill
        # autofluorescence photoelectrons
        bg_e = e_per_unit * af * ill
        e = np.maximum(sig_e + bg_e, 0.0)
        # shot noise scales as sqrt(signal): bright pixels are noisier
        noisy = e + np.sqrt(e) * tape["z_shot"][k] + read_e * tape["z_read"][k] + dark_e
        a = np.clip(noisy * adu_per_e + offset_adu, 0.0, 2 ** bit_depth - 1)
        adu[..., k] = np.rint(a) if quantise else a
    return adu