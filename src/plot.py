import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scene import _stretch, rotation_matrix




def tau_cmap(vmax=1.0):
    """The `plot_tau` colourmap, pinned to meaning: blue at tau=-1 (nucleus centre), white at
    0 (nuclear envelope), red at 1 (plasma membrane), yellow beyond (packing outgrowth)."""
    vmin, vmax = -1.0, max(float(vmax), 1.0)
    span = vmax - vmin
    nodes = [(0.0, "blue"), ((0.0 - vmin) / span, "white")]
    if vmax <= 1.0:
        nodes.append((1.0, "red"))
    else:
        nodes += [((1.0 - vmin) / span, "red"), (1.0, "yellow")]
    cm = mcolors.LinearSegmentedColormap.from_list("tau_custom", nodes)
    cm.set_bad(color="black")
    return cm, vmin, vmax


def plot_polar_phi(phi_img, background_mask=None):
    """
    Plots a polar coordinate image (phi) with a Cellpose-style rainbow colormap.
    
    Parameters:
    - phi_img: 2D numpy array of angles (typically -pi to pi or 0 to 2*pi).
    - background_mask: 2D boolean array where True indicates background. 
                       If None, assumes background is exactly 0.0.
    """
    # 1. Identify the background
    if background_mask is None:
        # Assuming background is exactly 0. 
        # (Be careful if actual cell coordinates can be exactly 0)
        background_mask = (phi_img == 0)
        
    # 2. Copy the image and set background to NaN to isolate it from valid angles
    phi_display = phi_img.copy().astype(float)
    phi_display[background_mask] = np.nan
    
    # 3. Create a cyclic HSV colormap
    cmap = plt.get_cmap('hsv').copy()
    
    # 4. Force NaN values (our background) to render as black
    cmap.set_bad(color='black')
    
    # 5. Plot the image
    fig, ax = plt.subplots()
    
    # vmin and vmax should match your angle bounds. 
    # Use -np.pi and np.pi if your angles are signed, or 0 and 2*np.pi if unsigned.
    im = ax.imshow(phi_display, cmap=cmap, vmin=-np.pi, vmax=np.pi, interpolation='nearest')
    
    plt.colorbar(im, label='Angle (Radians)')
    plt.axis('off')
    plt.show()
    return fig, ax

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_tau(tau_img, background_mask=None):
    """
    Plots a tau image with a custom colormap handling original boundaries and outgrowth.
    
    Parameters:
    - tau_img: 2D numpy array of tau values.
    - background_mask: 2D boolean array where True indicates background. 
    """
    if background_mask is None:
        background_mask = (tau_img == 0)
        
    # Isolate valid values from background
    tau_display = tau_img.copy().astype(float)
    tau_display[background_mask] = np.nan
    
    # 1. Define the exact values we want to pin colors to
    vmin = -1.0
    img_max = np.nanmax(tau_display)
    
    # Ensure vmax is at least 1.0 so we don't crush the standard cell colors
    vmax = img_max if img_max > 1.0 else 1.0  
    
    # 2. Calculate their relative positions between 0.0 and 1.0 for the colormap
    total_range = vmax - vmin
    pos_zero = (0.0 - vmin) / total_range  # Where 0 sits in the 0-1 scale
    pos_one = (1.0 - vmin) / total_range   # Where 1 sits in the 0-1 scale
    
    # 5. Plot the image
    fig, ax = plt.subplots()
    
    im = ax.imshow(tau_display, 
                   cmap=tau_cmap(vmax), 
                   interpolation='nearest',
                   vmin=vmin, 
                   vmax=vmax)
    
    # Add a descriptive colorbar
    cbar_label = 'Tau (Nuc<0, Peri=0, Memb=1)'
    if vmax > 1.0:
        cbar_label = 'Tau (Nuc<0, Peri=0, Memb=1, Growth>1)'
        
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.axis('off')
    
    return fig, ax

def surface_xyz_inline(rfn, elong, polar_deg, azim_deg, roll_deg, centre=(0, 0, 0), nt=160, npz=320):
    """Support function -> (X, Y, Z) grids for plot_surface, plus r for colouring."""
    th = np.linspace(0, np.pi, nt)
    ph = np.linspace(0, 2 * np.pi, npz)
    T, P = np.meshgrid(th, ph, indexing="ij")
    u = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1)
    r = rfn(u)                                                  # radius in the BODY frame
    p = r[..., None] * u                                        # 1. template point
    p = p * _stretch(elong)                                     # 2. stretch
    p = p @ rotation_matrix(polar_deg, azim_deg, roll_deg).T    # 3. rotate  (row vectors -> .T)
    p = p + np.asarray(centre, float)                           # 4. translate
    return p[..., 0], p[..., 1], p[..., 2], r


def plot_surface_xyz_inline(cell, elong, polar_deg, azim_deg, roll_deg):
    X,  Y,  Z,  r = surface_xyz_inline(cell["r_cell_fn"], elong = elong,
                                       polar_deg = polar_deg, azim_deg = azim_deg, 
                                       roll_deg = roll_deg
                                       )
    Xn, Yn, Zn, _ = surface_xyz_inline(cell["r_nuc_fn"], elong = elong, 
                                       polar_deg = polar_deg, azim_deg = azim_deg, 
                                       roll_deg = roll_deg,
                                       centre=cell["nuc_centre"]
                                       )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    n = (r - r.min()) / (np.ptp(r) + 1e-9)
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(n), rstride=2, cstride=2,
                    linewidth=0, antialiased=True, shade=False, alpha=0.45)
    ax.plot_surface(Xn, Yn, Zn, color="#111820", rstride=3, cstride=3, linewidth=0, shade=True)

    # equal aspect -- see below
    m   = np.array([X.mean(), Y.mean(), Z.mean()])
    rad = max(np.ptp(X), np.ptp(Y), np.ptp(Z)) / 2 * 1.05
    ax.set_xlim(m[0]-rad, m[0]+rad); ax.set_ylim(m[1]-rad, m[1]+rad); ax.set_zlim(m[2]-rad, m[2]+rad)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("3D Cell Boundary")
    plt.show()
    return fig

import plotly.graph_objects as go

def plot_surface_xyz_html(cell, elong, polar_deg, azim_deg, roll_deg, out_path):
    
    X,  Y,  Z,  r = surface_xyz_inline(cell["r_cell_fn"], elong = elong,
                                       polar_deg = polar_deg, azim_deg = azim_deg, 
                                       roll_deg = roll_deg
                                       )
    Xn, Yn, Zn, _ = surface_xyz_inline(cell["r_nuc_fn"], elong = elong, 
                                       polar_deg = polar_deg, azim_deg = azim_deg, 
                                       roll_deg = roll_deg,
                                       centre=cell["nuc_centre"]
                                       )
    fig = go.Figure()

    # 1. Add the Cell Membrane (Transparent, colored by radius)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=r,           # Color based on radius
        colorscale='Viridis',
        opacity=0.45,             # Matches your matplotlib alpha=0.45
        name='Cell Membrane',
        showscale=False           # Hides the colorbar
    ))

    # 2. Add the Nucleus (Solid dark color)
    # Plotly surfaces expect a colorscale. To make it a solid hex color like "#111820", 
    # we create a flat colorscale and pass a dummy surfacecolor array of zeros.
    solid_dark = [[0, '#111820'], [1, '#111820']]

    fig.add_trace(go.Surface(
        x=Xn, y=Yn, z=Zn,
        surfacecolor=np.zeros_like(Zn), 
        colorscale=solid_dark,
        opacity=1.0,
        name='Nucleus',
        showscale=False,
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2) # Approximates shade=True
    ))

    # 3. Configure the layout and 3D scene
    fig.update_layout(
        title="3D Cell Boundary",
        scene=dict(
            aspectmode='data',    # Automatically handles the equal aspect ratio bounding box!
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 4. Save to an interactive standalone HTML file
    output_file = "interactive_cell_3d.html"
    out_path.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path / output_file)
    print(f"Saved interactive plot to {output_file}")
    

def _centre(shape):
    """center of the cell volume"""
    return tuple(s // 2 for s in shape)

def ortho(vol, idx = None, cmap="viridis", vmin=None, vmax=None, title="", mask=None,
          figsize=(13, 4.4), axes=None):
    """Three orthogonal slices through the volume: XY (axial), XZ and YZ.

    `mask` (a boolean volume) blanks everything outside it to black, which is how the intrinsic
    coordinates should be read.
    """
    vol = np.asarray(vol)
    print(_centre(vol.shape))
    kz, ky, kx = idx if not idx is None else _centre(vol.shape)
    v = vol.astype(float)
    if mask is not None:
        v = np.where(mask, v, np.nan)
    if vmin is None:
        vmin = np.nanmin(v)
    if vmax is None:
        vmax = np.nanmax(v)
    cm = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
    cm.set_bad(color="black")

    panels = [(v[kz], f"XY  z={kz}", "x", "y"),
              (v[:, ky], f"XZ  y={ky}", "x", "z"),
              (v[:, :, kx], f"YZ  x={kx}", "y", "z")]
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig = axes[0].figure
    for ax, (im, t, xl, yl) in zip(axes, panels):
        h = ax.imshow(im, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest", origin="lower")
        ax.set_title(t, fontsize=10)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    fig.colorbar(h, ax=axes, fraction=0.025, pad=0.02)
    if title:
        fig.suptitle(title, fontsize=12)
    return fig, axes