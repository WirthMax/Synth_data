import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

def plot_tau(tau_img, background_mask=None):
    """
    Plots a tau image with a custom colormap handling original boundaries and outgrowth.
    
    Parameters:
    - tau_img: 2D numpy array of tau values (-1 to max_growth).
    - background_mask: 2D boolean array where True indicates background. 
    - max_growth: The expected maximum value of the outgrowth region.
    """
    if background_mask is None:
        background_mask = (tau_img == 0)
        
    # Isolate valid values from background
    tau_display = tau_img.copy().astype(float)
    tau_display[background_mask] = np.nan
    
    # 1. Define the exact values we want to pin colors to
    vmin = -1.0
    vmax = np.max(tau_img) if np.max(tau_img) > 1.0 else 1.0  # Ensure we capture outgrowth
    
    # 2. Calculate their relative positions between 0.0 and 1.0 for the colormap
    total_range = vmax - vmin
    pos_zero = (0.0 - vmin) / total_range  # Where 0 sits in the 0-1 scale
    pos_one = (1.0 - vmin) / total_range   # Where 1 sits in the 0-1 scale
    
    # 3. Define the colors for each key point
    # (-1 = Blue, 0 = White, 1 = Red, max = Gold)
    color_nodes = [
        (0.0, 'blue'),       # Maps to vmin (-1)
        (pos_zero, 'white'), # Maps to 0
        (pos_one, 'red'),    # Maps to 1 (Membrane)
        (1.0, 'yellow')        # Maps to vmax (Outgrowth limits)
    ]
    
    # 4. Generate the custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('tau_custom', color_nodes)
    cmap.set_bad(color='black') # Set background to black
    
    # 5. Plot the image
    fig, ax = plt.subplots()
    
    im = ax.imshow(tau_display, 
                   cmap=cmap, 
                   interpolation='nearest',
                   vmin=vmin, 
                   vmax=vmax)
    
    # Add a descriptive colorbar
    cbar_label = f'Tau\n(Nuc<0, Peri=0, Memb=1, Growth>{1})'
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.axis('off')
    
    return fig, ax