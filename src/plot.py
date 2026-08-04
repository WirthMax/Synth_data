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
    
    # 3. Define the colors for each key point dynamically
    color_nodes = [
        (0.0, 'blue'),       # Maps to vmin (-1)
        (pos_zero, 'white'), # Maps to 0
    ]
    
    # If there is no outgrowth, cap the colormap at red at position 1.0
    if vmax == 1.0:
        color_nodes.append((1.0, 'red'))
    else:
        # If there is outgrowth, red is partway through, and yellow caps it
        color_nodes.append((pos_one, 'red'))
        color_nodes.append((1.0, 'yellow'))
    
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
    cbar_label = 'Tau (Nuc<0, Peri=0, Memb=1)'
    if vmax > 1.0:
        cbar_label = 'Tau (Nuc<0, Peri=0, Memb=1, Growth>1)'
        
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.axis('off')
    
    return fig, ax