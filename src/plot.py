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
    Plots a tau image with a diverging colormap (e.g., RdGy).
    
    Parameters:
    - tau_img: 2D numpy array of tau values (typically -1 to 1).
    - background_mask: 2D boolean array where True indicates background. 
                       If None, assumes background is exactly 0.0.
    """
    # 1. Identify the background
    if background_mask is None:
        # Assuming background is exactly 0. 
        # (Be careful if actual cell coordinates can be exactly 0)
        background_mask = (tau_img == 0)
        
    # 2. Copy the image and set background to NaN to isolate it from valid tau values
    tau_display = tau_img.copy().astype(float)
    tau_display[background_mask] = np.nan
    
    
    # 3. Create a diverging colormap
    cmap = plt.get_cmap('RdGy').copy()
    
    # 4. Force NaN values (our background) to render as black
    cmap.set_bad(color='black')
    
    # 5. Plot the image
    fig, ax = plt.subplots()
    
    im = ax.imshow(tau_display, cmap=cmap, interpolation='nearest',
               vmin=-1, vmax=1)
    
    plt.colorbar(im, label='Tau Value')
    plt.axis('off')
    plt.show()
    return fig, ax
    