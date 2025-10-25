import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize
from GaussianSmoother import smooth_downsample_torch

# Load data
kappa_analytical = np.load("./Training_Data/analytical_kappa_map.npy")
region_map = np.load("./Training_Data/region_map.npy")

# Known parameters
kappa_bulk = 130.0
kappa_gb = 56.52

# Create full-resolution kappa map
kappa_map_full = np.zeros_like(region_map, dtype=np.float32)
kappa_map_full[region_map == 0] = kappa_bulk
kappa_map_full[region_map == 1] = kappa_gb

H, W = kappa_map_full.shape
min_dim = min(H, W)  # used in sigma guard: r = ceil(4*max_sigma) must be < min_dim



def print_callback(se):
    # se is a scalar when using 1-D optimization; handle array just in case
    se_val = float(se[0]) if hasattr(se, "__len__") else float(se)
    print(f"Step: sigma_eff={se_val:.6f} (pixels)")



# --- Optimization objective ---
def sigma_loss(sigma_eff):
    # scalar sigma_eff in pixels
    se = float(sigma_eff)
    if se <= 0.0:
        return 1e12

    # reflect-pad feasibility: r = ceil(4*se) < min_dim
    if 4.0 * se >= (min_dim - 1):
        return 1e12

    try:
        # If your smoother still expects (sigma_pump, sigma_probe),
        # pass sigma_eff to both:
        kappa_pred = smooth_downsample_torch(
            kappa_map_full,
            steps_y=kappa_analytical.shape[0],
            steps_x=kappa_analytical.shape[1],
            sigma_eff=se,
            device='cuda'
        )
        if hasattr(kappa_pred, "detach"):
            kappa_pred = kappa_pred.detach().cpu().numpy()
        mse = np.mean((kappa_pred - kappa_analytical) ** 2)
        return mse
    except Exception:
        return 1e12



# --- Run optimization ---
initial_guess = np.array([20.0])

result = minimize(
    lambda x: sigma_loss(x[0]),
    initial_guess,
    method='Nelder-Mead',
    callback=print_callback,
    options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-7, 'adaptive': True}
)



# --- Print and plot result ---
se = float(result.x[0])          
print("Best sigma_eff (pixels):", se)
print("Final MSE:", float(result.fun))

kappa_final = smooth_downsample_torch(
    kappa_map_full,
    steps_y=kappa_analytical.shape[0],
    steps_x=kappa_analytical.shape[1],
    sigma_eff=se,              
    device='cuda'
)




# Global style settings for publication-quality figures
mpl.rcParams.update({
    "font.size": 14,              # Default text size
    "axes.titlesize": 16,          # Title size
    "axes.labelsize": 16,          # Axes labels
    "xtick.labelsize": 14,         # X tick labels
    "ytick.labelsize": 14,         # Y tick labels
    "legend.fontsize": 14,         # Legend text
    "figure.titlesize": 18,        # Figure title size
    "axes.linewidth": 1.5,         # Thicker axes lines
    "xtick.direction": "in",       # Ticks pointing inward
    "ytick.direction": "in",
    "xtick.major.size": 6,         # Tick size
    "ytick.major.size": 6,
    "xtick.major.width": 1.2,      # Tick width
    "ytick.major.width": 1.2,
    "savefig.dpi": 600,            # High-resolution output
    "savefig.format": "png",       # You can also use 'pdf' or 'svg'
    "savefig.bbox": "tight"        # Trim white space
})


plt.figure()
plt.imshow(kappa_analytical, cmap='viridis')
plt.title("Analytical κ Map")
cbar = plt.colorbar(label="κ (W/(m·K))")
cbar.ax.tick_params(labelsize=14)
plt.savefig("Analytical_Kappa.png", dpi=600)
plt.show()

plt.figure()
plt.imshow(kappa_final, cmap='viridis')
plt.title("Best Gaussian Smoothed κ")
cbar = plt.colorbar(label="κ (W/(m·K))")
cbar.ax.tick_params(labelsize=14)
plt.savefig("Best_Smoothed_Kappa.png", dpi=600)
plt.show()

plt.figure()
plt.imshow(kappa_final - kappa_analytical, cmap='coolwarm')
plt.title("κ Error (Gaussian - Analytical)")
cbar = plt.colorbar(label="κ (W/(m·K))")
cbar.ax.tick_params(labelsize=14)
plt.savefig("Kappa_Smoothing_Error.png", dpi=600)
plt.show()
