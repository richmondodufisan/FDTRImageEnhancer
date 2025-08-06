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


def print_callback(xk):
    print(f"Step: sigma_pump={xk[0]:.4f}, sigma_probe={xk[1]:.4f}")


# --- Optimization objective ---
def sigma_loss(sigmas):
    sigma_pump, sigma_probe = sigmas

    # Enforce positivity
    if sigma_pump <= 0 or sigma_probe <= 0:
        return 1e9

    try:
        kappa_pred = smooth_downsample_torch(
            kappa_map_full,
            steps_y=kappa_analytical.shape[0],
            steps_x=kappa_analytical.shape[1],
            sigma_pump=sigma_pump,
            sigma_probe=sigma_probe,
            device='cuda'  # GPU enabled
        )
        mse = np.mean((kappa_pred - kappa_analytical) ** 2)
        return mse
    except Exception as e:
        print(f"Failure for sigmas {sigmas}: {e}")
        return 1e9


# --- Run optimization ---
initial_guess = [20.0, 20.0]
result = minimize(sigma_loss, initial_guess, method='Nelder-Mead', callback=print_callback, options={'maxiter': 100})

# --- Print and plot result ---
print("Best sigmas found:", result.x)
print("Final MSE:", result.fun)

# Visualize result
kappa_final = smooth_downsample_torch(
    kappa_map_full,
    steps_y=kappa_analytical.shape[0],
    steps_x=kappa_analytical.shape[1],
    sigma_pump=result.x[0],
    sigma_probe=result.x[1]
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
