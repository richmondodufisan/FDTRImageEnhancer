import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
import torch
import torch.nn.functional as F
import numpy as np






# small, module-level caches so we don't rebuild stuff every step
_GAUSS_KERNEL_CACHE = {}   # key: (sigma_eff, dtype, device) -> (k: [K], r)
_INDEX_CACHE = {}          # key: (H, W, steps_y, steps_x, device) -> (y_idx, x_idx)

def _max_reflect_radius(H: int, W: int) -> int:
    # reflect pad requires r < min(H,W)
    return min(H, W) - 1


def _get_cached_kernel(sigma_eff, dtype, device):
    # ensure tensor scalar on the right device/dtype
    s = torch.as_tensor(sigma_eff, dtype=dtype, device=device)

    # round a bit to stabilize the cache key against tiny fp jitter
    key = (float(torch.round(s * 1e6) / 1e6), str(dtype), device)
    if key in _GAUSS_KERNEL_CACHE:
        return _GAUSS_KERNEL_CACHE[key]

    r = int(torch.ceil(torch.tensor(4.0, dtype=dtype, device=device) * s).item())
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2 * s * s))
    k = k / k.sum()
    _GAUSS_KERNEL_CACHE[key] = (k, r)
    return k, r


def _get_cached_indices(H, W, steps_y, steps_x, device):
    key = (H, W, steps_y, steps_x, device)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    y_idx = torch.linspace(0, H-1, steps_y, device=device).round().long()
    x_idx = torch.linspace(0, W-1, steps_x, device=device).round().long()
    _INDEX_CACHE[key] = (y_idx, x_idx)
    return y_idx, x_idx

def _separable_blur2d(xHW, k1d, r):
    # xHW: [H, W]  --> returns blurred [H, W], fully differentiable
    # Reflect pad expects NCHW; keep 4D for conv2d, then squeeze at the end.
    x = F.pad(xHW.unsqueeze(0).unsqueeze(0), (r, r, r, r), mode='reflect')  # [1,1,H+2r,W+2r]

    kv = k1d.view(1, 1, -1, 1)  # vertical kernel
    x = F.conv2d(x, kv)

    kh = k1d.view(1, 1, 1, -1)  # horizontal kernel
    x = F.conv2d(x, kh)

    return x[0, 0]  # back to [H, W]






def smooth_downsample_torch_from_tensor(param_map_full, steps_y, steps_x, sigma_eff, device=None):

    # Fast Gaussian smoothing + downsample:
    # - Uses separable 1-D Gaussian passes (mathematically identical to 2-D Gaussian).
    # - Caches the 1-D kernel and the sampling indices.
    # - Automatically switches to FFT convolution for very large kernels.
    # Returns: torch.Tensor [steps_y, steps_x], keeps gradients.
    # Tested, significantly faster

    if device is None:
        device = param_map_full.device
    assert param_map_full.dim() == 2, "param_map_full should be [H, W]"

    # ensure tensor is on device/dtype we want
    xHW = param_map_full.to(device)

    H, W = xHW.shape
    dtype = xHW.dtype

    # 2) get 1-D kernel (cached) and radius r = 4*sigma
    k1d, r = _get_cached_kernel(sigma_eff, dtype, device)
    
    # Safety: reflect pad limit
    if r >= _max_reflect_radius(H, W):
        raise ValueError(f"Gaussian radius r={r} exceeds reflect-pad limit for {H}x{W}. "
                         f"Use a smaller sigma_eff or normalize by image size.")


        
    # blur   
    blurred =  _separable_blur2d(xHW, k1d, r)   

    # 4) downsample by sampling blurred map at the same centers as before (cached)
    y_idx, x_idx = _get_cached_indices(H, W, steps_y, steps_x, device)
    return blurred.index_select(0, y_idx).index_select(1, x_idx)













def smooth_downsample_torch_from_tensor_slow(param_map_full, steps_y, steps_x, sigma_eff, device=None):
    

    # Performs Gaussian smoothing and downsampling on a full-resolution parameter map using a PyTorch tensor input.
    # Keeps computation graph intact for gradient flow.
    
    # Parameters:
    # - param_map_full: torch.Tensor of shape [H, W], requiring gradients
    # - steps_y, steps_x: target output shape
    # - sigma_eff = gaussian standard deviation
    # - device: CUDA/CPU
    
    # Returns:
    # - torch.Tensor of shape [steps_y, steps_x]

    
    if device is None:
        device = param_map_full.device

    x = param_map_full.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    H, W = x.shape[-2:]

    r = int(np.ceil(4 * float(sigma_eff)))
    if r >= _max_reflect_radius(H, W):
        raise ValueError(f"Gaussian radius r={r} exceeds reflect-pad limit for {H}x{W}.")

    kernel = gaussian_kernel_2d(sigma_eff, r).to(device).unsqueeze(0).unsqueeze(0)  # [1,1,kH,kW]

    x = F.pad(x, [r, r, r, r], mode='reflect')
    smoothed = F.conv2d(x, kernel)

    x_centers = torch.linspace(0, W-1, steps_x, device=device).round().long()
    y_centers = torch.linspace(0, H-1, steps_y, device=device).round().long()

    return smoothed[0,0][y_centers[:,None], x_centers[None,:]]




def gaussian_kernel_2d(sigma_eff, r):
    
    # Construct a multiplicative pump × probe Gaussian kernel.
    
    size = 2 * r + 1
    y, x = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing='ij')
    dist_sq = x**2 + y**2

    kernel = torch.exp(-dist_sq / (2 * sigma_eff**2))
    kernel /= kernel.sum()  # Normalize to unit sum
    return kernel
    





def smooth_downsample_torch(param_map_full_np, steps_y, steps_x, sigma_eff, device=None):
    # NumPy in, NumPy out; single sigma_eff in pixels
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.as_tensor(param_map_full_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    H, W = x.shape[-2:]

    r = int(np.ceil(4 * float(sigma_eff)))
    if r >= _max_reflect_radius(H, W):
        raise ValueError(f"Gaussian radius r={r} exceeds reflect-pad limit for {H}x{W}.")

    kernel = gaussian_kernel_2d(sigma_eff, r).to(device).unsqueeze(0).unsqueeze(0)  # [1,1,kH,kW]

    x = F.pad(x, [r, r, r, r], mode='reflect')
    smoothed = F.conv2d(x, kernel)

    x_centers = torch.linspace(0, W-1, steps_x, device=device).round().long()
    y_centers = torch.linspace(0, H-1, steps_y, device=device).round().long()

    sampled = smoothed[0,0][y_centers[:,None], x_centers[None,:]]
    return sampled.detach().cpu().numpy()







# If running forward pass of this file just to demonstrate effectiveness of Gaussian Smoother
if __name__ == "__main__":
    # Step 1: Load region map (assume already created by Structural_Image_Clustering)
    region_map = np.load("./Training_Data/region_map.npy")  # shape (1000, 1000)

    # Step 2: Assign dummy kappa values for each region
    kappa_region_0 = 130.0  # e.g. GB
    kappa_region_1 = 56.52  # e.g. bulk

    # Assume region labels are 0 and 1
    kappa_map_full = np.zeros_like(region_map, dtype=np.float32)
    kappa_map_full[region_map == 0] = kappa_region_0
    kappa_map_full[region_map == 1] = kappa_region_1


    # Step 3: Apply Gaussian smoothing-based downsampling 
    # Example: sigma_eff = 50 (pixels) for demo; in practice compute from normalized or calibrated value
    kappa_downsampled = smooth_downsample_torch(
        kappa_map_full,
        steps_y=31,
        steps_x=31,
        sigma_eff=50
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

    # Step 4: Visualize
    plt.figure()
    plt.imshow(kappa_map_full, cmap='viridis')
    plt.title("Original 1000×1000 κ Map")
    cbar = plt.colorbar(label="κ (W/(m·K))")
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig("Original_Structure.png", dpi = 600)
    plt.show()
    
    plt.figure()
    plt.imshow(kappa_downsampled, cmap='viridis')
    plt.title("Smoothed 31×31 κ Map")
    cbar = plt.colorbar(label="κ (W/(m·K))")
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig("Smoothed_Structure.png", dpi = 600)
    plt.show()
    