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

def _sigma_eff(sigma_pump, sigma_probe, *, dtype, device):
    s_p = torch.as_tensor(sigma_pump, dtype=dtype, device=device)
    s_r = torch.as_tensor(sigma_probe, dtype=dtype, device=device)
    # (1/σ_eff^2) = (1/σ_p^2) + (1/σ_r^2)
    return ((1.0/(s_p*s_p) + 1.0/(s_r*s_r)) ** -0.5)

def _get_cached_kernel(sigma_eff, dtype, device):
    # round a bit to stabilize the cache key against tiny fp jitter
    key = (float(torch.round(sigma_eff*1e6)/1e6), str(dtype), device)
    if key in _GAUSS_KERNEL_CACHE:
        return _GAUSS_KERNEL_CACHE[key]
    r = int(torch.ceil(torch.tensor(4.0, dtype=dtype, device=device) * sigma_eff).item())
    x = torch.arange(-r, r+1, device=device, dtype=dtype)
    k = torch.exp(-(x*x) / (2*sigma_eff*sigma_eff))
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
    # xHW: [H,W]  --> returns blurred [H,W], fully differentiable
    # x = F.pad(xHW.unsqueeze(0).unsqueeze(0), (r, r, r, r), mode='reflect')
    f = F.pad(xHW.unsqueeze(0).unsqueeze(0), (r, r, r, r), mode='reflect')[0, 0]

    kv = k1d.view(1,1,-1,1)
    kh = k1d.view(1,1,1,-1)
    x = F.conv2d(x, kv)  # vertical 1-D
    x = F.conv2d(x, kh)  # horizontal 1-D
    return x[0,0]

def _fft_blur2d(xHW, sigma_eff, r):
    # 2-D FFT path for very large kernels (still autograd-friendly)
    H, W = xHW.shape
    y = torch.arange(-r, r+1, device=xHW.device, dtype=xHW.dtype)
    x = torch.arange(-r, r+1, device=xHW.device, dtype=xHW.dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    k = torch.exp(-(X*X + Y*Y) / (2*sigma_eff*sigma_eff))
    k = k / k.sum()

    f = F.pad(xHW, (r, r, r, r), mode='reflect')
    Ff = torch.fft.rfft2(f)

    K = torch.zeros_like(f)
    K[:2*r+1, :2*r+1] = k
    K = torch.roll(K, shifts=(-r, -r), dims=(0,1))
    FK = torch.fft.rfft2(K)

    out = torch.fft.irfft2(Ff * FK, s=f.shape[-2:])
    return out[r:H+r, r:W+r]
# === END HELPERS ===



def smooth_downsample_torch_from_tensor(param_map_full, steps_y, steps_x, sigma_pump=8.0, sigma_probe=9.0, device=None):
    """
    Fast Gaussian smoothing + downsample:
    - Uses separable 1-D Gaussian passes (mathematically identical to 2-D Gaussian).
    - Caches the 1-D kernel and the sampling indices.
    - Automatically switches to FFT convolution for very large kernels.
    Returns: torch.Tensor [steps_y, steps_x], keeps gradients.
    """
    if device is None:
        device = param_map_full.device
    assert param_map_full.dim() == 2, "param_map_full should be [H, W]"

    # ensure tensor is on device/dtype we want
    xHW = param_map_full.to(device)

    H, W = xHW.shape
    dtype = xHW.dtype

    # 1) compute effective sigma for Gaussian × Gaussian (pump×probe)
    sigma_eff = _sigma_eff(sigma_pump, sigma_probe, dtype=dtype, device=device)

    # 2) get 1-D kernel (cached) and radius r = 4*sigma
    k1d, r = _get_cached_kernel(sigma_eff, dtype, device)

    # 3) choose separable or FFT path
    if r > 128:  # threshold; tweak if you like
        blurred = _fft_blur2d(xHW, sigma_eff, r)
    else:
        blurred = _separable_blur2d(xHW, k1d, r)

    # 4) downsample by sampling blurred map at the same centers as before (cached)
    y_idx, x_idx = _get_cached_indices(H, W, steps_y, steps_x, device)
    return blurred.index_select(0, y_idx).index_select(1, x_idx)













def smooth_downsample_torch_from_tensor_slow(param_map_full, steps_y, steps_x, sigma_pump=8.0, sigma_probe=9.0, device=None):
    

    # Performs Gaussian smoothing and downsampling on a full-resolution parameter map using a PyTorch tensor input.
    # Keeps computation graph intact for gradient flow.
    
    # Parameters:
    # - param_map_full: torch.Tensor of shape [H, W], requiring gradients
    # - steps_y, steps_x: target output shape
    # - sigma_pump, sigma_probe: Gaussian standard deviations
    # - device: CUDA/CPU
    
    # Returns:
    # - torch.Tensor of shape [steps_y, steps_x]

    
    if device is None:
        device = param_map_full.device

    param_map_full = param_map_full.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
    pixels_y, pixels_x = param_map_full.shape[-2:]

    # Define kernel
    max_sigma = max(sigma_pump, sigma_probe)
    r = int(np.ceil(4 * max_sigma))
    kernel = gaussian_kernel_2d(sigma_pump, sigma_probe, r).to(device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, kH, kW]

    # Pad the input
    pad = [r, r, r, r]  # left, right, top, bottom
    param_padded = F.pad(param_map_full, pad, mode='reflect')

    # Convolve (Gaussian smoothing)
    smoothed = F.conv2d(param_padded, kernel)

    # Compute sampling coordinates
    x_centers = torch.linspace(0, pixels_x - 1, steps_x, device=device).round().long()
    y_centers = torch.linspace(0, pixels_y - 1, steps_y, device=device).round().long()

    # Sample from smoothed output — maintains gradient flow
    sampled = smoothed[0, 0][y_centers[:, None], x_centers[None, :]]
    return sampled




def gaussian_kernel_2d(sigma_pump, sigma_probe, r):
    
    # Construct a multiplicative pump × probe Gaussian kernel.
    
    size = 2 * r + 1
    y, x = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing='ij')
    dist_sq = x**2 + y**2

    pump = torch.exp(-dist_sq / (2 * sigma_pump**2))
    probe = torch.exp(-dist_sq / (2 * sigma_probe**2))
    kernel = pump * probe
    kernel /= kernel.sum()  # Normalize to unit sum
    return kernel
    


def smooth_downsample_torch(param_map_full_np, steps_y, steps_x, sigma_pump=8.0, sigma_probe=9.0, device=None):
    

    # PyTorch implementation of Gaussian smoothing and downsampling.
    # Uses convolution for fast GPU execution.

    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert to torch tensor
    param_map_full = torch.tensor(param_map_full_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

    pixels_y, pixels_x = param_map_full.shape[-2:]

    # Define kernel
    max_sigma = max(sigma_pump, sigma_probe)
    r = int(np.ceil(4 * max_sigma))
    kernel = gaussian_kernel_2d(sigma_pump, sigma_probe, r).to(device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, kH, kW]

    # Pad the input to keep edges valid
    pad = [r, r, r, r]  # left, right, top, bottom
    param_padded = F.pad(param_map_full, pad, mode='reflect')

    # Convolve (Gaussian smoothing)
    smoothed = F.conv2d(param_padded, kernel)




    # This just takes the coarse/convoluted version at pixels_y, pixels_x down to steps_y steps_x
    # # Resize using bilinear interpolation to match desired downsampling
    # smoothed_down = F.interpolate(smoothed, size=(steps_y, steps_x), mode='bilinear', align_corners=False)
    
    # return smoothed_down.squeeze().cpu().numpy()  # shape: [steps_y, steps_x]
    
    
    # More directly similar to other codes, the convolution only at each x/y center is used to make steps_y, steps_x map.
    # Compute sampling coordinates (center points)
    x_centers = torch.linspace(0, pixels_x - 1, steps_x, device=device).round().long()
    y_centers = torch.linspace(0, pixels_y - 1, steps_y, device=device).round().long()

    # Sample directly from smoothed map
    sampled = smoothed[0, 0][y_centers[:, None], x_centers[None, :]]
    return sampled.cpu().numpy()





def smooth_downsample_parallel(param_map_full, steps_y, steps_x, sigma_pump=50.0, sigma_probe=50.0, n_jobs=-1):
    pixels_y, pixels_x = param_map_full.shape
    ny, nx = steps_y, steps_x

    x_centers = np.linspace(0, pixels_x - 1, steps_x)
    y_centers = np.linspace(0, pixels_y - 1, steps_y)

    max_sigma = max(sigma_pump, sigma_probe)
    r = int(np.ceil(4 * max_sigma))
    kernel_size = 2 * r + 1

    Y_indices, X_indices = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing='ij')
    dist_sq = X_indices**2 + Y_indices**2

    gaussian_pump = np.exp(-dist_sq / (2 * sigma_pump**2))
    gaussian_probe = np.exp(-dist_sq / (2 * sigma_probe**2))
    gaussian_weights = gaussian_pump * gaussian_probe

    def process_pixel(i, j, yc, xc):
        yc_i = int(round(yc))
        xc_j = int(round(xc))

        y_min = max(yc_i - r, 0)
        y_max = min(yc_i + r + 1, pixels_y)
        x_min = max(xc_j - r, 0)
        x_max = min(xc_j + r + 1, pixels_x)

        sub_map = param_map_full[y_min:y_max, x_min:x_max]

        ky_min = r - (yc_i - y_min)
        ky_max = r + (y_max - yc_i)
        kx_min = r - (xc_j - x_min)
        kx_max = r + (x_max - xc_j)

        weights_crop = gaussian_weights[ky_min:ky_max, kx_min:kx_max]

        weighted_sum = np.sum(weights_crop * sub_map)
        total_weight = np.sum(weights_crop)
        return i, j, weighted_sum / total_weight if total_weight > 0 else 0.0

    # Generate all tasks
    tasks = [
        (i, j, yc, xc)
        for i, yc in enumerate(y_centers)
        for j, xc in enumerate(x_centers)
    ]

    # Run in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(process_pixel)(i, j, yc, xc) for i, j, yc, xc in tasks)

    # Fill the output array
    param_downsampled = np.zeros((ny, nx), dtype=np.float32)
    for i, j, val in results:
        param_downsampled[i, j] = val

    return param_downsampled






def smooth_downsample(param_map_full, steps_y, steps_x, sigma_pump=50.0, sigma_probe=50.0):
    

    # Downsample a full-resolution parameter map using finite-support Gaussian smoothing.
    # Gaussian width is defined separately for pump and probe, and kernel support extends to 4 * sigma.

    
    # Extract how many pixels are in full resolution image
    pixels_y, pixels_x = param_map_full.shape
    
    # Resolution of blurred (FEM/Experimental) image
    ny, nx = steps_y, steps_x

    # Create downsample grid
    # Centers of high resolution image that map to blurred image
    x_centers = np.linspace(0, pixels_x - 1, steps_x)
    y_centers = np.linspace(0, pixels_y - 1, steps_y)

    # Normalized Gaussian, gives weighting
    # Kernel size: max sigma * 4 (truncate when Gaussian ~ 0)
    max_sigma = max(sigma_pump, sigma_probe)
    r = int(np.ceil(4 * max_sigma))  # half-width
    kernel_size = 2 * r + 1          # force odd size so gaussian can be centered at each x_center/y_center

    # Precompute distance grid
    Y_indices, X_indices = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing='ij')
    dist_sq = X_indices**2 + Y_indices**2

    # Separate pump and probe Gaussians
    gaussian_pump = np.exp(-dist_sq / (2 * sigma_pump**2))
    gaussian_probe = np.exp(-dist_sq / (2 * sigma_probe**2))
    
    # Element-wise multiplication, this is now a gaussian weighted kernel ready for use
    gaussian_weights = gaussian_pump * gaussian_probe

    # Output map
    param_downsampled = np.zeros((ny, nx), dtype=np.float32)

    for i, yc in enumerate(y_centers):
        for j, xc in enumerate(x_centers):
            
            yc_i = int(round(yc))
            xc_j = int(round(xc))

            # Extract bounds from full image
            y_min = max(yc_i - r, 0)
            y_max = min(yc_i + r + 1, pixels_y)
            x_min = max(xc_j - r, 0)
            x_max = min(xc_j + r + 1, pixels_x)

            # Extract valid region from full image
            sub_map = param_map_full[y_min:y_max, x_min:x_max]

            # Match kernel crop
            ky_min = r - (yc_i - y_min)
            ky_max = r + (y_max - yc_i)
            kx_min = r - (xc_j - x_min)
            kx_max = r + (x_max - xc_j)
            weights_crop = gaussian_weights[ky_min:ky_max, kx_min:kx_max]

            # Weighted smoothing
            weighted_sum = np.sum(weights_crop * sub_map)
            total_weight = np.sum(weights_crop)
            param_downsampled[i, j] = weighted_sum / total_weight if total_weight > 0 else 0.0

    return param_downsampled




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
    # kappa_downsampled = smooth_downsample(
        # kappa_map_full,
        # steps_y=31,
        # steps_x=31,
        # sigma_pump=50.0,
        # sigma_probe=50.0
    # )   
    
    # kappa_downsampled = smooth_downsample_parallel(
        # kappa_map_full,
        # steps_y=31,
        # steps_x=31,
        # sigma_pump=50.0,
        # sigma_probe=50.0
    # )   
    
    kappa_downsampled = smooth_downsample_torch(
        kappa_map_full,
        steps_y=31,
        steps_x=31,
        sigma_pump=50,
        sigma_probe=50
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
    