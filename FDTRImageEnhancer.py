import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from GaussianSmoother import smooth_downsample_torch_from_tensor
from torch.amp import GradScaler, autocast
torch.backends.cudnn.benchmark = True  # lets cuDNN pick the fastest conv algo


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


#################################### STEP 1: PHYSICS-BASED PHASE GENERATION ####################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

steps_x = 31
steps_y = 31
n_freqs = 6

data_directory = "./Test_Data/"
data_file_prefix = "Phase_"
phase_range = ["1MHz", "2MHz", "4MHz", "6MHz", "8MHz", "10MHz"]
frequencies = [1, 2, 4, 6, 8, 10]  # MHz

# Read in raw phase map
def read_phase_data(file_name):
    data = np.loadtxt(file_name)
    return np.radians(data)  # degrees to radians

raw_phase_map = np.zeros((steps_y, steps_x, n_freqs))
for i, freq in enumerate(phase_range):
    filename = os.path.join(data_directory, f"{data_file_prefix}{freq}.txt")
    raw_phase_map[:, :, i] = read_phase_data(filename)


# Read analytical kappa map 
kappa_analytical = np.load("./Test_Data/analytical_kappa_map.npy")
G_analytical = np.load("./Test_Data/analytical_G_map.npy")

# Initialize fresh files
# with open("loss_log.csv", "w") as f:
    # f.write("epoch,total_loss,data_loss,physics_loss\n")

# with open("kappa_log.csv", "w") as f:
    # f.write("epoch,kappa_0,kappa_1,kappa_2\n")
    
    
# If continuing old training, continue from old files
# Only write headers if file does not already exist

if not os.path.exists("loss_log.csv"):
    with open("loss_log.csv", "w") as f:
        f.write("epoch,total_loss,data_loss,physics_loss\n")

if not os.path.exists("kappa_log.csv"):
    with open("kappa_log.csv", "w") as f:
        f.write("epoch,kappa_0,kappa_1,kappa_2\n")



#################################### STEP 2: STRUCTURAL REGION MAP ####################################

region_map = np.load("./Test_Data/region_map.npy")
pixels_y, pixels_x = region_map.shape
print(f"Image dimensions: pixels_y = {pixels_y}, pixels_x = {pixels_x}")

region_map = torch.tensor(region_map, dtype=torch.long).to(device)

n_struct_regions = torch.max(region_map).item() + 1  # assuming 0-indexed labels


#################################### STEP 3: MODEL DEFINITION ####################################

class FDTR_DenoisingNet(nn.Module):
    def __init__(self, n_struct_regions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.kappa = nn.Parameter(torch.ones(n_struct_regions) * 100)
        self.G = nn.Parameter(torch.tensor([G_analytical.mean()], dtype=torch.float32))
        
        # In the future, maybe fit G per-region as well
        # Can do so by adding another physics/parameter loss function to compare G maps
        # self.G = nn.Parameter(torch.ones(n_struct_regions) * G_init)

    def forward(self, x, y, f, region_map_flat):
        input_tensor = torch.cat([x, y, f], dim=1)
        output = self.net(input_tensor)
        return output

    def get_kappa_map(self, region_map_flat):
        return self.kappa[region_map_flat]


#################################### STEP 4: HELPER FUNCTIONS ####################################

  
    
def calc_data_loss(model, coords_tensor, freq_tensor, region_map_flat, raw_phase_map_tensor):
    

    # Computes the MSE loss between model predictions and raw phase map

    phase_pred = model(coords_tensor[:, 0:1], coords_tensor[:, 1:2], freq_tensor, region_map_flat)
    return F.mse_loss(phase_pred.squeeze(), raw_phase_map_tensor.squeeze())



def calc_physics_loss(model, region_map, kappa_analytical, steps_y, steps_x):
    
    # Compares structurally-constrained smoothed model kappa to already-downsampled analytical kappa.
    device = model.kappa.device  # ensure consistent device usage
    analytical_kappa_tensor = torch.tensor(kappa_analytical, dtype=torch.float32).to(device)

    # Create full-resolution map by mapping region indices to model parameters
    kappa_map_full = model.kappa[region_map].reshape(region_map.shape).to(device)  # shape: [pixels_y, pixels_x]

    # Use the version that maintains gradient flow
    smoothed_kappa = smooth_downsample_torch_from_tensor(
        kappa_map_full,
        steps_y=steps_y,
        steps_x=steps_x,
        sigma_pump=346.88457173,
        sigma_probe=320.35648512,
        device=device
    )

    return F.mse_loss(smoothed_kappa, analytical_kappa_tensor)





def save_checkpoint(model, optimizer, epoch, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth")

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")




def load_checkpoint(model, optimizer, checkpoint_dir="checkpoints"):
    # Look for latest checkpoint in folder
    if not os.path.exists(checkpoint_dir):
        
        print("No checkpoint directory found, starting from scratch")
        return 0

    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pth")]
    if not ckpts:
        
        print("No checkpoints found in directory, starting from scratch")
        return 0

    # Sort by epoch number (after 'checkpoint_' and before '.pth')
    
    ckpts.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    latest_ckpt = ckpts[-1]
    latest_path = os.path.join(checkpoint_dir, latest_ckpt)

    checkpoint = torch.load(latest_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1

    print(f"Resumed from {latest_ckpt} at epoch {checkpoint['epoch']}")
    return start_epoch








def visualize_params(model, region_map, pixels_y, pixels_x, epoch):

    # Constructs the full kappa map using model.kappa and region_map,
    # then visualizes and saves it.

    with torch.no_grad():
        # Build full kappa map from model parameters
        kappa_map_full = model.kappa[region_map].cpu().numpy()

        # Get predicted G
        G_val = model.G.item()

        # Plot
        plt.figure(figsize=(6, 5))
        plt.imshow(kappa_map_full, cmap='plasma', origin='lower')
        cbar = plt.colorbar(label="Predicted κ (W/(m·K))")
        cbar.ax.tick_params(labelsize=14)
        plt.title(f"Thermal Conductivity (κ) Map (Epoch {epoch})\nPredicted G = {G_val:.2e}")
        plt.axis('off')

        # Save
        os.makedirs("param_visuals", exist_ok=True)
        plt.savefig(f"param_visuals/kappa_map_epoch_{epoch:05d}.png", bbox_inches='tight', dpi = 600)
        plt.close()



#################################### STEP 5: TRAINING ####################################

# Step 5: Training

# Prepare inputs
coords = torch.stack(torch.meshgrid(
    torch.linspace(0, 1, steps_y),
    torch.linspace(0, 1, steps_x),
    indexing='ij'), dim=-1).reshape(-1, 2).to(device)  # [N, 2]
freqs = torch.tensor(frequencies, dtype=torch.float32).view(1, -1).repeat(coords.shape[0], 1).reshape(-1, 1).to(device)  # [N * F, 1]
coords = coords.repeat(n_freqs, 1)  # [N * F, 2]

# Region map flat (make it a 1D vector for use in NN)
region_map_flat = region_map.reshape(-1).repeat(n_freqs)

# Raw phase map
raw_phase_map_tensor = torch.tensor(raw_phase_map, dtype=torch.float32).reshape(-1, 1).to(device)

# Model
model = FDTR_DenoisingNet(n_struct_regions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Training loop
def train_model(model, coords, freqs, region_map, region_map_flat, raw_phase_map_tensor,
                kappa_analytical, steps_y, steps_x, pixels_y, pixels_x, epochs=1000, lr=1e-2):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    scaler = GradScaler()
    
    start_epoch = load_checkpoint(model, optimizer)
    
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()


        with autocast(device_type='cuda'):
            data_loss = calc_data_loss(model, coords, freqs, region_map_flat, raw_phase_map_tensor)
            physics_loss = calc_physics_loss(model, region_map, kappa_analytical, steps_y, steps_x)

            lambda_data = 1.0
            lambda_physics = 1.0
            
            loss = (lambda_data * data_loss) + (lambda_physics * physics_loss)


        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        
        
        if epoch % 50 == 0:
            
            # Save checkpoint in case it crashes
            save_checkpoint(model, optimizer, epoch)
            
            # Print and write loss & parameter evolution
            kappa_values = model.kappa.detach().cpu().numpy()
            kappa_str = ", ".join([f"kappa_{i} = {v:.5f}" for i, v in enumerate(kappa_values)])
            
            print(f"Epoch {epoch} | Total Loss: {loss.item():.6e} | Data Loss: {data_loss.item():.6e} | Physics Loss: {physics_loss.item():.6e} | {kappa_str}")
            
            visualize_params(model, region_map, pixels_y, pixels_x, epoch)

            with open("kappa_log.csv", "a") as f:
                f.write(f"{epoch}," + ",".join([f"{v:.8e}" for v in kappa_values]) + "\n")
                
            with open("loss_log.csv", "a") as f:
                f.write(f"{epoch},{loss.item():.6e},{data_loss.item():.6e},{physics_loss.item():.6e}\n")

        



#################################### STEP 6: EXECUTION ####################################

if __name__ == "__main__":
    model = FDTR_DenoisingNet(n_struct_regions).to(device)
    
    train_model(model, coords, freqs, region_map, region_map_flat, 
                raw_phase_map_tensor, kappa_analytical, 
                steps_y, steps_x, pixels_y, pixels_x,
                epochs=35000, lr=1e-2)
                
    print("Final kappa values:", model.kappa.data.cpu().numpy())
    print("Final G value:", model.G.item())

