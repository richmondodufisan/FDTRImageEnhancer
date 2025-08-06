import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Physical parameters
alpha_true = 0.5  # true thermal conductivity
L = 1.0           # length of the rod
T = 1.0           # total time
N_x = 50
N_t = 50

x = np.linspace(0, L, N_x)
t = np.linspace(0, T, N_t)
X, T_mesh = np.meshgrid(x, t)
X_flat = X.flatten()
T_flat = T_mesh.flatten()

# Analytical solution for initial-boundary conditions
def analytical_solution(x, t, alpha):
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)

u_exact = analytical_solution(X_flat, T_flat, alpha_true)
u_noisy = u_exact + 0.05 * np.random.randn(*u_exact.shape)  # simulate experimental noise

# Convert to torch
x_train = torch.tensor(X_flat, dtype=torch.float32).unsqueeze(1)
t_train = torch.tensor(T_flat, dtype=torch.float32).unsqueeze(1)
u_train = torch.tensor(u_noisy, dtype=torch.float32).unsqueeze(1)


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Learnable thermal conductivity
        self.alpha = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))  # initial guess

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))



def physics_loss(model, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    return torch.mean((u_t - model.alpha * u_xx) ** 2)



def train(model, x_train, t_train, u_train, epochs=5000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        optimizer.zero_grad()

        u_pred = model(x_train, t_train)
        loss_data = torch.mean((u_pred - u_train) ** 2)
        loss_phys = physics_loss(model, x_train, t_train)

        loss = loss_data + loss_phys
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Total Loss = {loss.item():.5f}, α = {model.alpha.item():.5f}")




model = PINN()
train(model, x_train, t_train, u_train)


with torch.no_grad():
    u_pred = model(x_train, t_train).numpy()

# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1)
# plt.title("Noisy Observed")
# plt.tricontourf(X_flat, T_flat, u_noisy, levels=100)
# plt.colorbar()

# plt.subplot(1,2,2)
# plt.title("PINN Predicted")
# plt.tricontourf(X_flat, T_flat, u_pred.flatten(), levels=100)
# plt.colorbar()
# plt.tight_layout()
# plt.savefig("noisy_v_PINN_temp_profile.png", dpi = 600)
# plt.show()


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
plt.title("Noisy Observed")
plt.tricontourf(X_flat, T_flat, u_noisy, levels=100)
plt.xlabel("x (position)")
plt.ylabel("t (time)")
cbar = plt.colorbar(label="Temperature")
cbar.ax.tick_params(labelsize=14)
plt.savefig("noisy_temp.png", dpi=600)

plt.figure()
plt.title("PINN Predicted")
plt.tricontourf(X_flat, T_flat, u_pred.flatten(), levels=100)
plt.xlabel("x (position)")
plt.ylabel("t (time)")
cbar = plt.colorbar(label="Temperature")
cbar.ax.tick_params(labelsize=14)
plt.savefig("PINN_temp.png", dpi=600)





# Range of alpha values to scan
alpha_vals = np.linspace(0.1, 1.0, 100)
pinn_total_losses = []
pinn_data_losses = []
pinn_phys_losses = []
analytic_losses = []


# Compute all losses across alpha values
for k in alpha_vals:
    
    # Set the current alpha in the model
    model.alpha.data = torch.tensor([k], dtype=torch.float32)

    # Evaluate PINN Losses
    u_pred = model(x_train, t_train)
    
    loss_data = torch.mean((u_pred - u_train) ** 2).item()
    loss_phys = physics_loss(model, x_train.clone(), t_train.clone()).item()
    loss_total = loss_data + loss_phys

    pinn_data_losses.append(loss_data)
    pinn_phys_losses.append(loss_phys)
    pinn_total_losses.append(loss_total)

    # Analytical Least Squares Loss
    u_analytic = np.sin(np.pi * X_flat) * np.exp(-k * np.pi**2 * T_flat)
    analytic_loss = np.mean((u_analytic - u_noisy) ** 2)
    analytic_losses.append(analytic_loss)


# Plot the loss surfaces
plt.figure(figsize=(10, 6))

plt.plot(alpha_vals, pinn_total_losses, label='PINN Total Loss')
# plt.plot(alpha_vals, pinn_data_losses, label='PINN Data Loss', linestyle='-.')
# plt.plot(alpha_vals, pinn_phys_losses, label='PINN Physics Loss', linestyle=':')
plt.plot(alpha_vals, analytic_losses, label='Least Squares (Analytic)', linestyle='--')

plt.axvline(alpha_true, color='red', linestyle=':', linewidth=2, label='True α')

plt.xlabel("Thermal Diffusivity (α)")
plt.ylabel("Loss Value")
plt.title("Loss Landscape Comparison")
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_landscape_comparison.png", dpi=600)
plt.show()


