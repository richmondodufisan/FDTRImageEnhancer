import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load data
df = pd.read_csv("kappa_log.csv")

# Rename columns for clarity
df.rename(columns={
    'kappa_0': 'kappa_left',
    'kappa_1': 'kappa_right',
    'kappa_2': 'kappa_GB'
}, inplace=True)


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

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(df['epoch'], df['kappa_left'], label=r'$\kappa_{left}$ (130 W/m·K)', color='red', s=8)
plt.scatter(df['epoch'], df['kappa_right'], label=r'$\kappa_{right}$ (100 W/m·K)', color='blue', s=8)
plt.scatter(df['epoch'], df['kappa_GB'], label=r'$\kappa_{GB}$ (75.5 W/m·K)', color='black', s=8)

# Ground truth horizontal lines with legend labels
plt.axhline(130, color='red', linestyle='--', linewidth=2, label=r'Ground Truth $\kappa_{left}$')
plt.axhline(100, color='blue', linestyle='--', linewidth=2, label=r'Ground Truth $\kappa_{right}$')
plt.axhline(75.5, color='black', linestyle='--', linewidth=2, label=r'Ground Truth $\kappa_{GB}$')

# Final touches
plt.xlabel("Epoch")
plt.ylabel("Thermal Conductivity (W/m·K)")
plt.title("Evolution of Thermal Conductivity Estimates")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.savefig("kappa_evolve.png")
plt.show()

