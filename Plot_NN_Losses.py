import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load the CSV file
df = pd.read_csv("loss_log.csv")

# Global style settings for publication-quality figures
mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "savefig.dpi": 600,
    "savefig.format": "png",
    "savefig.bbox": "tight"
})

# Total Loss 
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['total_loss'], color='purple')
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.title("Total Loss vs Epoch")
plt.grid(True)
plt.tight_layout()
plt.savefig("total_loss.png")
plt.show()

# Data Loss 
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['data_loss'], color='green')
plt.xlabel("Epoch")
plt.ylabel("Data Loss")
plt.title("Data Loss vs Epoch")
plt.grid(True)
plt.tight_layout()
plt.savefig("data_loss.png")
plt.show()

# Physics Loss
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['physics_loss'], color='blue')
plt.xlabel("Epoch")
plt.ylabel("Physics Loss")
plt.title("Physics Loss vs Epoch")
plt.grid(True)
plt.tight_layout()
plt.savefig("physics_loss.png")
plt.show()

# Physics Loss (log scale)
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['physics_loss'], color='blue')
plt.yscale("log")  # Logarithmic y-axis
plt.xlabel("Epoch")
plt.ylabel("Physics Loss")
plt.title("Physics Loss vs Epoch (Log Scale)")
plt.grid(True, which="both")  # Show grid for both major and minor ticks
plt.tight_layout()
plt.savefig("physics_loss_log.png")
plt.show()

