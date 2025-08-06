import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Directory containing the FEM .txt files
data_directory = "./"
data_file_prefix = "Phase_"

phase_range = ["1MHz", "2MHz", "4MHz", "6MHz", "8MHz", "10MHz"]

# Relative phase data container
relative_phase_profiles = []
x_indices = None
y_index = 4  # Choose center row for example (assuming 9x9)


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


def plot_phase_maps():
    global x_indices

    for freq in phase_range:
        filename = os.path.join(data_directory, f"{data_file_prefix}{freq}.txt")
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found.")
            continue

        # Load and convert from degrees to radians
        data_deg = np.loadtxt(filename)
        data_rad = np.radians(data_deg)

        # Save x-axis indices (assumes square map)
        if x_indices is None:
            x_indices = list(range(data_rad.shape[1]))
            
            
            

        # Plot full map
        plt.figure()
        plt.imshow(data_rad, cmap="magma", origin="lower")
        cbar = plt.colorbar(label="Phase (radians)")
        cbar.ax.tick_params(labelsize=14)
        plt.title(f"Raw Phase Map at {freq}")
        plt.xlabel("x-index")
        plt.ylabel("y-index")
        plt.tight_layout()
        plt.savefig(f"phase_map_{freq}.png", bbox_inches='tight', dpi = 600)
        plt.show()

        # Extract relative phase along the selected row
        row_phase = data_rad[y_index, :]
        relative_phase = row_phase - row_phase[0]
        relative_phase_profiles.append((freq, relative_phase))

    # Plot combined relative phase
    plt.figure()
    for freq, rel_phase in relative_phase_profiles:
        plt.plot(x_indices, rel_phase, marker='o', label=f"{freq}")

    plt.xlabel("x-index")
    plt.ylabel("Relative Phase (degrees)")
    plt.title(f"Relative Phase vs Position (Row y={y_index})")
    plt.legend(title="Frequencies")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Relative_Phase_Profile_Noisy.png", bbox_inches='tight', dpi = 600)
    plt.show()

if __name__ == "__main__":
    plot_phase_maps()
