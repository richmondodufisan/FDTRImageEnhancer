import numpy as np
import matplotlib.pyplot as plt

# Structure Dimensions (Fictional EBSD Image)
pixels_x = 10000
pixels_y = 10000
length_x = 36.0
length_y = 36.0
gb_thickness = 0.5  # physical thickness of the GB

# DERIVED PARAMETERS
dx = length_x / pixels_x
dy = length_y / pixels_y
gb_slope = 1.0  # 45 degree diagonal
gb_pixels = int(gb_thickness / np.sqrt(dx**2 + dy**2))

# GRID SETUP
X, Y = np.meshgrid(np.arange(pixels_x), np.arange(pixels_y))
m = gb_slope
c = pixels_y // 2 - m * (pixels_x // 2)  # Centered diagonal

# Distance from line: y = mx + c
dist = (Y - (m * X + c))

# Initialize structure map (default: 0 = material above GB)
structure = np.zeros((pixels_y, pixels_x), dtype=np.uint8)

# Assign material regions
structure[dist > gb_pixels / 2] = 2  # below GB
structure[np.abs(dist) <= gb_pixels / 2] = 1  # GB itself

plt.imsave("grain_structure.png", structure, cmap='bwr', origin='lower', dpi=600)