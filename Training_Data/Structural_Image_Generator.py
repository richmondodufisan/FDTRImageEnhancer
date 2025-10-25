import numpy as np
import matplotlib.pyplot as plt

# USER INPUTS
pixels_x = 1500          # number of pixels in x-direction
pixels_y = 1500          # number of pixels in y-direction
length_x = 36.0         # physical length in x-direction
length_y = 36.0         # physical length in y-direction
gb_thickness = 0.1      # thickness of grain boundary

# DERIVED QUANTITIES
dx = length_x / pixels_x  # physical size per pixel in x-direction
gb_pixels = int(gb_thickness / dx)  # how many pixels thick the GB is

# center GB, calculate GB range
gb_start = pixels_x // 2 - gb_pixels // 2
gb_end = gb_start + gb_pixels

# GENERATE STRUCTURE
structure = np.zeros((pixels_y, pixels_x))  # matrix of zeros
structure[:, gb_start:gb_end] = 1  # mark GB as ones

# SAVE WITHOUT AXES, TITLE, or WHITESPACE
# Use blue-white-red colormap to color the blue (zeros) and red (ones)
plt.imsave("grain_structure.png", structure, cmap='bwr', origin='lower', dpi = 600)
