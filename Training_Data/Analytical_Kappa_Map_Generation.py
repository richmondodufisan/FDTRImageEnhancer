import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from Analytical_Phase_Inversion import create_analytical_thermal_image


#################################### PHYSICS-BASED PHASE GENERATION ####################################

steps_x = 31
steps_y = 31
n_freqs = 6

data_directory = "./"
data_file_prefix = "Phase_"
phase_range = ["1MHz", "2MHz", "4MHz", "6MHz", "8MHz", "10MHz"]
frequencies = [1, 2, 4, 6, 8, 10]  # MHz

# Define rectangular regions of material properties
regions = [{"x_range": (0, 100), "y_range": (0, 100), "material_properties": [215, 19300, 128.5, 6180, 249.06]}]


def read_phase_data(file_name):
    data = np.loadtxt(file_name)
    return np.radians(data)  # degrees to radians

raw_phase_map = np.zeros((steps_y, steps_x, n_freqs))
raw_phase_map = raw_phase_map.astype(np.float32)


for i, freq in enumerate(phase_range):
    filename = os.path.join(data_directory, f"{data_file_prefix}{freq}.txt")
    raw_phase_map[:, :, i] = read_phase_data(filename)

kappa_map, G_map = create_analytical_thermal_image(
    steps_y=steps_y, steps_x=steps_x, freqs=frequencies, raw_phase_map=raw_phase_map, regions=regions
)

np.save("analytical_kappa_map.npy", kappa_map)
np.save("analytical_G_map.npy", G_map)


print("Saved analytical kappa map with shape:", kappa_map.shape)

