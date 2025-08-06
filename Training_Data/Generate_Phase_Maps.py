import pandas as pd
import numpy as np
import os
import pdb
import math
import matplotlib.pyplot as plt
import csv
import math

# USER-DEFINED NOISE LEVEL
noise_level = 0.005  # Set to 0 for no noise; increase for stronger noise

# Read the CSV files into pandas DataFrames
FDTR_data = pd.read_csv('FDTR_StepFunction_out.csv', skiprows=1, names=['x0', 'frequency', 'imag_part', 'real_part'])

# Extract lists of unique frequencies (in MHz) and unique x0 values
FDTR_freq_vals = FDTR_data['frequency'].unique().tolist()
FDTR_x0_vals = FDTR_data['x0'].unique().tolist()

# Skip first and last four x0_vals, now you should have data from -15 to 15
FDTR_x0_vals = FDTR_x0_vals[4:-4]



############################################# CALCULATING PHASE VALUES FROM DATA #############################################

# Dictionary of actual data. Each x0 value has a list phases for every frequency
# Key is x0 value, Value is list of phases for all frequencies
# Formatted this way to make thermal conductivity fitting easier
FDTR_phase_data = {} 

for x0 in FDTR_x0_vals:
    phase_vals = []
    
    for freq in FDTR_freq_vals:
        
        # Filter the original DataFrame to get the subset DataFrame for the specific (x0, frequency) pair
        subset_df = FDTR_data[(FDTR_data['x0'] == x0) & (FDTR_data['frequency'] == freq)][['imag_part', 'real_part']]
        
        # Check if subset_df is not empty
        if not subset_df.empty:
        
            # Calculate phase and amplitude
            imag_val = subset_df['imag_part'].iloc[0]
            real_val = subset_df['real_part'].iloc[0]
            
            phase = math.atan2(imag_val, real_val)
            phase = np.degrees(phase)
        
            amplitude = math.sqrt(imag_val**2 + real_val**2)
        
            # Save phase values
            phase_vals.append(phase)
        
    FDTR_phase_data[x0] = phase_vals


# Make a phase plot
# First, regroup phases by frequency
phase_by_freq = []

for i in range(0, len(FDTR_freq_vals)):
    phase_values = []
    
    for x0 in FDTR_x0_vals:
        phase_values.append(FDTR_phase_data[x0][i])

    phase_by_freq.append(phase_values)
    
# print(FDTR_freq_vals)
    
# Next, subtract all phase values by the value of the phase furthest from the GB    
for i in range(0, len(FDTR_freq_vals)):
    arr = np.array(phase_by_freq[i])
    relative_phase = arr - arr[0]
    plt.plot(FDTR_x0_vals, relative_phase, marker='o', markersize=5, label=str(FDTR_freq_vals[i]) + "MHz")

plt.xlabel('Pump/Probe Position')
plt.ylabel('Relative Phase')
plt.title("Relative Phase vs Position")
plt.legend(title='Frequencies')
plt.grid(True)
plt.savefig(f"Phase_Profile.png", bbox_inches='tight', dpi = 600)
plt.show()
    
############################################# END CALCULATING PHASE VALUES FROM DATA #############################################






############################################# CREATE PHASE MAPS #############################################
# Assuming phase_by_freq and FDTR_freq_vals are already defined from your earlier code
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

n_points = len(phase_by_freq[0])  # Number of x points, e.g., 31

# Loop through each frequency and make square maps
for i, freq in enumerate(FDTR_freq_vals):
    phase_1d = np.array(phase_by_freq[i])
    phase_2d = np.tile(phase_1d, (n_points, 1))  # Repeat rows to form a square
    
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, phase_2d.shape)
        phase_2d += noise

    # Save the 2D phase map
    filename = os.path.join(output_dir, f"Phase_{int(freq)}MHz.txt")
    np.savetxt(filename, phase_2d, fmt="%.6f")

    print(f"Saved: {filename}")    
############################################# CREATE PHASE MAPS #############################################