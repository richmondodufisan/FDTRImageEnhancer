import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from scipy.optimize import curve_fit
from Layered_Heat_Conduction import calc_thermal_response
import textwrap
import pdb
from joblib import Parallel, delayed


def get_material_properties(i, j, steps_x, steps_y, regions):
    
    for region in regions:
        x_range, y_range = region["x_range"], region["y_range"]
        scaled_i = y_range[1] * i / steps_y
        scaled_j = x_range[1] * j / steps_x
        
        if x_range[0] <= scaled_j < x_range[1] and y_range[0] <= scaled_i < y_range[1]:
            return region["material_properties"]
            
    return None
    
    
def fit_function_FDTR(freqs, fitting_properties, material_properties):
    phases = []

    kappa_2, conductance_12 = fitting_properties
    kappa_1, rho_1, c_1, rho_2, c_2 = material_properties

    for freq in freqs:
        # Define other parameters required by calc_thermal_response function
        # See comments in Layered_Heat_Conduction.py for documentation
        N_layers = 2
        layer2 = [100e-6, kappa_2, kappa_2, rho_2, c_2]
        layer1 = [90e-9, kappa_1, kappa_1, rho_1, c_1]
        layer_props = np.array([layer2, layer1])
        interface_props = [conductance_12]
        w_probe = 1.34e-6
        w_pump = 1.53e-6
        pump_power = 0.01
        freq = freq * 1e6

        # Calculate analytical phase 
        phase, _ = calc_thermal_response(N_layers, layer_props, interface_props, w_pump, w_probe, freq, pump_power)
        phases.append(phase)
        
    return np.array(phases)   
    


# In series (no parallelization)

# def create_analytical_thermal_image(steps_y, steps_x, freqs, raw_phase_map, regions):
    
    # fitting_map = np.zeros((steps_y, steps_x, 2)) # Assuming we're fitting 2 properties (kappa_2, conductance_12)
    
    # # Perform fitting at each position
    # for i in range(steps_y):
        # for j in range(steps_x):
            
            
            # # Get the material properties for the current position (i, j)
            # material_properties = get_material_properties(i, j, steps_x, steps_y, regions)
            # if material_properties is None:
                # print(f"No material properties found for position ({i}, {j})")
                # continue
            
            
            # FDTR_data = {
                # 'frequency': np.array(freqs),
                # 'phase': raw_phase_map[i, j, :]  # Adjust depending on tensor shape
            # }
            

            # # Define initial guesses and bounds for fitting properties
            # initial_guesses = [100, 30e6]  # Initial guesses for kappa_2 and conductance_12
            # bounds_lower = [0, 0]     # Lower bounds for the fitting properties
            # bounds_upper = [300, 500e6]  # Upper bounds for the fitting properties
            
            # # NB: use one initial guess/bounds for all regions, it'll find the correct value eventually

            # # pdb.set_trace()


                
            # def fit_wrapper(freqs, kappa_2, conductance_12):
                # return fit_function_FDTR(freqs, [kappa_2, conductance_12], material_properties)

            # # Fit the data to get the fitting properties
            # try:
                # popt, pcov = curve_fit(
                    # fit_wrapper,
                    # FDTR_data['frequency'],   # Frequency data
                    # FDTR_data['phase'],       # Phase data
                    # p0=initial_guesses,       # Initial guesses for the fitting properties
                    # bounds=(bounds_lower, bounds_upper),  # Bounds for the parameters
                    # method='trf',             # Trust Region Reflective algorithm
                    # maxfev=10000,             # Maximum function evaluations
                    # ftol=1e-12,
                    # xtol=1e-12,
                    # gtol=1e-12
                # )

                # fitting_map[i, j] = popt  # Store the fitted parameters (k_Si, conductance)
                # # pdb.set_trace()
                
                
                # # DEBUGGING - check fit
                # # Plot fit vs. experimental data for specific coordinate
                # if i == 0 and j == 0:  # Change this to the desired coordinate
                # # if (True):

                    # fit_phases = fit_function_FDTR((FDTR_data['frequency']), popt, material_properties)
                    
                    # # Define title text with LaTeX formatting and proper line breaks
                    # title_text = (
                        # f'Fit vs Experimental Data at ({i}, {j})\n'
                        # r'$\kappa$: ' + f'{popt[0]:.2f} W/(m$\cdot$K) '  # New line before κ
                        # r'G: ' + f'{(popt[1] / 1e6):.2f} MW/(m$^2\cdot$K)\n'  # New line before G
                        # r'MSE: ' + f'{np.mean((FDTR_data["phase"] - fit_phases) ** 2):.2e}'
                    # )
                    
                    
                    # plt.figure()
                    # plt.plot(FDTR_data['frequency'], FDTR_data['phase'], 'v-', label='Experimental', markersize=8)
                    # plt.plot(FDTR_data['frequency'], fit_phases, 'v-', label='Fit', markersize=8)
                    # plt.xscale('log')
                    # plt.xlabel('Frequency (Hz)')
                    # plt.ylabel('Phase (radians)')
                    # plt.title(title_text, fontsize=14, y=1.1)  # Adjusted title positioning
                    # plt.legend()
                    # plt.grid(True)
                    # plt.tight_layout()
                    # plt.savefig('fit_test.png')
                    # # plt.show()
                    
                    # # pdb.set_trace()

            # except Exception as e:
                # print(f"Fitting failed at position ({i}, {j}): {e}")
                # fitting_map[i, j] = np.nan

    # # Plot and Save Analytical κ and G (transposed for correct orientation)
    # fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # im1 = axs[0].imshow(fitting_map[:, :, 0], cmap="viridis", origin='lower')
    # axs[0].set_title("Analytical κ Map")
    # fig.colorbar(im1, ax=axs[0])

    # im2 = axs[1].imshow(fitting_map[:, :, 1], cmap="Reds", origin='lower')
    # axs[1].set_title("Analytical G Map")
    # fig.colorbar(im2, ax=axs[1])

    # plt.tight_layout()
    # plt.savefig("analytical_kappa_G.png")
    # plt.close()

            
                
    # return fitting_map[:, :, 0], fitting_map[:, :, 1]












# In Parallel

def create_analytical_thermal_image(steps_y, steps_x, freqs, raw_phase_map, regions):
    from joblib import Parallel, delayed

    fitting_map = np.zeros((steps_y, steps_x, 2), dtype=np.float32)

    def fit_pixel(i, j):
        material_properties = get_material_properties(i, j, steps_x, steps_y, regions)
        if material_properties is None:
            return (i, j, None)

        FDTR_data = {
            'frequency': np.array(freqs),
            'phase': raw_phase_map[i, j, :]
        }

        def fit_wrapper(freqs, kappa_2, conductance_12):
            return fit_function_FDTR(freqs, [kappa_2, conductance_12], material_properties)

        try:
            popt, _ = curve_fit(
                fit_wrapper,
                FDTR_data['frequency'],
                FDTR_data['phase'],
                p0=[100, 30e6],
                bounds=([0, 0], [300, 500e6]),
                method='trf',
                maxfev=5000,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10
            )
            return (i, j, popt)
        except Exception:
            return (i, j, None)

    # 🚀 Parallel fitting
    results = Parallel(n_jobs=-1)(delayed(fit_pixel)(i, j) for i in range(steps_y) for j in range(steps_x))

    for i, j, popt in results:
        if popt is not None:
            fitting_map[i, j] = popt
        else:
            fitting_map[i, j] = np.nan

    # Plot
    # fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # im1 = axs[0].imshow(fitting_map[:, :, 0], cmap="viridis", origin='lower')
    # axs[0].set_title("Analytical κ Map")
    # fig.colorbar(im1, ax=axs[0])
    
    # im2 = axs[1].imshow(fitting_map[:, :, 1], cmap="Reds", origin='lower')
    # axs[1].set_title("Analytical G Map")
    # fig.colorbar(im2, ax=axs[1])
    
    # plt.tight_layout()
    # plt.savefig("analytical_kappa_G.png", dpi=600)
    # plt.close()
    
    
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
    plt.imshow(fitting_map[:, :, 0], cmap="viridis", origin='lower')
    plt.title("Analytical κ Map")
    cbar = plt.colorbar(label="κ (W/(m·K))")
    cbar.ax.tick_params(labelsize=14)
    plt.savefig("Analytical_Kappa_Plot_Test.png", dpi=600)
    plt.show()
    
    plt.figure()
    plt.imshow(fitting_map[:, :, 1], cmap="Reds", origin='lower')
    plt.title("Analytical G Map")
    cbar = plt.colorbar(label="κ (W/(m·K))")
    cbar.ax.tick_params(labelsize=14)
    plt.savefig("Analytical_G_Plot_Test.png", dpi=600)
    plt.show()

    return fitting_map[:, :, 0], fitting_map[:, :, 1]
