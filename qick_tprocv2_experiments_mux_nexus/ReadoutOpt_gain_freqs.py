import numpy as np
import matplotlib.pyplot as plt
import os
from section_005_single_shot_ge import GainFrequencySweep
import datetime
import h5py
import time

from system_config import QICK_experiment

date_str = str(datetime.date.today())
outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30", str(datetime.date.today()))#f"/data/QICK_data/6transmon_run4a/{date_str}/readout_opt/Gain_Freq_Sweeps/"

# Ensure the output folder exists
#os.makedirs(outerFolder, exist_ok=True)

def create_folder_if_not_exists(folder_path):
    """Creates a folder at the given path if it doesn't already exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

create_folder_if_not_exists(outerFolder)

# Reference frequencies for each resonator in MHz
res_freq_ge = [6187.191, 5827.678, 6074.095, 5958.673], #MHz #[6191.439, 6216.0, 6292.261, 6405.79, 6432.899, 6468.501]

optimal_lengths = [2.53, 9.66, 3.50, 4.75, 4.57, 6.60]

# Define sweeping parameters
gain_range = [0.1,0.5]  # Gain range in a.u.
freq_steps = 40
gain_steps = 26

Qs = [3]#[2,3]
for QubitIndex in Qs:
#QubitIndex= 0
    print(f'Starting Qubit {QubitIndex + 1} measurements.')
    # Select the reference frequency for the current qubit
    print('QubitIndex',QubitIndex)
    print('res_freq_ge',res_freq_ge[0])
    reference_frequency = res_freq_ge[0][QubitIndex-1]#res_freq_ge[QubitIndex-1]#res_freq_ge[QubitIndex]
    freq_range = [reference_frequency-1, reference_frequency + 1]  # Frequency range in MHz

    experiment = QICK_experiment(outerFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10)
    sweep = GainFrequencySweep(QubitIndex, experiment , optimal_lengths=optimal_lengths, output_folder=outerFolder)
    results = sweep.run_sweep(freq_range, gain_range, freq_steps, gain_steps)
    results = np.array(results)

    # Save results and metadata in an HDF5 file
    timestamp = time.strftime("%H%M%S")
    h5_file = f"{outerFolder}Gain_Freq_Sweep_Qubit_{QubitIndex + 1}_{timestamp}.h5"
    with h5py.File(h5_file, "w") as f:
        # Store the data
        f.create_dataset("results", data=results)
        # Store metadata
        f.attrs["gain_range"] = gain_range
        f.attrs["freq_range"] = freq_range
        f.attrs["reference_frequency"] = reference_frequency
        f.attrs["freq_steps"] = freq_steps
        f.attrs["gain_steps"] = gain_steps

    print(f"Saved data for Qubit {QubitIndex + 1} to {h5_file}")

    plt.imshow(results, aspect='auto',
               extent=[gain_range[0], gain_range[1], freq_range[0] - reference_frequency, freq_range[1] - reference_frequency],
               origin='lower')
    plt.colorbar(label="Fidelity")
    plt.xlabel("Readout pulse gain (a.u.)")  # Gain on x-axis
    plt.ylabel("Readout frequency offset (MHz)")  # Frequency on y-axis
    plt.title(f"Gain-Frequency Sweep for Qubit {QubitIndex + 1}")
    #plt.show()
    # Save the plot
    file= f"{outerFolder}Gain_Freq_Sweep_Qubit_{QubitIndex + 1}_{timestamp}.png"
    plt.savefig(file, dpi=600, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory
    del results, sweep
    print(f"Saved plot for Qubit {QubitIndex + 1} to {file}")