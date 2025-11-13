import sys
import os
import numpy as np
import datetime
import h5py
import time
import matplotlib.pyplot as plt
import copy
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_005_single_shot_ge import TWPAGainFrequencySweep
from system_config import QICK_experiment
from expt_config import *
from windfreak import SynthHD

signal = 'None'
save_figs = True
live_plot = False
fit_data = False
unmask = True
FRIDGE = 'NEXUS'
number_of_qubits = 4
list_of_all_qubits = [0, 1, 2, 3]

run_name = 'run33d'
device_name = '4charge'
substudy = 'TWPA_opt_Q4'

outerFolder = os.path.join(f"/home/nexusadmin/Documents/Data/{run_name}/{device_name}/TWPA_optimization/{substudy}/{datetime.date.today()}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")

def create_folder_if_not_exists(folder_path):
    """Creates a folder at the given path if it doesn't already exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Where to save RR plots, if measured
outerfolder_plots = outerFolder + "/documentation/"

n = 1  # Number of rounds
n_loops = 4  # Number of ssf repetitions - currently unused

# List of qubits to measure
Qs = [3] #0, 1, 2, 3

# optimization outputs for NEXUS
res_leng_vals = [5.1, 4, 4, 3.6] #[9.5, 5.5, 6.25, 7.0] # Q1, Q4 reopt
res_gain = [0.115, 0.1, 0.15, 0.19] #[0.9, 0.7, 0.8, 0.8] # Q1, Q4 reopt
freq_offsets = [0.1364, 0, 0, -0.1364] #[-0.1429, -0.1429, 0, -0.04] # Q1, Q4 reopt

# Define sweeping parameters
gain_arr = np.linspace(-13.5, -11, 10)
freq_arr = np.linspace(7.80e9, 7.83e9, 10)  # Don't go above 7.9159 GHz

prev_TWPA_gain = -12.69
prev_TWPA_freq = 7.826e9

res_freq_ge = [None] * 4 # creates list where the script will be storing the freq of each resonator, to use in the 2d sweep

j=0 #round number, from RR code. Not really used here since we just run it once for each qubit


start = time.time()
for QubitIndex in Qs:
    # Get the config for this qubit
    experiment = QICK_experiment(outerFolder, DAC_attenuator1=10, DAC_attenuator2=15, qubit_DAC_attenuator1=5,
                                 qubit_DAC_attenuator2=4, ADC_attenuator=17,
                                 fridge=FRIDGE)

    # Mask out all other resonators except this one
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=number_of_qubits)
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

    ################################################## Res spec ####################################################

    res_spec = ResonanceSpectroscopy(QubitIndex, number_of_qubits, outerfolder_plots, j, True,
                                     experiment, unmasking_resgain=unmask)
    res_freqs, freq_pts, freq_center, amps, res_spec_config = res_spec.run()  # This is the line with the problem
    experiment.readout_cfg['res_freq_ge'] = res_freqs

    # incorporating offset (if you don't want to, then set all values inside freq_offsets to zero)
    offset = freq_offsets[QubitIndex]  # use optimized offset values
    offset_res_freqs = [r + offset for r in res_freqs]
    experiment.readout_cfg['res_freq_ge'] = offset_res_freqs

    # Used later when optimizing res gains and freqs, decide if you want to set the offsets to zero or not for the first round
    this_res_freq = offset_res_freqs[QubitIndex]
    res_freq_ge[QubitIndex] = float(this_res_freq)

    del res_spec

    ################################################## Qubit spec ##################################################

    q_spec = QubitSpectroscopy(QubitIndex, number_of_qubits, outerfolder_plots, j, signal,
                               True, experiment, live_plot, unmasking_resgain=unmask)
    qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq, qubit_spec_config = q_spec.run()

    # if these are None, fit didnt work
    if (qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None):
        print('QSpec fit didnt work, skipping the rest of this qubit')
        continue  # skip the rest of this qubit

    experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
    print('Qubit freq for qubit ', QubitIndex + 1, ' is: ', float(qubit_freq))
    del q_spec

    ###################################################### Rabi ####################################################
    increase_qubit_reps = False  # if you want to increase the reps for a qubit, set to True
    qubit_to_increase_reps_for = 0  # only has impact if previous line is True
    multiply_qubit_reps_by = 2  # only has impact if the line two above is True
    print('ge Rabi')
    rabi = AmplitudeRabiExperiment(QubitIndex, number_of_qubits, outerfolder_plots, j, signal,
                                   False, experiment=experiment, live_plot=live_plot,
                                   increase_qubit_reps=increase_qubit_reps,
                                   qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                   multiply_qubit_reps_by=multiply_qubit_reps_by, unmasking_resgain=unmask)
    rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_to_save = rabi.run()

    # if these are None, fit didnt work
    if (rabi_fit is None and pi_amp is None):
        print('Rabi fit didnt work, skipping the rest of this qubit')
        continue  # skip the rest of this qubit

    experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
    print('Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))

    # # MAKE DEEP COPY OF CONFIG, IMPORTANT!!!
    # tuned_experiment = copy.deepcopy(experiment)

    ##---------------------TWPA Sweep------------------------
    date_str = str(datetime.date.today())
    output_folder = outerFolder + "/study_data/Data_h5/2D_Power_Freq_Sweeps/"
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)


    print(f'Starting Qubit {QubitIndex + 1} TWPA Sweep')

    # experiment = copy.deepcopy(tuned_experiment)
    sweep = TWPAGainFrequencySweep(QubitIndex, number_of_qubits, list_of_all_qubits, experiment,
                               optimal_lengths=res_leng_vals, output_folder=output_folder, unmasking_resgain=unmask)
    results = sweep.run_sweep(freq_arr, gain_arr)
    results = np.array(results)

    timestamp = time.strftime("%H%M%S")
    h5_file = os.path.join(output_folder, f"2D_TWPA_Sweep_Qubit_{QubitIndex + 1}_{timestamp}.h5")

    with h5py.File(h5_file, "w") as f:
        # Store the data
        f.create_dataset("results", data=results)
        # Store metadata
        f.attrs["gain_arr"] = gain_arr
        f.attrs["freq_arr"] = freq_arr

    # print(f"Saved data for Qubit {QubitIndex + 1} to {h5_file}")

    plt.imshow(results, aspect='auto',
               extent=[gain_arr[0], gain_arr[-1], freq_arr[0]/1e9, freq_arr[-1]/1e9],
               origin='lower')
    plt.colorbar(label="Fidelity")
    plt.xlabel("Pump Power (dB)")  # Gain on x-axis
    plt.ylabel("Pump Frequency (GHz)")  # Frequency on y-axis
    plt.title(f"TWPA Pump-Frequency Sweep for Qubit {QubitIndex + 1}")
    # plt.show()
    path = os.path.join(outerfolder_plots, '2D_TWPAGainFreq_Sweep')
    create_folder_if_not_exists(path)
    file_nm = os.path.join(path, f'2D_TWPA_sweep_Q{QubitIndex + 1}_{timestamp}.png')
    plt.savefig(file_nm, dpi=600, bbox_inches='tight')

    plt.close()  # Close the plot to free up memory
    del results, sweep

end = time.time()
print('timetaken=', end - start)

synth = SynthHD('/dev/ttyACM0')
synth[0].frequency = prev_TWPA_freq
synth[0].gain = prev_TWPA_gain
synth[0].enable = True
time.sleep(5)

print('Back to previous TWPA setpoints: ', prev_TWPA_freq/1e9, prev_TWPA_gain)




