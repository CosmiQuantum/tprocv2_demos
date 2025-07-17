import sys
import os
import numpy as np
import datetime

sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_001_time_of_flight import TOFExperiment
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_005_single_shot_ge import GainFrequencySweep
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_005_single_shot_gef import SingleShot_ef
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from section_003_punch_out_ge_mux import PunchOut
from section_005_single_shot_ge import SingleShot
from system_config import QICK_experiment
from expt_config import *
import h5py
import time
import matplotlib.pyplot as plt
import copy

signal = 'None'  # 'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization has happened
save_figs = False  # save plots for everything as you go along the RR script?
live_plot = False  # for live plotting open http://localhost:8097/ on firefox
fit_data = False  # always set to False
FRIDGE = "QUIET"
number_of_qubits = 6  # for QUIET 6, for NEXUS 4
list_of_all_qubits = [0, 1, 2, 3, 4, 5]  # for QUIET [0, 1, 2, 3, 4, 5], for NEXUS [0, 1, 2, 3]


# For Nexus
# outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30", str(datetime.date.today())) #change run number in each new run

# For Quiet
# outerFolder = os.path.join("/data/QICK_data/6transmon_run6/", str(datetime.date.today()))
# outerFolder = os.path.join("/data/QICK_data/run6/6transmon/StarkShift/DAC0_check/Optimization/run2/", str(datetime.date.today()))
# outerFolder = os.path.join(f"/data/QICK_data/run6/6transmon/TLS_Comprehensive_Study/readout_optimization_{datetime.date.today().strftime('%Y-%m-%d')}", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def create_folder_if_not_exists(folder_path):
    """Creates a folder at the given path if it doesn't already exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)



def find_max_fidelity(file_path):
    with h5py.File(file_path, "r") as f:
        # Load the results dataset and metadata
        results = np.array(f["results"])
        gain_range = f.attrs["gain_range"]
        freq_range = f.attrs["freq_range"]
        reference_frequency = f.attrs["reference_frequency"]
        freq_steps = f.attrs["freq_steps"]
        gain_steps = f.attrs["gain_steps"]
        optimal_length = f.attrs["optimal_length"]

        # Calculate step sizes
        freq_step_size = (freq_range[1] - freq_range[0]) / freq_steps
        gain_step_size = (gain_range[1] - gain_range[0]) / gain_steps

        # Find the maximum fidelity value
        max_fidelity = np.max(results)

        # Find all indices where the maximum fidelity occurs
        max_indices = np.argwhere(results == max_fidelity)

        # Collect all configurations with the highest fidelity
        configurations = []
        for index in max_indices:
            freq_idx, gain_idx = index
            gain = gain_range[0] + gain_idx * gain_step_size
            freq_offset = (freq_range[
                               0] - reference_frequency) + freq_idx * freq_step_size  # this is doing freq - reference_frequency
            configurations.append((max_fidelity, gain, freq_offset, optimal_length))

        return max_fidelity, configurations


def find_configurations_below_threshold(file_path, threshold):
    """
    Returns *all* (fidelity, gain, freq_offset, optimal_length) points for which
    the gain is strictly below the given threshold.
    """
    with h5py.File(file_path, "r") as f:
        # Read results as a Python object. If 'results' is truly a list-of-lists,
        # do something like:
        results = f["results"][()]  # This loads the dataset into a NumPy array
        results = results.tolist()  # Convert to list of lists if it's not already.

        # Retrieve metadata
        gain_range = f.attrs["gain_range"]
        freq_range = f.attrs["freq_range"]
        reference_frequency = f.attrs["reference_frequency"]
        freq_steps = f.attrs["freq_steps"]
        gain_steps = f.attrs["gain_steps"]
        optimal_length = f.attrs["optimal_length"]

    # Calculate step sizes
    freq_step_size = (freq_range[1] - freq_range[0]) / freq_steps
    gain_step_size = (gain_range[1] - gain_range[0]) / gain_steps

    valid_configurations = []

    # Outer loop: each row in 'results' corresponds to one freq_idx
    for freq_idx, row_of_fidelities in enumerate(results):
        # Inner loop: each element in the row corresponds to one gain_idx
        for gain_idx, fidelity in enumerate(row_of_fidelities):
            # Reconstruct the actual gain from 'gain_idx'
            gain = gain_range[0] + gain_idx * gain_step_size

            if gain < threshold:
                # Compute frequency offset from 'freq_idx'
                freq_offset = (freq_range[0] - reference_frequency) + freq_idx * freq_step_size

                valid_configurations.append((fidelity, gain, freq_offset, optimal_length))

    return valid_configurations


# def optimize_readout_len(outerFolder, fid_states, old_res_leng_vals, old_res_gain, old_freq_offsets, lengs, Qs):
substudy = "junkyard"
outerFolder = os.path.join(
    f"/data/QICK_data/run7/6transmon/readout_optimization/{substudy}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")
# Where to save readout length sweep data
pprefix = str(datetime.date.today())
output_folder_length =outerFolder + "/study_data/Data_h5/ef_readout_length_optimization/"
create_folder_if_not_exists(output_folder_length)

#Where to save the RR plots
outerfolder_plots = outerFolder + "/documentation/"

n = 1  # Number of rounds
n_loops = 5  # Number of repetitions per length to average

# List of qubits and pulse lengths to measure
Qs =  [0,1,2,3,4,5]
number_of_qubits = len(Qs)  # for QUIET 6, for NEXUS 4
list_of_all_qubits = [0, 1, 2, 3, 4, 5]  # for QUIET [0, 1, 2, 3, 4, 5], for NEXUS [0, 1, 2, 3]

# Change for NEXUS vs QUIET
res_leng_vals =  [5.85, 2.85, 5.35, 2.35] # from 2/7/2025 optimization
res_gain =  [0.34, 0.3, 0.34, 0.3875] #from 1/31-2/1optimization, after implementing punchout threshold
freq_offsets = [-0.08, -0.16, -0.08, -0.08] #from 1/31-2/1 optimization, after implementing punchout threshold

optimal_lengths = [None] * 4  # creates list where the script will be storing the optimal readout lengths for each qubit. We currently have 6 qubits in total.
res_freq_ge = [None] * 4  # creates list where the script will be storing the freq of each resonator, to use in the 2d sweep

optimal_resgains1 = [None] * 4  # creates list where the script will be storing the optimal resonator gains
opt_offset_freqs1 = [None] * 4  # creates list where the script will be storing the optimal frequency offsets
j=0

for QubitIndex in Qs:
    # Get the config for this qubit
    experiment = QICK_experiment(outerFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10)

    # Mask out all other resonators except this one
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex])
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

    ################################################## Res spec ####################################################
    try:
        res_spec = ResonanceSpectroscopy(QubitIndex, number_of_qubits, list_of_all_qubits, outerfolder_plots, j,
                                         save_figs,
                                         experiment)
        res_freqs, freq_pts, freq_center, amps = res_spec.run(experiment.soccfg, experiment.soc)
        experiment.readout_cfg['res_freq_ge'] = res_freqs

        # incorporating offset (if you don't want to, then set all values inside freq_offsets to zero)
        offset = freq_offsets[QubitIndex]  # use optimized offset values
        offset_res_freqs = [r + offset for r in res_freqs]
        experiment.readout_cfg['res_freq_ge'] = offset_res_freqs

        # Used later when optimizing res gains and freqs, make sure you are setting all freq_offsets to zero beforehand
        this_res_freq = offset_res_freqs[QubitIndex]
        res_freq_ge[QubitIndex] = float(this_res_freq)

        del res_spec
    except Exception as e:
        print(f'Got the following error at Res Spec, continuing: {e}')
        continue  # skip the rest of this qubit

    ################################################## Qubit spec ##################################################
    try:
        q_spec = QubitSpectroscopy(QubitIndex, number_of_qubits, list_of_all_qubits, outerfolder_plots, j, signal,
                                   save_figs, experiment, live_plot)
        qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq = q_spec.run(experiment.soccfg,
                                                                                         experiment.soc)
        # if these are None, fit didnt work
        if (qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None):
            print('QSpec fit didnt work, skipping the rest of this qubit')
            continue  # skip the rest of this qubit

        experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
        print('Qubit freq for qubit ', QubitIndex + 1, ' is: ', float(qubit_freq))
        del q_spec

    except Exception as e:
        print(f'Got the following error, continuing: {e}')
        continue  # skip the rest of this qubit

    ###################################################### Rabi ####################################################
    increase_qubit_reps = False  # if you want to increase the reps for a qubit, set to True
    qubit_to_increase_reps_for = 0  # only has impact if previous line is True
    multiply_qubit_reps_by = 2  # only has impact if the line two above is True

    try:
        rabi = AmplitudeRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits, outerfolder_plots, j, signal,
                                       save_figs, experiment, live_plot,
                                       increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
        rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_to_save = rabi.run(experiment.soccfg,
                                                                                    experiment.soc)

        # if these are None, fit didnt work
        if (rabi_fit is None and pi_amp is None):
            print('Rabi fit didnt work, skipping the rest of this qubit')
            continue  # skip the rest of this qubit

        experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
        print('Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
        del rabi

    except Exception as e:
        print(f'Got the following error, continuing: {e}')
        continue  # skip the rest of this qubit

    # MAKE DEEP COPY OF CONFIG, IMPORTANT!!!
    tuned_experiment = copy.deepcopy(experiment)

    QubitIndex = int(QubitIndex)  # Ensure QubitIndex is an integer

    avg_fids = []
    rms_fids = []

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_filename = os.path.join(output_folder_length, f"qubit_{QubitIndex + 1}_data_{timestamp}.h5")
    with h5py.File(h5_filename, 'w') as h5_file:
        # Top-level group for the qubit
        qubit_group = h5_file.create_group(f"Qubit_{QubitIndex + 1}")
        # Store the data
        # f.create_dataset("results", data=results)
        # Store metadata
        # f.attrs["gain_range"] = gain_range
        # f.attrs["freq_range"] = freq_range
        # f.attrs["reference_frequency"] = reference_frequency
        # f.attrs["freq_steps"] = freq_steps
        # f.attrs["gain_steps"] = gain_steps
        # f.attrs["optimal_length"] = optimal_lengths[QubitIndex]
        fids = []  # Store fidelity values for each loop
        ground_iq_data = []  # Store ground state IQ data for each loop
        excited_iq_data = []  # Store excited state IQ data for each loop
        f_iq_data = []  # Store excited state IQ data for each loop

        ######### ge OR ef: CHOOOOSE ##########
        fid_states='ef' #'ge'

        # -----------Sweeping Readout Length----------------------------
        # Iterate over each readout pulse length
        lengs = np.arange(0.1, 15, 0.5)
        for leng in lengs:
            # Subgroup for each readout length within the round
            length_group = qubit_group.create_group(f"Length_{leng}")

            for k in range(n_loops):  # loops for each read out length
                # ------------------------Single Shot-------------------------
                # Initialize experiment for each loop iteration
                experiment = copy.deepcopy(tuned_experiment)
                # Set specific configuration values for each iteration
                experiment.readout_cfg['res_length'] = leng  # Set the current readout pulse length

                # Set gain for the current qubit
                gain = res_gain[QubitIndex]
                # res_gains = experiment.set_gain_filter_ge(QubitIndex, gain)  # Set gain for current qubit only
                res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=gain)
                experiment.readout_cfg['res_gain_ge'] = res_gains

                if fid_states == 'ge':
                    ss = SingleShot(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, save_figs,
                                    experiment)  # updated way
                    fid, angle, iq_list_g, iq_list_e = ss.run(experiment.soccfg, experiment.soc)
                    fids.append(fid)
                elif fid_states == 'ef':
                    ss = SingleShot_ef(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, save_figs,
                                       experiment)  # updated way
                    fid, angle, iq_list_e, iq_list_f = ss.fidelity_test(experiment.soccfg, experiment.soc)
                    fids.append(fid)

                # Append IQ data for each loop
                if fid_states == 'ge':
                    ground_iq_data.append(iq_list_g)
                    excited_iq_data.append(iq_list_e)
                elif fid_states == 'ef':
                    excited_iq_data.append(iq_list_e)
                    f_iq_data.append(iq_list_f)

                # Save individual fidelity and IQ data for this loop
                loop_group = length_group.create_group(f"Loop_{k + 1}")
                loop_group.create_dataset("fidelity", data=fid)
                if fid_states == 'ge':
                    loop_group.create_dataset("ground_iq_data", data=iq_list_g)
                    loop_group.create_dataset("excited_iq_data", data=iq_list_e)
                if fid_states == 'ef':
                    loop_group.create_dataset("excited_iq_data", data=iq_list_e)
                    loop_group.create_dataset("f_iq_data", data=iq_list_f)

            del experiment

            # Calculate average and RMS for fidelities across loops
            avg_fid = np.mean(fids)
            rms_fid = np.std(fids)
            avg_fids.append(avg_fid)
            rms_fids.append(rms_fid)

            # Calculate average IQ data across all loops
            avg_ground_iq = np.mean(ground_iq_data, axis=0)
            avg_excited_iq = np.mean(excited_iq_data, axis=0)

            fids.clear()
            ground_iq_data.clear()
            excited_iq_data.clear()

        # Save the averages and RMS to the HDF5 file for this length
        length_group.create_dataset("avg_fidelity", data=avg_fids)
        length_group.create_dataset("rms_fidelity", data=rms_fids)
        length_group.create_dataset("avg_ground_iq_data", data=avg_ground_iq)
        length_group.create_dataset("avg_excited_iq_data", data=avg_excited_iq)


    avg_max = max(avg_fids)


    avg_max_index = avg_fids.index(avg_max)
    max_len = lengs[avg_max_index]
    optimal_lengths[QubitIndex] = max_len

    # Plot the average fidelity vs. pulse length with error bars for each qubit
    plt.figure()
    plt.errorbar(lengs, avg_fids, yerr=rms_fids, fmt='-o', color='black')
    plt.axvline(x=max_len, linestyle="--", color="red")
    plt.text(max_len + 0.1, avg_fids[0], f'{max_len:.4f}', color='red')
    plt.xlabel('Readout and Pulse Length')
    plt.ylabel('Fidelity')
    plt.title(f'{fid_states} Avg Fidelity vs. Readout and Pulse Length for Qubit {QubitIndex + 1}, ({n_loops} repetitions)', fontsize=10)
    plt.savefig(os.path.join(outerfolder_plots, f'fidelity_Q{QubitIndex + 1}_{timestamp}.png'), dpi=300)
    print('res leng sweep plot saved to:', output_folder_length)
    # plt.show()
    plt.close()

    del avg_fids, rms_fids, avg_ground_iq, avg_excited_iq, loop_group, length_group


# def optimize_freq_gain(outerFolder, fid_states, old_res_leng_vals, resfreqs, gains, Qs):
    number_of_qubits = len(Qs)
    list_of_all_qubits = [0, 1, 2, 3, 4, 5]
    ##---------------------Res Gain and Res Freq Sweeps------------------------


    optimal_lengths =  [4, 14, 6, 10, 6, 7] #optional # DAC 2 optimization
    # optimal_lengths = [4.2, 5, 7.0, 6, 6, 7.5] # DAC 0 optimization
    date_str = str(datetime.date.today())
    output_folder = outerFolder + "/study_data/Data_h5/2D_Gain_Freq_Sweeps/"
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define sweeping parameters
    gain_range = [0.4, 1]  # Gain range in a.u.
    freq_steps = 8  # 20
    gain_steps = 15

    print(f'Starting Qubit {QubitIndex + 1} res gain and res freq measurements.')
    # Select the reference frequency for the current resonator

    # freq_range = [reference_frequency -0.2, (reference_frequency + 0.2) + 1]  # Frequency range in MHz
    reference_frequency = res_freq_ge[QubitIndex]


    freq_range = [reference_frequency - 0.6, reference_frequency + 0.6]  # Frequency range in MHz

    # avg_fids = []
    # rms_fids = []

    experiment = copy.deepcopy(tuned_experiment)
    sweep = GainFrequencySweep(QubitIndex, number_of_qubits, list_of_all_qubits, experiment,
                               optimal_lengths=optimal_lengths, output_folder=output_folder)
    results = sweep.run_sweep(freq_range, gain_range, freq_steps, fid_states)
    results = np.array(results)

    timestamp = time.strftime("%H%M%S")
    h5_file = os.path.join(output_folder, f"Gain_Freq_Sweep_Qubit_{QubitIndex + 1}_{timestamp}.h5")

    with h5py.File(h5_file, "w") as f:
        # Store the data
        f.create_dataset("results", data=results)
        # Store metadata
        f.attrs["gain_range"] = gain_range
        f.attrs["freq_range"] = freq_range
        f.attrs["reference_frequency"] = reference_frequency
        f.attrs["freq_steps"] = freq_steps
        f.attrs["gain_steps"] = gain_steps
        f.attrs["optimal_length"] = optimal_lengths[QubitIndex]

    # print(f"Saved data for Qubit {QubitIndex + 1} to {h5_file}")

    plt.imshow(results, aspect='auto',
               extent=[gain_range[0], gain_range[1], freq_range[0] - reference_frequency,
                       freq_range[1] - reference_frequency],
               origin='lower')
    plt.colorbar(label="Fidelity")
    plt.xlabel("Readout pulse gain (a.u.)")  # Gain on x-axis
    plt.ylabel("Readout frequency offset (MHz)")  # Frequency on y-axis
    plt.title(f"Gain-Frequency Sweep for Qubit {QubitIndex + 1}")
    # plt.show()
    file = f"Gain_Freq_Sweep_Qubit_{QubitIndex + 1}_{timestamp}.png"
    file_path = os.path.join(outerfolder_plots, file)
    plt.savefig(file_path, dpi=600, bbox_inches='tight')

    plt.close()  # Close the plot to free up memory
    del results, sweep



# Vd =
# Id =
# Vg =
# hemt_vd_id_vg = f'Vd_{Vd}__Id_{Id}__Vg_{Vg}'
# WarmAmpschain = ''
# MC_temp =

# with open(study_notes_path, "w", encoding="utf-8") as file:
#     file.write(
#         'Study Notes:  \n' + f'The MC temperature is {MC_temp}mK' + f'HEMT bias is Vg={Vg}, Vd={Vd}, Id={Id}. The warm amp  chain is {WarmAmpschain}' + '\n ')

