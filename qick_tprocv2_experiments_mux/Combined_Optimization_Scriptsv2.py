import sys
import os
import numpy as np
import datetime
#sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_005_single_shot_ge import GainFrequencySweep
from section_005_single_shot_ge import SingleShot
from section_008_save_data_to_h5 import Data_H5
from section_003_punch_out_ge_mux import PunchOut
from system_config import QICK_experiment
from expt_config import *
import h5py
import time
import matplotlib.pyplot as plt
import copy

signal = 'None' # Keep as 'None'
save_figs = True   # save plots for everything as you go along the RR script?
live_plot = False    # for live plotting open http://localhost:8097/ on firefox
fit_data = False # always set to False
unmask = True
FRIDGE = "QUIET"
use_iminuit_instead = True # fitting options: iminuit or curve fit
number_of_qubits = 6 # for QUIET 6, for NEXUS 4
list_of_all_qubits = [0,1,2,3,4,5] # for QUIET [0, 1, 2, 3, 4, 5], for NEXUS [0, 1, 2, 3]

# For Quiet
substudy = "opt_Q5_20dBDAC"
outerFolder = os.path.join(f"/data/QICK_data/run9/6transmon/readout_optimization/{substudy}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")

opt_flags = {"res_leng_sweep": False, "2d_sweep": True}

def create_folder_if_not_exists(folder_path):
    """Creates a folder at the given path if it doesn't already exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Where to save readout length sweep data
prefix = str(datetime.date.today())
output_folder_length =outerFolder + "/study_data/Data_h5/ge_readout_length_optimization/"
create_folder_if_not_exists(output_folder_length)

# Where to save the RR plots
outerfolder_plots = outerFolder + "/documentation/"

n = 1  # Number of rounds
n_loops = 3 # Number of repetitions per length to average

# List of qubits to measure
Qs = [4]

# For 25dB DAC
# res_leng_vals = [5.0, 6.75, 7.25, 6.0, 7.5, 5.0] # updated 4/11 except for Q5
# res_gain = [0.9571, 0.6700, 0.8583, 0.6821, 1.0, 0.9571] #4/11, 25dB
# freq_offsets = [-0.1385, -0.1385, -0.2308, -0.0462, 0, -0.1385] # 4/11, 25dB

# For 20dB DAC
res_leng_vals = [4.25, 5.25, 5.0, 5.0, 6.25, 4.5]  # 4/12, 20dB
res_gain = [0.6400, 0.7300, 0.7800, 0.3750, 0.45, 0.7600]  # 4/12, 20dB
freq_offsets = [0.2308, 0.1385, 0.0462, 0.1385, -0.1385, 0.1385]  # 4/12, 20dB

optimal_lengths = [None] * 6 # creates list where the script will be storing the optimal readout lengths for each qubit. We currently have 6 qubits in total.
res_freq_ge = [None] * 6 # creates list where the script will be storing the freq of each resonator, to use in the 2d sweep

j=0 # round number, from RR code. Not really used here since we just run it once for each qubit

lengs = np.arange(2.0, 8.0, 0.25)
start=time.time()

for QubitIndex in Qs:
    recycled_qfreq = False  # don't change

    # keep these as False
    ef_res_spec_survived = False
    ef_qspec_survived = False

    # Get the config for this qubit
    DAC_attenuator1 = 10
    DAC_attenuator2 = 10
    experiment = QICK_experiment(outerFolder, DAC_attenuator1=DAC_attenuator1, DAC_attenuator2=DAC_attenuator2,
                                 qubit_DAC_attenuator1=5,
                                 qubit_DAC_attenuator2=4, ADC_attenuator=17,
                                 fridge=FRIDGE)  # ADC_attenuator MUST be above 16dB
    experiment.create_folder_if_not_exists(outerFolder)
    print("DAC atten: ", DAC_attenuator1 + DAC_attenuator2)

    # Mask out all other resonators except this one
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_gain_ef'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

    ################################################# g-e Res spec ####################################################
    increase_geres_reps = False
    increase_geres_reps_to = None
    # if QubitIndex == 5:
    #     increase_geres_reps = True
    #     increase_geres_reps_to = 500 #400
    # if QubitIndex == 3:
    #     increase_geres_reps = True
    #     increase_geres_reps_to = 600
    # if QubitIndex == 4:
    #     increase_geres_reps = True
    #     increase_geres_reps_to = 500

    res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, outerfolder_plots, j, save_figs,
                                     increase_geres_reps,
                                     increase_geres_reps_to, experiment=experiment, unmasking_resgain=unmask)
    res_freqs, freq_pts, freq_center, amps, sys_config_rspec, meas_timestamp_resge = res_spec.run()
    offset = freq_offsets[QubitIndex]  # use optimized offset values or whats set at top of script based on pre_optimize flag
    offset_res_freqs = [r + offset for r in res_freqs]
    experiment.readout_cfg['res_freq_ge'] = offset_res_freqs

    # Used later when optimizing res gains and freqs, decide if you want to set the offsets to zero or not for the first round
    this_res_freq = offset_res_freqs[QubitIndex]
    res_freq_ge[QubitIndex] = float(this_res_freq)

    del res_spec

    ################################################## g-e Qubit spec ##################################################
    increase_qubit_reps_qspec = False
    increase_qspec_rounds = False
    qspecge_increase_reps_to = None
    increase_qspec_rounds_to = None

    # if QubitIndex == 5:
    #     increase_qubit_reps_qspec = True
    #     qspecge_increase_reps_to = 1600
    # increase_qspec_rounds = True
    # increase_qspec_rounds_to = 2

    if QubitIndex == 4:
        increase_qubit_reps_qspec = True
        qspecge_increase_reps_to = 1100
        increase_qspec_rounds = True
        increase_qspec_rounds_to = 3

    if QubitIndex == 3:
        increase_qubit_reps_qspec = True
        qspecge_increase_reps_to = 800
        # increase_qspec_rounds = True
        # increase_qspec_rounds_to = 3
    #
    # if QubitIndex == 2:
    #     increase_qubit_reps_qspec = True
    #     qspecge_increase_reps_to = 800

    q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, outerfolder_plots, j,
                               signal, save_figs, increase_reps=increase_qubit_reps_qspec,
                               increase_rounds=increase_qspec_rounds,
                               increase_reps_to=qspecge_increase_reps_to,
                               increase_rounds_to=increase_qspec_rounds_to,
                               plot_fit=True, experiment=experiment, live_plot=live_plot, unmasking_resgain=unmask)
    (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq, sys_config_qspec,
     meas_timestamp_qspecge) = q_spec.run()

    if qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None:
        print(f'No stored g-e qubit spec value for qubit {QubitIndex}; skipping iteration.')
        continue
    else:
        print(f"g-e Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
        experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)

    del q_spec

    ###################################################### g-e Rabi ####################################################
    increase_qubit_reps_gerabi = False  # if you want to increase the reps for a qubit, set to True
    qubit_to_increase_reps_for = None  # only has impact if previous line is True
    # if QubitIndex == 3:
    #     increase_qubit_reps_gerabi = True
    #     qubit_to_increase_reps_for = QubitIndex
    if QubitIndex == 4:
        increase_qubit_reps_gerabi = True
        qubit_to_increase_reps_for = QubitIndex
    # if QubitIndex == 5:
    #     increase_qubit_reps_gerabi = True
    #     qubit_to_increase_reps_for = QubitIndex
    multiply_qubit_reps_by = 2  # only has impact if increase_qubit_reps_gerabi is True. MUST be an integer.

    rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, outerfolder_plots, j, signal,
                                   save_figs=save_figs, save_shots=False,
                                   experiment=experiment, live_plot=live_plot,
                                   increase_qubit_reps=increase_qubit_reps_gerabi,
                                   qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                   multiply_qubit_reps_by=multiply_qubit_reps_by,
                                   unmasking_resgain=unmask)
    (rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_rabi, meas_timestamp_rabige) = rabi.run(
        thresholding=False, use_iminuit_instead=use_iminuit_instead)

    # if these are None, fit didnt work
    if (rabi_fit is None and pi_amp is None):
        print('g-e Rabi fit didnt work, skipping the rest of this qubit')
        continue  # skip the rest of this qubit
    else:
        print('g-e Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))

    experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)

    del rabi

    #MAKE DEEP COPY OF CONFIG, IMPORTANT!!!
    tuned_experiment = copy.deepcopy(experiment)

    if opt_flags["res_leng_sweep"]:
        # # #-----------Sweeping Readout Length----------------------------
        QubitIndex = int(QubitIndex)  # Ensure QubitIndex is an integer

        avg_fids = []
        rms_fids = []

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        h5_filename = os.path.join(output_folder_length, f"ge_readoutlength_sweep_Q{QubitIndex + 1}_data_{timestamp}.h5")
        with h5py.File(h5_filename, 'w') as h5_file:
            # Top-level group for the qubit
            qubit_group = h5_file.create_group(f"Qubit_{QubitIndex + 1}")
            fids = []  # Store fidelity values for each loop
            ground_iq_data = []  # Store ground state IQ data for each loop
            excited_iq_data = []  # Store excited state IQ data for each loop

            # Iterate over each readout pulse length
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
                    #res_gains = experiment.set_gain_filter_ge(QubitIndex, gain)  # Set gain for current qubit only
                    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=gain)
                    experiment.readout_cfg['res_gain_ge'] = res_gains

                    save_figs_ss = False
                    ss = SingleShot(QubitIndex, number_of_qubits, outerFolder,  j, save_figs_ss, experiment, unmasking_resgain = unmask)  # updated way
                    fid, angle, iq_list_g, iq_list_e, ss_config, _ = ss.run()
                    print(ss_config)
                    fids.append(fid)

                    # Append IQ data for each loop
                    ground_iq_data.append(iq_list_g)
                    excited_iq_data.append(iq_list_e)

                    # Save individual fidelity and IQ data for this loop
                    loop_group = length_group.create_group(f"Loop_{k + 1}")
                    loop_group.create_dataset("fidelity", data=fid)
                    loop_group.create_dataset("ground_iq_data", data=iq_list_g)
                    loop_group.create_dataset("excited_iq_data", data=iq_list_e)

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

        # avg_max = max(avg_fids[:10])
        avg_max = max(avg_fids)
        avg_max_index = avg_fids.index(avg_max)
        max_len = lengs[avg_max_index]
        optimal_lengths[QubitIndex] = max_len # auto mode

        # Plot the average fidelity vs. pulse length with error bars for each qubit
        plt.figure()
        plt.errorbar(lengs, avg_fids, yerr=rms_fids, fmt='-o', color='black')
        plt.axvline(x=max_len, linestyle="--", color="red")
        plt.text(max_len + 0.1, avg_fids[0], f'{max_len:.4f}', color='red')
        plt.xlabel('Readout and Pulse Length')
        plt.ylabel('Fidelity')
        plt.title(f'Avg Fidelity vs. Readout and Pulse Length for Qubit {QubitIndex + 1}, ({n_loops} repetitions)' , fontsize=10)
        path = os.path.join(outerfolder_plots, 'readout_length_ge')
        create_folder_if_not_exists(path)
        file_nm = os.path.join(path, f'ge_readoutlength_sweep_Q{QubitIndex + 1}_{timestamp}.png')
        plt.savefig(file_nm, dpi=300)
        print('res leng sweep plot saved to:', outerfolder_plots)
        #plt.show()
        plt.close()

        del avg_fids, rms_fids, avg_ground_iq, avg_excited_iq, loop_group, length_group

    if opt_flags["2d_sweep"]:
        # ##---------------------Res Gain and Res Freq Sweeps------------------------
        optimal_lengths = res_leng_vals #  can also just write the list here. # optional, used when not in auto mode.

        date_str = str(datetime.date.today())
        output_folder = outerFolder + "/study_data/Data_h5/2D_Gain_Freq_Sweeps/"

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        ## punchout thresholds: [1.0, 0.886, 1.0, 0.771, 1.0, 1.0] 25dB, [0.8, 0.75, 0.8, 0.5, 0.65, 0.8] 20dB
        # Define sweeping parameters
        if QubitIndex == 0:
            gain_range = [0.4, 0.8] #1.0
            gain_steps = 20
        elif QubitIndex == 1:
            gain_range = [0.35, 0.75]
            gain_steps = 20
        elif QubitIndex == 2:
            gain_range = [0.4, 0.8] # 0.9
            gain_steps = 20
        elif QubitIndex == 3:
            gain_range = [0.25, 0.5] # 0.8
            gain_steps = 8
        elif QubitIndex == 4: # 0.725
            gain_range = [0.3, 0.6]
            gain_steps = 8
        elif QubitIndex == 5:
            gain_range = [0.4, 0.8] #1.0
            gain_steps = 20

        freq_steps = 13

        print(f'Starting Qubit {QubitIndex + 1} res gain and res freq measurements.')
        # Select the reference frequency for the current resonator
        reference_frequency = res_freq_ge[QubitIndex]

        freq_range = [reference_frequency - 0.6, reference_frequency + 0.6] # Frequency range in MHz

        experiment = copy.deepcopy(tuned_experiment)
        save_figs_ss = False
        sweep = GainFrequencySweep(QubitIndex, number_of_qubits, list_of_all_qubits, experiment, optimal_lengths=optimal_lengths, output_folder=output_folder, unmasking_resgain = unmask,
                                   save_figs = save_figs_ss)
        results = sweep.run_sweep(freq_range, gain_range, freq_steps, gain_steps)
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

        #print(f"Saved data for Qubit {QubitIndex + 1} to {h5_file}")

        plt.imshow(results, aspect='auto',
                   extent=[gain_range[0], gain_range[1], freq_range[0] - reference_frequency,
                           freq_range[1] - reference_frequency],
                   origin='lower')
        plt.colorbar(label="Fidelity")
        plt.xlabel("Readout pulse gain (a.u.)")  # Gain on x-axis
        plt.ylabel("Readout frequency offset (MHz)")  # Frequency on y-axis
        plt.title(f"Gain-Frequency Sweep for Qubit {QubitIndex + 1}")
        # plt.show()
        path = os.path.join(outerfolder_plots, '2D_GainFreq_Sweep')
        create_folder_if_not_exists(path)
        file_nm = os.path.join(path, f'ge_gain_freqoffset_2Dsweep_Q{QubitIndex + 1}_{timestamp}.png')
        plt.savefig(file_nm, dpi=600, bbox_inches='tight')

        plt.close()  # Close the plot to free up memory
        del results, sweep

    end=time.time()
    print('timetaken=',end-start)
