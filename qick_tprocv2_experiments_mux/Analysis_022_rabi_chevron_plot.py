import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_002_res_spec_ef import ResonanceSpectroscopyEF
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_004_qubit_spec_ef import EFQubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_005_single_shot_gef import SingleShot_ef
from section_008_save_data_to_h5 import Data_H5
from section_006_amp_rabi_ef import EF_AmplitudeRabiExperiment
from section_007_T1_ef import EF_T1Measurement
from section_007_T1_ge import T1Measurement
from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from f_to_res_swap_spec import FtoResQubitSpectroscopy
from analysis_020_gef_ssf_fstate_plots import GEF_SSF_ANALYSIS
import copy
################################################ Run Configurations ####################################################
n= 1
save_r = 1            # how many rounds to save after
signal = 'None'       #'I', or 'Q' depending on where the signal is (after optimization). Put'None' if no optimization
save_figs = False    # save plots for RR experiments? Note; this is NOT asking whether to plot rabi chevron exp, that will always plot in this script
fig_quality = 200
live_plot = False      # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True      # fit the data here and save or plot the fits?
save_data_h5 = True   # save the data of the measurements you are taking to h5 files?
number_of_qubits = 6 # 4 for nexus, 6 for quiet

verbose=False
rr_logger=None
qick_verbose=False

Qs_to_look_at = [0] #only list the qubits you want to do the RR for

increase_qubit_reps = False #if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0 #only has impact if previous line is True
multiply_qubit_reps_by = 2 #only has impact if the line two above is True

path_saveplots = f"/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/RR_metrics/RabiChevronExp"
path_saveplots_chev = f"/exp/cosmiq/data/home/cosmiq/Analysis/acolonce/RR_metrics/Plots/RabiChevron/rabi_ge_chevron"

if not os.path.exists(path_saveplots): os.makedirs(path_saveplots)
if not os.path.exists(path_saveplots_chev): os.makedirs(path_saveplots_chev)

# set which of the following measurements you would like to take
run_flags = {"res_spec_ge": True, "q_spec_ge": True, "rabi_ge_chevron": True}

################################################ readout optimization vals ##################################################
res_leng_vals = [5.5, 7.5, 6.0, 6.5, 5.0, 6.0]
res_gain = [0.9, 0.95, 0.78, 0.58, 0.95, 0.57]
freq_offsets = [-0.1, 0.2, 0.1, -0.4, -0.1, -0.1]

default_qubit_freqs = [ 4189.8656, 3820.4723, 4161.3726, 4463.15226, 4471.446, 4997.86] # from 3/11, qfreqs to fall back on if qspec is set to False
default_res_freqs = [6216.9331, 6275.9373, 6335, 6407.0338, 6476.1256, 6538] #04/07, res freqs to fall back on if res spec is set to False
####################################################### RR #############################################################
def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num','Recycled QFreq',
              'Exp Config', 'Syst Config']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
chev_keys = ['I', 'Q', 'Gains', 'Freqs_MHz', 'q_center_freq_MHz', 'res_freq_ge_MHz']

#initialize a dictionary to store those values
res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
# chev_data = create_data_dict(chev_keys, save_r, list_of_all_qubits) # Not necessary, it is included below in the code instead

batch_num=0
j = 0
angles=[]

qubit_freqs_ge = np.zeros(6)
res_freq_ge = np.zeros(6)

for QubitIndex in Qs_to_look_at:
    #Get the config for this qubit
    experiment = QICK_experiment(path_saveplots, DAC_attenuator1 = 5, DAC_attenuator2 = 10, ADC_attenuator = 10, fridge=FRIDGE)
    #Mask out all other resonators except this one
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex])
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

    ################################################# g-e Res spec ####################################################
    if run_flags["res_spec_ge"]: #If set to false, it will fall back on res freqs already in the system config
        res_spec = ResonanceSpectroscopy(QubitIndex, number_of_qubits, path_saveplots, j, save_figs,
                                         experiment)
        res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
        experiment.readout_cfg['res_freq_ge'] = res_freqs

        # incorporating offset (if you don't want to, then set all values inside freq_offsets to zero)
        offset = freq_offsets[QubitIndex]
        offset_res_freqs = [r + offset for r in res_freqs]
        experiment.readout_cfg['res_freq_ge'] = offset_res_freqs

        this_res_freq = offset_res_freqs[QubitIndex]
        res_freq_ge[QubitIndex] = float(this_res_freq)

        print('Qubit ', QubitIndex + 1, ' g-e res freq: ', this_res_freq)
        del res_spec
    else:
        this_res_freq = default_res_freqs[QubitIndex]
        experiment.readout_cfg['res_freq_ge'][QubitIndex] = float(this_res_freq)
        res_freq_ge[QubitIndex] = float(this_res_freq)

    ################################################### g-e Qubit spec #############################################################
    if run_flags["q_spec_ge"]:
        q_spec = QubitSpectroscopy(QubitIndex, number_of_qubits, path_saveplots, j, signal, save_figs, experiment, live_plot, verbose = False, logger = None, qick_verbose = True, increase_reps = False, increase_reps_to = 500)
        qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq, sys_config_qspec = q_spec.run()

        qubit_freqs_ge[QubitIndex] = qubit_freq
        experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
        print('Qubit ', QubitIndex + 1, ' g-e freq: ', float(qubit_freq))
        del q_spec
    else:
        qubit_freq = default_qubit_freqs[QubitIndex]
        print(f"Q_spec disabled → using default g-e freq for Q{QubitIndex + 1}: {qubit_freq:.4f} MHz")
        experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)

    ############################################# Make a copy of og experiment: IMPORTANT #############################################
    experiment_template = copy.deepcopy(experiment)
    ####################################################### 2D sweep (Rabi Chevron) ######################################################
    if run_flags["rabi_ge_chevron"]:
        # frequency grid ±2 MHz around optimized qubit freq center
        freq_steps = 30
        freqs_mhz = np.linspace(qubit_freq - 2, qubit_freq + 2, freq_steps)

        signal_map = []  # will have shape (len(freqs), len(gains))
        all_rabi_I = [] # to save I data in h5 files
        all_rabi_Q = [] # to save Q data in h5 files
        for f in freqs_mhz:
            # get the “optimized” experiment defined above
            experiment = copy.deepcopy(experiment_template)

            # only change the qubit drive freq
            experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(f)

            # run the gain‐sweep Rabi
            rabi = AmplitudeRabiExperiment(QubitIndex, number_of_qubits, path_saveplots,0, signal, save_figs=save_figs, experiment=experiment,
                    live_plot=live_plot, increase_qubit_reps=increase_qubit_reps, qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                    multiply_qubit_reps_by=multiply_qubit_reps_by, verbose=verbose, logger=rr_logger, qick_verbose=qick_verbose)
            rabi_I, rabi_Q, rabi_gains, *_ = rabi.run()
            del rabi

            all_rabi_I.append(rabi_I)
            all_rabi_Q.append(rabi_Q)

            # compute signal magnitude √(I²+Q²)
            mag = np.sqrt(np.array(rabi_I) ** 2 + np.array(rabi_Q) ** 2)
            signal_map.append(mag)

        signal_map = np.vstack(signal_map)  # shape (freq_steps, len(rabi_gains))

        if save_data_h5:
            chev_data = create_data_dict(chev_keys, save_r, list_of_all_qubits)
            # also stack I and Q data into arrays of shape (freq_steps, n_gain_points)
            all_rabi_I = np.vstack(all_rabi_I)
            all_rabi_Q = np.vstack(all_rabi_Q)

            chev_data[QubitIndex]['I'][0] = all_rabi_I #[0] is the round number, always zero since we don't use that parameter in this script
            chev_data[QubitIndex]['Q'][0] = all_rabi_Q
            chev_data[QubitIndex]['Gains'][0] = rabi_gains
            chev_data[QubitIndex]['Freqs_MHz'][0] = freqs_mhz
            chev_data[QubitIndex]['q_center_freq_MHz'][0] = qubit_freq
            chev_data[QubitIndex]['res_freq_ge_MHz'][0] = this_res_freq

            saver_chev = Data_H5(path_saveplots_chev, chev_data, 0, save_r)
            saver_chev.save_to_h5('rabi_ge_chevron')
            del saver_chev

        # Plot chevron
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(signal_map, aspect='auto', origin='lower',
            extent=[
                rabi_gains[0],  # gain min
                rabi_gains[-1],  # gain max
                freqs_mhz[0],  # freq min (MHz)
                freqs_mhz[-1]  # freq max (MHz)
            ])
        ax.set_xlabel('Gain (amplitude)')
        ax.set_ylabel('Qubit Drive frequency (MHz)')
        ax.set_title(f'Rabi Chevron: Qubit {QubitIndex + 1}')
        plt.colorbar(im, ax=ax, label='IQ Signal Mag (a. u.)')
        plt.tight_layout()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rabichevron_Q{QubitIndex + 1}_{timestamp}.png"
        fig.savefig(os.path.join(path_saveplots_chev, filename), dpi=fig_quality)
        plt.close(fig)

        del experiment
        del experiment_template