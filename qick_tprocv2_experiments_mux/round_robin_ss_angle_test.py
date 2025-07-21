import sys
import os
import numpy as np
import datetime
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_001_time_of_flight import TOFExperiment
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_005_single_shot_ge import GainFrequencySweep
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from section_003_punch_out_ge_mux import PunchOut
from system_config import QICK_experiment
from expt_config import *
import h5py
import time
import matplotlib.pyplot as plt
import copy


signal = 'None'        #'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization has happened
save_figs = True   # save plots for everything as you go along the RR script?
live_plot = False    # for live plotting open http://localhost:8097/ on firefox
fit_data = False # always set to False
FRIDGE = "QUIET"
number_of_qubits = 6 #for QUIET 6, for NEXUS 4
list_of_all_qubits = [0,1,2,3,4,5] #for QUIET [0, 1, 2, 3, 4, 5], for NEXUS [0, 1, 2, 3]
n = 3  # Number of rounds on each qubit
Qs = [0,1,2,3,4,5] #, 1, 2, 3, 4, 5

# For Nexus
# outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30", str(datetime.date.today())) #change run number in each new run

# For Quiet
outerFolder = os.path.join("/data/QICK_data/6transmon_run6/", str(datetime.date.today()))

def create_folder_if_not_exists(folder_path):
    """Creates a folder at the given path if it doesn't already exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


# Where to save readout length sweep data
prefix = str(datetime.date.today())
output_folder =outerFolder + "/SingleShotAngle_Test/"
create_folder_if_not_exists(output_folder)

ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}
ss_data = create_data_dict(ss_keys, 1, list_of_all_qubits)

# Optimization parameters for resonator spectroscopy
res_leng_vals = [4.3, 5, 5, 4, 4.5, 9]
res_gain = [1, 1, 0.6, 0.6, 1, 0.6]
freq_offsets = [0.15, -0.35, 0.1, -0.4, 0.15, 0.3]

j=0
for QubitIndex in Qs:
    # Get the config for this qubit
    experiment = QICK_experiment(outerFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10,
                                 fridge=FRIDGE)

    # Mask out all other resonators except this one
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex])
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

    ################################################## Res spec ####################################################

    res_spec = ResonanceSpectroscopy(QubitIndex, number_of_qubits, outerFolder, j, True,
                                     experiment)
    res_freqs, freq_pts, freq_center, amps, res_spec_config = res_spec.run()
    experiment.readout_cfg['res_freq_ge'] = res_freqs

    # incorporating offset (if you don't want to, then set all values inside freq_offsets to zero)
    offset = freq_offsets[QubitIndex]  # use optimized offset values
    offset_res_freqs = [r + offset for r in res_freqs]
    experiment.readout_cfg['res_freq_ge'] = offset_res_freqs

    del res_spec

    ################################################## Qubit spec ##################################################

    q_spec = QubitSpectroscopy(QubitIndex, number_of_qubits, outerFolder, j, signal,
                               True, experiment, live_plot)
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


    rabi = AmplitudeRabiExperiment(QubitIndex, number_of_qubits, outerFolder, j, signal,
                                   True, experiment, live_plot,
                                   increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
    rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_to_save = rabi.run()

    # if these are None, fit didnt work
    if (rabi_fit is None and pi_amp is None):
        print('Rabi fit didnt work, skipping the rest of this qubit')
        continue  # skip the rest of this qubit

    experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
    print('Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))

    #-----------SS meaurements----------------------------
    for i in range(n):
        QubitIndex = int(QubitIndex)  # Ensure QubitIndex is an integer

        # Set specific configuration values for each iteration
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]  # Set the readout pulse length to what has been optimized previously

        # Set gain for the current qubit
        gain = res_gain[QubitIndex]
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=gain)
        experiment.readout_cfg['res_gain_ge'] = res_gains

        ss = SingleShot(QubitIndex, tot_num_of_qubits, outerFolder, j, save_figs, experiment=experiment)
        fid, angle, iq_list_g, iq_list_e, sys_config_ss = ss.run()
        I_g = iq_list_g[QubitIndex][0].T[0]
        Q_g = iq_list_g[QubitIndex][0].T[1]
        I_e = iq_list_e[QubitIndex][0].T[0]
        Q_e = iq_list_e[QubitIndex][0].T[1]

        fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(
            data=[I_g, Q_g, I_e, Q_e], cfg=ss.config, plot=save_figs)

        ss_data[QubitIndex]['Fidelity'][0] = fid
        ss_data[QubitIndex]['Angle'][0] = angle
        ss_data[QubitIndex]['Dates'][0] = (
            time.mktime(datetime.datetime.now().timetuple()))
        ss_data[QubitIndex]['I_g'][0] = I_g
        ss_data[QubitIndex]['Q_g'][0] = Q_g
        ss_data[QubitIndex]['I_e'][0] = I_e
        ss_data[QubitIndex]['Q_e'][0] = Q_e
        ss_data[QubitIndex]['Round Num'][0] = j
        ss_data[QubitIndex]['Batch Num'][0] = i
        ss_data[QubitIndex]['Exp Config'][0] = expt_cfg
        ss_data[QubitIndex]['Syst Config'][0] = sys_config_ss

        saver_ss = Data_H5(outerFolder, ss_data, 0, 1)
        saver_ss.save_to_h5('SS')
        del saver_ss
        del ss_data

        ss_data = create_data_dict(ss_keys, 1, list_of_all_qubits)

    del experiment
