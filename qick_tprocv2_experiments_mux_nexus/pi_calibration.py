import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes, copy pasted from QUIET
import datetime
import time
import visdom
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus/"))
from section_001_time_of_flight import TOFExperiment
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from system_config import QICK_experiment
from section_003_punch_out_ge_mux import PunchOut
from expt_config import expt_cfg

n= 1#000
save_r = 1            # how many rounds to save after
signal = 'None'       #'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization has happened
save_figs = True    # save plots for everything as you go along the RR script?
live_plot = True      # for live plotting open http://localhost:8097/ on firefox
fit_data = True      # fit the data here and save or plot the fits? Should be false! we fit in post processing. But spec and rabi always fit no matter what the flag is
save_data_h5 = True   # save all of the data to h5 files?

increase_qubit_reps = False #if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0 #only has impact if previous line is True
multiply_qubit_reps_by = 2 #only has impact if the line two above is True

outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30/", str(datetime.date.today())) #Change run number in each new run
custom_Ramsey=False

dictionary_qs = [3]#[0, 1, 2, 3] #needs to be the total number of qubits that you have
Qs_to_look_at = [3]#[0, 1, 2, 3] #only list the qubits you want to do the RR for

# optimization outputs
res_leng_vals = [5.6, 5.85, 6.35, 3.35] #from 1/31-2/1 optimization
res_gain = [0.34, 0.3, 0.34, 0.375] #from 1/31-2/1 optimization, after implementing punchout threshold
freq_offsets = [-0.32, -0.16, -0.08, -0.08] #from 1/31-2/1 optimization, after implementing punchout threshold


#NOTE: Everything below this line was copy pasted from QUIET, to update this RR script

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num']
t2r_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num']
t2e_keys = ['T2E', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num']

#initialize a dictionary to store those values
res_data = create_data_dict(res_keys, save_r, dictionary_qs)
qspec_data = create_data_dict(qspec_keys, save_r, dictionary_qs)
rabi_data = create_data_dict(rabi_keys, save_r, dictionary_qs)
ss_data = create_data_dict(ss_keys, save_r, dictionary_qs)
t1_data = create_data_dict(t1_keys, save_r, dictionary_qs)
t2r_data = create_data_dict(t2r_keys, save_r, dictionary_qs)
t2e_data = create_data_dict(t2e_keys, save_r, dictionary_qs)

batch_num=0
j = 0
angles=[]
while j < n:
    j += 1
    for QubitIndex in Qs_to_look_at:
        #Get the config for this qubit
        experiment = QICK_experiment(outerFolder, DAC_attenuator1 = 5, DAC_attenuator2 = 10, ADC_attenuator = 10)

        #Mask out all other resonators except this one
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex])
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        ###################################################### TOF #####################################################
        #tof        = TOFExperiment(QubitIndex, outerFolder, experiment, j, save_figs)
        #tof.run(experiment.soccfg, experiment.soc)
        #del tof

        # ################################################## Res spec ####################################################
        # try:
        #     res_spec = ResonanceSpectroscopy(QubitIndex, outerFolder, j, save_figs, experiment)
        #     res_freqs, freq_pts, freq_center, amps = res_spec.run(experiment.soccfg, experiment.soc)
        #     experiment.readout_cfg['res_freq_ge'] = res_freqs
        #     offset = freq_offsets[QubitIndex]  # use optimized offset values
        #     offset_res_freqs = [r + offset for r in res_freqs]
        #     experiment.readout_cfg['res_freq_ge'] = offset_res_freqs
        #     del res_spec
        # except Exception as e:
        #     print(f'Got the following error, continuing: {e}')
        #     continue  # skip the rest of this qubit
        #
        # # ############################################ Roll Signal into I ##############################################
        # # #get the average theta value, then use that to rotate the signal. Plug that value into system_config res_phase
        # # leng=4
        # # ss = SingleShot(QubitIndex, outerFolder, experiment, j, save_figs)
        # # fid, angle, iq_list_g, iq_list_e = ss.run(experiment.soccfg, experiment.soc)
        # # angles.append(angle)
        # # #print(angles)
        # # #print('avg theta: ', np.average(angles))
        # # del ss
        #
        # ################################################## Qubit spec ##################################################
        # try:
        #     q_spec = QubitSpectroscopy(QubitIndex, outerFolder, j, signal, save_figs, experiment, live_plot)
        #     qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq = q_spec.run(experiment.soccfg,
        #                                                                                      experiment.soc)
        #     # if these are None, fit didnt work
        #     if (qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None):
        #         print('QSpec fit didnt work, skipping the rest of this qubit')
        #         continue  # skip the rest of this qubit
        #
        #     experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
        #     print('Qubit freq for qubit ', QubitIndex + 1, ' is: ', float(qubit_freq))
        #     del q_spec
        #
        # except Exception as e:
        #     print(f'Got the following error, continuing: {e}')
        #     continue  # skip the rest of this qubit

        ###################################################### Rabi ####################################################
        try:
            rabi = AmplitudeRabiExperiment(QubitIndex, outerFolder, j, signal, save_figs, experiment, live_plot,
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

