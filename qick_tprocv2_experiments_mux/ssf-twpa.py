import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
import visdom
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))  ## Change for quiet vs nexus
# sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_001_time_of_flight import TOFExperiment
from section_003_punch_out_ge_mux import PunchOut
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement

from system_config_nexus import QICK_experiment  ## Change for quiet vs nexus
# from system_config import QICK_experiment
from expt_config_nexus import expt_cfg, list_of_all_qubits  ## Change for quiet vs nexus
from windfreak import SynthHD
import time
################################################ Run Configurations ####################################################

n= 1
save_r = 1            # how many rounds to save after
signal = 'None'       #'I', or 'Q' depending on where the signal is (after optimization). Put'None' if no optimization
save_figs = True    # save plots for everything as you go along the RR script?
live_plot = False      # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True      # fit the data here and save or plot the fits?
save_data_h5 = True   # save all of the data to h5 files?
number_of_qubits = 4  #currently 4 for NEXUS, 6 for QUIET
Qs_to_look_at = [0]#, 1, 2, 3] #only list the qubits you want to do the RR for

increase_qubit_reps = False #if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0 #only has impact if previous line is True
multiply_qubit_reps_by = 2 #only has impact if the line two above is True

custom_Ramsey=False

outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30/", str(datetime.date.today()))
#outerFolder = os.path.join("/data/QICK_data/", str(datetime.date.today()))

################################################ optimization outputs ##################################################
# optimization outputs for NEXUS
res_leng_vals = [3.6, 3.6, 6.35, 3.6] # from 2/8/2025 optimization, after punchout test
res_gain = [0.3, 0.2, 0.25, 0.25] # from 2/8/2025 optimization, after punchout test
freq_offsets = [-0.1333, -0.0667, -0.2667, 0.0] # from 2/8/2025 optimization, after punchout test
####################################################### RR #############################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files
# res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num']
# qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num']
# rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num']
# ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num']
# t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num']
# t2r_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num']
# t2e_keys = ['T2E', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num']
#
# #initialize a dictionary to store those values
# res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
# qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
# rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
# ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
# t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
# t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
# t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)

batch_num=0
j = 0
angles=[]
################################################ optimization outputs ##################################################
# optimization outputs for NEXUS
res_leng_vals = [3.6, 3.6, 6.35, 3.6] # from 2/8/2025 optimization, after punchout test
res_gain = [0.3, 0.2, 0.25, 0.25] # from 2/8/2025 optimization, after punchout test
freq_offsets = [-0.1333, -0.0667, -0.2667, 0.0] # from 2/8/2025 optimization, after punchout test
####################################################################################################################
QubitIndex=0
#Get the config for this qubit
experiment = QICK_experiment(outerFolder, DAC_attenuator1 = 5, DAC_attenuator2 = 10, ADC_attenuator = 10)

#Mask out all other resonators except this one
res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex])
experiment.readout_cfg['res_gain_ge'] = res_gains
experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]
###############################################################################################################
synth = SynthHD('/dev/ttyACM0')

synth[0].power = -12.5
synth[0].frequency = 7.8239e9
synth[0].enable = True
time.sleep(5)

###############################################################################################################
###################################################### TOF #####################################################
#tof        = TOFExperiment(QubitIndex, outerFolder, experiment, j, save_figs)
#tof.run(experiment.soccfg, experiment.soc)
#del tof
# while j < n:
#     j=j+1
#     ################################################## Res spec ####################################################
#     try:
#         res_spec   = ResonanceSpectroscopy(QubitIndex,number_of_qubits, list_of_all_qubits, outerFolder, j, save_figs, experiment)
#         res_freqs, freq_pts, freq_center, amps = res_spec.run(experiment.soccfg, experiment.soc)
#         experiment.readout_cfg['res_freq_ge'] = res_freqs
#         offset = freq_offsets[QubitIndex] #use optimized offset values
#         offset_res_freqs = [r + offset for r in res_freqs]
#         experiment.readout_cfg['res_freq_ge'] = offset_res_freqs
#         del res_spec
#     except Exception as e:
#         print(f'Got the following error at Res Spec, continuing: {e}')
#         continue ###skip the rest of this qubit
#
#     # ############################################ Roll Signal into I ##############################################
#     # #get the average theta value, then use that to rotate the signal. Plug that value into system_config res_phase
#     # leng=4
#     # ss = SingleShot(QubitIndex, outerFolder, experiment, j, save_figs)
#     # fid, angle, iq_list_g, iq_list_e = ss.run(experiment.soccfg, experiment.soc)
#     # angles.append(angle)
#     # #print(angles)
#     # #print('avg theta: ', np.average(angles))
#     # del ss
#
#     ################################################## Qubit spec ##################################################
#     try:
#         q_spec = QubitSpectroscopy(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, signal, save_figs, experiment, live_plot)
#         qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq = q_spec.run(experiment.soccfg,
#                                                                                          experiment.soc)
#         # if these are None, fit didnt work
#         if (qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None):
#             print('QSpec fit didnt work, skipping the rest of this qubit')
#             continue #skip the rest of this qubit
#
#         experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
#         print('Qubit freq for qubit ', QubitIndex + 1 ,' is: ',float(qubit_freq))
#         del q_spec
#
#     except Exception as e:
#         print(f'Got the following error, continuing: {e}')
#         continue #skip the rest of this qubit
#
#
#     ###################################################### Rabi ####################################################
#     try:
#         rabi = AmplitudeRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, signal,
#                                        save_figs, experiment, live_plot,
#                                        increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
#         rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_to_save = rabi.run(experiment.soccfg, experiment.soc)
#
#         # if these are None, fit didnt work
#         if (rabi_fit is None and pi_amp is None):
#             print('Rabi fit didnt work, skipping the rest of this qubit')
#             continue  # skip the rest of this qubit
#
#         experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
#         print('Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
#         del rabi
#     except Exception as e:
#         print(f'Got the following error, continuing: {e}')
#         continue #skip the rest of this qubit
#
#     ########################################## Single Shot Measurements ############################################
#     #try:
#     startF=7.9e9     #  7.8239e9 - 0.5e9
#     stopF= 8.2e9     #7.8239e9  + 0.5e9
#     Fnumpts=6
#     startA= -13 #-12.5 - 1
#     stopA= -10.5#-12.5 + 1
#     Anumpts=25
#
#     twpaFs = np.linspace(startF, stopF, Fnumpts)
#     twpaAs = np.linspace(startA, stopA, Anumpts)
#     num=0
#     fidelity=np.zeros((len(twpaFs),len(twpaAs)))
#     for fi in range(len(twpaFs)):
#
#         for Ai in range(len(twpaAs)):
#
#             synth = SynthHD('/dev/ttyACM0')
#             synth[0].power = twpaAs[Ai]
#             synth[0].frequency = twpaFs[fi]
#             synth[0].enable = True
#             time.sleep(5)
#
#             fids=[]
#             for i in range(5):#
#                 print('fnum=', num)
#                 #timestamp = time.strftime("%H%M%S")
#                 ss = SingleShot(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder,  j, save_figs, experiment)
#                 fid, angle, iq_list_g, iq_list_e = ss.run(experiment.soccfg, experiment.soc)
#                 # I_g = iq_list_g[QubitIndex][0].T[0]
#                 # Q_g = iq_list_g[QubitIndex][0].T[1]
#                 # I_e = iq_list_e[QubitIndex][0].T[0]
#                 # Q_e = iq_list_e[QubitIndex][0].T[1]
#                 #
#                 # fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(
#                 #     data=[I_g, Q_g, I_e, Q_e], cfg=ss.config, plot=save_figs)save_figs
#                 fids.append(fid)
#                 # ienews.append(ie_new)
#             fid_avg=np.mean(np.array(fids))
#             fidelity[fi][Ai]=fid_avg
#
#         num=num+1
#     np.savez(outerFolder +  f'TWPASSF_Q{QubitIndex + 1}' , fidelity=fidelity)
#     synth[0].power = -12.5
#     synth[0].frequency = 7.8239e9
#     synth[0].enable = True
#     time.sleep(5)

# except Exception as e:
#     print(f'Got the following error, continuing: {e}')
#     continue #skip the rest of this qubit

