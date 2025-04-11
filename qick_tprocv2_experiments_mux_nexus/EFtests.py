import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
import logging
import visdom
sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_001_time_of_flight import TOFExperiment
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from EFspec import EFQubitSpectroscopy
from f_to_res_swap_spec import FtoResQubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from EF_rabi import EF_AmplitudeRabiExperiment
from ftores_rabi import FtoResAmplitudeRabiExperiment

# # For QUIET
# from system_config import QICK_experiment
# from expt_config import expt_cfg, list_of_all_qubits
from NetDrivers import E36300
# For NEXUS
from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits

################################################## Configure logging ###################################################
# logging.basicConfig(
#     level=logging.DEBUG,  # log all of the things
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("RR_script.log", mode='a'),
#         # also output log to the console (remove if you want only the file)
#         logging.StreamHandler(sys.stdout)
#     ]
# )

################################################ Run Configurations ####################################################
n= 1#20#100000
save_r = 1            # how many rounds to save after
signal = 'None'       #'I', or 'Q' depending on where the signal is (after optimization). Put'None' if no optimization
save_figs = True    # save plots for everything as you go along the RR script?
live_plot = False      # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True      # fit the data here and save or plot the fits?
save_data_h5 = True   # save all of the data to h5 files?
number_of_qubits = 4 # 4 for nexus, 6 for quiet
Qs_to_look_at = [0]#[0,1,2,3] #only list the qubits you want to do the RR for


increase_qubit_reps = False #if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0 #only has impact if previous line is True
multiply_qubit_reps_by = 2 #only has impact if the line two above is True

outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30/", str(datetime.date.today()))
# outerFolder = os.path.join("/data/QICK_data/", str(datetime.date.today()))

################################################ optimization outputs ##################################################
# For NEXUS
res_leng_vals = [5.1, 3.3, 4.5, 3.25] # from 2/27/2025 optimization
res_gain = [0.3143, 0.1857, 0.1429, 0.1857]#[0.365, 0.295, 0.255, 0.325] # from 2/27/2025 optimization
freq_offsets = [0,0,0,0]#[0.1333, -0.0667, -0.0667, -0.6667] # from 2/27/2025 optimization
#res_phases=[0,0,0,0]
####################################################### RR #############################################################

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
res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)

#initialize a simple list to store the qspec values in incase a fit fails
max_index = max(Qs_to_look_at)
stored_qspec_list = [None] * (max_index + 1)
stored_efqspec_list = [None] * (max_index + 1)

timestamp = time.strftime("%H%M%S")
start_time = time.time()
resfs=np.zeros((4,n))

qgs=np.zeros((4,n))
rabis=np.zeros((4))
efrabis=np.zeros((4))
efs=np.zeros((4))
FtoRes_fs= np.zeros((4))
FtoResrabis= np.zeros((4))

batch_num=0
j = 0
angles=[]
#
Bias_PS_ip = ['192.168.0.44', '192.168.0.44', '192.168.0.44', '192.168.0.41']  # IP address of bias PS (qubits 1-3 are the same PS)
Bias_ch = [1, 2, 3, 1]  # Channel number of qubit 1-4 on associated PS
# for Q in range(4):
Q=0
BiasPS = E36300(Bias_PS_ip[Q], server_port=5025)
start_voltage = 0 #V
BiasPS.setVoltage(start_voltage, Bias_ch[Q])
BiasPS.enable(Bias_ch[Q])
# start_voltage = 0.1125 #V
# stop_voltage = 0.1125 #V
# voltage_pts = 500
#
# Bias_ch = [1, 2, 3, 1]  # Channel number of qubit 1-4 on associated PS
# vcent=[0.098, 0.105, 0.0125,0.109]
# vwide=[0.15 , 0.075, 0.075, 0.15]

while j < n:
    j += 1
    for QubitIndex in Qs_to_look_at:
        # Q=QubitIndex
        # start_voltage = 0.1125
        #BiasPS = E36300(Bias_PS_ip[QubitIndex], server_port=5025)
        #
        #
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

        ################################################# Res spec ####################################################
        #try:
        # nowres=time.time()
        res_spec   = ResonanceSpectroscopy(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, save_figs, experiment)
        #experiment.readout_cfg['res_phases'] = res_phases
        res_freqs, freq_pts, freq_center, amps = res_spec.run(experiment.soccfg, experiment.soc)
        print('res_freqs=',res_freqs)
        experiment.readout_cfg['res_freq_ge'] = res_freqs
        # offset = freq_offsets[QubitIndex] #use optimized offset values
        # offset_res_freqs = [r + offset for r in res_freqs]
        # experiment.readout_cfg['res_freq_ge'] = offset_res_freqs
        # resfs[QubitIndex][j-1]=res_freqs[QubitIndex]
        # if QubitIndex==0:
        #     r1time.append(nowres)
        # elif QubitIndex==1:
        #     r2time.append(nowres)
        # elif QubitIndex==2:
        #     r3time.append(nowres)
        # elif QubitIndex==3:
        #     r4time.append(nowres)
        del res_spec
        # except Exception as e:
        #     # logging.exception(f'Got the following error, continuing: {e}')
        #     print(f'Got the following error, continuing: {e}')
        #     continue #skip the rest of this qubit

        # # ############################################ Roll Signal into I ##############################################
        # # #get the average theta value, then use that to rotate the signal. Plug that value into system_config res_phase
        # # leng=4
        # # ss = SingleShot(QubitIndex, outerFolder, experiment, j, save_figs)
        # # fid, angle, iq_list_g, iq_list_e = ss.run(experiment.soccfg, experiment.soc)
        # # angles.append(angle)
        # # #logging.info(angles)
        # # #logging.info('avg theta: ', np.average(angles))
        # # del ss
        #
        # ################################################## Qubit spec ##################################################
        #try:
            #experiment.readout_cfg['res_phases'] = res_phases
            # nowq = time.time()
        q_spec = QubitSpectroscopy(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, signal, save_figs, experiment, live_plot)
        qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq = q_spec.run(experiment.soccfg,
                                                                                         experiment.soc)
        mag=np.sqrt(qspec_I**2 + qspec_Q**2)
        q_freq= qspec_freqs[np.argmax(mag)]
        experiment.qubit_cfg['qubit_freq_ge'] = q_freq
        # # if these are None, fit didnt work. use the last value
        # # if qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None:
        # #     if stored_qspec_list[QubitIndex] is not None:
        # #         experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)#stored_qspec_list[QubitIndex]
        # #         # logging.warning(f"Using previous stored value: {stored_qspec_list[QubitIndex]}")
        # #         print(f"Using previous stored value: {float(qubit_freq)}")
        # #     else:
        # #         # logging.warning('There were no previous qubit spec values stored, skipping rest of this qubit')
        # #         print('There were no previous qubit spec values stored, skipping rest of this qubit')
        # #         continue
        #
        # experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
        # # stored_qspec_list[QubitIndex] = float(qubit_freq)  # update the stored value
        # # logging.info('Qubit freq for qubit ', QubitIndex + 1 ,' is: ',float(qubit_freq))
        # print('Qubit freq for qubit ', QubitIndex + 1 ,' is: ',float(qubit_freq))
        # # qfs[QubitIndex][j - 1] = float(qubit_freq)
        # # if QubitIndex==0:
        # #     q1time.append(nowq)
        # # elif QubitIndex==1:
        # #     q2time.append(nowq)
        # # elif QubitIndex==2:
        # #     q3time.append(nowq)
        # # elif QubitIndex==3:
        # #     q4time.append(nowq)
        # del q_spec

        # except Exception as e:
        #     # logging.exception(f'Got the following error, continuing: {e}')
        #     print(f'Got the following error, continuing: {e}')
        #     continue #skip the rest of this qubit
        #

        ###################################################### Rabi ####################################################
        # for i in range(2):
        #     # if i ==0:
        #     #     BiasPS.setVoltage(vcent[QubitIndex], Bias_ch[QubitIndex])
        #     #     BiasPS.enable(Bias_ch[QubitIndex])
        #     # elif i==1:
        #     #     BiasPS.setVoltage(vwide[QubitIndex], Bias_ch[QubitIndex])
        #     #     BiasPS.enable(Bias_ch[QubitIndex])
        #try:
            #experiment.readout_cfg['res_phases'] = res_phases
        nowq = time.time()
        experiment = QICK_experiment(outerFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10)

        rabi = AmplitudeRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, signal, save_figs,
                                       experiment, live_plot,
                                       increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
        rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_to_save = rabi.run(experiment.soccfg, experiment.soc)
        experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
        print('pi_amp',pi_amp)
        rabis[QubitIndex]=pi_amp
        # if these are None, fit didnt work
        if (rabi_fit is None and pi_amp is None):
            # logging.info('Rabi fit didnt work, skipping the rest of this qubit')
            print('Rabi fit didnt work, skipping the rest of this qubit')
            continue  # skip the rest of this qubit

        del rabi

        # except Exception as e:
        #     # logging.exception(f'Got the following error, continuing: {e}')
        #     print(f'Got the following error, continuing: {e}')
        #     continue #skip the rest of this qubit
        #     # BiasPS.setVoltage(0, Bias_ch[QubitIndex])
        #BiasPS.enable(Bias_ch[QubitIndex])



        #     ################################################## EF-Qubit spec ##################################################
        #try:
            # experiment.readout_cfg['res_phases'] = res_phases
            # nowq = time.time()
        # experiment.qubit_cfg['pi_amp'][QubitIndex] = rabis[QubitIndex]
        ef_q_spec = EFQubitSpectroscopy(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, signal,
                                   save_figs, experiment, live_plot)
        efqspec_I, efqspec_Q, efqspec_freqs, efqspec_I_fit, efqspec_Q_fit, efqubit_freq = ef_q_spec.run(experiment.soccfg,
                                                                                         experiment.soc)

        magef = np.sqrt(efqspec_I ** 2 + efqspec_Q ** 2)
        efq_freq = efqspec_freqs[np.argmax(magef)]
        experiment.qubit_cfg['qubit_freq_ef'][QubitIndex]  = efq_freq
        # experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] = float(efqubit_freq)
        # efs[QubitIndex]=efqubit_freq
        # print('efqubit_freq',efqubit_freq)
        # # if these are None, fit didnt work. use the last value
        # # if efqspec_I_fit is None and efqspec_Q_fit is None and efqubit_freq is None:
        # #     if stored_qspec_list[QubitIndex] is not None:
        # #         #experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] = float(efqubit_freq)
        # #         # logging.warning(f"Using previous stored value: {stored_qspec_list[QubitIndex]}")
        # #         print(f"Using previous stored value: {float(efqubit_freq)}")
        # #     else:
        # #         # logging.warning('There were no previous qubit spec values stored, skipping rest of this qubit')
        # #         print('There were no previous qubit spec values stored, skipping rest of this qubit')
        # #         continue
        #
        # #experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] = float(efqubit_freq)
        # #stored_efqspec_list[QubitIndex] = float(efqubit_freq)  # update the stored value
        # # logging.info('Qubit freq for qubit ', QubitIndex + 1 ,' is: ',float(qubit_freq))
        # print('Qubit ef-freq for qubit ', QubitIndex + 1, ' is: ', float(efqubit_freq))
        #
        del ef_q_spec

        # except Exception as e:
        #     # logging.exception(f'Got the following error, continuing: {e}')
        #     print(f'Got the following error, continuing: {e}')
        #     continue  # skip the rest of this qubit



        ###################################################### EF Rabi ####################################################
        # for i in range(2):
        #     # if i ==0:
        #     #     BiasPS.setVoltage(vcent[QubitIndex], Bias_ch[QubitIndex])
        #     #     BiasPS.enable(Bias_ch[QubitIndex])
        #     # elif i==1:
        #     #     BiasPS.setVoltage(vwide[QubitIndex], Bias_ch[QubitIndex])
        #     #     BiasPS.enable(Bias_ch[QubitIndex])
        # try:
        ##experiment.readout_cfg['res_phases'] = res_phases
        # nowq = time.time()
        # experiment = QICK_experiment(outerFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10)
        #experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] =efs[QubitIndex]
        #experiment.qubit_cfg['pi_amp'][QubitIndex] = rabis[QubitIndex]
        efrabi = EF_AmplitudeRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, signal, save_figs,
                                       experiment, live_plot, increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
        efrabi_I, efrabi_Q, efrabi_gains, efrabi_fit, efpi_amp, ef_sys_config_to_save = efrabi.run(experiment.soccfg, experiment.soc)
        efrabis[QubitIndex] = efpi_amp
        experiment.qubit_cfg['pi_ef_amp'][QubitIndex] =  efpi_amp
        print('efpi_amp',efpi_amp)
        # #if these are None, fit didnt work
        # if (efrabi_fit is None and efpi_amp is None):
        #     # logging.info('Rabi fit didnt work, skipping the rest of this qubit')
        #     print('ef Rabi fit didnt work, skipping the rest of this qubit')
        #     continue  # skip the rest of this qubit

        del efrabi

        # except Exception as e:
        #     # logging.exception(f'Got the following error, continuing: {e}')
        #     print(f'Got the following error, continuing: {e}')
        #     continue #skip the rest of this qubit
        #     # BiasPS.setVoltage(0, Bias_ch[QubitIndex])
        # BiasPS.enable(Bias_ch[QubitIndex])
        #
        ###########################################################################################################
        ######################### FtoRes Spec #####################################################################
        #experiment.qubit_cfg['pi_ef_amp'][QubitIndex] = efrabis[QubitIndex]
        FtoResq_spec = FtoResQubitSpectroscopy(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, signal, save_figs, experiment, live_plot)
        FtoResqspec_I, FtoResqspec_Q, FtoResqspec_freqs, FtoResqspec_I_fit, FtoResqspec_Q_fit, FtoResqubit_freq = FtoResq_spec.run(experiment.soccfg,experiment.soc)
        # FtoRes_fs[QubitIndex] =  FtoResqubit_freq
        print('FtoResqubit_freq',FtoResqubit_freq)
        magFtoRes = np.sqrt(FtoResqspec_I ** 2 + FtoResqspec_Q ** 2)
        FtoResq_freq = FtoResqspec_freqs[np.argmin(magFtoRes)]
        experiment.qubit_cfg['qubit_freq_ftores'][QubitIndex] = FtoResq_freq

        ###################################################################################################################
        ######################################### FtoRes rabi ##########################################################
        #experiment.qubit_cfg['qubit_freq_ftores'][QubitIndex] = FtoRes_fs[QubitIndex]

        FtoResrabi = FtoResAmplitudeRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, signal,
                                            save_figs,
                                            experiment, live_plot,
                                            increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
        # FtoResrabi_I, FtoResrabi_Q, FtoResrabi_gains, FtoResrabi_fit, FtoRespi_amp, FtoRes_sys_config_to_save = FtoResrabi.run(experiment.soccfg,
        #                                                                                            experiment.soc)
        FtoResrabi_I, FtoResrabi_Q, FtoResrabi_gains = FtoResrabi.run(experiment.soccfg, experiment.soc)
        # FtoResrabis[QubitIndex] = FtoRespi_amp
        # print('FtoRespi_amp',FtoRespi_amp)
        # ########################################## Single Shot Measurements ############################################
        # if QubitIndex == 0:
        #     synth[0].power = -12.69
        #     synth[0].frequency = 7.826e9
        #     synth[0].enable = True
        #     time.sleep(0.5)
        #
        # elif QubitIndex == 1:
        #     synth[0].power = -12.85
        #     synth[0].frequency = 7.826e9
        #     synth[0].enable = True
        #     time.sleep(0.5)
        #
        # elif QubitIndex == 2:
        #     synth[0].power = -12.5
        #     synth[0].frequency = 7.8239e9
        #     synth[0].enable = True
        #     time.sleep(0.5)
        #
        # elif QubitIndex == 3:
        #     synth[0].power = -11.92
        #     synth[0].frequency = 7.771e9
        #     synth[0].enable = True
        #     time.sleep(0.5)
        # try:
        #     #experiment.readout_cfg['res_phases'] = res_phases
        #     #experiment.qubit_cfg['pi_amp'][QubitIndex] = angle
        #     timestamp = time.strftime("%H%M%S")
        #     ss = SingleShot(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder,  j, save_figs, experiment)
        #     fid, angle, iq_list_g, iq_list_e = ss.run(experiment.soccfg, experiment.soc)
        #     I_g = iq_list_g[QubitIndex][0].T[0]
        #     Q_g = iq_list_g[QubitIndex][0].T[1]
        #     I_e = iq_list_e[QubitIndex][0].T[0]
        #     Q_e = iq_list_e[QubitIndex][0].T[1]
        #
        #     fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(
        #         data=[I_g, Q_g, I_e, Q_e], cfg=ss.config, plot=save_figs)
        #     #experiment.qubit_cfg['res_phase'][QubitIndex] = angle
        #     #res_phases[QubitIndex]=angle
        #     #np.savez(outerFolder+timestamp+'ssf'+f'Q{QubitIndex+1}'+f'round{j}', fid=fid, threshold=threshold, angle=angle, ig_new=ig_new, ie_new=ie_new)
        #
        # except Exception as e:
        #     # logging.exception(f'Got the following error, continuing: {e}')
        #     print(f'Got the following error, continuing: {e}')
        #     continue #skip the rest of this qubit
fnV=0
for Q in range(4):
    BiasPS = E36300(Bias_PS_ip[Q], server_port=5025)

    BiasPS.setVoltage(fnV, Bias_ch[Q])
    BiasPS.enable(Bias_ch[Q])