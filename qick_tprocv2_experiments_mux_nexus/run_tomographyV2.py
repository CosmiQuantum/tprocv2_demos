import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus"))
from system_config import QICK_experiment
from tomography import TomographyMeasurement
from tomography import AllQubitTomographyMeasurement

#sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_001_time_of_flight import TOFExperiment
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement

# For NEXUS
from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits

import datetime

################################################ Run Configurations ####################################################
n= 100000
save_r = 1            # how many rounds to save after
signal = 'None'       #'I', or 'Q' depending on where the signal is (after optimization). Put'None' if no optimization
save_figs = True    # save plots for everything as you go along the RR script?
live_plot = False      # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True      # fit the data here and save or plot the fits?
save_data_h5 = True   # save all of the data to h5 files?
number_of_qubits = 4 # 4 for nexus, 6 for quiet
Qs_to_look_at = [0, 1, 2, 3] #only list the qubits you want to do the RR for

increase_qubit_reps = False #if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0 #only has impact if previous line is True
multiply_qubit_reps_by = 2 #only has impact if the line two above is True

outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30", str(datetime.date.today()))

################################################ optimization outputs ##################################################
# For NEXUS
res_leng_vals = [3.0, 3.0, 3.0, 3.0] # from 2/19/2025 optimization, after punchout test
res_gain = [0.3143, 0.1857, 0.1429, 0.1857] # from 2/19/2025 optimization, after punchout test
freq_offsets = [0.0, -0.0667, -0.2667, -0.400] # from 2/19/2025 optimization, after punchout test
####################################################### RR #############################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

#

#initialize a simple list to store the qspec values in incase a fit fails
max_index = max(Qs_to_look_at)
stored_qspec_list = [None] * (max_index + 1)


batch_num=0
j = 0
angles=[]
rfreqs=np.zeros(4)
qfreqs=np.zeros(4)
rabiGs=np.zeros(4)
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

        ################################################## Res spec ####################################################

        try:
            res_spec   = ResonanceSpectroscopy(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, save_figs, experiment)
            res_freqs, freq_pts, freq_center, amps = res_spec.run(experiment.soccfg, experiment.soc)
            experiment.readout_cfg['res_freq_ge'] = res_freqs
            offset = freq_offsets[QubitIndex] #use optimized offset values
            offset_res_freqs = [r + offset for r in res_freqs]
            experiment.readout_cfg['res_freq_ge'] = offset_res_freqs
            rfreqs[QubitIndex]=offset_res_freqs[QubitIndex]
            del res_spec
        except Exception as e:
            # logging.exception(f'Got the following error, continuing: {e}')
            print(f'Got the following error, continuing: {e}')
            continue #skip the rest of this qubit

        # ############################################ Roll Signal into I ##############################################
        # #get the average theta value, then use that to rotate the signal. Plug that value into system_config res_phase
        # leng=4
        # ss = SingleShot(QubitIndex, outerFolder, experiment, j, save_figs)
        # fid, angle, iq_list_g, iq_list_e = ss.run(experiment.soccfg, experiment.soc)
        # angles.append(angle)
        # #logging.info(angles)
        # #logging.info('avg theta: ', np.average(angles))
        # del ss

        ################################################## Qubit spec ##################################################
        try:
            q_spec = QubitSpectroscopy(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, signal, save_figs, experiment, live_plot)
            qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq = q_spec.run(experiment.soccfg,
                                                                                             experiment.soc)
            # if these are None, fit didnt work. use the last value
            if qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None:
                if stored_qspec_list[QubitIndex] is not None:
                    experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = stored_qspec_list[QubitIndex]
                    # logging.warning(f"Using previous stored value: {stored_qspec_list[QubitIndex]}")
                    print(f"Using previous stored value: {stored_qspec_list[QubitIndex]}")
                else:
                    # logging.warning('There were no previous qubit spec values stored, skipping rest of this qubit')
                    print('There were no previous qubit spec values stored, skipping rest of this qubit')
                    continue
            # print('float(qubit_freq)', float(qubit_freq))
            experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
            stored_qspec_list[QubitIndex] = float(qubit_freq)  # update the stored value
            print('float(qubit_freq)',float(qubit_freq))
            qfreqs[QubitIndex] = float(qubit_freq)
            # logging.info('Qubit freq for qubit ', QubitIndex + 1 ,' is: ',float(qubit_freq))
            print('Qubit freq for qubit ', QubitIndex + 1 ,' is: ',float(qubit_freq))
            del q_spec

        except Exception as e:
            # logging.exception(f'Got the following error, continuing: {e}')
            print(f'Got the following error, continuing: {e}')
            continue #skip the rest of this qubit

        ###################################################### Rabi ####################################################
        try:
            rabi = AmplitudeRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, signal, save_figs, experiment, live_plot,
                                           increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
            rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_to_save  = rabi.run(experiment.soccfg, experiment.soc)

            # if these are None, fit didnt work
            if (rabi_fit is None and pi_amp is None):
                # logging.info('Rabi fit didnt work, skipping the rest of this qubit')
                print('Rabi fit didnt work, skipping the rest of this qubit')
                continue  # skip the rest of this qubit

            experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)


            rabiGs[QubitIndex] = float(pi_amp)
            # logging.info('Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
            print('Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
            del rabi

        except Exception as e:
            # logging.exception(f'Got the following error, continuing: {e}')
            print(f'Got the following error, continuing: {e}')
            continue #skip the rest of this qubit





    ######################  Tomography  ##################################################

        ## Repeated All Qubit Tomography


    try:
        ## Repeated All Qubit Tomography
        start_voltage = 0  # V
        stop_voltage = 0.15  # 0.1 #V
        voltage_pts = 45

        ## Get num of rounds to use by total time you want, or just set manually below:
        run_time = 3  # hrs, 11pm to 730am, ~8.5 hrs
        round_time = 2.4  # min, actually more like 1.5 min but want to leave extra time
        round_num = int(run_time * 60 / round_time)

        #rounds = 25

        #experiment = QICK_experiment(outerFolder)
        experiment = QICK_experiment(outerFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10)
        experiment.readout_cfg['res_freq_ge'] = rfreqs
        experiment.qubit_cfg['qubit_freq_ge'] = qfreqs
        experiment.qubit_cfg['pi_amp'] = rabiGs

        qs_tomography = AllQubitTomographyMeasurement(outerFolder, number_of_qubits,experiment)
        qs_tomography.allq_run_tomography(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts,
                                          round_num, plot=False, save=True)

        del qs_tomography
        del experiment
    #
    except Exception as e:
        # logging.exception(f'Got the following error, continuing: {e}')
        print(f'Got the following error, continuing: {e}')
        continue  # skip the rest of this qubit




## Unblock one of the follow block to do single qubit tomography or repeated tomography for all qubits

# ## Single Qubit Tomography
#
# qubit = 3  #Qubit to Run
# start_voltage = 0 #V
# stop_voltage = 0.1 #V
# voltage_pts = 30
#
# experiment = QICK_experiment(outerFolder)
# tomography = TomographyMeasurement(qubit-1, outerFolder, experiment)
# tomography.run_tomography(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts, plot=True, save = False)
#
# del tomography
# del experiment



## Repeated All Qubit Tomography
# start_voltage = 0 #V
# stop_voltage = 0.1#0.1 #V
# voltage_pts = 30
#
# ## Get num of rounds to use by total time you want, or just set manually below:
# run_time = 3 # hrs, 11pm to 730am, ~8.5 hrs
# round_time = 2 #min, actually more like 1.5 min but want to leave extra time
# round_num = int(run_time*60/round_time)
#
# rounds = 25
#
# experiment = QICK_experiment(outerFolder)
# qs_tomography = AllQubitTomographyMeasurement(outerFolder, experiment)
# qs_tomography.allq_run_tomography(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts, round_num, plot=False, save=True)
#
# del qs_tomography
# del experiment