import sys
import os
import time
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus"))
from system_config import QICK_experiment
from tomography_modified  import TomographyMeasurement
from tomography_modified import AllQubitTomographyMeasurement
from expt_config import *
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
from windfreak import SynthHD
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

# outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30", str(datetime.date.today()))

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files

qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num']


#initialize a dictionary to store those values

qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)





#
# synth = SynthHD('/dev/ttyACM1')
#
# synth[0].power =     -12.85
# synth[0].frequency = 7.826e9
# synth[0].enable = False #True
# time.sleep(5)
################################################ optimization outputs ##################################################
# For NEXUS
res_leng_vals = [5.8, 3.8, 4, 4.6] #[6.15, 5.85, 6.45, 5.7] # from 2/19/2025 optimization, after punchout test
res_gain = [0, 0, 0, 0] #[0.38, 0.26, 0.28, 0.31]#[0.3143, 0.1857, 0.1429, 0.1857] # from 2/19/2025 optimization, after punchout test
freq_offsets = [0, 0, 0, 0] #[0.0, -0.0667, -0.2667, -0.400] # from 2/19/2025 optimization, after punchout test
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
resPhases=np.zeros(4)
fids=np.zeros(4)
# synth = SynthHD('/dev/ttyACM1')

substudy = 'Ba_sub_study' # 'Cs_3plates_sub_study' 'Cs_6plates_sub_study' 'Cs_9plates_sub_study'

while j < n:
    j += 1
    batch=j


    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")

    outerFolder = os.path.join(f"/home/nexusadmin/qick/NEXUS_sandbox/Data/Run31/Charge_Tomography/{substudy}", str(datetime.date.today()), f'batch_{batch}')
    saveFolder = outerFolder + '//repeated_tomoography' + formatted_datetime
    ssf_I_g = []
    ssf_I_e = []
    ssf_Q_g = []
    ssf_Q_e = []
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

        ################################################## Qubit spec ##################################################
        try:
            #experiment.readout_cfg['res_phases'] = res_phases
            # nowq = time.time()

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

            experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
            stored_qspec_list[QubitIndex] = float(qubit_freq)  # update the stored value
            # logging.info('Qubit freq for qubit ', QubitIndex + 1 ,' is: ',float(qubit_freq))
            print('Qubit freq for qubit ', QubitIndex + 1 ,' is: ',float(qubit_freq))
            # qfs[QubitIndex][j - 1] = float(qubit_freq)
            # if QubitIndex==0:
            #     q1time.append(nowq)
            # elif QubitIndex==1:
            #     q2time.append(nowq)
            # elif QubitIndex==2:
            #     q3time.append(nowq)
            # elif QubitIndex==3:
            #     q4time.append(nowq)
            del q_spec

        except Exception as e:
            # logging.exception(f'Got the following error, continuing: {e}')
            print(f'Got the following error, continuing: {e}')
            continue #skip the rest of this qubit

        ###################################################### Rabi ####################################################
        #try:
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

        # except Exception as e:
        #     # logging.exception(f'Got the following error, continuing: {e}')
        #     print(f'Got the following error, continuing: {e}')
        #     continue #skip the rest of this qubit

        # ########################################## Single Shot Measurements ############################################

        try:
            # experiment.readout_cfg['res_phases'] = res_phases
            # experiment.qubit_cfg['pi_amp'][QubitIndex] = angle
            timestamp = time.strftime("%H%M%S")
            ss = SingleShot(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, save_figs, experiment)
            fid, angle, iq_list_g, iq_list_e = ss.run(experiment.soccfg, experiment.soc)
            I_g = iq_list_g[QubitIndex][0].T[0]
            Q_g = iq_list_g[QubitIndex][0].T[1]
            I_e = iq_list_e[QubitIndex][0].T[0]
            Q_e = iq_list_e[QubitIndex][0].T[1]

            fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(
                data=[I_g, Q_g, I_e, Q_e], cfg=ss.config, plot=save_figs)
            # experiment.qubit_cfg['res_phase'][QubitIndex] = angle
            # res_phases[QubitIndex]=angle
            # np.savez(outerFolder+timestamp+'ssf'+f'Q{QubitIndex+1}'+f'round{j}', fid=fid, threshold=threshold, angle=angle, ig_new=ig_new, ie_new=ie_new)

        except Exception as e:
            # logging.exception(f'Got the following error, continuing: {e}')
            print(f'Got the following error, continuing: {e}')
            continue  # skip the rest of this qubit

        # ###################################################### T1 ######################################################
        try:
            t1 = T1Measurement(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, signal, save_figs, experiment, live_plot, fit_data,
                               increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
            t1_est, t1_err, t1_I, t1_Q, t1_delay_times, q1_fit_exponential = t1.run(experiment.soccfg, experiment.soc)
            del t1

        except Exception as e:
            # logging.exception(f'Got the following error, continuing: {e}')
            print(f'Got the following error, continuing: {e}')
            continue #skip the rest of this qubit

        if save_data_h5:
            print('collecting data')
            ##############----Collect QSpec Results----------------
            qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1]=time.mktime(datetime.datetime.now().timetuple())
            qspec_data[QubitIndex]['I'][j - batch_num * save_r - 1] = qspec_I
            qspec_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = qspec_Q
            qspec_data[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = qspec_freqs
            qspec_data[QubitIndex]['I Fit'][j - batch_num * save_r - 1] = qspec_I_fit
            qspec_data[QubitIndex]['Q Fit'][j - batch_num * save_r - 1] = qspec_Q_fit
            qspec_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            qspec_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num

            #---------------------Collect Single Shot Results----------------
            ss_data[QubitIndex]['Fidelity'][j - batch_num * save_r - 1] = fid
            ss_data[QubitIndex]['Angle'][j - batch_num * save_r - 1] = angle
            ss_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = time.mktime(datetime.datetime.now().timetuple())
            ss_data[QubitIndex]['I_g'][j - batch_num * save_r - 1] = I_g
            ss_data[QubitIndex]['Q_g'][j - batch_num * save_r - 1] = Q_g
            ss_data[QubitIndex]['I_e'][j - batch_num * save_r - 1] = I_e
            ss_data[QubitIndex]['Q_e'][j - batch_num * save_r - 1] = Q_e
            ss_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            ss_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num

            # ---------------------Collect T1 Results----------------
            t1_data[QubitIndex]['T1'][j - batch_num*save_r - 1] = t1_est
            t1_data[QubitIndex]['Errors'][j - batch_num*save_r - 1] = t1_err
            t1_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = time.mktime(datetime.datetime.now().timetuple())
            t1_data[QubitIndex]['I'][j - batch_num*save_r - 1] = t1_I
            t1_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = t1_Q
            t1_data[QubitIndex]['Delay Times'][j - batch_num*save_r - 1] = t1_delay_times
            t1_data[QubitIndex]['Fit'][j - batch_num*save_r - 1] = q1_fit_exponential
            t1_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            t1_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            print('collection complete')
    ################################################## Potentially Save ################################################
    save_data_h5=True
    if save_data_h5:
        # Check if you are at the right round number
        # If so, then save all of the data and change the round num so you replace data starting next round
        print('saving data')


        if j % save_r == 0:
            batch_num += 1


            # # --------------------------save QSpec-----------------------
            saver_qspec = Data_H5(outerFolder, qspec_data, batch_num, save_r)
            saver_qspec.save_to_h5('QSpec')
            del saver_qspec
            del qspec_data

            #
            # # --------------------------save SS-----------------------
            saver_ss = Data_H5(outerFolder, ss_data, batch_num, save_r)
            saver_ss.save_to_h5('SS')
            del saver_ss
            del ss_data
            #
            # # --------------------------save t1-----------------------
            saver_t1 = Data_H5(outerFolder, t1_data, batch_num, save_r)
            saver_t1.save_to_h5('T1')
            del saver_t1
            del t1_data

            # reset all dictionaries to none for safety

            qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
            ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
            t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)

            print('data saving complete')


            




    ######################  Tomography  ##################################################



    start_voltage = 0  # V
    stop_voltage = 0  # 0.1 #V
    voltage_pts = 30

    ## Get num of rounds to use by total time you want, or just set manually below:
    run_time = 0.1  # hrs, 11pm to 730am, ~8.5 hrs
    round_time = 2# min, actually more like 1.5 min but want to leave extra time
    round_num = int(run_time * 60 / round_time)


    experiment = QICK_experiment(outerFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10)


    start_time = time.time()
    qs_tomography = AllQubitTomographyMeasurement(saveFolder, outerFolder, number_of_qubits,experiment,  fids, ssf_I_g, ssf_I_e, ssf_Q_g, ssf_Q_e )
    qs_tomography.allq_run_tomography(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts,
                                      round_num, plot=False, save=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for the twpa sweep: {elapsed_time:.2f} seconds")
    print('experiment.soccfg',experiment.soccfg)

    del qs_tomography
    del experiment





