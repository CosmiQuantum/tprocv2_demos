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
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from system_config import QICK_experiment
from section_003_punch_out_ge_mux import PunchOut
from expt_config import expt_cfg, list_of_all_qubits

################################################## Configure logging ###################################################
logging.basicConfig(
    level=logging.DEBUG,  # log all of the things
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("RR_script.log", mode='a'),
        # also output log to the console (remove if you want only the file)
        logging.StreamHandler(sys.stdout)
    ]
)

################################################ Run Configurations ####################################################
n= 100000
save_r = 1            # how many rounds to save after
signal = 'None'       #'I', or 'Q' depending on where the signal is (after optimization). Put'None' if no optimization
save_figs = False    # save plots for everything as you go along the RR script?
live_plot = False      # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = False      # fit the data here and save or plot the fits?
save_data_h5 = True   # save all of the data to h5 files?
Qs_to_look_at = [1] #only list the qubits you want to do the RR for

increase_qubit_reps = True #if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0 #only has impact if previous line is True
multiply_qubit_reps_by = 2 #only has impact if the line two above is True

#outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30/", str(datetime.date.today()))
outerFolder = os.path.join("/data/QICK_data/", str(datetime.date.today()))

################################################ optimization outputs ##################################################

res_leng_vals = [3.25, 4.00, 2.25, 2.75, 3.5, 2.75] #Final decision, for Danso at 3.5V     2.75
# res_gain = [0.9, 0.95, 0.95, 0.95, 0.9, 0.95]
res_gain = [1,0.95,0.85,0.95,0.9,0.9]
freq_offsets = [0, 0.1333, -0.1333, -0.2000, -0.2000, -0.1333] #-0.2000

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



################################ Do Res spec once per qubit and store the value ########################################
experiment = QICK_experiment(outerFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10)

for QubitIndex in Qs_to_look_at:
    # Mask out all other resonators except this one
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex])
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

    res_spec   = ResonanceSpectroscopy(QubitIndex, outerFolder, 0, save_figs, experiment)
    res_freqs, freq_pts, freq_center, amps = res_spec.run(experiment.soccfg, experiment.soc)
    experiment.readout_cfg['res_freq_ge'] = res_freqs
    offset = freq_offsets[QubitIndex] #use optimized offset values
    offset_res_freqs = [r + offset for r in res_freqs]
    experiment.readout_cfg['res_freq_ge'] = offset_res_freqs
    del res_spec

    ################################# Do Qspec once per qubit and store the value ######################################
    q_spec = QubitSpectroscopy(QubitIndex,outerFolder, 0, signal, save_figs, experiment, live_plot)
    qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq = q_spec.run(experiment.soccfg,
                                                                                     experiment.soc)

    experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
    stored_qspec_list[QubitIndex] = float(qubit_freq)  # update the stored value
    logging.info('Qubit freq for qubit ', QubitIndex + 1 ,' is: ',float(qubit_freq))
    del q_spec

    ################################### Do Rabi once per qubit and store the value #####################################
    rabi = AmplitudeRabiExperiment(QubitIndex,outerFolder, 0, signal, save_figs, experiment, live_plot,
                                   increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
    rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_to_save  = rabi.run(experiment.soccfg, experiment.soc)

    experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
    logging.info('Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
    del rabi


batch_num=0
j = 0
angles=[]
while j < n:
    j += 1
    for QubitIndex in Qs_to_look_at:
        #Mask out all other resonators except this one
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex])
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        ################################################## Qubit spec ##################################################
        try:
            q_spec = QubitSpectroscopy(QubitIndex,outerFolder, j, signal, save_figs, experiment, live_plot)
            qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq = q_spec.run(experiment.soccfg,
                                                                                             experiment.soc)
            # if these are None, fit didnt work. use the last value
            if qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None:
                if stored_qspec_list[QubitIndex] is not None:
                    experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = stored_qspec_list[QubitIndex]
                    logging.warning(f"Using previous stored value: {stored_qspec_list[QubitIndex]}")
                else:
                    logging.warning('There were no previous qubit spec values stored, skipping rest of this qubit')
                    continue

            experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
            stored_qspec_list[QubitIndex] = float(qubit_freq)  # update the stored value
            logging.info('Qubit freq for qubit ', QubitIndex + 1 ,' is: ',float(qubit_freq))
            del q_spec

        except Exception as e:
            logging.exception(f'Got the following error, continuing: {e}')
            continue #skip the rest of this qubit


        ###################################################### T1 ######################################################
        try:
            t1 = T1Measurement(QubitIndex, outerFolder, j, signal, save_figs, experiment, live_plot, fit_data,
                               increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
            t1_est, t1_err, t1_I, t1_Q, t1_delay_times, q1_fit_exponential = t1.run(experiment.soccfg, experiment.soc)
            del t1

        except Exception as e:
            logging.exception(f'Got the following error, continuing: {e}')
            continue #skip the rest of this qubit


        ############################################### Collect Results ################################################
        if save_data_h5:
            # ---------------------Collect QSpec Results----------------
            qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1]=time.mktime(datetime.datetime.now().timetuple())
            qspec_data[QubitIndex]['I'][j - batch_num * save_r - 1] = qspec_I
            qspec_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = qspec_Q
            qspec_data[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = qspec_freqs
            qspec_data[QubitIndex]['I Fit'][j - batch_num * save_r - 1] = qspec_I_fit
            qspec_data[QubitIndex]['Q Fit'][j - batch_num * save_r - 1] = qspec_Q_fit
            qspec_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            qspec_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num

            #---------------------Collect T1 Results----------------
            t1_data[QubitIndex]['T1'][j - batch_num*save_r - 1] = t1_est
            t1_data[QubitIndex]['Errors'][j - batch_num*save_r - 1] = t1_err
            t1_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = time.mktime(datetime.datetime.now().timetuple())
            t1_data[QubitIndex]['I'][j - batch_num*save_r - 1] = t1_I
            t1_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = t1_Q
            t1_data[QubitIndex]['Delay Times'][j - batch_num*save_r - 1] = t1_delay_times
            t1_data[QubitIndex]['Fit'][j - batch_num*save_r - 1] = q1_fit_exponential
            t1_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            t1_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num

            # ---------------------save last system config and expt_cfg---------------------
            saver_config = Data_H5(outerFolder)
            saver_config.save_config(sys_config_to_save, expt_cfg)
            del saver_config

        del experiment

    ################################################## Potentially Save ################################################
    if save_data_h5:
        # Check if you are at the right round number
        # If so, then save all of the data and change the round num so you replace data starting next round
        if j % save_r == 0:
            batch_num+=1

            # --------------------------save QSpec-----------------------
            saver_qspec = Data_H5(outerFolder, qspec_data, batch_num, save_r)
            saver_qspec.save_to_h5('QSpec')
            del saver_qspec
            del qspec_data
            # --------------------------save t1-----------------------
            saver_t1 = Data_H5(outerFolder, t1_data, batch_num, save_r)
            saver_t1.save_to_h5('T1')
            del saver_t1
            del t1_data

            # reset all dictionaries to none for safety
            qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
            t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)



