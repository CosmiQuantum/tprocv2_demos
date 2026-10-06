import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
import logging
import visdom
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
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
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE


################################################ Run Configurations ####################################################
n= 10
save_r = 1                           # how many rounds to save after
signal = 'None'                      # 'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization
save_figs = False                     # save plots for everything as you go along the RR script?
live_plot = False                     # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True                      # fit the data here and save or plot the fits?
save_data_h5 = True                  # save all of the data to h5 files?
verbose = False                      # print everything to the console in real time, good for debugging, bad for memory
debug_mode = False                   # if True, it disables the continuing function of RR if an error pops up in a class -- errors now stop the RR script
increase_qubit_reps = False           # if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0       # only has impact if previous line is True
multiply_qubit_reps_by = 2           # only has impact if the line two above is True
Qs_to_look_at = [0, 1, 2, 3, 4, 5]   # only list the qubits you want to do the RR for
#Qs_to_look_at = [5]   # only list the qubits you want to do the RR for
# set which of the following you'd like to run to 'True'
run_flags = {"tof": False, "res_spec": True, "q_spec": True, "ss": True, "rabi": True,
             "t1": True, "t2r": True, "t2e": True}

#outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30/", str(datetime.date.today()))
outerFolder = os.path.join("/data/QICK_data/6transmon_run6/", str(datetime.date.today()))
outerFolder = os.path.join(outerFolder,'Stats')

################################################ optimization outputs ##################################################
res_leng_vals = [3.25, 4.00, 2.25, 2.75, 3.5, 2.75]
res_gain = [1,0.95,0.85,0.95,0.9,0.9]
freq_offsets = [0, 0.1333, -0.1333, -0.2000, -0.2000, -0.1333]

################################################## Configure logging ###################################################
''' We need to create a custom logger and disable propagation like this 
to remove the logs from the underlying qick from saving to the log file for RR'''

if not os.path.exists(outerFolder): os.makedirs(outerFolder)
log_file = os.path.join(outerFolder, "RR_script.log")
rr_logger = logging.getLogger("custom_logger_for_rr_only")
rr_logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

rr_logger.addHandler(file_handler)
rr_logger.propagate = False  #dont propagate logs from underlying qick package

####################################################### RR #############################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num','Recycled QFreq',
              'Exp Config', 'Syst Config']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t2r_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
t2e_keys = ['T2E', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']

# initialize a dictionary to store those values
res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)

#initialize a simple list to store the qspec values in incase a fit fails

if live_plot:
    # Check if visdom is connected right away, otherwise, throw an error
    if not (viz := visdom.Visdom()).check_connection(timeout_seconds=5):
        raise RuntimeError("Visdom server not connected!, Type \'visdom\' into the command line and go to "
                           "http://localhost:8097/ on firefox")

stored_qspec_list = [None] * tot_num_of_qubits
res_freqs_list=[None]*tot_num_of_qubits
pi_amp_list=[0.78,0.79,0.8,0.79,0.8,0.8]
for QubitIndex in Qs_to_look_at:
    recycled_qfreq = False

    #Get the config for this qubit
    experiment = QICK_experiment(outerFolder, DAC_attenuator1 = 5, DAC_attenuator2 = 10, ADC_attenuator = 10,
                                 fridge=FRIDGE)
    experiment.create_folder_if_not_exists(outerFolder)

    #Mask out all other resonators except this one
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]


    ################################################## Res spec ####################################################
    if run_flags["res_spec"]:
        try:
            res_spec   = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, outerFolder, 0, save_figs,
                                               experiment = experiment, verbose = verbose, logger = rr_logger)
            res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
            experiment.readout_cfg['res_freq_ge'] = res_freqs
            offset = freq_offsets[QubitIndex] #use optimized offset values
            offset_res_freqs = [r + offset for r in res_freqs]
            experiment.readout_cfg['res_freq_ge'] = offset_res_freqs
            res_freqs_list[QubitIndex] = res_freqs
            del res_spec

        except Exception as e:
            if debug_mode:
                raise e # In debug mode, re-raise the exception immediately
            else:
                rr_logger.exception(f'Got the following error, continuing: {e}')
                if verbose: print(f'Got the following error, continuing: {e}')
                continue #skip the rest of this qubit

    ################################################## Qubit spec ##################################################
    if run_flags["q_spec"]:
        try:
            q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, outerFolder, 0, signal, save_figs,
                                       experiment=experiment, live_plot=live_plot,
                                       verbose=verbose, logger=rr_logger)
            qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq, sys_config_qspec = q_spec.run(
                experiment.soccfg,
                experiment.soc)
            # if these are None, fit didnt work. use the last value
            if qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None:
                if stored_qspec_list[QubitIndex] is not None:
                    experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = stored_qspec_list[QubitIndex]
                    rr_logger.warning(f"Using previous stored value: {stored_qspec_list[QubitIndex]}")
                    recycled_qfreq = True
                    if verbose: print(f"Using previous stored value: {stored_qspec_list[QubitIndex]}")
                else:
                    rr_logger.warning(
                        'There were no previous qubit spec values stored, skipping rest of this qubit')
                    if verbose: print('There were no previous qubit spec values stored, '
                                      'skipping rest of this qubit')
                    continue

            experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
            stored_qspec_list[QubitIndex] = float(qubit_freq)  # update the stored value
            rr_logger.info(f"Qubit freq for qubit {QubitIndex + 1} is: {float(qubit_freq)}")
            if verbose: print('Qubit freq for qubit ', QubitIndex + 1, ' is: ', float(qubit_freq))
            del q_spec

        except Exception as e:
            if debug_mode:
                raise e  # In debug mode, re-raise the exception immediately
            else:
                rr_logger.exception(f'Got the following error, continuing: {e}')
                if verbose: print(f'Got the following error, continuing: {e}')
                continue  # skip the rest of this qubit
    # ###################################################### Rabi ####################################################
    # if run_flags["rabi"]:
    #     try:
    #         rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, outerFolder, 0, signal, save_figs,
    #                                        experiment=experiment, live_plot=live_plot,
    #                                        increase_qubit_reps=increase_qubit_reps,
    #                                        qubit_to_increase_reps_for=qubit_to_increase_reps_for,
    #                                        multiply_qubit_reps_by=multiply_qubit_reps_by,
    #                                        verbose=verbose, logger=rr_logger)
    #         (rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp,
    #          sys_config_rabi) = rabi.run(experiment.soccfg, experiment.soc)
    #
    #         # if these are None, fit didnt work
    #         if (rabi_fit is None and pi_amp is None):
    #             rr_logger.info('Rabi fit didnt work, skipping the rest of this qubit')
    #             if verbose: print('Rabi fit didnt work, skipping the rest of this qubit')
    #             continue  # skip the rest of this qubit
    #
    #         experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
    #         pi_amp_list[QubitIndex] = float(pi_amp)
    #         rr_logger.info(f'Pi amplitude for qubit {QubitIndex + 1} is: {float(pi_amp)}')
    #         if verbose: print('Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
    #         del rabi
    #
    #     except Exception as e:
    #         if debug_mode:
    #             raise e  # In debug mode, re-raise the exception immediately
    #         else:
    #             rr_logger.exception(f'Got the following error, continuing: {e}')
    #             if verbose: print(f'Got the following error, continuing: {e}')
    #             continue  # skip the rest of this qubit


batch_num=0
j = 0
angles=[]
while j < n:
    j += 1
    for QubitIndex in Qs_to_look_at:
        recycled_qfreq = False

        #Get the config for this qubit
        experiment = QICK_experiment(outerFolder, DAC_attenuator1 = 5, DAC_attenuator2 = 10, ADC_attenuator = 10,
                                     fridge=FRIDGE)
        experiment.create_folder_if_not_exists(outerFolder)

        #Mask out all other resonators except this one
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        experiment.readout_cfg['res_freq_ge'] = res_freqs_list[QubitIndex]
        offset = freq_offsets[QubitIndex]  # use optimized offset values
        offset_res_freqs = [r + offset for r in res_freqs_list[QubitIndex]]
        experiment.readout_cfg['res_freq_ge'] = offset_res_freqs


        ################################################## Qubit spec ##################################################
        if run_flags["q_spec"]:
            try:
                q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, outerFolder, j, signal, save_figs,
                                           experiment = experiment, live_plot = live_plot,
                                           verbose = verbose, logger = rr_logger)
                qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq, sys_config_qspec = q_spec.run()
                # if these are None, fit didnt work. use the last value
                if qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None:
                    if stored_qspec_list[QubitIndex] is not None:
                        experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = stored_qspec_list[QubitIndex]
                        rr_logger.warning(f"Using previous stored value: {stored_qspec_list[QubitIndex]}")
                        recycled_qfreq = True
                        if verbose: print(f"Using previous stored value: {stored_qspec_list[QubitIndex]}")
                    else:
                        rr_logger.warning('There were no previous qubit spec values stored, skipping rest of this qubit')
                        if verbose: print('There were no previous qubit spec values stored, '
                                                           'skipping rest of this qubit')
                        continue

                experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
                stored_qspec_list[QubitIndex] = float(qubit_freq)  # update the stored value
                rr_logger.info(f"Qubit freq for qubit {QubitIndex + 1} is: {float(qubit_freq)}")
                if verbose: print('Qubit freq for qubit ', QubitIndex + 1, ' is: ', float(qubit_freq))
                del q_spec

            except Exception as e:
                if debug_mode:
                    raise e # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue #skip the rest of this qubit

        experiment.qubit_cfg['pi_amp'][QubitIndex] = pi_amp_list[QubitIndex]

        ###################################################### T1 ######################################################
        if run_flags["t1"]:
            try:
                t1 = T1Measurement(QubitIndex, tot_num_of_qubits, outerFolder, j, signal, save_figs,
                                   experiment=experiment,
                                   live_plot=live_plot, fit_data=fit_data,
                                   increase_qubit_reps=increase_qubit_reps,
                                   qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                   multiply_qubit_reps_by=multiply_qubit_reps_by,
                                   verbose=verbose, logger=rr_logger)
                t1_est, t1_err, t1_I, t1_Q, t1_delay_times, q1_fit_exponential, sys_config_t1 = t1.run(
                    experiment.soccfg,
                    experiment.soc)
                del t1

            except Exception as e:
                if debug_mode:
                    raise e  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue  # skip the rest of this qubit

        ############################################### Collect Results ################################################
        if save_data_h5:

            # ---------------------Collect T1 Results----------------
            if run_flags["t1"]:
                t1_data[QubitIndex]['T1'][j - batch_num * save_r - 1] = t1_est
                t1_data[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t1_err
                t1_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t1_data[QubitIndex]['I'][j - batch_num * save_r - 1] = t1_I
                t1_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = t1_Q
                t1_data[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t1_delay_times
                t1_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = q1_fit_exponential
                t1_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t1_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t1_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                t1_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t1

        del experiment

    ################################################## Potentially Save ################################################
    if save_data_h5:
        # Check if you are at the right round number
        # If so, then save all of the data and change the round num so you replace data starting next round
        if j % save_r == 0:
            batch_num+=1

            # --------------------------save t1-----------------------
            if run_flags["t1"]:
                saver_t1 = Data_H5(outerFolder, t1_data, batch_num, save_r)
                saver_t1.save_to_h5('T1')
                del saver_t1
                del t1_data
            t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)




