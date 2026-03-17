import copy
import sys
import os
import numpy as np
from sphinx.addnodes import document

# from tprocv2_demos.qick_tprocv2_experiments_mux.long_qubit_spectroscopy import fh_config

np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import logging
import visdom
import gc, copy
import time
#sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
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
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from test_active_reset import Active_Reset_test


def RR_IntraTomo(tsFolder, rounds):
    ################################################ Run Configurations ####################################################
    st = time.time()
    #
    n= rounds

    save_r = 1                           # how many rounds to save after
    signal = 'None'                      # 'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization
    save_figs = True                     # save plots for everything as you go along the RR script?
    live_plot = False                    # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
    fit_data = True                      # fit the data here and save or plot the fits?
    save_data_h5 = True                  # save all of the data to h5 files?
    verbose = False                          # print everything to the console in real time, good for debugging, bad for memory
    qick_verbose = False                  # qick verbose prints the progress bar for each qick experiment as it is happening (the red bar that fills out as more experiment rounds/reps are being done)
    debug_mode = False                   # if True, it disables the continuing function of RR if an error pops up in a class -- errors now stop the RR script
    thresholding = False                 # use internal QICK threshold for ratio of Binary values on y for rabi/t1/t2r/t2e, or analog avg when false
    increase_qubit_reps = False          # if you want to increase the reps for a qubit, set to True
    unmask = True                        # Do you want to use the unmasking feature to increase resonator gain?
    qubit_to_increase_reps_for = 0       # only has impact if previous line is True
    multiply_qubit_reps_by = 2           # only has impact if the line two above is True

    Qs_to_look_at = [0, 1, 2, 3] #[0,1,2,3]    # only list the qubits you want to do the RR for

    #One round take 11.27 minutes for all 4 qubits: Rspec, Qspec, Rabi, SS, and T1

    #debug
    print(FRIDGE)

    #Data saving info
    substudy_txt_notes = ('Intra-tomography RR') #'DD off, rear shield hole closed, no colimator, 0V bias') #'0V bias, ssf edit')#('DD off, rear shield hole closed, 0V bias')

    # set which of the following you'd like to run to 'True'
    run_flags = {"tof": False, "res_spec": True, "q_spec": True, "ss": True, "rabi": True,
                 "t1": True, "t2r": False, "t2e": False}

    # optimization outputs from qick board, unmasking set to true
    res_leng_vals = [3, 5.25, 4.5, 4] #[5, 4.75, 5, 4.25] #[9.25, 5.5, 6.25, 7.5] #Q1,~Q2,Q4 opt
    res_gain = [0.625, 0.375, 0.475, 0.475] #[0.6, 0.25, 0.35, 0.5] #[0.116, 0.0935, 0.1162, 0.14] #[0.75, 0.7, 0.8, 0.75] #Q1,~Q2,Q4 optimized
    freq_offsets = [-0.25, 0.2, -0.3, 0] #[0.05, -0.225, -0.2, -0.075] #[-0.1429, -0.1429, 0, -0.1429] #Q1,~Q2,Q4 optimized


    qubit_freqs_ef = [None]*4
    number_of_qubits = 4
    figure_quality = 200
    ################################################ Data Saving Setup ##################################################
    #Folders
    data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    RRfolder = os.path.join(tsFolder, 'RR')
    RRdatasetFolder = os.path.join(RRfolder, data_set)
    dataFolder = os.path.join(RRdatasetFolder, 'data')
    documentationFolder = os.path.join(RRdatasetFolder, 'documentation')
    if not os.path.exists(RRfolder):
        os.makedirs(RRfolder)
    if not os.path.exists(RRdatasetFolder):
        os.makedirs(RRdatasetFolder)
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)
    if not os.path.exists(documentationFolder):
        os.makedirs(documentationFolder)

    file_path = os.path.join(documentationFolder, 'sub_study_notes.txt')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(substudy_txt_notes)

    ################################################## Configure logging ###################################################
    ''' We need to create a custom logger and disable propagation like this
    to remove the logs from the underlying qick from saving to the log file for RR'''

    log_file = os.path.join(documentationFolder, "RR_script.log")
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
    res_keys = ['Dates', 'freq_pts', 'freq_center', 'I', 'Q', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
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

    act_keys = [ 'actI', 'actQ','noactI', 'noactQ', 'Syst Config']
    #initialize a simple list to store the qspec values in incase a fit fails
    stored_qspec_list = [None] * tot_num_of_qubits
    # True
    if live_plot:
        # Check if visdom is connected right away, otherwise, throw an error
        if not (viz := visdom.Visdom()).check_connection(timeout_seconds=5):
            raise RuntimeError("Visdom server not connected!, Type \'visdom\' into the command line and go to "
                               "http://localhost:8097/ on firefox")

    # initialize a dictionary to store those values
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)

    t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
    t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
    t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)

    act_data = create_data_dict(act_keys, save_r, list_of_all_qubits)

    RR_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    batch_num=0
    j = 0
    angles=[]
    while j < n:
        j += 1
        for QubitIndex in Qs_to_look_at:
            recycled_qfreq = False

            #Get the config for this qubit
            experiment = QICK_experiment(documentationFolder, DAC_attenuator1 = 10, DAC_attenuator2 = 15, qubit_DAC_attenuator1 = 5,
                                         qubit_DAC_attenuator2 = 4, ADC_attenuator = 17, fridge=FRIDGE) # ADC_attenuator MUST be above 16dB
            experiment.create_folder_if_not_exists(documentationFolder)

            #Mask out all other resonators except this one
            res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
            experiment.readout_cfg['res_gain_ge'] = res_gains
            experiment.readout_cfg['res_gain_ef'] = res_gains
            experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

            ###################################################### TOF #####################################################
            if run_flags["tof"]:
                tof        = TOFExperiment(QubitIndex, documentationFolder, experiment, j, save_figs, unmasking_resgain = unmask)
                tof.run()
                del tof

            ################################################# g-e Res spec ####################################################
            if run_flags["res_spec"]:
                try:
                    res_spec   = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, documentationFolder, j, save_figs,
                                                       experiment = experiment, verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                    res_freqs, freq_pts, freq_center, res_Iarr, res_Qarr, amps, sys_config_rspec = res_spec.run()
                    offset = freq_offsets[QubitIndex] #use optimized offset values or whats set at top of script based on pre_optimize flag
                    offset_res_freqs = [r + offset for r in res_freqs]
                    experiment.readout_cfg['res_freq_ge'] = offset_res_freqs
                    del res_spec

                except Exception as e:
                    if debug_mode:
                        raise e # In debug mode, re-raise the exception immediately
                    else:
                        rr_logger.exception(f'Got the following error, continuing: {e}')
                        if verbose: print(f'Got the following error, continuing: {e}')
                        continue #skip the rest of this qubit

            ################################################## g-e Qubit spec ##################################################
            if run_flags["q_spec"]:
                try:
                    q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, documentationFolder, j,
                                               signal, save_figs, plot_fit=True,experiment=experiment,
                                               live_plot=live_plot, verbose=verbose, logger=rr_logger, unmasking_resgain = unmask)
                    (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit,
                     qspec_Q_fit, qubit_freq, sys_config_qspec) = q_spec.run()

                    if qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None:
                        if stored_qspec_list[QubitIndex] is not None:
                            experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = stored_qspec_list[QubitIndex]
                            rr_logger.warning(f"Using previous stored value: {stored_qspec_list[QubitIndex]}")
                            recycled_qfreq = True
                            qubit_freq = stored_qspec_list[QubitIndex]
                            experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
                            stored_qspec_list[QubitIndex] = float(qubit_freq)
                            if verbose:
                                print(f"Using previous stored value: {qubit_freq}")
                        else:
                            rr_logger.warning(f"No stored g-e qubit spec value for qubit {QubitIndex}; skipping iteration.")
                            if verbose:
                                print('No stored g-e qubit spec value for qubit {QubitIndex}; skipping iteration.')
                            del q_spec

                        continue
                    else:
                        experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
                        stored_qspec_list[QubitIndex] = float(qubit_freq)
                    rr_logger.info(f"g-e Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                    if verbose:
                        print(f"g-e Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                    del q_spec

                except Exception as e:
                    if debug_mode:
                        raise e
                    rr_logger.exception(f"RR g-e QSpec error on qubit {QubitIndex}: {e}")
                    if verbose:
                        print(f"RR g-e QSpec error on qubit {QubitIndex}: {e}")
                    continue
            ###################################################### g-e Rabi ####################################################
            if run_flags["rabi"]:
                # try:
                rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, documentationFolder, j, signal, save_figs=save_figs,save_shots=False,
                                               experiment = experiment, live_plot = live_plot,
                                               increase_qubit_reps = increase_qubit_reps,
                                               qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                               multiply_qubit_reps_by = multiply_qubit_reps_by,
                                               verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                (rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp,
                 sys_config_rabi)  = rabi.run(thresholding=thresholding)

                # if these are None, fit didnt work
                if (rabi_fit is None and pi_amp is None):
                    rr_logger.info('g-e Rabi fit didnt work, skipping the rest of this qubit')
                    if verbose: print('g-e Rabi fit didnt work, skipping the rest of this qubit')
                    continue  # skip the rest of this qubit

                experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
                rr_logger.info(f'g-e Pi amplitude for qubit {QubitIndex + 1} is: {float(pi_amp)}')
                if verbose: print('g-e Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
                del rabi

                # except Exception as e:
                #     if debug_mode:
                #         raise e # In debug mode, re-raise the exception immediately
                #     else:
                #         rr_logger.exception(f'Got the following error, continuing: {e}')
                #         if verbose: print(f'Got the following error, continuing: {e}')
                #         continue #skip the rest of this qubit

            ########################################## g-e Single Shot Measurements ############################################
            if run_flags["ss"]:
                # try:
                ss = SingleShot(QubitIndex, tot_num_of_qubits, documentationFolder, j, save_figs, experiment = experiment,
                                verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                fid, angle, iq_list_g, iq_list_e, sys_config_ss = ss.run()
                print('fid', fid)
                I_g = iq_list_g[QubitIndex][0].T[0]
                Q_g = iq_list_g[QubitIndex][0].T[1]
                I_e = iq_list_e[QubitIndex][0].T[0]
                Q_e = iq_list_e[QubitIndex][0].T[1]

                # fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(
                #     data=[I_g, Q_g, I_e, Q_e], cfg=ss.config, plot=save_figs)
                #print(sys_config_ss)
                # except Exception as e:
                #     if debug_mode:
                #         raise  # In debug mode, re-raise the exception immediately
                #     else:
                #         rr_logger.exception(f'Got the following error, continuing: {e}')
                #         if verbose: print(f'Got the following error, continuing: {e}')
                #         continue #skip the rest of this qubit

            ###################################################### g-e T1 ######################################################
            if run_flags["t1"]:
                try:
                    t1 = T1Measurement(QubitIndex, tot_num_of_qubits, documentationFolder, j, signal, save_figs,
                                       experiment = experiment,
                                       live_plot = live_plot, fit_data = fit_data,
                                       increase_qubit_reps = increase_qubit_reps,
                                       qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                       multiply_qubit_reps_by = multiply_qubit_reps_by,
                                       verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                    t1_est, t1_err, t1_I, t1_Q, t1_delay_times, q1_fit_exponential, sys_config_t1 = t1.run(
                        thresholding=thresholding)
                    del t1

                except Exception as e:
                    if debug_mode:
                        raise e # In debug mode, re-raise the exception immediately
                    else:
                        rr_logger.exception(f'Got the following error, continuing: {e}')
                        if verbose: print(f'Got the following error, continuing: {e}')
                        continue #skip the rest of this qubit

            ###################################################### g-e T2R #####################################################
            if run_flags["t2r"]:
                try:
                    t2r = T2RMeasurement(QubitIndex, tot_num_of_qubits, documentationFolder, j, signal, save_figs,
                                         experiment = experiment, live_plot = live_plot, fit_data = fit_data,
                                         increase_qubit_reps = increase_qubit_reps,
                                         qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                         multiply_qubit_reps_by = multiply_qubit_reps_by,
                                         verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                    t2r_est, t2r_err, t2r_I, t2r_Q, t2r_delay_times, fit_ramsey, sys_config_t2r = t2r.run(
                        thresholding=thresholding)
                    del t2r

                except Exception as e:
                    if debug_mode:
                        raise e # In debug mode, re-raise the exception immediately
                    else:
                        rr_logger.exception(f'Got the following error, continuing: {e}')
                        if verbose: print(f'Got the following error, continuing: {e}')
                        continue #skip the rest of this qubit

            ##################################################### g-e T2E ######################################################
            if run_flags["t2e"]:
                try:
                    t2e = T2EMeasurement(QubitIndex, tot_num_of_qubits, documentationFolder, j, signal, save_figs,
                                         experiment = experiment, live_plot = live_plot, fit_data = fit_data,
                                         increase_qubit_reps = increase_qubit_reps,
                                         qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                         multiply_qubit_reps_by = multiply_qubit_reps_by,
                                         verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                    (t2e_est, t2e_err, t2e_I, t2e_Q, t2e_delay_times,
                     fit_t2e, sys_config_t2e) = t2e.run(thresholding=thresholding)
                    del t2e

                except Exception as e:
                    if debug_mode:
                        raise e # In debug mode, re-raise the exception immediately
                    else:
                        rr_logger.exception(f'Got the following error, continuing: {e}')
                        if verbose: print(f'Got the following error, continuing: {e}')
                        continue #skip the rest of this qubit

            ############################################### Collect Results ################################################
            if save_data_h5:
                # ---------------------Collect g-e Res Spec Results----------------
                if run_flags["res_spec"]:
                    res_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                        time.mktime(datetime.datetime.now().timetuple()))
                    res_data[QubitIndex]['freq_pts'][j - batch_num * save_r - 1] = freq_pts
                    res_data[QubitIndex]['freq_center'][j - batch_num * save_r - 1] = freq_center
                    res_data[QubitIndex]['I'][j - batch_num * save_r - 1] = res_Iarr
                    res_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = res_Qarr
                    res_data[QubitIndex]['Amps'][j - batch_num * save_r - 1] = amps
                    res_data[QubitIndex]['Found Freqs'][j - batch_num * save_r - 1] = res_freqs
                    res_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                    res_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                    res_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                    res_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rspec

                # ---------------------Collect g-e QSpec Results----------------
                if run_flags["q_spec"]:
                    qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1]=(
                        time.mktime(datetime.datetime.now().timetuple()))
                    qspec_data[QubitIndex]['I'][j - batch_num * save_r - 1] = qspec_I
                    qspec_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = qspec_Q
                    qspec_data[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = qspec_freqs
                    qspec_data[QubitIndex]['I Fit'][j - batch_num * save_r - 1] = qspec_I_fit
                    qspec_data[QubitIndex]['Q Fit'][j - batch_num * save_r - 1] = qspec_Q_fit
                    qspec_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                    qspec_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                    qspec_data[QubitIndex]['Recycled QFreq'][j - batch_num * save_r - 1] = recycled_qfreq
                    qspec_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                    qspec_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_qspec

                # ---------------------Collect g-e Rabi Results----------------
                if run_flags["rabi"]:
                    rabi_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                        time.mktime(datetime.datetime.now().timetuple()))
                    rabi_data[QubitIndex]['I'][j - batch_num * save_r - 1] = rabi_I
                    rabi_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = rabi_Q
                    rabi_data[QubitIndex]['Gains'][j - batch_num * save_r - 1] = rabi_gains
                    rabi_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = rabi_fit
                    rabi_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                    rabi_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                    rabi_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                    rabi_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rabi

                # ---------------------Collect g-e Single Shot Results----------------
                if run_flags["ss"]:
                    ss_data[QubitIndex]['Fidelity'][j - batch_num * save_r - 1] = fid
                    ss_data[QubitIndex]['Angle'][j - batch_num * save_r - 1] = angle
                    ss_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                        time.mktime(datetime.datetime.now().timetuple()))
                    ss_data[QubitIndex]['I_g'][j - batch_num * save_r - 1] = I_g
                    ss_data[QubitIndex]['Q_g'][j - batch_num * save_r - 1] = Q_g
                    ss_data[QubitIndex]['I_e'][j - batch_num * save_r - 1] = I_e
                    ss_data[QubitIndex]['Q_e'][j - batch_num * save_r - 1] = Q_e
                    ss_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                    ss_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                    ss_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                    ss_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_ss

                #---------------------Collect g-e T1 Results----------------
                if run_flags["t1"]:
                    t1_data[QubitIndex]['T1'][j - batch_num*save_r - 1] = t1_est
                    t1_data[QubitIndex]['Errors'][j - batch_num*save_r - 1] = t1_err
                    t1_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = (
                        time.mktime(datetime.datetime.now().timetuple()))
                    t1_data[QubitIndex]['I'][j - batch_num*save_r - 1] = t1_I
                    t1_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = t1_Q
                    t1_data[QubitIndex]['Delay Times'][j - batch_num*save_r - 1] = t1_delay_times
                    t1_data[QubitIndex]['Fit'][j - batch_num*save_r - 1] = q1_fit_exponential
                    t1_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                    t1_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                    t1_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                    t1_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t1

                #---------------------Collect g-e T2 Results----------------
                if run_flags["t2r"]:
                    t2r_data[QubitIndex]['T2'][j - batch_num*save_r - 1] = t2r_est
                    t2r_data[QubitIndex]['Errors'][j - batch_num*save_r - 1] = t2r_err
                    t2r_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = (
                        time.mktime(datetime.datetime.now().timetuple()))
                    t2r_data[QubitIndex]['I'][j - batch_num*save_r - 1] = t2r_I
                    t2r_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = t2r_Q
                    t2r_data[QubitIndex]['Delay Times'][j - batch_num*save_r - 1] = t2r_delay_times
                    t2r_data[QubitIndex]['Fit'][j - batch_num*save_r - 1] = fit_ramsey
                    t2r_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                    t2r_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                    t2r_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                    t2r_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2r

                #---------------------Collect g-e T2E Results----------------
                if run_flags["t2e"]:
                    t2e_data[QubitIndex]['T2E'][j - batch_num*save_r - 1] = t2e_est
                    t2e_data[QubitIndex]['Errors'][j - batch_num*save_r - 1] = t2e_err
                    t2e_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = (
                        time.mktime(datetime.datetime.now().timetuple()))
                    t2e_data[QubitIndex]['I'][j - batch_num*save_r - 1] = t2e_I
                    t2e_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = t2e_Q
                    t2e_data[QubitIndex]['Delay Times'][j - batch_num*save_r - 1] = t2e_delay_times
                    t2e_data[QubitIndex]['Fit'][j - batch_num*save_r - 1] = fit_t2e
                    t2e_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                    t2e_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                    t2e_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                    t2e_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2e

            del experiment

        ################################################## Potentially Save ################################################
        if save_data_h5:
            # Check if you are at the right round number
            # If so, then save all of the data and change the round num so you replace data starting next round
            if j % save_r == 0:
                batch_num+=1

                # --------------------------save g-e Res Spec-----------------------
                if run_flags["res_spec"]:
                    saver_res = Data_H5(dataFolder, res_data, batch_num, save_r)
                    saver_res.save_to_h5('res_ge', ts = RR_timestamp)
                    del saver_res
                    del res_data

                # --------------------------save g-e QSpec-----------------------
                if run_flags["q_spec"]:
                    saver_qspec = Data_H5(dataFolder, qspec_data, batch_num, save_r)
                    saver_qspec.save_to_h5('qspec_ge', ts = RR_timestamp)
                    del saver_qspec
                    del qspec_data

                # --------------------------save g-e Rabi-----------------------
                if run_flags["rabi"]:
                    saver_rabi = Data_H5(dataFolder, rabi_data, batch_num, save_r)
                    saver_rabi.save_to_h5('rabi_ge', ts = RR_timestamp)
                    del saver_rabi
                    del rabi_data

                # --------------------------save g-e SS-----------------------
                if run_flags["ss"]:
                    saver_ss = Data_H5(dataFolder, ss_data, batch_num, save_r)
                    saver_ss.save_to_h5('ss_ge', ts = RR_timestamp)
                    del saver_ss
                    del ss_data

                # --------------------------save g-e t1-----------------------
                if run_flags["t1"]:
                    saver_t1 = Data_H5(dataFolder, t1_data, batch_num, save_r)
                    saver_t1.save_to_h5('t1_ge', ts = RR_timestamp)
                    del saver_t1
                    del t1_data

                #--------------------------save g-e t2r-----------------------
                if run_flags["t2r"]:
                    saver_t2r = Data_H5(dataFolder, t2r_data, batch_num, save_r)
                    saver_t2r.save_to_h5('t2_ge', ts = RR_timestamp)
                    del saver_t2r
                    del t2r_data

                #--------------------------save g-e t2e-----------------------
                if run_flags["t2e"]:
                    saver_t2e = Data_H5(dataFolder, t2e_data, batch_num, save_r)
                    saver_t2e.save_to_h5('t2e_ge', ts = RR_timestamp)
                    del saver_t2e
                    del t2e_data

        # reset all dictionaries to none for safety
        res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
        qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
        ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
        t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
        t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
        t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)

    en=time.time()
    print('timetaken=',en-st)

    return()