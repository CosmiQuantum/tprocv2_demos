#---------------------------------------------------------------
# Fast Relaxation + Excitation measurement tests
# ryan linehan, 4/30/2025
#---------------------------------------------------------------

# Relevant imports (systemwide, non-QICK)
import sys
import os
import numpy as np
import datetime
import time
import logging
import gc, copy
import matplotlib.pyplot as plt
np.set_printoptions(threshold=int(1e15))
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))

# Relevant imports (QICK-related) 
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_008_save_data_to_h5 import Data_H5
from system_config import QICK_experiment
from starkshift import StarkShiftSpec
from starkshift import ResStarkShiftSpec
from starkshift import StarkShift2D
from starkshift import ResStarkShift2D
from section_002_res_spec_ef import ResonanceSpectroscopyEF
from section_004_qubit_spec_ef import EFQubitSpectroscopy
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from section_011_qubit_temperatures_efRabipt3 import Temps_EFAmpRabiExperiment
from analysis_optimization_report import optimization_report_ge
from analysis_optimization_report import optimization_report_ef

#---------------------------------------------------------------
# Run Configurations and Optimization Params

#---- General/overarching parameters
max_datataking_round_index = 10
n_rounds_to_save_after = 1  # how many rounds to save after
signal = 'None'  # 'I', or 'Q' depending on where the signal is
save_figs = False  # whether to save plots
fit_data = False  # fit data during the run?
save_data_h5 = True  # save data to h5 files?
verbose = True  # verbose output
qick_verbose = False
debug_mode = True  # if True, errors will stop the run immediately
study = 'TLS_Comprehensive_Study'
sub_study = 'source_on_substudy7'
substudy_txt_notes = '137Cs run with highest rate configuration, added declare_gen statement for starkshiftspec.'
Qs_to_look_at = [0,4]  # list of qubits to process

#---- Res spec parameters
# Optimization parameters for resonator spectroscopy
# 04/13 parameters
res_leng_vals = [5.5, 7.5, 6.0, 6.5, 5.0, 6.0]
res_gain = [0.9, 0.95, 0.78, 0.58, 0.95, 0.57]
freq_offsets = [-0.1, 0.2, 0.1, -0.4, -0.1, -0.1]
qubit_freqs_ef = [None]*6
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']

#---- Qubit spec parameters
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit','Round Num', 'Batch Num', 'Recycled QFreq', 'Exp Config', 'Syst Config']

#---- Amplitude rabi parameters
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']

#---- Resonator Offset optimization parameters
ssf_avgs_per_opt_pt = 3
freq_offset_steps = 15
offset_keys = ['Res Frequency', 'Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']

#---- (Lone) SSF parameters
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']

#----  T1 parameters
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']

#---- Stark Spec Keys
starkspec_keys = ['Dates', 'I', 'Q', 'P', 'shots','Gain Sweep','Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
stark2D_keys = ['Dates', 'I', 'Q', 'Qu Frequency Sweep', 'Res Gain Sweep','Round Num', 'Batch Num', 'Exp Config', 'Syst Config']

#---- Fast Relaxation/Excitation Measurement
forego_excitation_measurement = False
fastrelex_keys = ['Dates', 'I', 'Q', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']





#---------------------------------------------------------------
#Generally useful functions
#---------------------------------------------------------------

# Defining the (general) structure of a dictionary to be saved each measurement round
def create_data_dict(keys, n_rounds_to_save_after, qs):
    return {Q: {key: np.empty(n_rounds_to_save_after, dtype=object) for key in keys} for Q in qs}

# Function that iterates over various frequency offsets relative to the baseline res_spec value to find one that has maximized SSF
def sweep_frequency_offset(experiment, QubitIndex, offset_values, n_loops=10, number_of_qubits=6,
                           outerFolder="", studyDocumentationFolder="",optimizationFolder="", j=0):
    baseline_freq = experiment.readout_cfg['res_freq_ge'][QubitIndex]

    ssf_dict = {}
    for offset in offset_values:        
        fids = []
        # repeat n times for each offset
        for i in range(n_loops):

            #Create an actual copy of the experiment, set the resonator frequency based on the offset value in the sweep
            exp_copy = copy.deepcopy(experiment)  # python is python, doesnt overwrite things properly
            res_freqs = exp_copy.readout_cfg['res_freq_ge']
            res_freqs[QubitIndex] = baseline_freq + offset
            exp_copy.readout_cfg['res_freq_ge'] = res_freqs

            #Run the single shot measurement and extract data
            ss = SingleShot(QubitIndex, number_of_qubits, outerFolder, 0, save_figs, exp_copy)
            fid, angle, iq_list_g, iq_list_e, ss_config = ss.run()
            fids.append(fid)
            del exp_copy

            #Attempt to save the data
            try:
                offset_data = create_data_dict(offset_keys, n_rounds_to_save_after, list_of_all_qubits)

                I_g = iq_list_g[QubitIndex][0].T[0]
                Q_g = iq_list_g[QubitIndex][0].T[1]
                I_e = iq_list_e[QubitIndex][0].T[0]
                Q_e = iq_list_e[QubitIndex][0].T[1]

                offset_data[QubitIndex]['Res Frequency'][0] = offset
                offset_data[QubitIndex]['Fidelity'][0] = fid
                offset_data[QubitIndex]['Angle'][0] = angle
                offset_data[QubitIndex]['Dates'][0] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                offset_data[QubitIndex]['I_g'][0] = I_g
                offset_data[QubitIndex]['Q_g'][0] = Q_g
                offset_data[QubitIndex]['I_e'][0] = I_e
                offset_data[QubitIndex]['Q_e'][0] = Q_e
                offset_data[QubitIndex]['Round Num'][0] = i
                offset_data[QubitIndex]['Batch Num'][0] = j
                offset_data[QubitIndex]['Exp Config'][0] = expt_cfg
                offset_data[QubitIndex]['Syst Config'][0] = ss_config

                saver_offset = Data_H5(outerFolder, offset_data, 0, n_rounds_to_save_after)
                saver_offset.save_to_h5('offset')
                del saver_offset
                del offset_data
                gc.collect()
                offset_data = create_data_dict(offset_keys, n_rounds_to_save_after, list_of_all_qubits)

            #If any one of these fails, it's probably okay as long as all of them don't fail
            except Exception as e:
                if debug_mode:
                    raise  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')

        # find avg ssf
        avg_fid = np.mean(fids)
        ssf_dict[offset] = avg_fid
        if verbose:
            print(f"Offset: {offset} -> Average SSF: {avg_fid:.4f}")


    # Determine the offset value that yielded the best (highest) average SSF.
    optimal_offset = max(ssf_dict, key=ssf_dict.get)
    if verbose:
        print(
            f"Optimal frequency offset for Qubit {QubitIndex + 1}: {optimal_offset} (Avg SSF: {ssf_dict[optimal_offset]:.4f})")

    return optimal_offset, ssf_dict

# Make the directory tree for this study (with study, substudy, and other folders
def run_directory_tree_creation():
    
    #Folders
    if not os.path.exists("/data/QICK_data/run6/"):
        os.makedirs("/data/QICK_data/run6/")
    if not os.path.exists("/data/QICK_data/run6/6transmon/"):
        os.makedirs("/data/QICK_data/run6/6transmon/")

    studyFolder = os.path.join("/data/QICK_data/run6/6transmon/", study)
    if not os.path.exists(studyFolder):
        os.makedirs(studyFolder)

    subStudyFolder = os.path.join(studyFolder, sub_study)
    if not os.path.exists(subStudyFolder):
        os.makedirs(subStudyFolder)

    file_path = os.path.join(subStudyFolder, 'sub_study_notes.txt')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(substudy_txt_notes)

    #Now set up the logger (which requires this directory tree info)
    #Now that substudies are defined, Set up the logger
    #Logging
    log_file = os.path.join(subStudyFolder, "RR_Comprehensive_TLS_script.log")
    rr_logger = logging.getLogger("custom_logger_for_rr_only")
    rr_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    rr_logger.addHandler(file_handler)
    rr_logger.propagate = False


#---------------------------------------------------------------
# optimization: run the block that finds basic stuff (resspec,
# qspec, etc.)
#---------------------------------------------------------------
def run_optimization(QubitIndex, experiment):
    
    # Set resonator configuration for this qubit. This sets the resonator gain for this given res/qubit pair to the res_gain chosen
    # during the optimization. This res_gain parameter is a global variable that is at the top of this file. Res_gains is a mask where
    # only the QubitIndex qubit is active, and the rest are off. Res_len_vals (also at top of script) is used to set the length based
    # on initial manual optimization.
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

    # Verbose output
    rr_logger.info(f"Starting optimization for Qubit {QubitIndex + 1}")
    if verbose:
        print(f"Starting optimization for Qubit {QubitIndex + 1}")

    # Within optimization, we will do X different tests, with a goal of keeping optimization short
    # 1. Res spec x1
    # 2. Qubit spec x1
        
    ################################################ Res Spec ################################################
    t0 = time.perf_counter()
    rr_logger.info("----------------- Optimization: Starting Res Spec g-e  -----------------")
    if verbose:
        print("----------------- Optimization: Starting Res Spec g-e  -----------------")

    # Now create our data dictionary
    res_data = create_data_dict(res_keys, n_rounds_to_save_after, list_of_all_qubits)
    res_freqs_samples = []

    #Attempt to run a res spec measurement.
    try:
        res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, optimizationFolder, 0,
                                         save_figs, experiment=experiment, verbose=verbose,
                                         logger=rr_logger, qick_verbose=qick_verbose)
        res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
        res_freqs_samples.append(res_freqs)
        rr_logger.info(f"ResSpec sample {sample} for qubit {QubitIndex + 1}: {res_freqs}")

        res_data[QubitIndex]['Dates'][0] = (time.mktime(datetime.datetime.now().timetuple()))
        res_data[QubitIndex]['freq_pts'][0] = freq_pts
        res_data[QubitIndex]['freq_center'][0] = freq_center
        res_data[QubitIndex]['Amps'][0] = amps
        res_data[QubitIndex]['Found Freqs'][0] = res_freqs
        res_data[QubitIndex]['Round Num'][0] = 0
        res_data[QubitIndex]['Batch Num'][0] = 0
        res_data[QubitIndex]['Exp Config'][0] = expt_cfg
        res_data[QubitIndex]['Syst Config'][0] = sys_config_rspec

        saver_res = Data_H5(optimizationFolder, res_data, sample, n_rounds_to_save_after)  # save
        saver_res.save_to_h5('res_ge')
        del saver_res
        del res_data
        gc.collect()
        res_data = create_data_dict(res_keys, n_rounds_to_save_after, list_of_all_qubits)  # initialize again to a blank for saftey
        
        del res_spec

    #If it fails, continue
    except Exception as e:
        if debug_mode:
            raise  # In debug mode, re-raise the exception immediately
        rr_logger.exception(f"ResSpec error on qubit {QubitIndex +1} sample {sample}: {e}")
        continue

    #Average the resonator frequencies and use the result as the resulting "central" resonator spec for the rest of the dataset
    if res_freqs_samples:
        # Average the resonator frequency values across samples
        avg_res_freqs = np.mean(np.array(res_freqs_samples), axis=0).tolist()
    else:
        rr_logger.error(f"No resonator spectroscopy data collected for qubit {QubitIndex +1}.")
        return
    experiment.readout_cfg['res_freq_ge'][QubitIndex] = avg_res_freqs[QubitIndex]

    #Logging, verbosity, and printing to study notes
    rr_logger.info(f"Avg. resonator frequencies for qubit {QubitIndex +1}: {avg_res_freqs}")
    if verbose:
        print(f"Avg. resonator frequencies for qubit {QubitIndex +1}: {avg_res_freqs}")
    with open(study_notes_path, "a", encoding="utf-8") as file:
        file.write("\n" + f'Averaged Resonator Frequency Used for study: {avg_res_freqs[QubitIndex]}')

    #Computing time at the end of execution, and printing
    t1 = time.perf_counter()    
    print(f"g-e res spec took {t1 - t0:.4f} seconds")


    
    ################################################ Qubit Spec ################################################
    rr_logger.info("----------------- Optimization: Starting Qubit Spec g-e  -----------------")
    if verbose:
        print("----------------- Optimization: Starting Qubit Spec g-e  -----------------")

    #Create a data dictionary for qubit spec. This takes the keys for which data to save and creates a dictionary with them.
    #See the top of this script for details.
    qspec_data = create_data_dict(qspec_keys, n_rounds_to_save_after, list_of_all_qubits)

    #Attempt to run a res spec.
    try:
        q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, 0,
                                   signal, save_figs=save_figs, experiment=experiment,
                                   live_plot=False, verbose=verbose, logger=rr_logger,
                                   qick_verbose=qick_verbose)
        (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit,
         qubit_freq, sys_config_qspec) = q_spec.run()
        
        #If our qubit frequency measurement fails, then break
        if qubit_freq is None:
            rr_logger.info(f"Optimization block Qubit {QubitIndex + 1} qspec_ge failed on round {i} using stored qspec")
            if verbose:
                print(f"Optimization block Qubit {QubitIndex + 1} qspec_ge failed on round {i} using stored qspec")
                return
            
        #Set the qubit spec result to be the qubit frequency for the experiment (and print)
        experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)            
        rr_logger.info(f"Tune-up: Qubit {QubitIndex +1} frequency: {stored_qspec}")
        if verbose:
            print(f"Tune-up: Qubit {QubitIndex +1} frequency: {stored_qspec}")
                
        del q_spec
        qspec_data[QubitIndex]['Dates'][0] = (time.mktime(datetime.datetime.now().timetuple()))
        qspec_data[QubitIndex]['I'][0] = qspec_I
        qspec_data[QubitIndex]['Q'][0] = qspec_Q
        qspec_data[QubitIndex]['Frequencies'][0] = qspec_freqs
        qspec_data[QubitIndex]['I Fit'][0] = qspec_I_fit
        qspec_data[QubitIndex]['Q Fit'][0] = qspec_Q_fit
        qspec_data[QubitIndex]['Round Num'][0] = 0
        qspec_data[QubitIndex]['Batch Num'][0] = 0
        qspec_data[QubitIndex]['Recycled QFreq'][0] = False  # no rr so no recycling here
        qspec_data[QubitIndex]['Exp Config'][0] = expt_cfg
        qspec_data[QubitIndex]['Syst Config'][0] = sys_config_qspec
        
        saver_qspec = Data_H5(optimizationFolder, qspec_data, 0, n_rounds_to_save_after)
        saver_qspec.save_to_h5('qspec_ge')
        del saver_qspec
        del qspec_data
        gc.collect() #REL added 4/23/2025

    # If it fails, we need to fully return
    except Exception as e:
        if debug_mode:
            raise  # In debug mode, re-raise the exception immediately
        rr_logger.exception(f"QubitSpectroscopyGE error on qubit {QubitIndex +1}: {e}")
        print(f"Qubit Spectroscopy error on qubit {QubitIndex +1}")
        return

    # Save the time after this experiment and print.
    t2 = time.perf_counter()
    print(f"g-e Qspec took {t2 - t1:.4f} seconds")


    ################################################ g-e amp rabi ################################################
    rr_logger.info("----------------- Optimization: Starting Amplitude Rabi g-e  -----------------")
    if verbose:
        print("----------------- Optimization: Starting Amplitude Rabi g-e  -----------------")

    #Create a data dictionary for the amplitude rabi. This takes the keys for which data to save and creates a dictionary with them.
    #See the top of this script for details.    
    rabi_data = create_data_dict(rabi_keys, n_rounds_to_save_after, list_of_all_qubits)    
    
    #Try the amplitude rabi experiment
    try:
        rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, 0,
                                       signal, save_figs=save_figs, experiment=experiment,
                                       live_plot=False,
                                       increase_qubit_reps=False,
                                       qubit_to_increase_reps_for=0,
                                       multiply_qubit_reps_by=2,
                                       verbose=verbose, logger=rr_logger,
                                       qick_verbose=qick_verbose)
        (rabi_I, rabi_Q, rabi_gains, rabi_fit, stored_pi_amp, sys_config_rabi) = rabi.run()

        #Save the results in the experiment's qubit config dict
        experiment.qubit_cfg['pi_amp'][QubitIndex] = float(stored_pi_amp)
        rr_logger.info(f"Tune-up: Pi amplitude for qubit {QubitIndex +1}: {float(stored_pi_amp)}")

        #Verbose print
        if verbose:
            print(f"Tune-up: Pi amplitude for qubit {QubitIndex +1}: {float(stored_pi_amp)}")
        with open(study_notes_path, "a", encoding="utf-8") as file:
            file.write("\n" + f'Pi Amplitude Used for study: {float(stored_pi_amp)}')

        #Save the data that is returned from this
        rabi_data[QubitIndex]['Dates'][0] = (time.mktime(datetime.datetime.now().timetuple()))
        rabi_data[QubitIndex]['I'][0] = rabi_I
        rabi_data[QubitIndex]['Q'][0] = rabi_Q
        rabi_data[QubitIndex]['Gains'][0] = rabi_gains
        rabi_data[QubitIndex]['Fit'][0] = rabi_fit
        rabi_data[QubitIndex]['Round Num'][0] = 0
        rabi_data[QubitIndex]['Batch Num'][0] = 0
        rabi_data[QubitIndex]['Exp Config'][0] = expt_cfg
        rabi_data[QubitIndex]['Syst Config'][0] = sys_config_rabi

        #Save the data to h5
        saver_rabi = Data_H5(optimizationFolder, rabi_data, 0, n_rounds_to_save_after)
        saver_rabi.save_to_h5('rabi_ge')

        #Delete the experiment
        del rabi
        del saver_rabi
        del rabi_data
        
    #If the rabi fails, then this needs to trigger a full fail
    except Exception as e:
        if debug_mode:
            raise  # In debug mode, re-raise the exception immediately
        rr_logger.exception(f"Rabi error on qubit {QubitIndex +1}: {e}")
        print(f"Amplitude Rabi error on qubit {QubitIndex +1}")
        return

    #Calculate and print the time
    t3 = time.perf_counter()
    print(f"g-e Rabi took {t3 - t2:.4f} seconds")


    
    ################################################ optimize ################################################
    rr_logger.info("----------------- Optimization: Starting Resonator Offset Frequency  -----------------")
    if verbose:
        print("----------------- Optimization: Starting Resonator Offset Frequency  -----------------")

    #Get the "central" frequency from which offsets are calculated, and then produce an offset range
    reference_frequency = float(avg_res_freqs[QubitIndex])
    freq_range = np.linspace(-1, 1, freq_offset_steps)

    #Compute the optimal offset by running several sets of SSF measurements
    optimal_offset, ssf_dict = sweep_frequency_offset(experiment, QubitIndex, freq_range, n_loops=ssf_avgs_per_opt_pt, number_of_qubits=6,
                           outerFolder=optimizationFolder, studyDocumentationFolder=studyDocumentationFolder, j=0)

    #For each qubit, update the "central" resonator frequency with the offset value
    offset_res_freqs = [r + optimal_offset for r in avg_res_freqs]
    experiment.readout_cfg['res_freq_ge'][QubitIndex] = offset_res_freqs[QubitIndex]    #update with offset added
    print(experiment.readout_cfg)

    #Write to study note
    with open(study_notes_path, "a", encoding="utf-8") as file:
        file.write("\n" + f'Offset Frequency used for study: {offset_res_freqs[QubitIndex]}')

    #Save the time
    t4 = time.perf_counter()
    print(f"Freq offset Optimization took {t4 - t3:.4f} seconds")

    ################################################ repeated ss ################################################
    rr_logger.info("----------------- Optimization: Starting SSF  -----------------")
    if verbose:
        print("----------------- Optimization: Starting SSF  -----------------")

    #Create a data dictionary for the SSF measurement. This takes the keys for which data to save and creates a dictionary with them.
    #See the top of this script for details.    
    ss_data = create_data_dict(ss_keys, n_rounds_to_save_after, list_of_all_qubits)
    angles =[]
    thresholds=[]

    #Attempt the SSF measurement
    try:
        ss = SingleShot(QubitIndex, tot_num_of_qubits, optimizationFolder, 0, False, experiment=experiment,
                        verbose=verbose, logger=rr_logger)
        fid, angle, iq_list_g, iq_list_e, sys_config_ss = ss.run()
        I_g = iq_list_g[QubitIndex][0].T[0]
        Q_g = iq_list_g[QubitIndex][0].T[1]
        I_e = iq_list_e[QubitIndex][0].T[0]
        Q_e = iq_list_e[QubitIndex][0].T[1]
        
        fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(data=[I_g, Q_g, I_e, Q_e], cfg=ss.config, plot=save_figs)
        angles.append(angle)
        thresholds.append(threshold)

        ss_data[QubitIndex]['Fidelity'][0] = fid
        ss_data[QubitIndex]['Angle'][0] = angle
        ss_data[QubitIndex]['Dates'][0] = (time.mktime(datetime.datetime.now().timetuple()))
        ss_data[QubitIndex]['I_g'][0] = I_g
        ss_data[QubitIndex]['Q_g'][0] = Q_g
        ss_data[QubitIndex]['I_e'][0] = I_e
        ss_data[QubitIndex]['Q_e'][0] = Q_e
        ss_data[QubitIndex]['Round Num'][0] = 0
        ss_data[QubitIndex]['Batch Num'][0] = 0
        ss_data[QubitIndex]['Exp Config'][0] = expt_cfg
        ss_data[QubitIndex]['Syst Config'][0] = sys_config_ss

        #Save the data to h5 format
        saver_ss = Data_H5(optimizationFolder, ss_data, 0, n_rounds_to_save_after)
        saver_ss.save_to_h5('ss_ge')

        #Delete the SS saver and data, and then reinitialize again to a blank for safety
        del saver_ss
        del ss_data
        gc.collect()

    #If it fails, break
    except Exception as e:
        if debug_mode:
            raise  # In debug mode, re-raise the exception immediately
        else:
            rr_logger.exception(f'Got the following error, breaking: {e}')
            if verbose: print(f'Got the following error, breaking: {e}')
            return

    # average angles and thresholds to use
    avg_angle = np.mean(angles)
    avg_thresh = np.mean(thresholds)
    experiment.readout_cfg['res_phase'][QubitIndex] = avg_angle * 180/np.pi #need it to be a list of 6, the other qubits dont matter so just amke them the same val
    experiment.readout_cfg['threshold'][QubitIndex] = avg_thresh

    #Save info to study notes
    with open(study_notes_path, "a", encoding="utf-8") as file:
        file.write("\n" + f'Rotation angle into I component of signal used for study: {avg_angle * 180/np.pi}')
        file.write("\n" + f'I signal component threshold used for study: {avg_thresh}')

    #Print out time
    t5 = time.perf_counter()
    print(f"Repeated SS took {t5 - t4:.4f} seconds")


    ################################################ single t1 g-e with long relax delay #########################
    rr_logger.info("----------------- Optimization: Starting T1 g-e with long relax delay -----------------")
    if verbose:
        print("----------------- Optimization: Starting T1 g-e with long relax delay -----------------")

    #Create a data dictionary for the long-relax-delay T1 measurement. This takes the keys for which data to save and creates a dictionary with them.
    #See the top of this script for details.            
    t1_long_relax_data = create_data_dict(t1_keys, n_rounds_to_save_after, list_of_all_qubits)

    #Attempt a single long-relax-delay T1 measurement.
    try:
        t1_long_relax = T1Measurement(QubitIndex, tot_num_of_qubits, optimizationFolder, 0,
                                      signal, save_figs, experiment=experiment,
                                      live_plot=False, fit_data=fit_data,
                                      increase_qubit_reps=False,
                                      qubit_to_increase_reps_for=0,
                                      multiply_qubit_reps_by=2,
                                      verbose=verbose, logger=rr_logger, qick_verbose=qick_verbose, save_shots=True,
                                      set_relax_delay=True, relax_delay=1000)
        (t1_long_relax_est, t1_long_relax_err, t1_long_relax_I, t1_long_relax_Q, t1_long_relax_delay_times,
         q1_fit_exponential_long_relax, sys_config_t1_long_relax) = t1_long_relax.run(thresholding=False)

        t1_long_relax_data[QubitIndex]['T1'][0] = t1_long_relax_est
        t1_long_relax_data[QubitIndex]['Errors'][0] = t1_long_relax_err
        t1_long_relax_data[QubitIndex]['Dates'][0] = (time.mktime(datetime.datetime.now().timetuple()))
        t1_long_relax_data[QubitIndex]['I'][0] = t1_long_relax_I
        t1_long_relax_data[QubitIndex]['Q'][0] = t1_long_relax_Q
        t1_long_relax_data[QubitIndex]['Delay Times'][0] = t1_long_relax_delay_times
        t1_long_relax_data[QubitIndex]['Fit'][0] = q1_fit_exponential_long_relax
        t1_long_relax_data[QubitIndex]['Round Num'][0] = 0
        t1_long_relax_data[QubitIndex]['Batch Num'][0] = 0
        t1_long_relax_data[QubitIndex]['Exp Config'][0] = expt_cfg
        t1_long_relax_data[QubitIndex]['Syst Config'][0] = sys_config_t1_long_relax

        #Save the data to h5 format
        saver_t1_long_relax = Data_H5(optimizationFolder, t1_long_relax_data, 0, n_rounds_to_save_after)
        saver_t1_long_relax.save_to_h5('t1_ge_long_relax')

        #Delete the saver
        del t1_long_relax
        del saver_t1_long_relax
        del t1_long_relax_data

    #If this fails, kill the code immediately. No real reason to do this except to make sure things are just working very well
    except Exception as e:
        if debug_mode:
            raise  # In debug mode, re-raise the exception immediately
        rr_logger.exception(f"t1 long relax error on qubit {QubitIndex + 1}: {e}")
        return

    #Take and then print the time
    t6 = time.perf_counter()
    print(f"g-e T1 with long relax delay took {t6 - t5:.4f} seconds")










#---------------------------------------------------------------
# dataset block: run the block that takes the data "of interest"
# (here the relaxation and excitation measurements)
#---------------------------------------------------------------    
def run_dataset(Qs_to_look_at, experiment, this_datataking_round_index):

    #In this measurement, it's incredibly important to have the stark spec data be taken as immediately as possible
    #before the FastRelEx data, since we will be relying on the stark spec data's results to actually know where our TLS is.
    #However, we can still loop over qubits, as long as each qubit does both measurements sequentially before switching to a
    #Different qubit.
    
    # create dictionaries
    starkspec_data = create_data_dict(starkspec_keys, n_rounds_to_save_after, Qs_to_look_at)
    fastrelex_data = create_data_dict(fastrelex_keys, n_rounds_to_save_after, Qs_to_look_at)
    
    
    for QubitIndex in Qs_to_look_at:

        # Set resonator configuration for this qubit
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]
        
        #Double check that other relevant input parameters (i.e. qubit frequency and amplitude rabi) are set properly here
        
        
        
        #------------------------------------------------------------
        # First, try the stark spec measurement. Start a timer
        t0 = time.perf_counter()        
        try:
            timestamp_starkspec = time.mktime(datetime.datetime.now().timetuple())
            stark_shift_spec = StarkShiftSpec(QubitIndex, tot_num_of_qubits, studyFolder, save_figs=False,
                                              experiment=experiment)
            starkspec_I, starkspec_Q, starkspec_P, starkspec_shots, starkspec_gain_sweep, sys_config_starkspec = stark_shift_spec.run_with_qick_sweep()
            del stark_shift_spec

        # If the stark spec measurement fails, then we need to flag that we should *only* run the relaxation-style measurements
        except Exception as e:
            if debug_mode:
                raise e  # In debug mode, re-raise the exception immediately
            else:
                rr_logger.exception(f'Got the following error, continuing: {e}')
                if verbose: print(f'Got the following error, continuing: {e}')

        # If the stark spec measurement does not give a reasonable output, then 

        #Print timing
        t1 = time.perf_counter()        
        print(f"Data taking: Stark Spec took {t1 - t0:.4f} seconds")        


        #Online analysis to determine a TLS onto which we tune. Output should be a frequency in MHz. Otherwise we return -1.
        
        target_tls_frequency = -1;
        
        
        #------------------------------------------------------------        
        # Next, run the FastRelEx measurement. We'll have 
        try:
            timestamp_fastrelex = time.mktime(datetime.datetime.now().timetuple())            

            #Raw I and Q returned from the fast rel/ex measurements
            fastrelex_I = []
            fastrelex_Q = []
            meas_is_rel = True
            
            #If the target tls frequency is negative, it means we didn't find an appropriate TLS onto which we could tune. In this case, pass a flag
            #to just do a relaxation measurement, sans excitation.
            if target_tls_frequency < 0 or forego_excitation_measurement == True:
                meas_is_rel = True
                fast_rel_ex = FastRelaxationExcitation(QubitIndex,tot_num_of_qubits,studyFolder,this_datataking_round_index,save_figs=False,
                                                       experiment=experiment,target_tls=False,tls_gain=target_tls_gain,tls_detuning=target_tls_detuning)
                
                #Returns the I/Q raw values from the run, and the config
                fastrelex_I, fastrelex_Q, sys_config_fastrelex = fast_rel_ex.run()
                

            else:
                #Otherwise, we try to tune our relaxation/excitation measurement onto the TLS.
                meas_is_rel = False
                fast_rel_ex = FastRelaxationExcitation(QubitIndex,tot_num_of_qubits,studyFolder,this_datataking_round_index,save_figs=False,
                                                       experiment=experiment,target_tls=True,tls_gain=target_tls_gain,tls_detuning=target_tls_detuning)
                
                #Returns the I/Q raw values from the run, and the config
                fastrelex_I, fastrelex_Q, sys_config_fastrelex = fast_rel_ex.run()

                
            
            
            
        # If this fails, break. This is the whole reason we're writing this script
        except Exception as e:
            if debug_mode:
                raise  # In debug mode, re-raise the exception immediately
            rr_logger.exception(f'Single Shot error on qubit {QubitIndex +1} at round {this_datataking_round_index}: {e}')
            continue

        #Measure and print timing
        t2 = time.perf_counter()
        print(f"Data taking: FastRelEx took {t2 - t1:.4f} seconds")



        #------------------------------------------------------------
        # Save data for this round
        if save_data_h5:
            idx = 0

            #Save stark spec data
            starkspec_data[QubitIndex]['Dates'][idx] = timestamp_starkspec
            starkspec_data[QubitIndex]['I'][idx] = starkspec_I
            starkspec_data[QubitIndex]['Q'][idx] = starkspec_Q
            starkspec_data[QubitIndex]['P'][idx] = starkspec_P
            starkspec_data[QubitIndex]['shots'][idx] = starkspec_shots
            starkspec_data[QubitIndex]['Gain Sweep'][idx] = starkspec_gain_sweep
            starkspec_data[QubitIndex]['Round Num'][idx] = this_datataking_round_index
            starkspec_data[QubitIndex]['Batch Num'][idx] = batch_num
            starkspec_data[QubitIndex]['Exp Config'][idx] = expt_cfg
            starkspec_data[QubitIndex]['Syst Config'][idx] = sys_config_starkspec

            #Save FastRelEx data
            fastrelex_data[QubitIndex]['Dates'][idx] = timestamp_fastrelex
            fastrelex_data[QubitIndex]['I'][idx] = fastrelex_I
            fastrelex_data[QubitIndex]['Q'][idx] = fastrelex_Q
            #fastrelex_data[QubitIndex]['P'][idx] = starkspec_P
            #fastrelex_data[QubitIndex]['shots'][idx] = starkspec_shots
            #fastrelex_data[QubitIndex]['Gain Sweep'][idx] = starkspec_gain_sweep
            starkspec_data[QubitIndex]['Round Num'][idx] = this_datataking_round_index
            starkspec_data[QubitIndex]['Batch Num'][idx] = batch_num
            starkspec_data[QubitIndex]['Exp Config'][idx] = expt_cfg
            starkspec_data[QubitIndex]['Syst Config'][idx] = sys_config_fastrelex



    #--------------------------------------------------------------------
    #once all measurements are taken for each qubit and dictionaries filled with data, save h5 files and delete dictionaries.
    #For this, it'll happen after every round -- in this case this block is relevant because we'll save multiple qubits' returned data
    #in the same file
    if save_data_h5:
        if this_datataking_round_index % n_rounds_to_save_after == 0:

            #Save stark spec data
            saver_starkspec = Data_H5(studyFolder, starkspec_data, batch_num, n_rounds_to_save_after)
            saver_starkspec.save_to_h5('starkspec_ge')
            del saver_starkspec
            del starkspec_data
            gc.collect()

            #Save FastRelEx data
            saver_fastrelex = Data_H5(studyFolder, fastrelex_data, batch_num, n_rounds_to_save_after)
            saver_fastrelex.save_to_h5('fastrelex_ge')
            del saver_fastrelex
            del fastrelex_data
            gc.collect()


            
    # create dictionaries for reinitialization
    starkspec_data = create_data_dict(starkspec_keys, n_rounds_to_save_after, Qs_to_look_at)
    fastrelex_data = create_data_dict(fastrelex_keys, n_rounds_to_save_after, Qs_to_look_at)

    
    gc.collect()
    t3 = time.perf_counter()
    print(f"Data saving took {t3 - t2:.4f} seconds")



#---------------------------------------------------------------
# main function: run the whole thing
#---------------------------------------------------------------

#Set up directory tree to make sure study and substudy folders are present
run_directory_tree_creation():

# Set up the dataset organization
formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
dataSetFolder = os.path.join(subStudyFolder, formatted_datetime)
optimizationFolder = os.path.join(dataSetFolder, 'optimization')
studyFolder = os.path.join(dataSetFolder, 'study_data')
studyDocumentationFolder = os.path.join(dataSetFolder, 'documentation')
study_notes_path = os.path.join(studyDocumentationFolder, 'study_notes.txt')

# With this experiment, we need to 
experiment = QICK_experiment( dataSetFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10, fridge=FRIDGE )
experiment.create_folder_if_not_exists(dataSetFolder)
experiment.create_folder_if_not_exists(optimizationFolder)
experiment.create_folder_if_not_exists(studyFolder)
experiment.create_folder_if_not_exists(studyDocumentationFolder)
with open(study_notes_path, "w", encoding="utf-8") as file:
    file.write('Study Notes:')

#For time benchmarking
t_start = time.perf_counter()

#-------------------------------------------------------------------------
#run complete optimization for each qubit as a separate data block
for QubitIndex in Qs_to_look_at:
    run_optimization(QubitIndex,experiment)
    try:
        optimization_report_ge(subStudyFolder, formatted_datetime, QubitIndex)
    except Exception as e:
        if debug_mode:
            raise
        continue

    gc.collect()


#-------------------------------------------------------------------------
#Run the datataking block, which consists of N rounds of:
# 1. 1D stark shift sweep to find a gain at which there is a TLS
# 2. A period (minute? 10 min?) during which the relaxation/excitation measurements are being made
rr_logger.info("----------------- Starting data block part of dataset -----------------")
if verbose:
    print("----------------- Starting data block part of dataset -----------------")

#Round index
datataking_round_index=0
while datataking_round_index < max_datataking_round_index:
    t_datataking_round_start = time.time()
    run_dataset(Qs_to_look_at, experiment, datataking_round_index)
    datataking_round_index+=1
    gc.collect()

del experiment
gc.collect()
t_end = time.perf_counter()
print(f"Full script took {t_end - t_start:.4f} seconds")
