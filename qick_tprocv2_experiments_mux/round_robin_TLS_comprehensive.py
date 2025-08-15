import sys
import os
import numpy as np
import datetime
import time
import logging
import gc, copy

import visdom
import matplotlib.pyplot as plt
np.set_printoptions(threshold=int(1e15))
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))

# Import experiments and configurations
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

################################################
# Run Configurations and Optimization Params
################################################
ssf_avgs_per_opt_pt = 3
freq_offset_steps = 15
res_sample_number = 20
ef_res_sample_number = 1
ss_sample_number = 1
qspec_sample_number = 50 #1
med_gain = 0.5 #set medium gain qspec gain
high_gain = 1.0 #set high gain qspec gain
n = 80 # number of rounds for fast repetitive runs
save_r = 1  # how many rounds to save after
signal = 'None'  # 'I', or 'Q' depending on where the signal is
save_figs = False  # whether to save plots
live_plot = False  # use visdom for live plotting?
fit_data = False  # fit data during the run?
save_data_h5 = True  # save data to h5 files?
verbose = True  # verbose output
qick_verbose = False
debug_mode = True  # if True, errors will stop the run immediately
increase_qubit_reps = False
qubit_to_increase_reps_for = 0
multiply_qubit_reps_by = 2
# increase_qubit_steps_ef = False #if you want to increase the steps for all qubits, set to True, if you only want to set it to true for 1 qubit, see e-f qubit spec section
increase_steps_to_ef = 600
study = 'TLS_Comprehensive_Study'
sub_study = 'source_off_substudy7'
substudy_txt_notes = 'Source removed, Qubit 5 qubit freq found as minimum of qspec ge. Post science run data-taking in between other R&D'
Qs_to_look_at = [0,4]  # list of qubits to process

# Set which experiments to run
run_flags = {"q_spec": True, "hi_gain_q_spec": True, "med_gain_q_spec": False, "ss": True, "t1": True, "starkSpec": True, "resStarkSpec": False}

# Optimization parameters for resonator spectroscopy
# 04/13 parameters
res_leng_vals = [5.5, 7.5, 6.0, 6.5, 5.0, 6.0]
res_gain = [0.9, 0.95, 0.78, 0.58, 0.95, 0.57]
freq_offsets = [-0.1, 0.2, 0.1, -0.4, -0.1, -0.1]
qubit_freqs_ef = [None]*6

# Dictionaries
def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in qs}

res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit','Round Num', 'Batch Num', 'Recycled QFreq', 'Exp Config', 'Syst Config']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
offset_keys = ['Res Frequency', 'Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
starkspec_keys = ['Dates', 'I', 'Q', 'P', 'shots','Gain Sweep','Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
stark2D_keys = ['Dates', 'I', 'Q', 'Qu Frequency Sweep', 'Res Gain Sweep','Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
rabi_keys_ef_Qtemps = ['Dates', 'Qfreq_ge', 'I1', 'Q1', 'Gains1', 'Fit1', 'I2', 'Q2', 'Gains2', 'Fit2', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']

def get_min_qspec(I, Q, probe_freqs):

    #find max signal in I or Q
    diff_I = np.abs(np.max(I) - np.min(I))
    diff_Q = np.abs(np.max(Q) - np.min(Q))
    if diff_I > diff_Q:
        signal = I
    else:
        signal = Q

    baseline = np.mean(signal[0:3]) #sample points at beginning of qspec scan for baseline
    diff_max = np.abs(np.max(signal) - baseline) #find if max or min of signal is different from baseline
    diff_min = np.abs(np.min(signal) - baseline)
    if diff_max > diff_min: #max of signal differs from baseline
        idx = np.argmax(signal)
    else: #min of signal differs from baseline
        idx = np.argmin(signal)

    qubit_freq = probe_freqs[idx] #get qubit frequency at selected min/max of signal

    return qubit_freq

def sweep_frequency_offset(experiment, QubitIndex, offset_values, n_loops=10, number_of_qubits=6,
                           outerFolder="", studyDocumentationFolder="",optimizationFolder="", j=0):
    baseline_freq = experiment.readout_cfg['res_freq_ge'][QubitIndex]

    ssf_dict = {}
    for offset in offset_values:
        fids = []
        # repeat n times for each offset
        for i in range(n_loops):

            exp_copy = copy.deepcopy(experiment)  # python is python, doesnt overwrite things properly

            res_freqs = exp_copy.readout_cfg['res_freq_ge']
            res_freqs[QubitIndex] = baseline_freq + offset
            exp_copy.readout_cfg['res_freq_ge'] = res_freqs

            ss = SingleShot(QubitIndex, number_of_qubits, outerFolder, 0, save_figs, exp_copy)
            fid, angle, iq_list_g, iq_list_e, ss_config = ss.run()
            fids.append(fid)
            del exp_copy

            try:
                offset_data = create_data_dict(offset_keys, save_r, list_of_all_qubits)

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

                saver_offset = Data_H5(outerFolder, offset_data, 0, save_r)
                saver_offset.save_to_h5('offset')
                del saver_offset
                del offset_data
                gc.collect()
                offset_data = create_data_dict(offset_keys, save_r, list_of_all_qubits)

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

    # plt.figure()
    # offsets_sorted = sorted(ssf_dict.keys())
    # ssf_values = [ssf_dict[offset] for offset in offsets_sorted]
    # plt.plot(offsets_sorted, ssf_values, marker='o')
    # plt.xlabel('Frequency Offset')
    # plt.ylabel('Average SSF')
    # plt.title(f'SSF vs Frequency Offset for Qubit {QubitIndex + 1}')
    # plt.grid(True)
    # if outerFolder:
    #     os.makedirs(outerFolder, exist_ok=True)
    #     plot_path = os.path.join(studyDocumentationFolder, f"SSF_vs_offset_Q{QubitIndex + 1}.png")
    #     plt.savefig(plot_path)
    #     if verbose:
    #         print(f"Plot saved to {plot_path}")
    # plt.close()

    # Determine the offset value that yielded the best (highest) average SSF.
    optimal_offset = max(ssf_dict, key=ssf_dict.get)
    if verbose:
        print(
            f"Optimal frequency offset for Qubit {QubitIndex + 1}: {optimal_offset} (Avg SSF: {ssf_dict[optimal_offset]:.4f})")

    return optimal_offset, ssf_dict

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

#Logging
log_file = os.path.join(subStudyFolder, "RR_Comprehensive_TLS_script.log")
rr_logger = logging.getLogger("custom_logger_for_rr_only")
rr_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
rr_logger.addHandler(file_handler)
rr_logger.propagate = False


def run_optimization(QubitIndex, ss_sample_number, res_sample_number, experiment):

    # Set resonator configuration for this qubit
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

    rr_logger.info(f"Starting optimization for Qubit {QubitIndex + 1}")
    if verbose:
        print(f"Starting optimization for Qubit {QubitIndex + 1}")

    ################################################ Res Spec ################################################
    t0 = time.perf_counter()
    rr_logger.info("----------------- Optimization: Starting Res Spec g-e  -----------------")
    if verbose:
        print("----------------- Optimization: Starting Res Spec g-e  -----------------")

    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    res_freqs_samples = []
    for sample in range(res_sample_number):
        try:
            res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, optimizationFolder, sample,
                                             save_figs, experiment=experiment, verbose=verbose,
                                             logger=rr_logger, qick_verbose=qick_verbose)
            res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
            res_freqs_samples.append(res_freqs)
            rr_logger.info(f"ResSpec sample {sample} for qubit {QubitIndex + 1}: {res_freqs}")

            res_data[QubitIndex]['Dates'][0] = (
                time.mktime(datetime.datetime.now().timetuple()))
            res_data[QubitIndex]['freq_pts'][0] = freq_pts
            res_data[QubitIndex]['freq_center'][0] = freq_center
            res_data[QubitIndex]['Amps'][0] = amps
            res_data[QubitIndex]['Found Freqs'][0] = res_freqs
            res_data[QubitIndex]['Round Num'][0] = sample
            res_data[QubitIndex]['Batch Num'][0] = 0
            res_data[QubitIndex]['Exp Config'][0] = expt_cfg
            res_data[QubitIndex]['Syst Config'][0] = sys_config_rspec

            saver_res = Data_H5(optimizationFolder, res_data, sample, save_r)  # save
            saver_res.save_to_h5('res_ge')
            del saver_res
            del res_data
            gc.collect()
            res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)  # initialize again to a blank for saftey

            del res_spec

        except Exception as e:
            if debug_mode:
                raise  # In debug mode, re-raise the exception immediately
            rr_logger.exception(f"ResSpec error on qubit {QubitIndex +1} sample {sample}: {e}")
            continue


    if res_freqs_samples:
        # Average the resonator frequency values across samples
        avg_res_freqs = np.mean(np.array(res_freqs_samples), axis=0).tolist()
    else:
        rr_logger.error(f"No resonator spectroscopy data collected for qubit {QubitIndex +1}.")
        return

    experiment.readout_cfg['res_freq_ge'][QubitIndex] = avg_res_freqs[QubitIndex] #start with offset of 0, lets try like this and see if works

    rr_logger.info(f"Avg. resonator frequencies for qubit {QubitIndex +1}: {avg_res_freqs}")
    if verbose:
        print(f"Avg. resonator frequencies for qubit {QubitIndex +1}: {avg_res_freqs}")

    with open(study_notes_path, "a", encoding="utf-8") as file:
        file.write("\n" + f'Averaged Resonator Frequency Used for study: {avg_res_freqs[QubitIndex]}')

    t1 = time.perf_counter()
    print(f"g-e res spec took {t1 - t0:.4f} seconds")
    ################################################ Qubit Spec ################################################
    rr_logger.info("----------------- Optimization: Starting Qubit Spec g-e  -----------------")
    if verbose:
        print("----------------- Optimization: Starting Qubit Spec g-e  -----------------")

    stored_qspec = copy.deepcopy(experiment.qubit_cfg['qubit_freq_ge'][QubitIndex])
    for i in np.arange(0, qspec_sample_number):
        qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
        try:
            q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, 0,
                                   signal, save_figs=save_figs, experiment=experiment,
                                   live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                   qick_verbose=qick_verbose)
            (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit,
            qubit_freq, sys_config_qspec) = q_spec.run()

            if QubitIndex == 4:
                qubit_freq = get_min_qspec(qspec_I, qspec_Q, qspec_freqs) #replace with min for Qubit 5 only
                stored_qspec = float(qubit_freq)

            else: #for qubit 1, continue normal catch for a failed fit
                if qubit_freq is None:
                #fit failed save previous value and set flag to True
                    recycled_qfreq = True
                    qubit_freq = stored_qspec
                    rr_logger.info(f"Optimization block Qubit {QubitIndex + 1} qspec_ge failed on round {i} using stored qspec")
                    if verbose:
                        print(f"Optimization block Qubit {QubitIndex + 1} qspec_ge failed on round {i} using stored qspec")
                else:
                    recycled_qfreq = False
                    stored_qspec = float(qubit_freq)

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
            qspec_data[QubitIndex]['Round Num'][0] = i
            qspec_data[QubitIndex]['Batch Num'][0] = 0
            qspec_data[QubitIndex]['Recycled QFreq'][0] = False  # no rr so no recycling here
            qspec_data[QubitIndex]['Exp Config'][0] = expt_cfg
            qspec_data[QubitIndex]['Syst Config'][0] = sys_config_qspec

            saver_qspec = Data_H5(optimizationFolder, qspec_data, 0, save_r)
            saver_qspec.save_to_h5('qspec_ge')
            del saver_qspec
            del qspec_data
            gc.collect() #REL added 4/23/2025
            
        except Exception as e:
            if debug_mode:
                raise  # In debug mode, re-raise the exception immediately
            rr_logger.exception(f"QubitSpectroscopyGE error on qubit {QubitIndex +1}: {e}")
            return

    t2 = time.perf_counter()
    print(f"g-e Qspec took {t2 - t1:.4f} seconds")

    #   ################################################ Extended Qubit Spec ################################################
    #   rr_logger.info("----------------- Optimization: Starting Extended Freq Range Qubit Spec g-e  -----------------")
    #   if verbose:
    #       print("----------------- Optimization: Starting Extended Freq Range Qubit Spec g-e -----------------")
    #
    #   ext_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    #   qubit_gain_temp = experiment.qubit_cfg['qubit_gain_ge']
    #   try:
    #       experiment.qubit_cfg['qubit_gain_ge'] = np.ones(len(qubit_gain_temp))
    #       ext_q_spec = QubitSpectroscopyGE(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, 0,
    #                                  signal, save_figs=save_figs, experiment=experiment,
    #                                  live_plot=live_plot, verbose=verbose, logger=rr_logger,
    #                                  qick_verbose=qick_verbose, increase_reps = True, increase_reps_to = 500,
    #                                  ext_q_spec=True, fit_data=False)
    #       (ext_qspec_I, ext_qspec_Q, ext_qspec_freqs, ext_qspec_I_fit, ext_qspec_Q_fit,
    #        ext_qubit_freq, sys_config_ext_qspec) = ext_q_spec.run()
    #
    #       del ext_q_spec
    #
    #       ext_qspec_data[QubitIndex]['Dates'][0] = (
    #           time.mktime(datetime.datetime.now().timetuple()))
    #       ext_qspec_data[QubitIndex]['I'][0] = ext_qspec_I
    #       ext_qspec_data[QubitIndex]['Q'][0] = ext_qspec_Q
    #       ext_qspec_data[QubitIndex]['Frequencies'][0] = ext_qspec_freqs
    #       ext_qspec_data[QubitIndex]['Round Num'][0] = 0
    #       ext_qspec_data[QubitIndex]['Batch Num'][0] = 0
    #       ext_qspec_data[QubitIndex]['Exp Config'][0] = expt_cfg
    #       ext_qspec_data[QubitIndex]['Syst Config'][0] = sys_config_ext_qspec
    #
    #       saver_ext_qspec = Data_H5(optimizationFolder, ext_qspec_data, 0, save_r)
    #       saver_ext_qspec.save_to_h5('extended_qspec_ge')
    #       del saver_ext_qspec
    #       del ext_qspec_data
    #
    #   except Exception as e:
    #       if debug_mode:
    #           raise  # In debug mode, re-raise the exception immediately
    #       rr_logger.exception(f"Extended QubitSpectroscopyGE error on qubit {QubitIndex +1}: {e}")
    #       return
    #   experiment.qubit_cfg['qubit_gain_ge'] = qubit_gain_temp
    #
    t3 = time.perf_counter()
    print(f"Extended Qspec took {t3 - t2:.4f} seconds")
#
#   ################################################ g-e amp rabi ################################################
    rr_logger.info("----------------- Optimization: Starting Amplitude Rabi g-e  -----------------")
    if verbose:
        print("----------------- Optimization: Starting Amplitude Rabi g-e  -----------------")

    rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    try:
        rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, 0,
                                       signal, save_figs=save_figs, experiment=experiment,
                                       live_plot=live_plot,
                                       increase_qubit_reps=increase_qubit_reps,
                                       qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                       multiply_qubit_reps_by=multiply_qubit_reps_by,
                                       verbose=verbose, logger=rr_logger,
                                       qick_verbose=qick_verbose)
        (rabi_I, rabi_Q, rabi_gains, rabi_fit, stored_pi_amp, sys_config_rabi) = rabi.run()
        experiment.qubit_cfg['pi_amp'][QubitIndex] = float(stored_pi_amp)
        rr_logger.info(f"Tune-up: Pi amplitude for qubit {QubitIndex +1}: {float(stored_pi_amp)}")
        if verbose:
            print(f"Tune-up: Pi amplitude for qubit {QubitIndex +1}: {float(stored_pi_amp)}")
        with open(study_notes_path, "a", encoding="utf-8") as file:
            file.write("\n" + f'Pi Amplitude Used for study: {float(stored_pi_amp)}')


        rabi_data[QubitIndex]['Dates'][0] = (
            time.mktime(datetime.datetime.now().timetuple()))
        rabi_data[QubitIndex]['I'][0] = rabi_I
        rabi_data[QubitIndex]['Q'][0] = rabi_Q
        rabi_data[QubitIndex]['Gains'][0] = rabi_gains
        rabi_data[QubitIndex]['Fit'][0] = rabi_fit
        rabi_data[QubitIndex]['Round Num'][0] = 0
        rabi_data[QubitIndex]['Batch Num'][0] = 0
        rabi_data[QubitIndex]['Exp Config'][0] = expt_cfg
        rabi_data[QubitIndex]['Syst Config'][0] = sys_config_rabi
        saver_rabi = Data_H5(optimizationFolder, rabi_data, 0, save_r)
        saver_rabi.save_to_h5('rabi_ge')
        del rabi
        del saver_rabi
        del rabi_data
    except Exception as e:
        if debug_mode:
            raise  # In debug mode, re-raise the exception immediately
        rr_logger.exception(f"Rabi error on qubit {QubitIndex +1}: {e}")
        return

    t4 = time.perf_counter()
    print(f"g-e Rabi took {t4 - t3:.4f} seconds")

    ################################################ optimize ################################################
    rr_logger.info("----------------- Optimization: Starting Resonator Offset Frequency  -----------------")
    if verbose:
        print("----------------- Optimization: Starting Resonator Offset Frequency  -----------------")

    reference_frequency = float(avg_res_freqs[QubitIndex])
    freq_range = np.linspace(-1, 1, freq_offset_steps)

    optimal_offset, ssf_dict = sweep_frequency_offset(experiment, QubitIndex, freq_range, n_loops=ssf_avgs_per_opt_pt, number_of_qubits=6,
                           outerFolder=optimizationFolder, studyDocumentationFolder=studyDocumentationFolder, j=0)

    offset_res_freqs = [r + optimal_offset for r in avg_res_freqs]
    experiment.readout_cfg['res_freq_ge'][QubitIndex] = offset_res_freqs[QubitIndex]    #update with offset added
    print(experiment.readout_cfg)

    with open(study_notes_path, "a", encoding="utf-8") as file:
        file.write("\n" + f'Offset Frequency used for study: {offset_res_freqs[QubitIndex]}')

    t5 = time.perf_counter()
    print(f"Freq offset Optimization took {t5 - t4:.4f} seconds")

    ################################################ repeated ss ################################################
    rr_logger.info("----------------- Optimization: Starting SSF  -----------------")
    if verbose:
        print("----------------- Optimization: Starting SSF  -----------------")

    ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
    angles =[]
    thresholds=[]
    for ss_round in range(ss_sample_number):
        try:
            ss = SingleShot(QubitIndex, tot_num_of_qubits, optimizationFolder, 0, False, experiment=experiment,
                            verbose=verbose, logger=rr_logger)
            fid, angle, iq_list_g, iq_list_e, sys_config_ss = ss.run()
            I_g = iq_list_g[QubitIndex][0].T[0]
            Q_g = iq_list_g[QubitIndex][0].T[1]
            I_e = iq_list_e[QubitIndex][0].T[0]
            Q_e = iq_list_e[QubitIndex][0].T[1]

            fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(
                data=[I_g, Q_g, I_e, Q_e], cfg=ss.config, plot=save_figs)
            angles.append(angle)
            thresholds.append(threshold)

            ss_data[QubitIndex]['Fidelity'][0] = fid
            ss_data[QubitIndex]['Angle'][0] = angle
            ss_data[QubitIndex]['Dates'][0] = (
                time.mktime(datetime.datetime.now().timetuple()))
            ss_data[QubitIndex]['I_g'][0] = I_g
            ss_data[QubitIndex]['Q_g'][0] = Q_g
            ss_data[QubitIndex]['I_e'][0] = I_e
            ss_data[QubitIndex]['Q_e'][0] = Q_e
            ss_data[QubitIndex]['Round Num'][0] = ss_round
            ss_data[QubitIndex]['Batch Num'][0] = 0
            ss_data[QubitIndex]['Exp Config'][0] = expt_cfg
            ss_data[QubitIndex]['Syst Config'][0] = sys_config_ss

            saver_ss = Data_H5(optimizationFolder, ss_data, 0, save_r)
            saver_ss.save_to_h5('ss_ge')
            del saver_ss
            del ss_data
            gc.collect()
            ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
        except Exception as e:
            if debug_mode:
                raise  # In debug mode, re-raise the exception immediately
            else:
                rr_logger.exception(f'Got the following error, continuing: {e}')
                if verbose: print(f'Got the following error, continuing: {e}')
                continue  # skip the rest of this

    # average angles and thresholds to use
    avg_angle = np.mean(angles)
    avg_thresh = np.mean(thresholds)
    experiment.readout_cfg['res_phase'][QubitIndex] = avg_angle * 180/np.pi #need it to be a list of 6, the other qubits dont matter so just amke them the same val
    experiment.readout_cfg['threshold'][QubitIndex] = avg_thresh

    with open(study_notes_path, "a", encoding="utf-8") as file:
        file.write("\n" + f'Rotation angle into I component of signal used for study: {avg_angle * 180/np.pi}')
        file.write("\n" + f'I signal component threshold used for study: {avg_thresh}')

    t6 = time.perf_counter()
    print(f"Repeated SS took {t6 - t5:.4f} seconds")


    ################################################ single t1 g-e with long relax delay #########################

    rr_logger.info("----------------- Optimization: Starting T1 g-e with long relax delay -----------------")
    if verbose:
        print("----------------- Optimization: Starting T1 g-e with long relax delay -----------------")

    t1_long_relax_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
    try:
        t1_long_relax = T1Measurement(QubitIndex, tot_num_of_qubits, optimizationFolder, 0,
                                   signal, save_figs, experiment=experiment,
                                   live_plot=live_plot, fit_data=fit_data,
                                   increase_qubit_reps=increase_qubit_reps,
                                   qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                   multiply_qubit_reps_by=multiply_qubit_reps_by,
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

        saver_t1_long_relax = Data_H5(optimizationFolder, t1_long_relax_data, 0, save_r)
        saver_t1_long_relax.save_to_h5('t1_ge_long_relax')

        del t1_long_relax
        del saver_t1_long_relax
        del t1_long_relax_data

    except Exception as e:
        if debug_mode:
            raise  # In debug mode, re-raise the exception immediately
        rr_logger.exception(f"t1 long relax error on qubit {QubitIndex + 1}: {e}")
        return

    t7 = time.perf_counter()
    print(f"g-e T1 with long relax delay took {t7 - t6:.4f} seconds")

    #    ############################################ stark shift calibration ########################################
    #
    #    rr_logger.info("----------------- Optimization: Starting stark shift calibration   -----------------")
    #    if verbose:
    #        print("----------------- Optimization: Starting stark shift calibration  -----------------")
    #
    #    stark_pos_data = create_data_dict(stark2D_keys, save_r, list_of_all_qubits) #for positive detuning
    #    stark_neg_data = create_data_dict(stark2D_keys, save_r, list_of_all_qubits) #for negative detuning
    #
    #    try:
    #        stark_shift_2D_pos = StarkShift2D(QubitIndex, tot_num_of_qubits, optimizationFolder, save_figs, experiment=experiment)
    #        stark_pos_I, stark_pos_Q, stark_pos_qu_freq_sweep, stark_pos_gain_sweep, sys_config_stark_pos = stark_shift_2D_pos.run(set_pos_detuning=True)
    #
    #        stark_pos_data[QubitIndex]['Dates'][0] = time.mktime(datetime.datetime.now().timetuple())
    #        stark_pos_data[QubitIndex]['I'][0] = stark_pos_I
    #        stark_pos_data[QubitIndex]['Q'][0] = stark_pos_Q
    #        stark_pos_data[QubitIndex]['Qu Frequency Sweep'][0] = stark_pos_qu_freq_sweep
    #        stark_pos_data[QubitIndex]['Res Gain Sweep'][0] = stark_pos_gain_sweep
    #        stark_pos_data[QubitIndex]['Round Num'][0] = 0
    #        stark_pos_data[QubitIndex]['Batch Num'][0] = 0
    #        stark_pos_data[QubitIndex]['Exp Config'][0] = expt_cfg
    #        stark_pos_data[QubitIndex]['Syst Config'][0] = sys_config_stark_pos
    #
    #        saver_stark_pos = Data_H5(optimizationFolder, stark_pos_data, 0, save_r)
    #        saver_stark_pos.save_to_h5('stark_pos_detuning_calibration')
    #
    #        del saver_stark_pos
    #        del stark_shift_2D_pos
    #        del stark_pos_data
    #
    #        stark_shift_2D_neg = StarkShift2D(QubitIndex, tot_num_of_qubits, optimizationFolder, save_figs, experiment=experiment)
    #        stark_neg_I, stark_neg_Q, stark_neg_qu_freq_sweep, stark_neg_gain_sweep, sys_config_stark_neg = stark_shift_2D_neg.run(set_pos_detuning=False)
    #
    #        stark_neg_data[QubitIndex]['Dates'][0] = time.mktime(datetime.datetime.now().timetuple())
    #        stark_neg_data[QubitIndex]['I'][0] = stark_neg_I
    #        stark_neg_data[QubitIndex]['Q'][0] = stark_neg_Q
    #        stark_neg_data[QubitIndex]['Qu Frequency Sweep'][0] = stark_neg_qu_freq_sweep
    #        stark_neg_data[QubitIndex]['Res Gain Sweep'][0] = stark_neg_gain_sweep
    #        stark_neg_data[QubitIndex]['Round Num'][0] = 0
    #        stark_neg_data[QubitIndex]['Batch Num'][0] = 0
    #        stark_neg_data[QubitIndex]['Exp Config'][0] = expt_cfg
    #        stark_neg_data[QubitIndex]['Syst Config'][0] = sys_config_stark_neg
    #
    #        saver_stark_neg = Data_H5(optimizationFolder, stark_neg_data, 0, save_r)
    #        saver_stark_neg.save_to_h5('stark_neg_detuning_calibration')
    #
    #        del saver_stark_neg
    #        del stark_shift_2D_neg
    #        del stark_neg_data
    #        gc.collect()
    #
    #    except Exception as e:
    #        if debug_mode:
    #            raise  # In debug mode, re-raise the exception immediately
    #        else:
    #            rr_logger.exception(f'Got the following error, continuing: {e}')
    #            if verbose: print(f'Got the following error, continuing: {e}')
    #
    t8 = time.perf_counter()
    print(f"fixed detuning stark shift calibration took {t8 - t7:.4f} seconds")
    #
    #    ############################################ resonator stark shift calibration ########################################
    #
    #    rr_logger.info("----------------- Optimization: Starting resonator stark shift calibration   -----------------")
    #    if verbose:
    #            print("----------------- Optimization: Starting resonator stark shift calibration  -----------------")
    #
    #    res_stark_data = create_data_dict(stark2D_keys, save_r, list_of_all_qubits)
    #    res_freq_stark = copy.deepcopy(experiment.readout_cfg['res_freq_ge'])
    #    res_freq_stark.append(res_freq_stark[QubitIndex])
    #
    #    res_phase_stark = copy.deepcopy(experiment.readout_cfg['res_phase'])
    #    res_phase_stark.append(res_phase_stark[QubitIndex])
    #
    #    try:
    #            res_stark_shift_2D = ResStarkShift2D(QubitIndex, tot_num_of_qubits, optimizationFolder, res_freq_stark, res_phase_stark, save_figs, experiment=experiment)
    #            stark_res_I, stark_res_Q, stark_res_qu_freq_sweep, stark_res_gain_sweep, sys_config_stark_res = res_stark_shift_2D.run()
    #
    #            res_stark_data[QubitIndex]['Dates'][0] = time.mktime(datetime.datetime.now().timetuple())
    #            res_stark_data[QubitIndex]['I'][0] = stark_res_I
    #            res_stark_data[QubitIndex]['Q'][0] = stark_res_Q
    #            res_stark_data[QubitIndex]['Qu Frequency Sweep'][0] = stark_res_qu_freq_sweep
    #            res_stark_data[QubitIndex]['Res Gain Sweep'][0] = stark_res_gain_sweep
    #            res_stark_data[QubitIndex]['Round Num'][0] = 0
    #            res_stark_data[QubitIndex]['Batch Num'][0] = 0
    #            res_stark_data[QubitIndex]['Exp Config'][0] = expt_cfg
    #            res_stark_data[QubitIndex]['Syst Config'][0] = sys_config_stark_res
    #
    #            saver_stark_res = Data_H5(optimizationFolder, res_stark_data, 0, save_r)
    #            saver_stark_res.save_to_h5('stark_res_calibration')
    #
    #            del saver_stark_res
    #            del res_stark_shift_2D
    #            del res_stark_data
    #            gc.collect()
    #
    #    except Exception as e:
    #            if debug_mode:
    #                raise  # In debug mode, re-raise the exception immediately
    #            else:
    #                rr_logger.exception(f'Got the following error, continuing: {e}')
    #                if verbose: print(f'Got the following error, continuing: {e}')
    #
    t9 = time.perf_counter()
    print(f"resonator stark shift calibration took {t9 - t8:.4f} seconds")
    #
    ############################################## res spec ef ####################################################
    rr_logger.info("----------------- Optimization: Starting Res Spec EF  -----------------")
    if verbose:
        print("----------------- Optimization: Starting Res Spec EF  -----------------")

    ef_res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    ef_res_freqs_samples = []
    for sample in range(ef_res_sample_number):
        try:
            ef_res_spec = ResonanceSpectroscopyEF(QubitIndex, tot_num_of_qubits, optimizationFolder, sample,
                                             save_figs, experiment=experiment, verbose=verbose,
                                             logger=rr_logger, qick_verbose=qick_verbose)
            ef_res_freqs, ef_freq_pts, ef_freq_center, ef_amps, sys_config_rspec_ef = ef_res_spec.run()
            ef_res_freqs_samples.append(ef_res_freqs)
            rr_logger.info(f"EF ResSpec sample {sample} for qubit {QubitIndex +1}: {ef_res_freqs}")

            ef_res_data[QubitIndex]['Dates'][0] = (
                time.mktime(datetime.datetime.now().timetuple()))
            ef_res_data[QubitIndex]['freq_pts'][0] = ef_freq_pts
            ef_res_data[QubitIndex]['freq_center'][0] = ef_freq_center
            ef_res_data[QubitIndex]['Amps'][0] = ef_amps
            ef_res_data[QubitIndex]['Found Freqs'][0] = ef_res_freqs
            ef_res_data[QubitIndex]['Round Num'][0] = sample
            ef_res_data[QubitIndex]['Batch Num'][0] = 0
            ef_res_data[QubitIndex]['Exp Config'][0] = expt_cfg
            ef_res_data[QubitIndex]['Syst Config'][0] = sys_config_rspec_ef

            saver_ef_res = Data_H5(optimizationFolder, ef_res_data, sample, save_r)  # save
            saver_ef_res.save_to_h5('res_ef')
            del saver_ef_res
            del ef_res_data
            gc.collect()
            ef_res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)  # initialize again to a blank for saftey

            del ef_res_spec

        except Exception as e:
            if debug_mode:
                raise  # In debug mode, re-raise the exception immediately
            rr_logger.exception(f"EF ResSpec error on qubit {QubitIndex +1} sample {sample}: {e}")
            continue

    if ef_res_freqs_samples:
        # Average the resonator frequency values across samples
        avg_ef_res_freqs = np.mean(np.array(ef_res_freqs_samples), axis=0).tolist()
    else:
        rr_logger.error(f"No resonator spectroscopy data collected for qubit {QubitIndex +1}.")
        return

    experiment.readout_cfg['res_freq_ef'] = ef_res_freqs_samples[-1]  # use the last e-f res spec frequency to update the sys config

    rr_logger.info(f"Avg. EF resonator frequencies for qubit {QubitIndex +1}: {avg_ef_res_freqs[QubitIndex]}")
    if verbose:
        print(f"Avg. EF resonator frequencies for qubit {QubitIndex +1}: {avg_ef_res_freqs}")

    with open(study_notes_path, "a", encoding="utf-8") as file:
        file.write("\n" + f'Averaged EF Resonator Frequency Used for study: {avg_ef_res_freqs[QubitIndex]}')

    t10 = time.perf_counter()
    print(f"e-f res spec took {t10 - t9:.4f} seconds")
    ################################################ Qubit Spec EF ################################################
    rr_logger.info("----------------- Optimization: Starting Qubit Spec EF  -----------------")
    if verbose:
        print("----------------- Optimization: Starting Qubit Spec EF  -----------------")

    ef_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)

    try:
        increase_qubit_steps_ef = False
        # Qubit 4 needs more steps for e-f spec
        if QubitIndex == 3:
            increase_qubit_steps_ef = True  # if you want to increase the steps for a qubit, set to True

        number_of_qubits = 6
        j = 0
        ef_q_spec = EFQubitSpectroscopy(QubitIndex, number_of_qubits, list_of_all_qubits, optimizationFolder, j, signal,
                                        save_figs, experiment, live_plot, increase_qubit_steps_ef,
                                        increase_steps_to_ef)
        efqspec_I, efqspec_Q, efqspec_freqs, efqspec_I_fit, efqspec_Q_fit, efqubit_freq, sys_config_qspec_ef = ef_q_spec.run(
            experiment.soccfg,
            experiment.soc)
        qubit_freqs_ef[QubitIndex] = efqubit_freq
        experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] = float(efqubit_freq)

        rr_logger.info(f"EF Qubit {QubitIndex +1} frequency: {efqubit_freq}")
        if verbose:
            print(f"EF Qubit {QubitIndex +1} frequency: {efqubit_freq}")
        del ef_q_spec

        ef_qspec_data[QubitIndex]['Dates'][0] = (
            time.mktime(datetime.datetime.now().timetuple()))
        ef_qspec_data[QubitIndex]['I'][0] = efqspec_I
        ef_qspec_data[QubitIndex]['Q'][0] = efqspec_Q
        ef_qspec_data[QubitIndex]['Frequencies'][0] = efqspec_freqs
        ef_qspec_data[QubitIndex]['I Fit'][0] = efqspec_I_fit
        ef_qspec_data[QubitIndex]['Q Fit'][0] = efqspec_Q_fit
        ef_qspec_data[QubitIndex]['Round Num'][0] = 0
        ef_qspec_data[QubitIndex]['Batch Num'][0] = 0
        ef_qspec_data[QubitIndex]['Recycled QFreq'][0] = False  # no rr so no recycling here
        ef_qspec_data[QubitIndex]['Exp Config'][0] = expt_cfg
        ef_qspec_data[QubitIndex]['Syst Config'][0] = sys_config_qspec_ef

        saver_ef_qspec = Data_H5(optimizationFolder, ef_qspec_data, 0, save_r)
        saver_ef_qspec.save_to_h5('qspec_ef')
        del saver_ef_qspec
        del ef_qspec_data

    except Exception as e:
        if debug_mode:
            raise  # In debug mode, re-raise the exception immediately
        rr_logger.exception(f"EF QubitSpectroscopyGE error on qubit {QubitIndex +1}: {e}")
        return

    t11 = time.perf_counter()
    print(f"e-f Qspec took {t11 - t10:.4f} seconds")

    ################################################ e-f amp rabi pop meas. ################################################
    rr_logger.info("----------------- Optimization: Starting EF Amplitude Rabi Population Measurements -----------------")
    if verbose:
        print("----------------- Optimization: EF Amplitude Rabi Population Measurements  -----------------")

    rabi_data_ef_Qtemps = create_data_dict(rabi_keys_ef_Qtemps, save_r, list_of_all_qubits)
    try:
        number_of_qubits = 6
        j = 0
        efAmprabi_Qtemps = Temps_EFAmpRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits,
                                                     optimizationFolder,
                                                     j,
                                                     signal, save_figs,
                                                     experiment, live_plot,
                                                     increase_qubit_reps, qubit_to_increase_reps_for,
                                                     multiply_qubit_reps_by)
        (I1_qtemp, Q1_qtemp, gains1_qtemp, fit_cosine1_qtemp, pi_amp1_qtemp, A_amplitude1, amp_fit1,
         I2_qtemp, Q2_qtemp, gains2_qtemp, fit_cosine2_qtemp, pi_amp2_qtemp, A_amplitude2, amp_fit2,
         sysconfig_efrabi_Qtemps) = efAmprabi_Qtemps.run(experiment.soccfg, experiment.soc)

        rr_logger.info(
            f"RPM Amplitudes for qubit {QubitIndex +1}: A1 = {float(A_amplitude1)}, A2 = {float(A_amplitude2)}")
        if verbose:
            print(f"RPM Amplitudes for qubit {QubitIndex +1}: A1 = {float(A_amplitude1)}, A2 = {float(A_amplitude2)}")
        with open(study_notes_path, "a", encoding="utf-8") as file:
            file.write(
                "\n" + f"RPM Amplitudes for qubit {QubitIndex +1}: A1 = {float(A_amplitude1)}, A2 = {float(A_amplitude2)}")

        batch_num = 0
        rabi_data_ef_Qtemps[QubitIndex]['Dates'][0] = (time.mktime(datetime.datetime.now().timetuple()))
        rabi_data_ef_Qtemps[QubitIndex]['Qfreq_ge'][0] = qubit_freq  # save the g-e qubit freq too for this qubit

        rabi_data_ef_Qtemps[QubitIndex]['I1'][0] = I1_qtemp
        rabi_data_ef_Qtemps[QubitIndex]['Q1'][0] = Q1_qtemp
        rabi_data_ef_Qtemps[QubitIndex]['Gains1'][0] = gains1_qtemp
        rabi_data_ef_Qtemps[QubitIndex]['Fit1'][0] = fit_cosine1_qtemp

        rabi_data_ef_Qtemps[QubitIndex]['I2'][0] = I2_qtemp
        rabi_data_ef_Qtemps[QubitIndex]['Q2'][0] = Q2_qtemp
        rabi_data_ef_Qtemps[QubitIndex]['Gains2'][0] = gains2_qtemp
        rabi_data_ef_Qtemps[QubitIndex]['Fit2'][0] = fit_cosine2_qtemp

        rabi_data_ef_Qtemps[QubitIndex]['Round Num'][0] = j
        rabi_data_ef_Qtemps[QubitIndex]['Batch Num'][0] = batch_num
        rabi_data_ef_Qtemps[QubitIndex]['Exp Config'][0] = expt_cfg
        rabi_data_ef_Qtemps[QubitIndex]['Syst Config'][0] = sysconfig_efrabi_Qtemps

        saver_rabi_Qtemps = Data_H5(optimizationFolder, rabi_data_ef_Qtemps, batch_num, save_r)
        saver_rabi_Qtemps.save_to_h5('q_temperatures')
        del efAmprabi_Qtemps
        del saver_rabi_Qtemps
        del rabi_data_ef_Qtemps

    except Exception as e:
        if debug_mode:
           raise  # In debug mode, re-raise the exception immediately
        rr_logger.exception(f"EF Rabi Population Measurements error on qubit {QubitIndex +1}: {e}")
        return

    t12 = time.perf_counter()
    print(f"ef Rabi Pop. Measurements took {t12 - t11:.4f} seconds")

    ################################################
    # Fast Repetitive Runs for this Qubit
    ################################################

def run_dataset(Qs_to_look_at, experiment, j, batch_num):

    # create dictionaries
    qspec_data = create_data_dict(qspec_keys, save_r, Qs_to_look_at)
    ss_data = create_data_dict(ss_keys, save_r, Qs_to_look_at)
    t1_data = create_data_dict(t1_keys, save_r, Qs_to_look_at)
    high_gain_qspec_data = create_data_dict(qspec_keys, save_r, Qs_to_look_at)
    med_gain_qspec_data = create_data_dict(qspec_keys, save_r, Qs_to_look_at)
    starkspec_data = create_data_dict(starkspec_keys, save_r, Qs_to_look_at)
    res_starkspec_data = create_data_dict(starkspec_keys, save_r, Qs_to_look_at)

    for QubitIndex in Qs_to_look_at:

        # Set resonator configuration for this qubit
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        # Qubit Spectroscopy
        t0 = time.perf_counter()
        if run_flags["q_spec"]:
            try:
                timestamp_qspec = time.mktime(datetime.datetime.now().timetuple())
                q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyFolder, j,
                                           signal, save_figs, experiment=experiment,
                                           live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                           qick_verbose=qick_verbose, high_gain_q_spec=False)
                (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit,
                 qspec_Q_fit, qubit_freq, sys_config_qspec) = q_spec.run()

                if QubitIndex == 4: #if qubit 5, replace fit frequency with minimum of qspec
                    qubit_freq = get_min_qspec(qspec_I, qspec_Q, qspec_freqs)
                    recycled_qfreq = False
                    experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq) #update experimental config
                    rr_logger.info(f"RR: Qubit {QubitIndex + 1} frequency: {float(qubit_freq)} MHz")
                    if verbose:
                        print(f"Qubit {QubitIndex + 1} frequency found as minimum: {float(qubit_freq)} MHz")

                else: #for qubit 1, continue with usual handling of qubit freq fit result
                    if qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None:
                        # Use the previously stored qubit frequency in the expt config if the fit fails
                        #experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = stored_qspec
                        recycled_qfreq = True
                        #qubit_freq = stored_qspec
                        rr_logger.info(f"RR: Qubit {QubitIndex}: Using previous stored value: {experiment.qubit_cfg['qubit_freq_ge'][QubitIndex]}")
                        if verbose:
                            print(f"Using previous stored value: {experiment.qubit_cfg['qubit_freq_ge'][QubitIndex]}")
                    else:
                        recycled_qfreq = False
                        experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
                        #stored_qspec = float(qubit_freq)
                        rr_logger.info(f"RR: Qubit {QubitIndex +1} frequency: {float(qubit_freq)}")
                
                del q_spec
                gc.collect()
            
            except Exception as e:
                if debug_mode:
                    raise  # In debug mode, re-raise the exception immediately
                rr_logger.exception(f"RR QSpec error on qubit {QubitIndex +1}: {e}")
                continue
        t1 = time.perf_counter()
        print(f"Data taking: g-e Qspec took {t1 - t0:.4f} seconds")

        # Single Shot Measurement
        if run_flags["ss"]:
            try:
                timestamp_ss = time.mktime(datetime.datetime.now().timetuple())
                ss = SingleShot(QubitIndex, tot_num_of_qubits, studyFolder, j, save_figs,
                                experiment=experiment, verbose=verbose, logger=rr_logger, qick_verbose=qick_verbose)
                fid, angle, iq_list_g, iq_list_e, sys_config_ss = ss.run()
                I_g = iq_list_g[QubitIndex][0].T[0]
                Q_g = iq_list_g[QubitIndex][0].T[1]
                I_e = iq_list_e[QubitIndex][0].T[0]
                Q_e = iq_list_g[QubitIndex][0].T[1]  # or iq_list_e depending on your config
                fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(
                    data=[I_g, Q_g, I_e, Q_e], cfg=ss.config, plot=save_figs)
            except Exception as e:
                if debug_mode:
                    raise  # In debug mode, re-raise the exception immediately
                rr_logger.exception(f'Single Shot error on qubit {QubitIndex +1} at round {j}: {e}')
                continue
        t2 = time.perf_counter()
        print(f"Data taking: SS took {t2 - t1:.4f} seconds")

        # T1 Measurement
        if run_flags["t1"]:
            try:
                timestamp_t1 = time.mktime(datetime.datetime.now().timetuple())
                t1 = T1Measurement(QubitIndex, tot_num_of_qubits, studyFolder, j,
                                   signal, save_figs, experiment=experiment,
                                   live_plot=live_plot, fit_data=fit_data,
                                   increase_qubit_reps=increase_qubit_reps,
                                   qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                   multiply_qubit_reps_by=multiply_qubit_reps_by,
                                   verbose=verbose, logger=rr_logger, qick_verbose=qick_verbose, save_shots=True)
                (t1_est, t1_err, t1_I, t1_Q, t1_delay_times,
                 q1_fit_exponential, sys_config_t1) = t1.run(thresholding=False)
                del t1
                gc.collect()
            except Exception as e:
                if debug_mode:
                    raise  # In debug mode, re-raise the exception immediately
                rr_logger.exception(f"RR T1 error on qubit {QubitIndex +1} at round {j}: {e}")
                continue
        t3 = time.perf_counter()
        print(f"Data taking: T1 took {t3 - t2:.4f} seconds")

        # med gain qspec
        if run_flags["med_gain_q_spec"]:
            qubit_gain_temp = experiment.qubit_cfg['qubit_gain_ge']  # save current config parameters
            try:
                timestamp_med_gain_qspec = time.mktime(datetime.datetime.now().timetuple())
                experiment.qubit_cfg['qubit_gain_ge'] = np.ones(len(qubit_gain_temp)) * med_gain  # set med gain

                med_gain_q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyFolder, 0,
                                                    signal, plot_fit=False, save_figs=False,
                                                     experiment=experiment,
                                                     live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                                     qick_verbose=True, high_gain_q_spec=True, fit_data=False)
                (mgqspec_I, mgqspec_Q, mgqspec_freqs, mgqspec_I_fit, mgqspec_Q_fit, mgqubit_freq, sys_config_mgqspec) = med_gain_q_spec.run()


                del med_gain_q_spec

            except Exception as e:
                if debug_mode:
                    raise  # In debug mode, re-raise the exception immediately
                rr_logger.exception(f"medium gain QubitSpectroscopyGE error on qubit {QubitIndex +1}: {e}")

            experiment.qubit_cfg['qubit_gain_ge'] = qubit_gain_temp  # restore parameters for regular qspec


        # high gain qspec
        if run_flags["hi_gain_q_spec"]:
            qubit_gain_temp = experiment.qubit_cfg['qubit_gain_ge']  # save current config parameters
            try:
                timestamp_high_gain_qspec = time.mktime(datetime.datetime.now().timetuple())
                experiment.qubit_cfg['qubit_gain_ge'] = np.ones(len(qubit_gain_temp)) * high_gain  # set high gain

                high_gain_q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyFolder, 0,
                                                    signal, plot_fit=False, save_figs=False,
                                                     experiment=experiment,
                                                     live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                                     qick_verbose=True, high_gain_q_spec=True, fit_data=False)
                (hgqspec_I, hgqspec_Q, hgqspec_freqs, hgqspec_I_fit, hgqspec_Q_fit, hgqubit_freq, sys_config_hgqspec) = high_gain_q_spec.run()


                del high_gain_q_spec

            except Exception as e:
                if debug_mode:
                    raise  # In debug mode, re-raise the exception immediately
                rr_logger.exception(f"high gain QubitSpectroscopyGE error on qubit {QubitIndex +1}: {e}")

            experiment.qubit_cfg['qubit_gain_ge'] = qubit_gain_temp  # restore parameters for regular qspec

        t4 = time.perf_counter()
        print(f"Data taking: Med and High Gain Qspec took {t4 - t3:.4f} seconds")

        # resonator stark spec
        if run_flags["resStarkSpec"]:
            try:
                timestamp_res_starkspec = time.mktime(datetime.datetime.now().timetuple())
                res_freq_stark = copy.deepcopy(experiment.readout_cfg['res_freq_ge'])
                res_freq_stark.append(res_freq_stark[QubitIndex])
                res_phase_stark = copy.deepcopy(experiment.readout_cfg['res_phase'])
                res_phase_stark.append(res_phase_stark[QubitIndex])

                res_stark_shift_spec = ResStarkShiftSpec(QubitIndex, tot_num_of_qubits, studyFolder, res_freq_stark,
                                                             res_phase_stark, save_figs=False,
                                                             experiment=experiment)
                res_starkspec_I, res_starkspec_Q, res_starkspec_P, res_starkspec_shots, res_starkspec_gain_sweep, sys_config_res_starkspec = res_stark_shift_spec.run()
                del res_stark_shift_spec

            except Exception as e:
                if debug_mode:
                    raise e  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
        t5 = time.perf_counter()
        print(f"Data taking: Res Stark Spec took {t5 - t4:.4f} seconds")

        # stark spec
        if run_flags["starkSpec"]:
             try:
                timestamp_starkspec = time.mktime(datetime.datetime.now().timetuple())
                stark_shift_spec = StarkShiftSpec(QubitIndex, tot_num_of_qubits, studyFolder, save_figs=False,
                                                      experiment=experiment)
                starkspec_I, starkspec_Q, starkspec_P, starkspec_shots, starkspec_gain_sweep, sys_config_starkspec = stark_shift_spec.run_with_qick_sweep()
                del stark_shift_spec

             except Exception as e:
                if debug_mode:
                    raise e  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
        t6 = time.perf_counter()
        print(f"Data taking: Stark Spec took {t6 - t5:.4f} seconds")
    ################################################ saving ################################################
        if save_data_h5:
            #idx = j - batch_num * save_r - 1
            idx = 0

            if run_flags["q_spec"]:
                qspec_data[QubitIndex]['Dates'][idx] = timestamp_qspec
                qspec_data[QubitIndex]['I'][idx] = qspec_I
                qspec_data[QubitIndex]['Q'][idx] = qspec_Q
                qspec_data[QubitIndex]['Frequencies'][idx] = qspec_freqs
                qspec_data[QubitIndex]['I Fit'][idx] = qspec_I_fit
                qspec_data[QubitIndex]['Q Fit'][idx] = qspec_Q_fit
                qspec_data[QubitIndex]['Round Num'][idx] = j
                qspec_data[QubitIndex]['Batch Num'][idx] = batch_num
                qspec_data[QubitIndex]['Recycled QFreq'][idx] = recycled_qfreq
                qspec_data[QubitIndex]['Exp Config'][idx] = expt_cfg
                qspec_data[QubitIndex]['Syst Config'][idx] = sys_config_qspec

            if run_flags["hi_gain_q_spec"]:
                high_gain_qspec_data[QubitIndex]['Dates'][idx] = timestamp_high_gain_qspec
                high_gain_qspec_data[QubitIndex]['I'][idx] = hgqspec_I
                high_gain_qspec_data[QubitIndex]['Q'][idx] = hgqspec_Q
                high_gain_qspec_data[QubitIndex]['Frequencies'][idx] = hgqspec_freqs
                high_gain_qspec_data[QubitIndex]['I Fit'][idx] = None
                high_gain_qspec_data[QubitIndex]['Q Fit'][idx] = None
                high_gain_qspec_data[QubitIndex]['Round Num'][idx] = j
                high_gain_qspec_data[QubitIndex]['Batch Num'][idx] = batch_num
                high_gain_qspec_data[QubitIndex]['Recycled QFreq'][idx] = False
                high_gain_qspec_data[QubitIndex]['Exp Config'][idx] = expt_cfg
                high_gain_qspec_data[QubitIndex]['Syst Config'][idx] = sys_config_hgqspec

            if run_flags["med_gain_q_spec"]:
                med_gain_qspec_data[QubitIndex]['Dates'][idx] = timestamp_med_gain_qspec
                med_gain_qspec_data[QubitIndex]['I'][idx] = mgqspec_I
                med_gain_qspec_data[QubitIndex]['Q'][idx] = mgqspec_Q
                med_gain_qspec_data[QubitIndex]['Frequencies'][idx] = mgqspec_freqs
                med_gain_qspec_data[QubitIndex]['I Fit'][idx] = None
                med_gain_qspec_data[QubitIndex]['Q Fit'][idx] = None
                med_gain_qspec_data[QubitIndex]['Round Num'][idx] = j
                med_gain_qspec_data[QubitIndex]['Batch Num'][idx] = batch_num
                med_gain_qspec_data[QubitIndex]['Recycled QFreq'][idx] = False
                med_gain_qspec_data[QubitIndex]['Exp Config'][idx] = expt_cfg
                med_gain_qspec_data[QubitIndex]['Syst Config'][idx] = sys_config_mgqspec

            if run_flags["ss"]:
                ss_data[QubitIndex]['Fidelity'][idx] = fid
                ss_data[QubitIndex]['Angle'][idx] = angle
                ss_data[QubitIndex]['Dates'][idx] = timestamp_ss
                ss_data[QubitIndex]['I_g'][idx] = I_g
                ss_data[QubitIndex]['Q_g'][idx] = Q_g
                ss_data[QubitIndex]['I_e'][idx] = I_e
                ss_data[QubitIndex]['Q_e'][idx] = Q_e
                ss_data[QubitIndex]['Round Num'][idx] = j
                ss_data[QubitIndex]['Batch Num'][idx] = batch_num
                ss_data[QubitIndex]['Exp Config'][idx] = expt_cfg
                ss_data[QubitIndex]['Syst Config'][idx] = sys_config_ss

            if run_flags["t1"]:
                t1_data[QubitIndex]['T1'][idx] = t1_est
                t1_data[QubitIndex]['Errors'][idx] = t1_err
                t1_data[QubitIndex]['Dates'][idx] = timestamp_t1
                t1_data[QubitIndex]['I'][idx] = t1_I
                t1_data[QubitIndex]['Q'][idx] = t1_Q
                t1_data[QubitIndex]['Delay Times'][idx] = t1_delay_times
                t1_data[QubitIndex]['Fit'][idx] = q1_fit_exponential
                t1_data[QubitIndex]['Round Num'][idx] = j
                t1_data[QubitIndex]['Batch Num'][idx] = batch_num
                t1_data[QubitIndex]['Exp Config'][idx] = expt_cfg
                t1_data[QubitIndex]['Syst Config'][idx] = sys_config_t1

            if run_flags["starkSpec"]:
                starkspec_data[QubitIndex]['Dates'][idx] = timestamp_starkspec
                starkspec_data[QubitIndex]['I'][idx] = starkspec_I
                starkspec_data[QubitIndex]['Q'][idx] = starkspec_Q
                starkspec_data[QubitIndex]['P'][idx] = starkspec_P
                starkspec_data[QubitIndex]['shots'][idx] = starkspec_shots
                starkspec_data[QubitIndex]['Gain Sweep'][idx] = starkspec_gain_sweep
                starkspec_data[QubitIndex]['Round Num'][idx] = j
                starkspec_data[QubitIndex]['Batch Num'][idx] = batch_num
                starkspec_data[QubitIndex]['Exp Config'][idx] = expt_cfg
                starkspec_data[QubitIndex]['Syst Config'][idx] = sys_config_starkspec

            if run_flags["resStarkSpec"]:
                res_starkspec_data[QubitIndex]['Dates'][idx] = timestamp_res_starkspec
                res_starkspec_data[QubitIndex]['I'][idx] = res_starkspec_I
                res_starkspec_data[QubitIndex]['Q'][idx] = res_starkspec_Q
                res_starkspec_data[QubitIndex]['P'][idx] = res_starkspec_P
                res_starkspec_data[QubitIndex]['shots'][idx] = res_starkspec_shots
                res_starkspec_data[QubitIndex]['Gain Sweep'][idx] = res_starkspec_gain_sweep
                res_starkspec_data[QubitIndex]['Round Num'][idx] = j
                res_starkspec_data[QubitIndex]['Batch Num'][idx] = batch_num
                res_starkspec_data[QubitIndex]['Exp Config'][idx] = expt_cfg
                res_starkspec_data[QubitIndex]['Syst Config'][idx] = sys_config_res_starkspec

    #once all measurements are taken for each qubit and dictionaries filled with data, save h5 files and delete dictionaries
    if save_data_h5:
        if j % save_r == 0:
            # batch_num += 1 #commented out to keep batch_num at zero
            if run_flags["q_spec"]:
                saver_qspec = Data_H5(studyFolder, qspec_data, batch_num, save_r)
                saver_qspec.save_to_h5('qspec_ge')
                del saver_qspec
                del qspec_data
                gc.collect()

            if run_flags["ss"]:
                saver_ss = Data_H5(studyFolder, ss_data, batch_num, save_r)
                saver_ss.save_to_h5('ss_ge')
                del saver_ss
                del ss_data

            if run_flags["t1"]:
                saver_t1 = Data_H5(studyFolder, t1_data, batch_num, save_r)
                saver_t1.save_to_h5('t1_ge')
                del saver_t1
                del t1_data
                gc.collect()

            # --------------------------save high gain QSpec-----------------------
            if run_flags["hi_gain_q_spec"]:
                saver_high_gain_qspec = Data_H5(studyFolder, high_gain_qspec_data, batch_num, save_r)
                saver_high_gain_qspec.save_to_h5('high_gain_qspec_ge')
                del saver_high_gain_qspec
                del high_gain_qspec_data
                gc.collect()

            # --------------------------save med gain QSpec-----------------------
            if run_flags["med_gain_q_spec"]:
                saver_med_gain_qspec = Data_H5(studyFolder, med_gain_qspec_data, batch_num, save_r)
                saver_med_gain_qspec.save_to_h5('med_gain_qspec_ge')
                del saver_med_gain_qspec
                del med_gain_qspec_data
                gc.collect()

        # --------------------------save starkSpec-----------------------
            if run_flags["starkSpec"]:
                saver_starkspec = Data_H5(studyFolder, starkspec_data, batch_num, save_r)
                saver_starkspec.save_to_h5('starkspec_ge')
                del saver_starkspec
                del starkspec_data
                gc.collect()

            if run_flags["resStarkSpec"]:
                saver_res_starkspec = Data_H5(studyFolder, res_starkspec_data, batch_num, save_r)
                saver_res_starkspec.save_to_h5('res_starkspec_ge')
                del saver_res_starkspec
                del res_starkspec_data
                gc.collect()

    rr_logger.info(f"Round {j} on qubit {QubitIndex +1} took {time.time() - inner_start:.2f} seconds")
    if verbose:
        print(f"Round {j} on qubit {QubitIndex +1} took {time.time() - inner_start:.2f} seconds")

    # create dictionaries
    qspec_data = create_data_dict(qspec_keys, save_r, Qs_to_look_at)
    ss_data = create_data_dict(ss_keys, save_r, Qs_to_look_at)
    t1_data = create_data_dict(t1_keys, save_r, Qs_to_look_at)
    high_gain_qspec_data = create_data_dict(qspec_keys, save_r, Qs_to_look_at)
    starkspec_data = create_data_dict(starkspec_keys, save_r, Qs_to_look_at)
    res_starkspec_data = create_data_dict(starkspec_keys, save_r, Qs_to_look_at)
    gc.collect()
    t7 = time.perf_counter()
    print(f"Data saving took {t7 - t6:.4f} seconds")

################################################
# Run the Full Shabang
################################################

formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
dataSetFolder = os.path.join(subStudyFolder, formatted_datetime)
optimizationFolder = os.path.join(dataSetFolder, 'optimization')
studyFolder = os.path.join(dataSetFolder, 'study_data')
studyDocumentationFolder = os.path.join(dataSetFolder, 'documentation')
study_notes_path = os.path.join(studyDocumentationFolder, 'study_notes.txt')

experiment = QICK_experiment(
    dataSetFolder,
    DAC_attenuator1=5,
    DAC_attenuator2=10,
    ADC_attenuator=10,
    fridge=FRIDGE
)
experiment.create_folder_if_not_exists(dataSetFolder)
experiment.create_folder_if_not_exists(optimizationFolder)
experiment.create_folder_if_not_exists(studyFolder)
experiment.create_folder_if_not_exists(studyDocumentationFolder)
with open(study_notes_path, "w", encoding="utf-8") as file:
    file.write('Study Notes:')

t_fullshebang1 = time.perf_counter()

#run complete optimization for each qubit as a separate data block
for QubitIndex in Qs_to_look_at:
    run_optimization(QubitIndex, ss_sample_number, res_sample_number, experiment)
    try:
        optimization_report_ge(subStudyFolder, formatted_datetime, QubitIndex)
        optimization_report_ef(subStudyFolder, formatted_datetime, QubitIndex)
    except Exception as e:
        if debug_mode:
            raise
        continue

    gc.collect()

rr_logger.info("----------------- Starting repeated measurements (TLS) Step -----------------")
if verbose:
    print("----------------- Starting repeated measurements (TLS) Step -----------------")

j = 0
batch_num = 0
recycled_qfreq = False
while j < n:
    inner_start = time.time()
    run_dataset(Qs_to_look_at, experiment, j, batch_num)
    j+=1
    gc.collect()

del experiment
gc.collect()
t_fullshebang2 = time.perf_counter()
print(f"Full shebang took {t_fullshebang2 - t_fullshebang1:.4f} seconds")
