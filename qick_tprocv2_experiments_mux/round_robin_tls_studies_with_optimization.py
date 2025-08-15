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

################################################
# Run Configurations and Optimization Params
################################################
ssf_avgs_per_opt_pt = 10
freq_offset_steps = 20
res_sample_number = 330
ss_sample_number = 10
n = 250  # number of rounds for fast repetitive runs
save_r = 1  # how many rounds to save after
signal = 'None'  # 'I', or 'Q' depending on where the signal is
save_figs = False  # whether to save plots
live_plot = False  # use visdom for live plotting?
fit_data = False  # fit data during the run?
save_data_h5 = True  # save data to h5 files?
verbose = False  # verbose output
qick_verbose = False
debug_mode = False  # if True, errors will stop the run immediately
increase_qubit_reps = False
qubit_to_increase_reps_for = 0
multiply_qubit_reps_by = 2
study = 'TLS_PSD_Studies'
sub_study = 'source_on'
substudy_txt_notes = 'Fixed res gain and readout pulse/window length, offset frequency optimised in each batch'
Qs_to_look_at = [3]  # list of qubits to process

# Set which experiments to run
run_flags = {"q_spec": True, "rabi": False, "ss": False, "t1": True}

# Optimization parameters for resonator spectroscopy
res_leng_vals = [4.3, 5, 5, 4, 4.5, 9]
res_gain = [1, 1, 0.6, 0.6, 1, 0.6]
freq_offsets = [0.15, -0.35, 0.1, -0.4, 0.15, 0.3]


# Dictionaries
def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in qs}

res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit',
              'Round Num', 'Batch Num', 'Recycled QFreq', 'Exp Config', 'Syst Config']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']


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

            ss = SingleShot(QubitIndex, number_of_qubits, outerFolder, j, save_figs, exp_copy)
            fid, angle, iq_list_g, iq_list_e, ss_config = ss.run()
            fids.append(fid)
            del exp_copy

            try:
                ss = SingleShot(QubitIndex, tot_num_of_qubits, optimizationFolder, 0, save_figs, experiment=experiment,
                                verbose=verbose, logger=rr_logger)
                fid, angle, iq_list_g, iq_list_e, sys_config_ss = ss.run()
                I_g = iq_list_g[QubitIndex][0].T[0]
                Q_g = iq_list_g[QubitIndex][0].T[1]
                I_e = iq_list_e[QubitIndex][0].T[0]
                Q_e = iq_list_e[QubitIndex][0].T[1]

                ss_data[QubitIndex]['Fidelity'][0] = fid
                ss_data[QubitIndex]['Angle'][0] = angle
                ss_data[QubitIndex]['Dates'][0] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                ss_data[QubitIndex]['I_g'][0] = I_g
                ss_data[QubitIndex]['Q_g'][0] = Q_g
                ss_data[QubitIndex]['I_e'][0] = I_e
                ss_data[QubitIndex]['Q_e'][0] = Q_e
                ss_data[QubitIndex]['Round Num'][0] = i
                ss_data[QubitIndex]['Batch Num'][0] = 0
                ss_data[QubitIndex]['Exp Config'][0] = expt_cfg
                ss_data[QubitIndex]['Syst Config'][0] = sys_config_ss

                saver_ss = Data_H5(optimizationFolder, ss_data, 0, save_r)
                saver_ss.save_to_h5('SS')
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

        # find avg ssf
        avg_fid = np.mean(fids)
        ssf_dict[offset] = avg_fid
        if verbose:
            print(f"Offset: {offset} -> Average SSF: {avg_fid:.4f}")

    plt.figure()
    offsets_sorted = sorted(ssf_dict.keys())
    ssf_values = [ssf_dict[offset] for offset in offsets_sorted]
    plt.plot(offsets_sorted, ssf_values, marker='o')
    plt.xlabel('Frequency Offset')
    plt.ylabel('Average SSF')
    plt.title(f'SSF vs Frequency Offset for Qubit {QubitIndex+1}')
    plt.grid(True)
    if outerFolder:
        os.makedirs(outerFolder, exist_ok=True)
        plot_path = os.path.join(studyDocumentationFolder, f"SSF_vs_offset_Q{QubitIndex+1}.png")
        plt.savefig(plot_path)
        if verbose:
            print(f"Plot saved to {plot_path}")
    plt.close()


    # Determine the offset value that yielded the best (highest) average SSF.
    optimal_offset = max(ssf_dict, key=ssf_dict.get)
    if verbose:
        print(
            f"Optimal frequency offset for Qubit {QubitIndex}: {optimal_offset} (Avg SSF: {ssf_dict[optimal_offset]:.4f})")

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
log_file = os.path.join(subStudyFolder, "RR_TLS_script.log")
rr_logger = logging.getLogger("custom_logger_for_rr_only")
rr_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
rr_logger.addHandler(file_handler)
rr_logger.propagate = False


def run_full_cycle_for_qubit(QubitIndex, ss_sample_number, res_sample_number):
    rr_logger.info(f"Starting full cycle for qubit {QubitIndex}")
    formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataSetFolder = os.path.join(subStudyFolder, formatted_datetime)
    optimizationFolder = os.path.join(dataSetFolder, 'optimization')
    studyFolder = os.path.join(dataSetFolder, 'study')
    studyDocumentationFolder = os.path.join(dataSetFolder, 'study_documentation')
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

    # Set resonator configuration for this qubit
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

    ################################################ Res Spec ################################################
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    res_freqs_samples = []
    for sample in range(res_sample_number):
        try:
            res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, optimizationFolder, sample,
                                             save_figs, experiment=experiment, verbose=verbose,
                                             logger=rr_logger, qick_verbose=qick_verbose)
            res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
            res_freqs_samples.append(res_freqs)
            rr_logger.info(f"ResSpec sample {sample} for qubit {QubitIndex}: {res_freqs}")

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
            saver_res.save_to_h5('Res')
            del saver_res
            del res_data
            gc.collect()
            res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)  # initialize again to a blank for saftey

            del res_spec

        except Exception as e:
            rr_logger.exception(f"ResSpec error on qubit {QubitIndex} sample {sample}: {e}")
            continue


    if res_freqs_samples:
        # Average the resonator frequency values across samples
        avg_res_freqs = np.mean(np.array(res_freqs_samples), axis=0).tolist()
    else:
        rr_logger.error(f"No resonator spectroscopy data collected for qubit {QubitIndex}.")
        return

    experiment.readout_cfg['res_freq_ge'] = avg_res_freqs #start with offset of 0, lets try like this and see if works
    rr_logger.info(f"Avg. resonator frequencies for qubit {QubitIndex}: {avg_res_freqs}")
    with open(study_notes_path, "a", encoding="utf-8") as file:
        file.write("\n" + f'Averaged Resonator Frequency Used for study: {avg_res_freqs[QubitIndex]}')

    rr_logger.info("----------------- Moving to Optimization Step -----------------")
    if verbose:
        print("----------------- Moving to Optimization Step -----------------")
    ################################################ Qubit Spec ################################################
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    try:
        q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, 0,
                                   signal, save_figs=True, experiment=experiment,
                                   live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                   qick_verbose=qick_verbose, increase_reps = True, increase_reps_to = 500)
        (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit,
         qubit_freq, sys_config_qspec) = q_spec.run()
        experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
        stored_qspec = float(qubit_freq)
        rr_logger.info(f"Tune-up: Qubit {QubitIndex + 1} frequency: {stored_qspec}")
        del q_spec


        qspec_data[QubitIndex]['Dates'][0] = (
            time.mktime(datetime.datetime.now().timetuple()))
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

        saver_qspec = Data_H5(optimizationFolder, qspec_data, 0, save_r)
        saver_qspec.save_to_h5('QSpec')
        del saver_qspec
        del qspec_data

    except Exception as e:
        rr_logger.exception(f"QubitSpectroscopyGE error on qubit {QubitIndex}: {e}")
        return


    ################################################ amp rabi ################################################
    rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    try:
        rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, 0,
                                       signal, save_figs=True, experiment=experiment,
                                       live_plot=live_plot,
                                       increase_qubit_reps=increase_qubit_reps,
                                       qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                       multiply_qubit_reps_by=multiply_qubit_reps_by,
                                       verbose=verbose, logger=rr_logger,
                                       qick_verbose=qick_verbose)
        (rabi_I, rabi_Q, rabi_gains, rabi_fit, stored_pi_amp, sys_config_rabi) = rabi.run()
        experiment.qubit_cfg['pi_amp'][QubitIndex] = float(stored_pi_amp)
        rr_logger.info(f"Tune-up: Pi amplitude for qubit {QubitIndex + 1}: {float(stored_pi_amp)}")
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
        saver_rabi.save_to_h5('Rabi')
        del rabi
        del saver_rabi
        del rabi_data
    except Exception as e:
        rr_logger.exception(f"Rabi error on qubit {QubitIndex}: {e}")
        return

    ################################################ optimize ################################################
    reference_frequency = float(avg_res_freqs[QubitIndex])
    freq_range = np.linspace(-1, 1, freq_offset_steps)

    optimal_offset, ssf_dict = sweep_frequency_offset(experiment, QubitIndex, freq_range, n_loops=ssf_avgs_per_opt_pt, number_of_qubits=6,
                           outerFolder=optimizationFolder, studyDocumentationFolder=studyDocumentationFolder, j=0)

    offset_res_freqs = [r + optimal_offset for r in avg_res_freqs]
    experiment.readout_cfg['res_freq_ge'] = offset_res_freqs    #update with ofset added

    with open(study_notes_path, "a", encoding="utf-8") as file:
        file.write("\n" + f'Offset Frequency used for study: {offset_res_freqs[QubitIndex]}')

    rr_logger.info("----------------- Moving to repeat SS Step -----------------")
    if verbose:
        print("----------------- Moving to repeat SS Step -----------------")
    ################################################ repeated ss ################################################
    ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
    angles =[]
    thresholds=[]
    for ss_round in range(ss_sample_number):
        try:
            ss = SingleShot(QubitIndex, tot_num_of_qubits, optimizationFolder, 0, save_figs, experiment=experiment,
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
            saver_ss.save_to_h5('SS')
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
    experiment.readout_cfg['res_phase']=[avg_angle * 180/np.pi] * 6 #need it to be a list of 6, the other qubits dont matter so just amke them the same val
    experiment.readout_cfg['threshold'] = [avg_thresh] * 6

    with open(study_notes_path, "a", encoding="utf-8") as file:
        file.write("\n" + f'Rotation angle into I component of signal used for study: {avg_angle * 180/np.pi}')
        file.write("\n" + f'I signal component threshold used for study: {avg_thresh}')

    rr_logger.info("----------------- Moving to repeated measurements (TLS) Step -----------------")
    if verbose:
        print("----------------- Moving to repeated measurements (TLS) Step -----------------")
    ################################################
    # Fast Repetitive Runs for this Qubit
    ################################################
    qspec_data = create_data_dict(qspec_keys, save_r, [QubitIndex])
    rabi_data = create_data_dict(rabi_keys, save_r, [QubitIndex])
    ss_data = create_data_dict(ss_keys, save_r, [QubitIndex])
    t1_data = create_data_dict(t1_keys, save_r, [QubitIndex])

    j = 0
    batch_num = 0
    while j < n:
        inner_start = time.time()
        j += 1
        recycled_qfreq = False

        # Qubit Spectroscopy
        if run_flags["q_spec"]:
            try:
                q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyFolder, j,
                                           signal, save_figs, experiment=experiment,
                                           live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                           qick_verbose=qick_verbose)
                (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit,
                 qspec_Q_fit, qubit_freq, sys_config_qspec) = q_spec.run()

                if qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None:
                    # Use the previously stored qubit frequency if the fit fails
                    experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = stored_qspec
                    recycled_qfreq = True
                    qubit_freq = stored_qspec
                    if verbose:
                        print(f"Using previous stored value: {qubit_freq}")
                else:
                    experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
                    stored_qspec = float(qubit_freq)
                rr_logger.info(f"RR: Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                del q_spec
                gc.collect()
            except Exception as e:
                rr_logger.exception(f"RR QSpec error on qubit {QubitIndex}: {e}")
                continue

        # Rabi Measurement
        if run_flags["rabi"]:
            try:
                rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyFolder, j,
                                               signal, save_figs, experiment=experiment,
                                               live_plot=live_plot,
                                               increase_qubit_reps=increase_qubit_reps,
                                               qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                               multiply_qubit_reps_by=multiply_qubit_reps_by,
                                               verbose=verbose, logger=rr_logger, qick_verbose=qick_verbose)
                (rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_rabi) = rabi.run()

                if rabi_fit is None and pi_amp is None:
                    rr_logger.info(f"Rabi fit failed on qubit {QubitIndex} at round {j}; skipping iteration.")
                    del rabi
                    gc.collect()
                    continue

                experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
                rr_logger.info(f"RR: Pi amplitude for qubit {QubitIndex + 1}: {float(pi_amp)}")
                del rabi
                gc.collect()
            except Exception as e:
                rr_logger.exception(f"RR Rabi error on qubit {QubitIndex} at round {j}: {e}")
                continue
        # else:
        #     experiment.qubit_cfg['pi_amp'][QubitIndex] = float(stored_pi_amp)  #no need anymore, saved already in experiment

        # Single Shot Measurement
        if run_flags["ss"]:
            try:
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
                rr_logger.exception(f'Single Shot error on qubit {QubitIndex} at round {j}: {e}')
                continue

        # T1 Measurement
        if run_flags["t1"]:
            try:
                t1 = T1Measurement(QubitIndex, tot_num_of_qubits, studyFolder, j,
                                   signal, save_figs, experiment=experiment,
                                   live_plot=live_plot, fit_data=fit_data,
                                   increase_qubit_reps=increase_qubit_reps,
                                   qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                   multiply_qubit_reps_by=multiply_qubit_reps_by,
                                   verbose=verbose, logger=rr_logger, qick_verbose=qick_verbose)
                (t1_est, t1_err, t1_I, t1_Q, t1_delay_times,
                 q1_fit_exponential, sys_config_t1) = t1.run(thresholding=True)
                del t1
                gc.collect()
            except Exception as e:
                rr_logger.exception(f"RR T1 error on qubit {QubitIndex} at round {j}: {e}")
                continue

        ################################################ saving ################################################
        if save_data_h5:
            if run_flags["q_spec"]:
                idx = j - batch_num * save_r - 1
                qspec_data[QubitIndex]['Dates'][idx] = time.mktime(datetime.datetime.now().timetuple())
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
            if run_flags["rabi"]:
                idx = j - batch_num * save_r - 1
                rabi_data[QubitIndex]['Dates'][idx] = time.mktime(datetime.datetime.now().timetuple())
                rabi_data[QubitIndex]['I'][idx] = rabi_I
                rabi_data[QubitIndex]['Q'][idx] = rabi_Q
                rabi_data[QubitIndex]['Gains'][idx] = rabi_gains
                rabi_data[QubitIndex]['Fit'][idx] = rabi_fit
                rabi_data[QubitIndex]['Round Num'][idx] = j
                rabi_data[QubitIndex]['Batch Num'][idx] = batch_num
                rabi_data[QubitIndex]['Exp Config'][idx] = expt_cfg
                rabi_data[QubitIndex]['Syst Config'][idx] = sys_config_rabi
            if run_flags["ss"]:
                ss_data[QubitIndex]['Fidelity'][j - batch_num * save_r - 1] = fid
                ss_data[QubitIndex]['Angle'][j - batch_num * save_r - 1] = angle
                ss_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = time.mktime(datetime.datetime.now().timetuple())
                ss_data[QubitIndex]['I_g'][j - batch_num * save_r - 1] = I_g
                ss_data[QubitIndex]['Q_g'][j - batch_num * save_r - 1] = Q_g
                ss_data[QubitIndex]['I_e'][j - batch_num * save_r - 1] = I_e
                ss_data[QubitIndex]['Q_e'][j - batch_num * save_r - 1] = Q_e
                ss_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                ss_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                ss_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                ss_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_ss
            if run_flags["t1"]:
                idx = j - batch_num * save_r - 1
                t1_data[QubitIndex]['T1'][idx] = t1_est
                t1_data[QubitIndex]['Errors'][idx] = t1_err
                t1_data[QubitIndex]['Dates'][idx] = time.mktime(datetime.datetime.now().timetuple())
                t1_data[QubitIndex]['I'][idx] = t1_I
                t1_data[QubitIndex]['Q'][idx] = t1_Q
                t1_data[QubitIndex]['Delay Times'][idx] = t1_delay_times
                t1_data[QubitIndex]['Fit'][idx] = q1_fit_exponential
                t1_data[QubitIndex]['Round Num'][idx] = j
                t1_data[QubitIndex]['Batch Num'][idx] = batch_num
                t1_data[QubitIndex]['Exp Config'][idx] = expt_cfg
                t1_data[QubitIndex]['Syst Config'][idx] = sys_config_t1

            if j % save_r == 0:
                batch_num += 1
                if run_flags["q_spec"]:
                    saver_qspec = Data_H5(studyFolder, qspec_data, batch_num, save_r)
                    saver_qspec.save_to_h5('QSpec')
                    del saver_qspec
                    gc.collect()
                if run_flags["rabi"]:
                    saver_rabi = Data_H5(studyFolder, rabi_data, batch_num, save_r)
                    saver_rabi.save_to_h5('Rabi')
                    del saver_rabi
                    gc.collect()
                if run_flags["ss"]:
                    saver_ss = Data_H5(studyFolder, ss_data, batch_num, save_r)
                    saver_ss.save_to_h5('SS')
                    del saver_ss
                    del ss_data
                if run_flags["t1"]:
                    saver_t1 = Data_H5(studyFolder, t1_data, batch_num, save_r)
                    saver_t1.save_to_h5('T1')
                    del saver_t1
                    gc.collect()

                #saftey reinitializing
                qspec_data = create_data_dict(qspec_keys, save_r, [QubitIndex])
                rabi_data = create_data_dict(rabi_keys, save_r, [QubitIndex])
                ss_data = create_data_dict(ss_keys, save_r, [QubitIndex])
                t1_data = create_data_dict(t1_keys, save_r, [QubitIndex])
        rr_logger.info(f"Round {j} on qubit {QubitIndex + 1} took {time.time() - inner_start:.2f} seconds")
        if verbose:
            print(f"Round {j} on qubit {QubitIndex + 1} took {time.time() - inner_start:.2f} seconds")
        gc.collect()

    # End of RR loop for this qubit; clean up
    del experiment
    gc.collect()
    rr_logger.info(f"Completed full cycle for qubit {QubitIndex}")

################################################
# Run the Full Shabang for Each Qubit Sequentially
################################################

for QubitIndex in Qs_to_look_at:
    run_full_cycle_for_qubit(QubitIndex, ss_sample_number, res_sample_number)
    gc.collect()
