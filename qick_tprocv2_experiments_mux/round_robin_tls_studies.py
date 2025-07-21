import sys
import os
import numpy as np
import datetime
import time
import logging
import gc
import visdom

# Increase print threshold so that all data is shown when needed
np.set_printoptions(threshold=int(1e15))

# Add the local repository path
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
n = 600 # for two hours, 17 seconds per round w/ rabi
save_r = 1  # how many rounds to save after
signal = 'None'  # 'I', or 'Q' depending on where the signal is (or 'None' if no optimization)
save_figs = False  # whether to save plots
live_plot = False  # use visdom for live plotting?
fit_data = False  # fit data during the run?
save_data_h5 = True  # save data to h5 files?
verbose = False  # verbose output
debug_mode = False  # if True, errors will stop the run immediately
increase_qubit_reps = False  # option to increase reps for a qubit
qick_verbose=False
qubit_to_increase_reps_for = 0  # only has effect if increase_qubit_reps is True
multiply_qubit_reps_by = 2  # only has effect if the previous flag is True
Qs_to_look_at = [5] # # only process these qubits

# Set which experiments to run
run_flags = {"q_spec": True, "rabi": False, "ss":False, "t1": True}

# Optimization parameters for resonator spectroscopy
res_leng_vals = [4.3, 5, 5, 4, 4.5, 9]
res_gain = [1, 1, 0.6, 0.6, 1, 0.6]
freq_offsets = [0.15, -0.35, 0.1, -0.4, 0.15, 0.3]


################################################
# Configure Logging
################################################
outerFolder = os.path.join("/data/QICK_data/6transmon_run6/", 'TLS_Studies')
if not os.path.exists(outerFolder):
    os.makedirs(outerFolder)
log_file = os.path.join(outerFolder, "RR_TLS_script.log")
rr_logger = logging.getLogger("custom_logger_for_rr_only")
rr_logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
rr_logger.addHandler(file_handler)
rr_logger.propagate = False  # prevent propagation of logs from underlying packages

################################################
# Utility: Data Dictionary Creator
################################################
def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Keys for each experiment type
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit',
              'Round Num', 'Batch Num', 'Recycled QFreq', 'Exp Config', 'Syst Config']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times',
           'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']

################################################
# Stage 1: Tune-up (Preliminary Measurements)
################################################
def tune_qubits():

    stored_qspec_list = [None] * tot_num_of_qubits
    res_freqs_list = [None] * tot_num_of_qubits
    pi_amp_list = [None] * tot_num_of_qubits

    if live_plot:
        viz = visdom.Visdom()
        if not viz.check_connection(timeout_seconds=5):
            raise RuntimeError("Visdom server not connected! Start it with 'visdom' and open http://localhost:8097/.")

    for QubitIndex in Qs_to_look_at:
        # Create and/or verify folder for this qubit
        qubitFolder = os.path.join("/data/QICK_data/6transmon_run6/TLS_Studies/", f'Q{QubitIndex}')
        if not os.path.exists(qubitFolder):
            os.makedirs(qubitFolder)

        recycled_qfreq = False

        # Create experiment configuration for this qubit
        experiment = QICK_experiment(
            qubitFolder,
            DAC_attenuator1=5,
            DAC_attenuator2=10,
            ADC_attenuator=10,
            fridge=FRIDGE
        )
        experiment.create_folder_if_not_exists(qubitFolder)

        # Set resonator configuration for this qubit only
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        
        ##################################################
        # Resonator Spectroscopy
        ##################################################
        try:
            res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, qubitFolder, 0,
                                             save_figs, experiment=experiment, verbose=verbose, logger=rr_logger,
                                             qick_verbose=qick_verbose)
            
            res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
            # Update readout config using optimized offset
            offset = freq_offsets[QubitIndex]
            offset_res_freqs = [r + offset for r in res_freqs]
            experiment.readout_cfg['res_freq_ge'] = offset_res_freqs
            res_freqs_list[QubitIndex] = res_freqs
            del res_spec
            gc.collect()
        except Exception as e:
            if debug_mode:
                raise e
            rr_logger.exception(f"ResSpec error on qubit {QubitIndex}: {e}")
            if verbose:
                print(f"ResSpec error on qubit {QubitIndex}: {e}")
                del experiment
                gc.collect()
            continue  # Skip this qubit if resonator spectroscopy fails
  
        ##################################################
        # Qubit Spectroscopy
        ##################################################
        try:
            q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, qubitFolder, 0,
                                       signal, save_figs, experiment=experiment,
                                       live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                       qick_verbose=qick_verbose)
            (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit,
             qubit_freq, sys_config_qspec) = q_spec.run()
  
            # If the fit fails, revert to a stored value if available
            if qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None:
                if stored_qspec_list[QubitIndex] is not None:
                    experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = stored_qspec_list[QubitIndex]
                    rr_logger.warning(f"Using previous stored value: {stored_qspec_list[QubitIndex]}")
                    recycled_qfreq = True
                    if verbose:
                        print(f"Using previous stored value: {stored_qspec_list[QubitIndex]}")
                else:
                    rr_logger.warning(f"No previous qubit spec values stored for qubit {QubitIndex}; skipping.")
                    del q_spec, experiment
                    gc.collect()
                    continue
  
            experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
            stored_qspec_list[QubitIndex] = float(qubit_freq)
            rr_logger.info(f"Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
            if verbose:
                print(f"Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                del q_spec
                gc.collect()
        except Exception as e:
            if debug_mode:
                raise e
            rr_logger.exception(f"QubitSpec error on qubit {QubitIndex}: {e}")
            if verbose:
                print(f"QubitSpec error on qubit {QubitIndex}: {e}")
                del experiment
                gc.collect()
            continue
  
        ##################################################
        # Amplitude Rabi Measurement
        ##################################################
        try:
            rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, qubitFolder, 0,
                                           signal, save_figs, experiment=experiment,
                                           live_plot=live_plot,
                                           increase_qubit_reps=increase_qubit_reps,
                                           qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                           multiply_qubit_reps_by=multiply_qubit_reps_by,
                                           verbose=verbose, logger=rr_logger,
                                           qick_verbose=qick_verbose)
            (rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_rabi) = rabi.run()
  
            # If fit fails, skip the qubit
            if rabi_fit is None and pi_amp is None:
                rr_logger.info(f"Rabi fit failed on qubit {QubitIndex}; skipping.")
                if verbose:
                    print(f"Rabi fit failed on qubit {QubitIndex}; skipping.")
                    del rabi, experiment
                    gc.collect()
                continue
  
            experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
            pi_amp_list[QubitIndex] = float(pi_amp)
            rr_logger.info(f"Pi amplitude for qubit {QubitIndex + 1}: {float(pi_amp)}")
            if verbose:
                print(f"Pi amplitude for qubit {QubitIndex + 1}: {float(pi_amp)}")
                del rabi
                gc.collect()
        except Exception as e:
            if debug_mode:
                raise e
            rr_logger.exception(f"Rabi error on qubit {QubitIndex}: {e}")
            if verbose:
                print(f"Rabi error on qubit {QubitIndex}: {e}")
                del experiment
                gc.collect()
            continue
  
        # End of this qubit’s tune-up: clean up and force garbage collection.
        del experiment
        gc.collect()
  
    return stored_qspec_list, res_freqs_list, pi_amp_list

################################################
# Stage 2: Fast Repetitive Runs (RR)
################################################
def fast_RR_loop(stored_qspec_list, res_freqs_list, pi_amp_list):
    # Initialize data dictionaries for storing experimental results
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
    t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)

    for QubitIndex in Qs_to_look_at:
        batch_num = 0
        j = 0

        qubitFolder = os.path.join("/data/QICK_data/6transmon_run6/TLS_Studies/", f'Q{QubitIndex}')

        # Create a new experiment configuration for the fast RR loop
        experiment = QICK_experiment(
            qubitFolder,
            DAC_attenuator1=5,
            DAC_attenuator2=10,
            ADC_attenuator=10,
            fridge=FRIDGE
        )
        experiment.create_folder_if_not_exists(qubitFolder)

        # Set resonator configuration using pre-tuned values
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        # Use pre-tuned resonator frequency values with optimized offset
        experiment.readout_cfg['res_freq_ge'] = res_freqs_list[QubitIndex]
        offset = freq_offsets[QubitIndex]
        offset_res_freqs = [r + offset for r in res_freqs_list[QubitIndex]]
        experiment.readout_cfg['res_freq_ge'] = offset_res_freqs

        # Use pre-tuned pi amplitude
        experiment.qubit_cfg['pi_amp'][QubitIndex] = pi_amp_list[QubitIndex]

        while j < n:
            inner_start = time.time()
            j += 1
            recycled_qfreq = False

            ##################################################
            # Qubit Spectroscopy 
            ##################################################
            if run_flags["q_spec"]:
                try:
                    q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, qubitFolder, j,
                                               signal, save_figs, experiment=experiment,
                                               live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                               qick_verbose=qick_verbose)
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
                            rr_logger.warning(f"No stored qubit spec value for qubit {QubitIndex}; skipping iteration.")
                            del q_spec
                            gc.collect()
                            continue
                    else:
                        experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
                        stored_qspec_list[QubitIndex] = float(qubit_freq)
                    rr_logger.info(f"Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                    if verbose:
                        print(f"Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                    del q_spec
                    gc.collect()
                except Exception as e:
                    if debug_mode:
                        raise e
                    rr_logger.exception(f"RR QSpec error on qubit {QubitIndex}: {e}")
                    if verbose:
                        print(f"RR QSpec error on qubit {QubitIndex}: {e}")
                    continue

            ##################################################
            # Rabi Measurement 
            ##################################################
            if run_flags["rabi"]:
                try:
                    rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, qubitFolder, j,
                                                   signal, save_figs, experiment=experiment,
                                                   live_plot=live_plot,
                                                   increase_qubit_reps=increase_qubit_reps,
                                                   qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                                   multiply_qubit_reps_by=multiply_qubit_reps_by,
                                                   verbose=verbose, logger=rr_logger, qick_verbose=qick_verbose)
                    (rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp,
                     sys_config_rabi) = rabi.run()

                    if rabi_fit is None and pi_amp is None:
                        rr_logger.info(f"Rabi fit failed on qubit {QubitIndex} at round {j}; skipping iteration.")
                        del rabi
                        gc.collect()
                        continue

                    experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
                    rr_logger.info(f"Pi amplitude for qubit {QubitIndex + 1}: {float(pi_amp)}")
                    if verbose:
                        print(f"Pi amplitude for qubit {QubitIndex + 1}: {float(pi_amp)}")
                    del rabi
                    gc.collect()
                except Exception as e:
                    if debug_mode:
                        raise e
                    rr_logger.exception(f"RR Rabi error on qubit {QubitIndex} at round {j}: {e}")
                    if verbose:
                        print(f"RR Rabi error on qubit {QubitIndex} at round {j}: {e}")
                    continue
            ##################################################
            # Single Shot Measurement
            ##################################################
            if run_flags["ss"]:
                try:
                    ss = SingleShot(QubitIndex, tot_num_of_qubits, outerFolder, j, save_figs,
                                    experiment=experiment,
                                    verbose=verbose, logger=rr_logger, qick_verbose=qick_verbose)
                    fid, angle, iq_list_g, iq_list_e, sys_config_ss = ss.run()
                    I_g = iq_list_g[QubitIndex][0].T[0]
                    Q_g = iq_list_g[QubitIndex][0].T[1]
                    I_e = iq_list_e[QubitIndex][0].T[0]
                    Q_e = iq_list_e[QubitIndex][0].T[1]

                    fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(
                        data=[I_g, Q_g, I_e, Q_e], cfg=ss.config, plot=save_figs)

                except Exception as e:
                    if debug_mode:
                        raise  # In debug mode, re-raise the exception immediately
                    else:
                        rr_logger.exception(f'Got the following error, continuing: {e}')
                        if verbose: print(f'Got the following error, continuing: {e}')
                        continue  # skip the rest of this qubit

            ##################################################
            # T1 Measurement 
            ##################################################
            if run_flags["t1"]:
                try:
                    t1 = T1Measurement(QubitIndex, tot_num_of_qubits, qubitFolder, j,
                                       signal, save_figs, experiment=experiment,
                                       live_plot=live_plot, fit_data=fit_data,
                                       increase_qubit_reps=increase_qubit_reps,
                                       qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                       multiply_qubit_reps_by=multiply_qubit_reps_by,
                                       verbose=verbose, logger=rr_logger, qick_verbose=qick_verbose)
                    (t1_est, t1_err, t1_I, t1_Q, t1_delay_times,
                     q1_fit_exponential, sys_config_t1) = t1.run()
                    del t1
                    gc.collect()
                except Exception as e:
                    if debug_mode:
                        raise e
                    rr_logger.exception(f"RR T1 error on qubit {QubitIndex} at round {j}: {e}")
                    if verbose:
                        print(f"RR T1 error on qubit {QubitIndex} at round {j}: {e}")
                    continue

            ##################################################
            # Data Collection and Saving
            ##################################################
            try:
                if save_data_h5:
                    # Collect QSpec results
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

                    # Collect Rabi results
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

                    # Collect Single Shot Results
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

                    # Collect T1 results
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

                    # Save data every 'save_r' rounds
                    if j % save_r == 0:
                        batch_num += 1
                        if run_flags["q_spec"]:
                            saver_qspec = Data_H5(qubitFolder, qspec_data, batch_num, save_r)
                            saver_qspec.save_to_h5('QSpec')
                            del saver_qspec
                            gc.collect()
                        if run_flags["rabi"]:
                            saver_rabi = Data_H5(qubitFolder, rabi_data, batch_num, save_r)
                            saver_rabi.save_to_h5('Rabi')
                            del saver_rabi
                            gc.collect()
                        if run_flags["ss"]:
                            saver_ss = Data_H5(outerFolder, ss_data, batch_num, save_r)
                            saver_ss.save_to_h5('SS')
                            del saver_ss
                            del ss_data

                        if run_flags["t1"]:
                            saver_t1 = Data_H5(qubitFolder, t1_data, batch_num, save_r)
                            saver_t1.save_to_h5('T1')
                            del saver_t1
                            gc.collect()

                        # Reinitialize data dictionaries after saving
                        qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
                        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
                        ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
                        t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)

                rr_logger.info(f"Round {j} on qubit {QubitIndex + 1} took {time.time() - inner_start:.2f} seconds")
                if verbose:
                    print(f"Round {j} on qubit {QubitIndex + 1} took {time.time() - inner_start:.2f} seconds")
                gc.collect()
            except Exception as e:
                rr_logger.exception(f"Data collection error on qubit {QubitIndex} at round {j}: {e}")
                continue

        # End of fast RR loop for this qubit
        del experiment
        gc.collect()
        j = 0
        batch_num = 0

################################################
# Run everything here
################################################
#Stage 1: Tune up qubits once and store key calibration parameters
stored_qspec_list, res_freqs_list, pi_amp_list = tune_qubits()

## Stage 2: Run the fast repetitive measurement loop using tuned parameters
fast_RR_loop(stored_qspec_list, res_freqs_list, pi_amp_list)
