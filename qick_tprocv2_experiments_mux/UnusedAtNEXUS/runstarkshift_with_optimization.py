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
from T2R_stark import starkT2RMeasurement
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_005_single_shot_ge import SingleShot
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_008_save_data_to_h5 import Data_H5
from starkshift import StarkShift2D
from starkshift import ResStarkShift2D
from starkshift import ResStarkShiftSpec
from starkshift import StarkShiftSpec
from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE

################################################
# Run Configurations and Optimization Params
################################################
ssf_avgs_per_opt_pt = 1
freq_offset_steps = 15
res_sample_number = 1 #10
ss_sample_number = 3
n = 5  # number of rounds for fast repetitive runs
save_r = 1  # how many rounds to save after
signal = 'None'  # 'I', or 'Q' depending on where the signal is
save_figs = True  # whether to save plots
live_plot = False  # use visdom for live plotting?
fit_data = False  # fit data during the run?
save_data_h5 = False  # save data to h5 files?
verbose = True  # verbose output
qick_verbose = False
debug_mode = False  # if True, errors will stop the run immediately
increase_qubit_reps = False
qubit_to_increase_reps_for = 0
multiply_qubit_reps_by = 2
Qs_to_look_at = [4]  # list of qubits to process

# Set which experiments to run
run_flags = {"optimization": False, "stark2D": True, "starkRamsey": False, "starkSpec": False}

# Optimization parameters for resonator spectroscopy
#res_leng_vals = [4.3, 5, 5, 4, 5.8, 4.5]
res_leng_vals  = [4.2, 5, 7.0, 6, 6, 7.5] #DAC 0 optimization
res_gain = [0.96, 1, 0.76, 0.58, 0.75, 0.57]
#res_gain = [1, 1, 0.6, 0.6, 1, 0.6]
#freq_offsets = [0.15, -0.35, 0.1, -0.4, 0.15, 0.3]

#Logging
now = datetime.datetime.now()
formatted_datetime = now.strftime("%Y-%m-%d")
outerFolder = os.path.join("/data/QICK_data/run6/6transmon/StarkShift/calibration_test_5x_pi_pulse/", f"{formatted_datetime}")
if not os.path.exists(outerFolder):
    os.makedirs(outerFolder)
log_file = os.path.join(outerFolder, "starkshift_script.log")
logging.basicConfig(
    level=logging.INFO,  # log all of the things
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        # also output log to the console (remove if you want only the file)
        logging.StreamHandler(sys.stdout)
    ]
)

# Dictionaries
def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in qs}

res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config','Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config','Syst Config']
stark2D_keys = ['Dates', 'I', 'Q', 'Qu Frequency Sweep', 'Res Gain Sweep','Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
starkSpec_keys = ['Dates', 'I', 'Q', 'P', 'shots','Gain Sweep','Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
starkRamsey_keys = ['Ramsey Freq', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']

def sweep_frequency_offset(experiment, QubitIndex, offset_values, n_loops=10, number_of_qubits=6,outerFolder="", j=0):
    baseline_freq = experiment.readout_cfg['res_freq_ge'][QubitIndex]

    ssf_dict = {}
    for offset in offset_values:
        fids = []
        # repeat n times for each offset
        for _ in range(n_loops):
            exp_copy = copy.deepcopy(experiment)  # python is python, doesnt overwrite things properly

            res_freqs = exp_copy.readout_cfg['res_freq_ge']
            res_freqs[QubitIndex] = baseline_freq + offset
            exp_copy.readout_cfg['res_freq_ge'] = res_freqs

            ss = SingleShot(QubitIndex, number_of_qubits, outerFolder, j, save_figs, exp_copy)
            fid, angle, iq_list_g, iq_list_e, ss_config = ss.run()
            fids.append(fid)
            del exp_copy

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
        plot_path = os.path.join(outerFolder, f"SSF_vs_offset_Q{QubitIndex+1}.png")
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

def run_optimization(experiment, QubitIndex, ss_sample_number, res_sample_number):
    logging.info(f"Starting optimization for qubit {QubitIndex}")
    qubitFolder = os.path.join(outerFolder, f'Q{QubitIndex}/optimization')
    if not os.path.exists(qubitFolder):
        os.makedirs(qubitFolder)
    experiment.create_folder_if_not_exists(qubitFolder)

#   ################################################ Res Spec ################################################
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    res_freqs_samples = []

    for sample in range(res_sample_number):
        try:
            res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, qubitFolder, sample, save_figs,
                                             experiment=experiment, verbose=verbose,
                                             logger=logging, qick_verbose=qick_verbose)
            res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
            res_freqs_samples.append(res_freqs)
            logging.info(f"ResSpec sample {sample} for qubit {QubitIndex}: {res_freqs}")

            res_data[QubitIndex]['Dates'][0] = (time.mktime(datetime.datetime.now().timetuple()))
            res_data[QubitIndex]['freq_pts'][0] = freq_pts
            res_data[QubitIndex]['freq_center'][0] = freq_center
            res_data[QubitIndex]['Amps'][0] = amps
            res_data[QubitIndex]['Found Freqs'][0] = res_freqs
            res_data[QubitIndex]['Round Num'][0] = sample
            res_data[QubitIndex]['Batch Num'][0] = 0
            res_data[QubitIndex]['Exp Config'][0] = expt_cfg
            res_data[QubitIndex]['Syst Config'][0] = sys_config_rspec

            saver_res = Data_H5(outerFolder, res_data, sample, save_r)  # save
            saver_res.save_to_h5('Res')
            del saver_res
            del res_data
            gc.collect()
            res_data = create_data_dict(res_keys, save_r,
                                        list_of_all_qubits)  # initialize again to a blank for saftey
            del res_spec

        except Exception as e:
            logging.exception(f"ResSpec error on qubit {QubitIndex} sample {sample}: {e}")
            continue

    if res_freqs_samples:
        # Average the resonator frequency values across samples
        avg_res_freqs = np.mean(np.array(res_freqs_samples), axis=0).tolist()
    else:
        logging.error(f"No resonator spectroscopy data collected for qubit {QubitIndex}.")
        return

    experiment.readout_cfg['res_freq_ge'] = avg_res_freqs  # start with offset of 0, lets try like this and see if works
    logging.info(f"Avg. resonator frequencies for qubit {QubitIndex}: {avg_res_freqs}")

    logging.info("----------------- Moving to Qubit Spec -----------------")
    if verbose:
        print("----------------- Moving to Qubit Spec -----------------")

     ################################################ Qubit Spec ################################################
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    try:
        q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, qubitFolder, 0, signal, save_figs=True, experiment=experiment,
                                   live_plot=live_plot, verbose=verbose, logger=logging, qick_verbose=qick_verbose, increase_reps=True, increase_reps_to=500)
        (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq, sys_config_qspec) = q_spec.run()
        experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
        stored_qspec = float(qubit_freq)
        logging.info(f"Tune-up: Qubit {QubitIndex} frequency: {stored_qspec}")
        del q_spec

        qspec_data[QubitIndex]['Dates'][0] = (time.mktime(datetime.datetime.now().timetuple()))
        qspec_data[QubitIndex]['I'][0] = qspec_I
        qspec_data[QubitIndex]['Q'][0] = qspec_Q
        qspec_data[QubitIndex]['Frequencies'][0] = qspec_freqs
        qspec_data[QubitIndex]['I Fit'][0] = qspec_I_fit
        qspec_data[QubitIndex]['Q Fit'][0] = qspec_Q_fit
        qspec_data[QubitIndex]['Round Num'][0] = 0
        qspec_data[QubitIndex]['Batch Num'][0] = 0
        qspec_data[QubitIndex]['Exp Config'][0] = expt_cfg
        qspec_data[QubitIndex]['Syst Config'][0] = sys_config_qspec

        saver_qspec = Data_H5(outerFolder, qspec_data, 0, save_r)
        saver_qspec.save_to_h5('QSpec')
        del saver_qspec
        del qspec_data

    except Exception as e:
        logging.exception(f"QubitSpectroscopyGE error on qubit {QubitIndex}: {e}")
        return

    logging.info("----------------- Moving to Rabi -----------------")
    if verbose:
        print("----------------- Moving to Rabi -----------------")

    ################################################ amp rabi ################################################
    rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    try:
        rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, qubitFolder, 0,signal, save_figs=True, experiment=experiment,
                                        live_plot=live_plot,
                                        increase_qubit_reps=increase_qubit_reps,
                                        qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                        multiply_qubit_reps_by=multiply_qubit_reps_by,
                                        verbose=verbose, logger=logging,
                                        qick_verbose=qick_verbose)
        (rabi_I, rabi_Q, rabi_gains, rabi_fit, stored_pi_amp, sys_config_rabi) = rabi.run()
        experiment.qubit_cfg['pi_amp'][QubitIndex] = float(stored_pi_amp)
        logging.info(f"Tune-up: Pi amplitude for qubit {QubitIndex}: {float(stored_pi_amp)}")
        rabi_data[QubitIndex]['Dates'][0] = (time.mktime(datetime.datetime.now().timetuple()))
        rabi_data[QubitIndex]['I'][0] = rabi_I
        rabi_data[QubitIndex]['Q'][0] = rabi_Q
        rabi_data[QubitIndex]['Gains'][0] = rabi_gains
        rabi_data[QubitIndex]['Fit'][0] = rabi_fit
        rabi_data[QubitIndex]['Round Num'][0] = 0
        rabi_data[QubitIndex]['Batch Num'][0] = 0
        rabi_data[QubitIndex]['Exp Config'][0] = expt_cfg
        rabi_data[QubitIndex]['Syst Config'][0] = sys_config_rabi
        saver_rabi = Data_H5(outerFolder, rabi_data, 0, save_r)
        saver_rabi.save_to_h5('Rabi')
        del rabi
        del saver_rabi
        del rabi_data
    except Exception as e:
        logging.exception(f"Rabi error on qubit {QubitIndex}: {e}")
        return

    ############################################## optimize ################################################
    reference_frequency = float(avg_res_freqs[QubitIndex])
    freq_range = np.linspace(-1, 1, freq_offset_steps)
    optimal_offset, ssf_dict = sweep_frequency_offset(experiment, QubitIndex, freq_range,n_loops=ssf_avgs_per_opt_pt, number_of_qubits=6,
                                                      outerFolder=outerFolder, j=0)
    offset_res_freqs = [r + optimal_offset for r in avg_res_freqs]
    experiment.readout_cfg['res_freq_ge'] = offset_res_freqs  # update with offset added
    logging.info(f'optimal offset frequency for resonator {QubitIndex}: {optimal_offset} MHz')
    logging.info(f'resonator {QubitIndex} frequency with offset: {offset_res_freqs[QubitIndex]} MHz')

    logging.info("----------------- Moving to repeat SS Step -----------------")
    if verbose:
        print("----------------- Moving to repeat SS Step -----------------")

    ################################################ repeated ss ################################################
    ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
    angles = []
    thresholds = []
    for ss_round in range(ss_sample_number):
        try:
            ss = SingleShot(QubitIndex, tot_num_of_qubits, outerFolder, 0, save_figs, experiment=experiment,
                            verbose=verbose, logger=logging)
            fid, angle, iq_list_g, iq_list_e, sys_config_ss = ss.run()
            I_g = iq_list_g[QubitIndex][0].T[0]
            Q_g = iq_list_g[QubitIndex][0].T[1]
            I_e = iq_list_e[QubitIndex][0].T[0]
            Q_e = iq_list_e[QubitIndex][0].T[1]

            fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(data=[I_g, Q_g, I_e, Q_e], cfg=ss.config,plot=save_figs)
            angles.append(angle)
            thresholds.append(threshold)

            ss_data[QubitIndex]['Fidelity'][0] = fid
            ss_data[QubitIndex]['Angle'][0] = angle
            ss_data[QubitIndex]['Dates'][0] = (time.mktime(datetime.datetime.now().timetuple()))
            ss_data[QubitIndex]['I_g'][0] = I_g
            ss_data[QubitIndex]['Q_g'][0] = Q_g
            ss_data[QubitIndex]['I_e'][0] = I_e
            ss_data[QubitIndex]['Q_e'][0] = Q_e
            ss_data[QubitIndex]['Round Num'][0] = ss_round
            ss_data[QubitIndex]['Batch Num'][0] = 0
            ss_data[QubitIndex]['Exp Config'][0] = expt_cfg
            ss_data[QubitIndex]['Syst Config'][0] = sys_config_ss

            saver_ss = Data_H5(outerFolder, ss_data, 0, save_r)
            saver_ss.save_to_h5('SS')
            del saver_ss
            del ss_data
            gc.collect()
            ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
        except Exception as e:
            if debug_mode:
                raise  # In debug mode, re-raise the exception immediately
            else:
                logging.exception(f'Got the following error, continuing: {e}')
                if verbose:
                    print(f'Got the following error, continuing: {e}')
                continue  # skip the rest of this

    # average angles and thresholds to use
    avg_angle = np.mean(angles)
    avg_thresh = np.mean(thresholds)
    experiment.readout_cfg['res_phase'] = [avg_angle * 180 / np.pi] * 6  # need it to be a list of 6, the other qubits dont matter so just amke them the same val
    experiment.readout_cfg['threshold'] = [avg_thresh] * 6

    logging.info(f"Average angle for Qubit {QubitIndex}: {avg_angle}")
    logging.info(f"Average threshold for Qubit {QubitIndex}: {avg_thresh}")

    logging.info("----------------- Optimization Complete -----------------")
    if verbose:
        print("----------------- Optimization Complete -----------------")

    gc.collect()

def run_stark2D(experiment, QubitIndex):
    logging.info(f"Starting initial optimization for qubit {QubitIndex}")
    qubitFolder = os.path.join(outerFolder, f'Q{QubitIndex}/stark2D')
    if not os.path.exists(qubitFolder):
        os.makedirs(qubitFolder)

    experiment.create_folder_if_not_exists(qubitFolder)

    res_freq_stark = copy.deepcopy(experiment.readout_cfg['res_freq_ge'])
    res_freq_stark.append(res_freq_stark[QubitIndex])

    res_phase_stark = copy.deepcopy(experiment.readout_cfg['res_phase'])
    res_phase_stark.append(res_phase_stark[QubitIndex])

    batch_num = 0
    j = 0

    stark2D_data = create_data_dict(stark2D_keys, save_r, list_of_all_qubits)

    while j < n:
        inner_start = time.time()
        j += 1

        try:
            #stark_shift_2D = ResStarkShift2D(QubitIndex, tot_num_of_qubits, qubitFolder, res_freq_stark, res_phase_stark, save_figs, experiment=experiment)
            stark_shift_2D = StarkShift2D(QubitIndex, tot_num_of_qubits, qubitFolder, save_figs, experiment = experiment)
            I, Q, qu_freq_sweep, res_gain_sweep, sys_config = stark_shift_2D.run(set_pos_detuning = True)
            stark_shift_2D.plot(I, Q, qu_freq_sweep, res_gain_sweep)

            stark2D_data[QubitIndex]['Dates'][0] = time.mktime(datetime.datetime.now().timetuple())
            stark2D_data[QubitIndex]['I'][0] = I
            stark2D_data[QubitIndex]['Q'][0] = Q
            stark2D_data[QubitIndex]['Qu Frequency Sweep'][0] = qu_freq_sweep
            stark2D_data[QubitIndex]['Res Gain Sweep'][0] = res_gain_sweep
            stark2D_data[QubitIndex]['Round Num'][0] = j
            stark2D_data[QubitIndex]['Batch Num'][0] = batch_num
            stark2D_data[QubitIndex]['Exp Config'][0] = expt_cfg
            stark2D_data[QubitIndex]['Syst Config'][0] = sys_config

            saver_2D = Data_H5(qubitFolder, stark2D_data, batch_num, save_r)
            saver_2D.save_to_h5('StarkShift2D')
            del saver_2D
            del stark_shift_2D
            gc.collect()
            stark2D_data = create_data_dict(stark2D_keys, save_r, list_of_all_qubits)

        except Exception as e:
            logging.exception(f'Got the following error, continuing: {e}')
            gc.collect()
            continue  # skip the rest of this qubit

        logging.info(
            f"Stark Shift 2D Round {j} on qubit {QubitIndex} took {time.time() - inner_start:.2f} seconds")
        if verbose:
            print(f"Stark Shift 2D Round {j} on qubit {QubitIndex} took {time.time() - inner_start:.2f} seconds")

    gc.collect()

def run_starkSpec(experiment, QubitIndex):
    batch_num = 0
    j = 0
    qubitFolder = os.path.join(outerFolder, f'Q{QubitIndex}/starkSpec')

    #starkSpec_data = create_data_dict(starkSpec_keys, save_r, list_of_all_qubits)
    starkSpec_data = create_data_dict(starkSpec_keys, n, list_of_all_qubits)

    while j < n:
        inner_start = time.time()
        j += 1

        try:
            res_freq_stark = copy.deepcopy(experiment.readout_cfg['res_freq_ge'])
            res_freq_stark.append(res_freq_stark[QubitIndex])

            res_phase_stark = copy.deepcopy(experiment.readout_cfg['res_phase'])
            res_phase_stark.append(res_phase_stark[QubitIndex])

            stark_shift_spec = StarkShiftSpec(QubitIndex, tot_num_of_qubits, qubitFolder, save_figs,
                                             experiment=experiment)
            I, Q, P, shots, gain_sweep, sys_config = stark_shift_spec.run_with_qick_sweep()
            stark_shift_spec.plot(P, gain_sweep)
            stark_shift_spec.plot_shots(I, Q, shots, gain_sweep, gain_index=0)

            # res_stark_shift_spec = ResStarkShiftSpec(QubitIndex, tot_num_of_qubits, qubitFolder, res_freq_stark, res_phase_stark, save_figs,
            #                                  experiment=experiment)
            # I, Q, P, shots, gain_sweep, sys_config = res_stark_shift_spec.run()
            # res_stark_shift_spec.plot(P, gain_sweep)
            # res_stark_shift_spec.plot_shots(I, Q, shots, gain_sweep)

            idx = j - batch_num * save_r - 1
            starkSpec_data[QubitIndex]['Dates'][idx] = time.mktime(datetime.datetime.now().timetuple())
            starkSpec_data[QubitIndex]['I'][idx] = I
            starkSpec_data[QubitIndex]['Q'][idx] = Q
            starkSpec_data[QubitIndex]['P'][idx] = P
            starkSpec_data[QubitIndex]['shots'][idx] = shots
            starkSpec_data[QubitIndex]['Gain Sweep'][idx] = gain_sweep
            starkSpec_data[QubitIndex]['Round Num'][idx] = j
            starkSpec_data[QubitIndex]['Batch Num'][idx] = batch_num
            starkSpec_data[QubitIndex]['Exp Config'][idx] = expt_cfg
            starkSpec_data[QubitIndex]['Syst Config'][idx] = sys_config

        except Exception as e:
            logging.exception(f'Got the following error, continuing: {e}')
            del stark_shift_spec
            continue  # skip the rest o

    saver_spec = Data_H5(qubitFolder, starkSpec_data, batch_num, save_r)
    saver_spec.save_to_h5('StarkSpec')
    del saver_spec
    del stark_shift_spec
    gc.collect()
        #starkSpec_data = create_data_dict(starkSpec_keys, save_r, list_of_all_qubits)
    starkSpec_data = create_data_dict(starkSpec_keys, n, list_of_all_qubits)


    logging.info(
            f"Stark Shift Spec Round {j} on qubit {QubitIndex} took {time.time() - inner_start:.2f} seconds")
    if verbose:
        print(
                f"Stark Shift Spec Round {j} on qubit {QubitIndex} took {time.time() - inner_start:.2f} seconds")

    gc.collect()

def run_starkRamsey(experiment, QubitIndex):
    batch_num = 0
    j = 0

    qubitFolder = os.path.join(outerFolder, f'Q{QubitIndex}/starkRamsey')
    starkRamsey_data = create_data_dict(starkRamsey_keys, save_r, list_of_all_qubits)

    try:
        t2r = starkT2RMeasurement(QubitIndex, tot_num_of_qubits, qubitFolder, j, signal, save_figs,
                             experiment=experiment, fit_data=True, verbose=verbose, logger=logging)
        t2r_est, t2r_err, f_est, f_err, t2r_I, t2r_Q, t2r_delay_times, fit_ramsey, sys_config_t2r = t2r.run(
            thresholding=False)

        starkRamsey_data[QubitIndex]['Ramsey Freq'][j - batch_num * save_r - 1] = f_est
        starkRamsey_data[QubitIndex]['Errors'][j - batch_num * save_r - 1] = f_err
        starkRamsey_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (time.mktime(datetime.datetime.now().timetuple()))
        starkRamsey_data[QubitIndex]['I'][j - batch_num * save_r - 1] = t2r_I
        starkRamsey_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = t2r_Q
        starkRamsey_data[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t2r_delay_times
        starkRamsey_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fit_ramsey
        starkRamsey_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
        starkRamsey_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
        starkRamsey_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
        starkRamsey_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2r

    except Exception as e:
        if debug_mode:
            raise e  # In debug mode, re-raise the exception immediately
        else:
            logging.exception(f'Got the following error, continuing: {e}')
            if verbose: print(f'Got the following error, continuing: {e}')

    saver_spec = Data_H5(qubitFolder, starkRamsey_data, batch_num, save_r)
    saver_spec.save_to_h5('StarkRamsey')
    del saver_spec
    del t2r
    gc.collect()
    starkRamsey_data = create_data_dict(starkSpec_keys, n, list_of_all_qubits)

###############################################
# Run
################################################

for QubitIndex in Qs_to_look_at:

    experiment = QICK_experiment(
        outerFolder,
        DAC_attenuator1=5,
        DAC_attenuator2=10,
        ADC_attenuator=10,
        fridge=FRIDGE,
    )

    # Set resonator configuration for this qubit
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

    if run_flags["optimization"]:
        run_optimization(experiment, QubitIndex, ss_sample_number, res_sample_number)

    if run_flags["stark2D"]:
        run_stark2D(experiment, QubitIndex)

    if run_flags["starkSpec"]:
        run_starkSpec(experiment, QubitIndex)

    if run_flags["starkRamsey"]:
        run_starkRamsey(experiment, QubitIndex)

    del experiment
    gc.collect()

