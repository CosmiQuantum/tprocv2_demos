import sys
import os
from copy import deepcopy
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
import logging
import visdom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_005_single_shot_ge import SingleShot
from section_009_T2R_ge import T2RMeasurement
from section_008_save_data_to_h5 import Data_H5
from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
################################################ Run Configurations ####################################################
zero_qubit_drive_gain = False
constant_zeno_pulse = True
adapt_starked_qubit_freq = False
wait_for_res_ring_up = True
n= 1
unmask = True
save_r = 1                           # how many rounds to save after
signal = 'None'                      # 'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization

save_figs = True                     # save plots for everything as you go along the RR script?
fit_data = False                     # fit the data here and save or plot the fits?
save_data_h5 = True                  # save all of the data to h5 files?

verbose = True                       # print everything to the console in real time, good for debugging, bad for memory
debug_mode = True                  # if True, it disables the continuing function of RR if an error pops up in a class -- errors now stop the RR script
thresholding = False                 # use internal QICK threshold for ratio of Binary values on y for rabi/t1/t2r/t2e, or analog avg when false
increase_qubit_reps = False          # if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0       # only has impact if previous line is True
multiply_qubit_reps_by = 2           # only has impact if the line two above is True
save_shots_gerabi = False

Qs_to_look_at = [0]        # only list the qubits you want to do the RR for

#Data saving info
run_name = 'run9d'
device_name = '6transmon'
substudy_txt_notes = ('testing active reset')

run_flags = {"res_spec": True, "q_spec": True, "rabi": True, "ss": True}

################################################ optimization outputs ##################################################
# Optimization parameters for resonator spectroscopy
res_leng_vals = [5.4, 6.4, 6.2, 6.2, 6.8, 7.0]
res_gain = [0.8164, 0.8, 0.8156,0.6156, 0.8125, 0.8375]
freq_offsets = [-0.2000, -0.1111, -0.1111,-0.1111,-0.3000,-0.0667]

# To save how long each measurement took for each qubit
meas_time_RR = {}
################################################ Data Saving Setup ##################################################
# Folders
study = 'active_reset' #qubit_checkouts, round_robin_benchmark
sub_study = 'junk' #batch2_post_1stopt_25dbDAC, opt_sigmas_gains_reps_steps
data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if not os.path.exists(f"/data/QICK_data/{run_name}/"):
    os.makedirs(f"/data/QICK_data/{run_name}/")
if not os.path.exists(f"/data/QICK_data/{run_name}/{device_name}/"):
    os.makedirs(f"/data/QICK_data/{run_name}/{device_name}/")
studyFolder = os.path.join(f"/data/QICK_data/{run_name}/{device_name}/", study)
if not os.path.exists(studyFolder):
    os.makedirs(studyFolder)
subStudyFolder = os.path.join(studyFolder, sub_study)
if not os.path.exists(subStudyFolder):
    os.makedirs(subStudyFolder)

dataSetFolder = os.path.join(subStudyFolder, data_set)
optimizationFolder = os.path.join(dataSetFolder, 'optimization')
studyFolder = os.path.join(dataSetFolder, 'study_data')
studyDocumentationFolder = os.path.join(dataSetFolder, 'documentation')
subStudyDataFolder = os.path.join(dataSetFolder, 'study_data')
if not os.path.exists(studyDocumentationFolder):
    os.makedirs(studyDocumentationFolder)
if not os.path.exists(optimizationFolder):
    os.makedirs(optimizationFolder)
if not os.path.exists(subStudyDataFolder):
    os.makedirs(subStudyDataFolder)

file_path = os.path.join(studyDocumentationFolder, 'sub_study_notes.txt')
with open(file_path, "w", encoding="utf-8") as file:
    file.write(substudy_txt_notes)

################################################## Configure logging ###################################################
''' We need to create a custom logger and disable propagation like this
to remove the logs from the underlying qick from saving to the log file for RR'''

log_file = os.path.join(studyDocumentationFolder, "RR_script.log")
rr_logger = logging.getLogger("custom_logger_for_rr_only")
rr_logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

rr_logger.addHandler(file_handler)
rr_logger.propagate = False  # dont propagate logs from underlying qick package

############################################# Pre-configure saving ###############################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num','Recycled QFreq',
              'Exp Config', 'Syst Config','ss_Q_e', 'ss_Q_g','ss_I_e', 'ss_I_g', 'I_shots', 'Q_shots', 'Gains']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config',  'ss_Q_e', 'ss_Q_g','ss_I_e', 'ss_I_g', 'I_shots', 'Q_shots']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config', 'I_shots', 'Q_shots']

#initialize a simple list to store the qspec values in incase a fit fails
stored_qspec_list = [None] * tot_num_of_qubits

res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)

batch_num=0
j = 0
for QubitIndex in Qs_to_look_at:

    # Get the config for this qubit
    DAC_attenuator1 = 15
    DAC_attenuator2 = 10
    experiment = QICK_experiment(optimizationFolder, DAC_attenuator1=DAC_attenuator1, DAC_attenuator2=DAC_attenuator2,
                                 qubit_DAC_attenuator1=5,
                                 qubit_DAC_attenuator2=4, ADC_attenuator=17,
                                 fridge=FRIDGE)  # ADC_attenuator MUST be above 16dB

    experiment.create_folder_if_not_exists(optimizationFolder)

    experiment.readout_cfg['res_gain_ge'] = res_gain[QubitIndex]
    experiment.readout_cfg['res_gain_ef'] = res_gain[QubitIndex]
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]
    experiment.readout_cfg['res_freq_ge'] = experiment.readout_cfg['res_freq_ge'][QubitIndex]

    experiment.qubit_cfg['qubit_freq_ge'] = experiment.qubit_cfg['qubit_freq_ge'][QubitIndex]
    experiment.qubit_cfg['qubit_gain_ge'] = experiment.qubit_cfg['qubit_gain_ge'][QubitIndex]

    experiment.readout_cfg['res_freq_ge'] = freq_offsets[QubitIndex] + 7287.57 # where did this number come from
    experiment.qubit_cfg['qubit_freq_ge'] = float(3095.192) # where did this number come from
    experiment.qubit_cfg['pi_amp'] =  experiment.qubit_cfg['pi_amp'][QubitIndex]

    ############################### Do Res spec once per qubit and store the value ####################################
    ################################################## g-e Res spec ####################################################
    if run_flags["res_spec"]:
        res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
        t0 = time.perf_counter()
        try:
            increase_geres_reps = False
            increase_geres_reps_to = None
            use_savgol_smoothing_rspec = False

            res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs,
                                             increase_geres_reps,
                                             increase_geres_reps_to, experiment=experiment, verbose=verbose,
                                             logger=rr_logger, unmasking_resgain=unmask,
                                             use_savgol_smoothing=use_savgol_smoothing_rspec)
            res_freqs, freq_pts, freq_center, amps, sys_config_rspec, meas_timestamp_resge = res_spec.run()
            offset = freq_offsets[
                QubitIndex]  # use optimized offset values or whats set at top of script based on pre_optimize flag
            offset_res_freqs = [r + offset for r in res_freqs]
            experiment.readout_cfg['res_freq_ge'] = offset_res_freqs
            del res_spec

        except Exception as e:
            if debug_mode:
                raise e  # In debug mode, re-raise the exception immediately
            else:
                rr_logger.exception(f'Got the following error, continuing: {e}')
                if verbose: print(f'Got the following error during ge res spec, continuing: {e}')
                continue  # skip the rest of this qubit

        end_time = time.perf_counter()
        meas_time_RR[QubitIndex]["res_spec_ge"] = end_time - t0

        res_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (time.mktime(datetime.datetime.now().timetuple()))
        res_data[QubitIndex]['freq_pts'][j - batch_num * save_r - 1] = freq_pts
        res_data[QubitIndex]['freq_center'][j - batch_num * save_r - 1] = freq_center
        res_data[QubitIndex]['Amps'][j - batch_num * save_r - 1] = amps
        res_data[QubitIndex]['Found Freqs'][j - batch_num * save_r - 1] = res_freqs
        res_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
        res_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
        res_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
        res_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rspec
        res_data[QubitIndex]['measurement_timestamp'][j - batch_num * save_r - 1] = meas_timestamp_resge

        saver_res = Data_H5(optimizationFolder, res_data, batch_num, save_r)  # save
        saver_res.save_to_h5('res_ge')
        del saver_res
        del res_data
        res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)  # initialize again to a blank for safety

    ######################################################## g-e Qubit Spec ###################################################################
    if run_flags["q_spec"]:
        qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
        t0 = time.perf_counter()
        try:
            increase_qubit_reps_qspec = False
            increase_qspec_rounds = False
            qspecge_increase_reps_to = None
            increase_qspec_rounds_to = None

            # if QubitIndex == 0:
            #     increase_qubit_reps_qspec = True
            #     qspecge_increase_reps_to = 800

            q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j,
                                       signal, save_figs, increase_reps=increase_qubit_reps_qspec,
                                       increase_rounds=increase_qspec_rounds,
                                       increase_reps_to=qspecge_increase_reps_to,
                                       increase_rounds_to=increase_qspec_rounds_to,
                                       plot_fit=True, experiment=experiment, live_plot=False, verbose=verbose,
                                       logger=rr_logger, unmasking_resgain=unmask)
            (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq, sys_config_qspec,
             meas_timestamp_qspecge) = q_spec.run()

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
                        print(f'No stored g-e qubit spec value for qubit {QubitIndex}; skipping iteration.')
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
            else:
                rr_logger.exception(f"RR g-e QSpec error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"g-e QSpec error on qubit {QubitIndex}: {e}")
                continue

        end_time = time.perf_counter()
        meas_time_RR[QubitIndex]["qspec_ge"] = end_time - t0

        qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (time.mktime(datetime.datetime.now().timetuple()))
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
        qspec_data[QubitIndex]['measurement_timestamp'][j - batch_num * save_r - 1] = meas_timestamp_qspecge

        saver_qspec = Data_H5(optimizationFolder, qspec_data, batch_num, save_r)
        saver_qspec.save_to_h5('qspec_ge')
        del saver_qspec
        del qspec_data

        rr_logger.info(f"g-e Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
        if verbose:
            print(f"g-e Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")

        qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits) # reset for safety

    ##################################################### g-e power rabi ############################################################
    if run_flags["rabi"]:
        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
        t0 = time.perf_counter()
        try:
            increase_qubit_reps_gerabi = False  # if you want to increase the reps for a qubit, set to True
            qubit_to_increase_gerabi_reps_for = None  # only has impact if previous line is True
            multiply_gerabi_reps_by = 1
            reduce_rlx_delay_gerabi = False
            reduce_rlx_delay_gerabi_to = None

            # if QubitIndex == 0:
            #     increase_qubit_reps_gerabi = True
            #     qubit_to_increase_gerabi_reps_for = QubitIndex
            #     multiply_gerabi_reps_by = 2

            rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                           save_figs=save_figs, save_shots=save_shots_gerabi,
                                           experiment=experiment, live_plot=False,
                                           increase_qubit_reps=increase_qubit_reps_gerabi,
                                           qubit_to_increase_reps_for=qubit_to_increase_gerabi_reps_for,
                                           multiply_qubit_reps_by=multiply_gerabi_reps_by,
                                           verbose=verbose, logger=rr_logger, unmasking_resgain=unmask,
                                           reduce_rlx_delay=reduce_rlx_delay_gerabi,
                                           reduce_rlx_delay_to=reduce_rlx_delay_gerabi_to)
            (rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_rabi, meas_timestamp_rabige) = rabi.run(
                thresholding=thresholding, use_iminuit_instead=True)

            # if these are None, fit didnt work
            if (rabi_fit is None and pi_amp is None):
                rr_logger.info('g-e Rabi fit didnt work, skipping the rest of this qubit')
                if verbose: print('g-e Rabi fit didnt work, skipping the rest of this qubit')
                continue  # skip the rest of this qubit

            experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
            rr_logger.info(f'g-e Pi amplitude for qubit {QubitIndex + 1} is: {float(pi_amp)}')
            if verbose: print('g-e Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
            del rabi

        except Exception as e:
            if debug_mode:
                raise e  # In debug mode, re-raise the exception immediately
            else:
                rr_logger.exception(f'Got the following error in ge rabi, continuing: {e}')
                if verbose: print(f'Got the following error in ge rabi, continuing: {e}')
                continue  # skip the rest of this qubit

        end_time = time.perf_counter()
        meas_time_RR[QubitIndex]["power_rabi_ge"] = end_time - t0

        rabi_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (time.mktime(datetime.datetime.now().timetuple()))
        rabi_data[QubitIndex]['I'][j - batch_num * save_r - 1] = rabi_I
        rabi_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = rabi_Q
        rabi_data[QubitIndex]['Gains'][j - batch_num * save_r - 1] = rabi_gains
        rabi_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = rabi_fit
        rabi_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
        rabi_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
        rabi_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
        rabi_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rabi
        rabi_data[QubitIndex]['measurement_timestamp'][j - batch_num * save_r - 1] = meas_timestamp_rabige

        saver_rabi = Data_H5(optimizationFolder, rabi_data, batch_num, save_r)
        saver_rabi.save_to_h5('rabi_ge')
        del saver_rabi
        del rabi_data
        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)

    ############################### g-e Single Shot Measurement, get the rotation angle and update config ############################
    if run_flags["ss"]:
        ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
        t0 = time.perf_counter()
        try:
            reduce_rlx_delay_ssf = False
            reduce_rlx_delay_ssf_to = None
            if QubitIndex == 5:
                reduce_rlx_delay_ssf = True
                reduce_rlx_delay_ssf_to = 650

            max_tries = 1  # 5
            try_num = 0
            fid_check = 0

            ssf_thresholds = [0.85, 0.80, 0.80, 0.8, 0.20,0.75]  # all Qs are generally above these unless something is wrong
            ssf_threshold = ssf_thresholds[QubitIndex]

            while fid_check < ssf_threshold and try_num < max_tries:
                ## after the loop finishes, the code only keeps the data from the last attempt that ran
                try_num += 1

                ss = SingleShot(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs,
                                experiment=experiment, verbose=verbose, logger=rr_logger, unmasking_resgain=unmask,
                                reduce_rlx_delay=reduce_rlx_delay_ssf, reduce_rlx_delay_to=reduce_rlx_delay_ssf_to)
                fid, angle, _, iq_list_g, iq_list_e, sys_config_ss, meas_timestamp_ssge = ss.run()

                fid_check = fid
                # if fid_check < ssf_threshold: # checks if SSF is bad, if it is it tries again
                #     rr_logger.warning(
                #         f"Q{QubitIndex + 1} SSF fid={fid_check:.3f} below {ssf_threshold}, retrying "
                #         f"({try_num}/{max_tries})")

            # if fid_check < ssf_threshold:
            #     rr_logger.warning(f"Q{QubitIndex + 1} SSF never reached {ssf_threshold}. Keeping last attempt.")

            I_g = iq_list_g[QubitIndex][0].T[0]
            Q_g = iq_list_g[QubitIndex][0].T[1]
            I_e = iq_list_e[QubitIndex][0].T[0]
            Q_e = iq_list_e[QubitIndex][0].T[1]

            theta = -np.arctan2(np.median(Q_e) - np.median(Q_g), np.median(I_e) - np.median(I_g))
            # update config ro_phase to rotate blobs onto I for future experiments below this
            experiment.readout_cfg['ro_phase'] = np.degrees(theta)

            ss_data[QubitIndex]['Fidelity'][j - batch_num * save_r - 1] = fid
            ss_data[QubitIndex]['Angle'][j - batch_num * save_r - 1] = angle
            ss_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (time.mktime(datetime.datetime.now().timetuple()))
            ss_data[QubitIndex]['I_g'][j - batch_num * save_r - 1] = I_g
            ss_data[QubitIndex]['Q_g'][j - batch_num * save_r - 1] = Q_g
            ss_data[QubitIndex]['I_e'][j - batch_num * save_r - 1] = I_e
            ss_data[QubitIndex]['Q_e'][j - batch_num * save_r - 1] = Q_e
            ss_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            ss_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            ss_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
            ss_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_ss
            ss_data[QubitIndex]['measurement_timestamp'][j - batch_num * save_r - 1] = meas_timestamp_ssge

            saver_ss = Data_H5(subStudyDataFolder, ss_data, batch_num, save_r)
            saver_ss.save_to_h5('ss_ge_unpre_rotated')
            del saver_ss
            del ss_data
            del ss

            ##################### g-e Single Shot Measurement, should be rotated all into I ########################
            ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
            ss = SingleShot(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs,
                            experiment=experiment, verbose=verbose, logger=rr_logger, unmasking_resgain=unmask,
                            reduce_rlx_delay=reduce_rlx_delay_ssf, reduce_rlx_delay_to=reduce_rlx_delay_ssf_to)
            fid, angle, thresh, iq_list_g, iq_list_e, sys_config_ss, meas_timestamp_ssge = ss.run()
            I_g = iq_list_g[QubitIndex][0].T[0]
            Q_g = iq_list_g[QubitIndex][0].T[1]
            I_e = iq_list_e[QubitIndex][0].T[0]
            Q_e = iq_list_e[QubitIndex][0].T[1]

            experiment.readout_cfg['threshold'] = thresh
            experiment.readout_cfg['g_center'] = g_center[0] # 0 is I, 1 is Q
            experiment.readout_cfg['e_center'] = e_center[0]
            print(thresh, e_center, g_center)

            ss_data[QubitIndex]['Fidelity'][j - batch_num * save_r - 1] = fid
            ss_data[QubitIndex]['Angle'][j - batch_num * save_r - 1] = angle
            ss_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (time.mktime(datetime.datetime.now().timetuple()))
            ss_data[QubitIndex]['I_g'][j - batch_num * save_r - 1] = I_g
            ss_data[QubitIndex]['Q_g'][j - batch_num * save_r - 1] = Q_g
            ss_data[QubitIndex]['I_e'][j - batch_num * save_r - 1] = I_e
            ss_data[QubitIndex]['Q_e'][j - batch_num * save_r - 1] = Q_e
            ss_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            ss_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            ss_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
            ss_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_ss
            ss_data[QubitIndex]['measurement_timestamp'][j - batch_num * save_r - 1] = meas_timestamp_ssge

            saver_ss = Data_H5(subStudyDataFolder, ss_data, batch_num, save_r)
            saver_ss.save_to_h5('ss_ge_phase_fixed')
            del saver_ss
            del ss_data
            del ss

            ###################### g-e Single Shot Measurement, active reset ########################
            ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
            ss = SingleShot(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs, experiment=experiment,
                            verbose=verbose, logger=rr_logger, unmasking_resgain=unmask)
            fid, angle, thresh, iq_list_g, iq_list_e, sys_config_ss, g_center, e_center = ss.run(active_reset=True)
            I_g = iq_list_g[QubitIndex][0].T[0]
            Q_g = iq_list_g[QubitIndex][0].T[1]
            I_e = iq_list_e[QubitIndex][0].T[0]
            Q_e = iq_list_e[QubitIndex][0].T[1]

            experiment.readout_cfg['threshold'] = thresh
            experiment.readout_cfg['g_center'] = g_center[0]  # 0 is I, 1 is Q
            experiment.readout_cfg['e_center'] = e_center[0]

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
            ss_data[QubitIndex]['measurement_timestamp'][j - batch_num * save_r - 1] = meas_timestamp_ssge

            saver_ss = Data_H5(subStudyDataFolder, ss_data, batch_num, save_r)
            saver_ss.save_to_h5('ss_ge_active_reset')
            del saver_ss
            del ss_data
            del ss

        except Exception as e:
            if debug_mode:
                raise e  # In debug mode, re-raise the exception immediately
            else:
                rr_logger.exception(f'Got the following error in ge ssf, continuing: {e}')
                if verbose: print(f'Got the following error in ge ssf, continuing: {e}')
                continue  # skip the rest of this qubit

        ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)

    ##################### active reset Rabi with 0 correction to compare to ########################
    if run_flags["act_reset_0corr"]:
        experiment.readout_cfg['n_resets'] = 0
        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)

        increase_qubit_reps_gerabi = False  # if you want to increase the reps for a qubit, set to True
        qubit_to_increase_gerabi_reps_for = None  # only has impact if previous line is True
        multiply_gerabi_reps_by = 1
        reduce_rlx_delay_gerabi = False
        reduce_rlx_delay_gerabi_to = None

        rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                       save_figs=save_figs, save_shots=save_shots_gerabi,
                                       experiment=experiment, live_plot=False,
                                       increase_qubit_reps=increase_qubit_reps_gerabi,
                                       qubit_to_increase_reps_for=qubit_to_increase_gerabi_reps_for,
                                       multiply_qubit_reps_by=multiply_gerabi_reps_by,
                                       verbose=verbose, logger=rr_logger, unmasking_resgain=unmask,
                                       reduce_rlx_delay=reduce_rlx_delay_gerabi,
                                       reduce_rlx_delay_to=reduce_rlx_delay_gerabi_to)

        (rabi_I_corrected0, rabi_Q_corrected0, rabi_gains_corrected0, rabi_fit_corrected0, pi_amp_corrected0,
         sys_config_rabi_corrected0, ss_Q_e20, ss_Q_g20, ss_I_e20, ss_I_g20, I_shots_rabi_corr0,
         Q_shots_rabi_corr0) = rabi.run_active_reset(scaling=True, control_test=True)

        rabi_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (time.mktime(datetime.datetime.now().timetuple()))
        rabi_data[QubitIndex]['I'][j - batch_num * save_r - 1] = rabi_I_corrected0
        rabi_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = rabi_Q_corrected0
        rabi_data[QubitIndex]['Gains'][j - batch_num * save_r - 1] = rabi_gains_corrected0
        rabi_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = rabi_fit_corrected0
        rabi_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
        rabi_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
        rabi_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
        rabi_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rabi_corrected0
        rabi_data[QubitIndex]['ss_Q_e'][j - batch_num * save_r - 1] = ss_Q_e20
        rabi_data[QubitIndex]['ss_Q_g'][j - batch_num * save_r - 1] = ss_Q_g20
        rabi_data[QubitIndex]['ss_I_e'][j - batch_num * save_r - 1] = ss_I_e20
        rabi_data[QubitIndex]['ss_I_g'][j - batch_num * save_r - 1] = ss_I_g20
        rabi_data[QubitIndex]['I_shots'][j - batch_num * save_r - 1] = I_shots_rabi_corr0
        rabi_data[QubitIndex]['Q_shots'][j - batch_num * save_r - 1] = Q_shots_rabi_corr0

        saver_rabi = Data_H5(optimizationFolder, rabi_data, batch_num, save_r)
        saver_rabi.save_to_h5('ge_rabi_corrected_0')
        del saver_rabi
        del rabi_data
        del rabi
        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)

    ###################### active reset Rabi with 1 correction ########################
    if run_flags["act_reset_0corr"]:
        experiment.readout_cfg['n_resets'] = 1
        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
        rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                       save_figs=save_figs, save_shots=False,
                                       experiment=experiment, live_plot=False,
                                       increase_qubit_reps=increase_qubit_reps,
                                       qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                       multiply_qubit_reps_by=multiply_qubit_reps_by,
                                       verbose=verbose, logger=rr_logger, unmasking_resgain=unmask,)

        (rabi_I_corrected1, rabi_Q_corrected1, rabi_gains_corrected1, rabi_fit_corrected1, pi_amp_corrected1,
         sys_config_rabi_corrected1, ss_Q_e21, ss_Q_g21, ss_I_e21, ss_I_g21, I_shots_rabi_corr1, Q_shots_rabi_corr1) = rabi.run_active_reset(scaling=True)

        rabi_data[QubitIndex]['Dates'][j - batch_num * save_r - 1]  = (
            time.mktime(datetime.datetime.now().timetuple()))
        rabi_data[QubitIndex]['I'][j - batch_num * save_r - 1]  = rabi_I_corrected1
        rabi_data[QubitIndex]['Q'][j - batch_num * save_r - 1]  = rabi_Q_corrected1
        rabi_data[QubitIndex]['Gains'][j - batch_num * save_r - 1]  = rabi_gains_corrected1
        rabi_data[QubitIndex]['Fit'][j - batch_num * save_r - 1]  = rabi_fit_corrected1
        rabi_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1]  = j
        rabi_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1]  = batch_num
        rabi_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1]  = expt_cfg
        rabi_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1]  = sys_config_rabi_corrected1
        rabi_data[QubitIndex]['ss_Q_e'][j - batch_num * save_r - 1]  = ss_Q_e21
        rabi_data[QubitIndex]['ss_Q_g'][j - batch_num * save_r - 1]  = ss_Q_g21
        rabi_data[QubitIndex]['ss_I_e'][j - batch_num * save_r - 1]  = ss_I_e21
        rabi_data[QubitIndex]['ss_I_g'][j - batch_num * save_r - 1]  = ss_I_g21
        rabi_data[QubitIndex]['I_shots'][j - batch_num * save_r - 1]  = I_shots_rabi_corr1
        rabi_data[QubitIndex]['Q_shots'][j - batch_num * save_r - 1]  = Q_shots_rabi_corr1

        saver_rabi = Data_H5(optimizationFolder, rabi_data, batch_num, save_r)
        saver_rabi.save_to_h5('ge_rabi_corrected_1')
        del saver_rabi
        del rabi_data
        del rabi
        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)

    # ##################### active reset Rabi with 2 corrections ########################
    # experiment.readout_cfg['n_resets'] = 2
    # rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    # rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
    #                                save_figs=save_figs, save_shots=False,
    #                                experiment=experiment, live_plot=False,
    #                                increase_qubit_reps=increase_qubit_reps,
    #                                qubit_to_increase_reps_for=qubit_to_increase_reps_for,
    #                                multiply_qubit_reps_by=multiply_qubit_reps_by,
    #                                verbose=verbose, logger=rr_logger, unmasking_resgain=unmask, )
    #
    # (rabi_I_corrected2, rabi_Q_corrected2, rabi_gains_corrected2, rabi_fit_corrected2, pi_amp_corrected2,
    #  sys_config_rabi_corrected2, ss_Q_e22, ss_Q_g22, ss_I_e22, ss_I_g22, I_shots_rabi_corr2,
    #  Q_shots_rabi_corr2) = rabi.run_active_reset(
    #     scaling=True)
    #
    # rabi_data[QubitIndex]['Dates'][0] = (
    #     time.mktime(datetime.datetime.now().timetuple()))
    # rabi_data[QubitIndex]['I'][0] = rabi_I_corrected2
    # rabi_data[QubitIndex]['Q'][0] = rabi_Q_corrected2
    # rabi_data[QubitIndex]['Gains'][0] = rabi_gains_corrected2
    # rabi_data[QubitIndex]['Fit'][0] = rabi_fit_corrected2
    # rabi_data[QubitIndex]['Round Num'][0] = 0
    # rabi_data[QubitIndex]['Batch Num'][0] = 0
    # rabi_data[QubitIndex]['Exp Config'][0] = expt_cfg
    # rabi_data[QubitIndex]['Syst Config'][0] = sys_config_rabi_corrected2
    # rabi_data[QubitIndex]['ss_Q_e'][0] = ss_Q_e22
    # rabi_data[QubitIndex]['ss_Q_g'][0] = ss_Q_g22
    # rabi_data[QubitIndex]['ss_I_e'][0] = ss_I_e22
    # rabi_data[QubitIndex]['ss_I_g'][0] = ss_I_g22
    # rabi_data[QubitIndex]['I_shots'][0] = I_shots_rabi_corr2
    # rabi_data[QubitIndex]['Q_shots'][0] = Q_shots_rabi_corr2
    #
    # saver_rabi = Data_H5(optimizationFolder, rabi_data, 0, save_r)
    # saver_rabi.save_to_h5('Rabi_corrected_2')
    # del saver_rabi
    # del rabi_data
    # del rabi

    ##################### active reset Rabi with 5 corrections ########################
    # experiment.readout_cfg['n_resets'] = 5
    # rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    # rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
    #                                save_figs=save_figs, save_shots=False,
    #                                experiment=experiment, live_plot=False,
    #                                increase_qubit_reps=increase_qubit_reps,
    #                                qubit_to_increase_reps_for=qubit_to_increase_reps_for,
    #                                multiply_qubit_reps_by=multiply_qubit_reps_by,
    #                                verbose=verbose, logger=rr_logger, unmasking_resgain=unmask, )
    #
    # (rabi_I_corrected5, rabi_Q_corrected5, rabi_gains_corrected5, rabi_fit_corrected5, pi_amp_corrected5,
    #  sys_config_rabi_corrected5, ss_Q_e25, ss_Q_g25, ss_I_e25, ss_I_g25, I_shots_rabi_corr5,
    #  Q_shots_rabi_corr5) = rabi.run_active_reset(
    #     scaling=True)
    #
    # rabi_data[QubitIndex]['Dates'][0] = (
    #     time.mktime(datetime.datetime.now().timetuple()))
    # rabi_data[QubitIndex]['I'][0] = rabi_I_corrected5
    # rabi_data[QubitIndex]['Q'][0] = rabi_Q_corrected5
    # rabi_data[QubitIndex]['Gains'][0] = rabi_gains_corrected5
    # rabi_data[QubitIndex]['Fit'][0] = rabi_fit_corrected5
    # rabi_data[QubitIndex]['Round Num'][0] = 0
    # rabi_data[QubitIndex]['Batch Num'][0] = 0
    # rabi_data[QubitIndex]['Exp Config'][0] = expt_cfg
    # rabi_data[QubitIndex]['Syst Config'][0] = sys_config_rabi_corrected5
    # rabi_data[QubitIndex]['ss_Q_e'][0] = ss_Q_e25
    # rabi_data[QubitIndex]['ss_Q_g'][0] = ss_Q_g25
    # rabi_data[QubitIndex]['ss_I_e'][0] = ss_I_e25
    # rabi_data[QubitIndex]['ss_I_g'][0] = ss_I_g25
    # rabi_data[QubitIndex]['I_shots'][0] = I_shots_rabi_corr5
    # rabi_data[QubitIndex]['Q_shots'][0] = Q_shots_rabi_corr5
    #
    # saver_rabi = Data_H5(optimizationFolder, rabi_data, 0, save_r)
    # saver_rabi.save_to_h5('Rabi_corrected_5')
    # del saver_rabi
    # del rabi_data
    # del rabi
    ##################### Comparison plot: all Rabi curves ########################
    # fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    # plt.rcParams.update({'font.size': 14})
    #
    # rabi_runs = [
    #     (0, rabi_I_corrected0, rabi_Q_corrected0, rabi_gains_corrected0, ss_I_e20, ss_I_g20, ss_Q_e20, ss_Q_g20, 'No reset'),
    #     # (1, rabi_I_corrected1, rabi_Q_corrected1, rabi_gains_corrected1, ss_I_e21, ss_I_g21, ss_Q_e21, ss_Q_g21, '1 active reset'),
    #     # (2, rabi_I_corrected2, rabi_Q_corrected2, rabi_gains_corrected2, ss_I_e22, ss_I_g22, ss_Q_e22, ss_Q_g22, '2 active resets'),
    #     (3, rabi_I_corrected5, rabi_Q_corrected5, rabi_gains_corrected5, ss_I_e25, ss_I_g25, ss_Q_e25, ss_Q_g25, '5 active resets'),
    #
    # ]
    #
    # colors = cm.viridis(np.linspace(0, 0.9, len(rabi_runs)))
    #
    # for idx, (n, I, Q, gains, Ie, Ig, Qe, Qg, label) in enumerate(rabi_runs):
    #     e = np.mean(np.array(Ie) + 1j * np.array(Qe))
    #     g = np.mean(np.array(Ig) + 1j * np.array(Qg))
    #     pop = np.real(((np.array(I) + 1j * np.array(Q)) - g) * np.conj(e - g) / np.abs(e - g) ** 2)
    #
    #     ax.plot(gains, pop, 'o-', color=colors[idx], linewidth=1.5, markersize=3, alpha=0.8, label=label)
    #
    # ax.set_xlabel('Gain (a.u.)', fontsize=16)
    # ax.set_ylabel('Qubit Population', fontsize=16)
    # ax.set_title(f'Amplitude Rabi Q{QubitIndex + 1} Active Reset Comparison', fontsize=18)
    # ax.legend(fontsize=12, loc='best')
    # ax.tick_params(axis='both', which='major', labelsize=13)
    # plt.tight_layout()
    #
    # now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # comparison_file = os.path.join(studyDocumentationFolder,
    #                                 f"Q_{QubitIndex + 1}_active_reset_rabi_comparison_{now}.png")
    # fig.savefig(comparison_file, dpi=150, bbox_inches='tight')
    # if verbose:
    #     print(f"Saved active reset comparison plot to {comparison_file}")
    # plt.close(fig)