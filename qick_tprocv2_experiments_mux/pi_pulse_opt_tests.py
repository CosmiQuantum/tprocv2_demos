import copy
import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15))  # need this so it saves absolutely everything returned from the classes
import datetime
import time
import logging
import visdom
import gc, copy
import time
#sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_001_time_of_flight import TOFExperiment
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_002_res_spec_ef import ResonanceSpectroscopyEF
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_004_qubit_spec_ef import EFQubitSpectroscopy
from section_006_amp_rabi_ef import EF_AmplitudeRabiExperiment
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_011_qubit_temperatures_efRabipt3 import Temps_EFAmpRabiExperiment  # must be pt3 version, do not change
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
# from section_005_single_shot_ef import SingleShot_ef # Kester way
from section_005_single_shot_gef import SingleShot_ef # Arianna way: Fix for example Unmask
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from analysis_021_plot_allRR_noqick import PlotRR_noQick
from analysis_014_temp_calcsandplots_cosmiqgpvm import SSFTempCalcAndPlots

################################################ Run Configurations ####################################################
st = time.time()

n = 1 # number of rounds
use_iminuit_instead = True # for fitting, curve fit when False, iminuit when True
pre_optimize = False # ignore
freq_offset_steps = 10 # ignore
ssf_avgs_per_opt_pt = 5 # ignore
ef_res_sample_number = 1 # keep this as one
save_r = 1  # how many rounds to save after. KEEP THIS AS ONE!!!!!!!!
qubit_freqs_ef = [None] * 6 # don't change this!!! These get updated later on in the code.
number_of_qubits = 6 # total
figure_quality = 200

signal = 'None'  # 'I', or 'Q' depending on where the signal is (after optimization). Keep as None
save_figs = True  # save plots for everything as you go along the RR script?
live_plot = False  # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True  # fit the data here and save or plot the fits?
save_data_h5 = True  # save all of the data to h5 files?

verbose = True  # print everything to the console in real time, good for debugging, bad for memory
qick_verbose = True  # qick verbose prints the progress bar for each qick experiment as it is happening (the red bar that fills out as more experiment rounds/reps are being done)
debug_mode = False  # if True, it disables the continuing function of RR if an error pops up in a class -- errors now stop the RR script

thresholding = False  # use internal QICK threshold for ratio of Binary values on y for rabi/t1/t2r/t2e, or analog avg when false
unmask = True  # Do you want to use the unmasking feature to increase resonator gain? This may not apply to LOUD
save_shots_gerabi = False  # save IQ shots instead of averaged IQ data? for ge rabi
save_shots_efrabi = False  # NOT implemented yet in this experiment. If you want to use this add code block to ef rabi experiment.

Qs_to_look_at = [0] # only list the qubits you want to do the RR for

# Data saving info
run_name = 'run9c'
device_name = '6transmon'
substudy_txt_notes = ('Reverted back to warm filtering setup a the beginning of the run.\n')

# set which of the following you'd like to run to 'True'
# run_flags = {"tof": False, "res_spec": True, "q_spec": True, "rabi": True, "ss": True, "ss_gef": False,
#              "t1": True, "t2r": True, "t2e": True, "ef_res_spec": True, "ef_q_spec": True,
#              "rabi_pop_meas": True, "ef_Rabi": False}
run_flags = {"tof": False, "res_spec": True, "q_spec": True, "rabi": True, "ss": True, "ss_gef": False,
             "t1": False, "t2r": False, "t2e": False, "ef_res_spec": False, "ef_q_spec": False,
             "rabi_pop_meas": False, "ef_Rabi": False}

# For 25dB DAC, 6/14/2026
res_leng_vals = [5.75, 7.2500, 6.25, 7.25, 7.0, 6.75]  # 5.63, 25dB [5.5, 6.0, 5.7, 6.8, 7.0, 8.0] , [5.6, 6.0, 5.7, 6.85, 5.0, 8.5]
res_gain = [0.76, 0.7788, 0.8419,0.5894, 0.825, 0.8375]  # 0.8125,25dB, [0.8125, 0.836, 0.915, 0.6218, 0.95, 0.97], [0.825, 0.835, 0.915, 0.634, 0.95, 0.97]
freq_offsets = [-0.1556,0.0667,-0.0222,-0.0222,0,-0.1111]  # -0.1556,0.0222,-0.2000,-0.2000,0.0, -0.1200

#DO NOT CHANGE THESE: They are flags to keep track of what happened in RR along the way
ef_res_any = False # did ef res spec run succesfully for any of the qubits?
ef_qspec_any = False # what about ef qspec?
rpm_any = False # and rabi population measurements?

# To save how long each measurement took for each qubit
meas_time_RR = {}

################################################ Data Saving Setup ##################################################
# Folders
study = 'DAC_filtering_tests_reverted_setup' #qubit_checkouts, DAC_filtering_tests
sub_study = 'original_filtering_prog' #Day4_base_not_fully_opt_yet_25dBDAC
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

####################################################### RR #############################################################
def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config', 'measurement_timestamp']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num', 'Recycled QFreq',
              'Exp Config', 'Syst Config', 'measurement_timestamp']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config', 'measurement_timestamp']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config', 'measurement_timestamp']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Ishots', 'Qshots', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config', 'measurement_timestamp']
t2r_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config', 'measurement_timestamp']
t2e_keys = ['T2E', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config', 'measurement_timestamp']
rabi_keys_ef_Qtemps = ['Dates', 'Qfreq_ge', 'I1', 'Q1', 'Gains1', 'Ishots1', 'Qshots1', 'Fit1', 'I2', 'Q2', 'Gains2', 'Ishots2', 'Qshots2', 'Fit2', 'Round Num',
                       'Batch Num', 'Exp Config', 'Syst Config', 'measurement_timestamp']

# initialize a simple list to store the qspec values in incase a fit fails
stored_qspec_list = [None] * tot_num_of_qubits

if live_plot:
    # Check if visdom is connected right away, otherwise, throw an error
    if not (viz := visdom.Visdom()).check_connection(timeout_seconds=5):
        raise RuntimeError("Visdom server not connected!, Type \'visdom\' into the command line and go to "
                           "http://localhost:8097/ on firefox")

# initialize a dictionary to store the values
res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)

ef_res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
ef_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
ef_rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
rabi_data_ef_Qtemps = create_data_dict(rabi_keys_ef_Qtemps, save_r, list_of_all_qubits)

batch_num = 0 # keep as zero
j = 0 # keep as zero
angles = []
while j < n:
    j += 1
    for QubitIndex in Qs_to_look_at:
        meas_time_RR[QubitIndex] = {}

        recycled_qfreq = False # don't change

        # keep these as False
        ef_res_spec_survived = False
        ef_qspec_survived = False

        # Get the config for this qubit
        DAC_attenuator1 = 15
        DAC_attenuator2 = 10
        experiment = QICK_experiment(optimizationFolder, DAC_attenuator1=DAC_attenuator1, DAC_attenuator2=DAC_attenuator2,
                                     qubit_DAC_attenuator1=5,
                                     qubit_DAC_attenuator2=4, ADC_attenuator=17,
                                     fridge=FRIDGE)  # ADC_attenuator MUST be above 16dB
        experiment.create_folder_if_not_exists(optimizationFolder)
        print("DAC atten: ", DAC_attenuator1 + DAC_attenuator2)

        # Mask out all other resonators except this one
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_gain_ef'] = res_gains
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        ###################################################### TOF #####################################################
        if run_flags["tof"]:
            tof = TOFExperiment(QubitIndex, studyDocumentationFolder, experiment, j, save_figs,
                                unmasking_resgain=unmask)
            tof.run()
            del tof

        ################################################# g-e Res spec ####################################################
        if run_flags["res_spec"]:
            t0 = time.perf_counter()
            try:
                increase_geres_reps = False
                increase_geres_reps_to = None
                use_savgol_smoothing_rspec = False

                if QubitIndex == 5:
                    increase_geres_reps = True
                    increase_geres_reps_to = 400 #400
                # if QubitIndex == 3:
                #     increase_geres_reps = True
                #     increase_geres_reps_to = 600
                # if QubitIndex == 4:
                #     use_savgol_smoothing_rspec = True
                #     increase_geres_reps = True
                #     increase_geres_reps_to = 1800

                res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs, increase_geres_reps,
                                                 increase_geres_reps_to, experiment=experiment, verbose=verbose, logger=rr_logger, unmasking_resgain=unmask,
                                                 use_savgol_smoothing = use_savgol_smoothing_rspec)
                res_freqs, freq_pts, freq_center, amps, sys_config_rspec, meas_timestamp_resge = res_spec.run()
                offset = freq_offsets[QubitIndex]  # use optimized offset values or whats set at top of script based on pre_optimize flag
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

        ################################################## g-e Qubit spec ##################################################
        if run_flags["q_spec"]:
            t0 = time.perf_counter()
            try:
                increase_qubit_reps_qspec = False
                increase_qspec_rounds = False
                qspecge_increase_reps_to = None
                increase_qspec_rounds_to = None

                # if QubitIndex == 5:
                    # increase_qubit_reps_qspec = True
                    # qspecge_increase_reps_to = 650
                    # increase_qspec_rounds = True
                    # increase_qspec_rounds_to = 2

                if QubitIndex == 4:
                    increase_qubit_reps_qspec = True
                    qspecge_increase_reps_to = 1400
                    increase_qspec_rounds = True
                    increase_qspec_rounds_to = 2

                # if QubitIndex == 3:
                #     increase_qubit_reps_qspec = True
                #     qspecge_increase_reps_to = 600
                #     # increase_qspec_rounds = True
                #     # increase_qspec_rounds_to = 3
                #
                # if QubitIndex == 2:
                #     increase_qubit_reps_qspec = True
                #     qspecge_increase_reps_to = 700

                if QubitIndex == 1:
                    increase_qubit_reps_qspec = True
                    qspecge_increase_reps_to = 650

                if QubitIndex == 0:
                    increase_qubit_reps_qspec = True
                    qspecge_increase_reps_to = 650

                q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j,
                                           signal, save_figs, increase_reps = increase_qubit_reps_qspec, increase_rounds =increase_qspec_rounds,
                                           increase_reps_to = qspecge_increase_reps_to, increase_rounds_to = increase_qspec_rounds_to,
                                           plot_fit=True, experiment=experiment, live_plot=live_plot, verbose=verbose,
                                           logger=rr_logger, unmasking_resgain=unmask)
                (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq, sys_config_qspec, meas_timestamp_qspecge) = q_spec.run()

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

        ###################################################### g-e Rabi ####################################################
        if run_flags["rabi"]:
            t0 = time.perf_counter()
            try:
                increase_qubit_reps_gerabi = False  # if you want to increase the reps for a qubit, set to True
                qubit_to_increase_gerabi_reps_for = None  # only has impact if previous line is True
                multiply_gerabi_reps_by = 1
                reduce_rlx_delay_gerabi = False
                reduce_rlx_delay_gerabi_to = None

                # if QubitIndex == 3:
                #     increase_qubit_reps_gerabi = True
                #     qubit_to_increase_gerabi_reps_for = QubitIndex

                if QubitIndex == 4:
                    increase_qubit_reps_gerabi = True
                    qubit_to_increase_gerabi_reps_for = QubitIndex
                    multiply_gerabi_reps_by = 2

                if QubitIndex == 5:
                #     increase_qubit_reps_gerabi = True
                #     qubit_to_increase_gerabi_reps_for = QubitIndex
                    reduce_rlx_delay_gerabi = True
                    reduce_rlx_delay_gerabi_to = 700

                rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                               save_figs=save_figs, save_shots=save_shots_gerabi,
                                               experiment=experiment, live_plot=live_plot,
                                               increase_qubit_reps=increase_qubit_reps_gerabi,
                                               qubit_to_increase_reps_for=qubit_to_increase_gerabi_reps_for,
                                               multiply_qubit_reps_by=multiply_gerabi_reps_by,
                                               verbose=verbose, logger=rr_logger, unmasking_resgain=unmask,
                                               reduce_rlx_delay = reduce_rlx_delay_gerabi, reduce_rlx_delay_to = reduce_rlx_delay_gerabi_to)
                (rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_rabi, meas_timestamp_rabige) = rabi.run(thresholding=thresholding, use_iminuit_instead = use_iminuit_instead)

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

        ########################################## g-e Single Shot Measurements ############################################
        if run_flags["ss"]:
            t0 = time.perf_counter()
            try:
                reduce_rlx_delay_ssf = False
                reduce_rlx_delay_ssf_to = None
                if QubitIndex == 5:
                    reduce_rlx_delay_ssf = True
                    reduce_rlx_delay_ssf_to = 650

                # ------------------------------------------------------------
                # Try a few pi-amp offsets after the Rabi fit.
                # These are absolute gain offsets added to the fitted pi_amp.
                # Keep this list short so SSF does not take forever.
                # ------------------------------------------------------------
                base_pi_amp = float(pi_amp)
                pi_amp_offsets = [-0.016, -0.012, -0.008, -0.004]

                # Optional: include exactly the Rabi-fit value too.
                # This makes 5 SSFs total instead of 4.
                # pi_amp_offsets = [0.0, -0.02, -0.01, 0.01, 0.02]

                best_ssf = {
                    "fid": -np.inf,
                    "angle": None,
                    "iq_list_g": None,
                    "iq_list_e": None,
                    "sys_config_ss": None,
                    "meas_timestamp_ssge": None,
                    "pi_amp_used": None,
                    "pi_amp_offset": None,
                }

                ssf_thresholds = [0.85, 0.80, 0.80, 0.8, 0.20, 0.75]
                ssf_threshold = ssf_thresholds[QubitIndex]

                for pi_offset in pi_amp_offsets:
                    pi_amp_test = base_pi_amp + pi_offset
                    experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp_test)

                    rr_logger.info(
                        f"Q{QubitIndex + 1} SSF testing pi_amp={pi_amp_test:.6f} "
                        f"(Rabi pi_amp={base_pi_amp:.6f}, offset={pi_offset:+.6f})"
                    )
                    if verbose:
                        print(
                            f"Q{QubitIndex + 1} SSF testing pi_amp={pi_amp_test:.6f} "
                            f"(offset={pi_offset:+.6f})"
                        )

                    ss = SingleShot(
                        QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs,
                        experiment=experiment, verbose=verbose, logger=rr_logger,
                        unmasking_resgain=unmask,
                        reduce_rlx_delay=reduce_rlx_delay_ssf,
                        reduce_rlx_delay_to=reduce_rlx_delay_ssf_to
                    )

                    fid, angle, iq_list_g, iq_list_e, sys_config_ss, meas_timestamp_ssge = ss.run()
                    del ss

                    rr_logger.info(
                        f"Q{QubitIndex + 1} SSF result: fid={fid:.4f}, "
                        f"pi_amp={pi_amp_test:.6f}, offset={pi_offset:+.6f}"
                    )
                    if verbose:
                        print(
                            f"Q{QubitIndex + 1} SSF result: fid={fid:.4f}, "
                            f"pi_amp={pi_amp_test:.6f}, offset={pi_offset:+.6f}"
                        )

                    if fid > best_ssf["fid"]:
                        best_ssf.update({
                            "fid": fid,
                            "angle": angle,
                            "iq_list_g": iq_list_g,
                            "iq_list_e": iq_list_e,
                            "sys_config_ss": sys_config_ss,
                            "meas_timestamp_ssge": meas_timestamp_ssge,
                            "pi_amp_used": pi_amp_test,
                            "pi_amp_offset": pi_offset,
                        })

                    # Optional early stop if it is already good enough.
                    # Comment this out if you always want all 4 offsets.
                    # if fid >= ssf_threshold:
                    #     rr_logger.info(
                    #         f"Q{QubitIndex + 1} SSF reached threshold {ssf_threshold:.3f}; "
                    #         f"stopping pi_amp offset scan early."
                    #     )
                    #     break

                # if best_ssf["fid"] < ssf_threshold:
                #     rr_logger.warning(
                #         f"Q{QubitIndex + 1} SSF never reached {ssf_threshold}. "
                #         f"Keeping best attempt: fid={best_ssf['fid']:.4f}, "
                #         f"pi_amp={best_ssf['pi_amp_used']:.6f}, "
                #         f"offset={best_ssf['pi_amp_offset']:+.6f}"
                #     )

                # Restore the config to the best pi_amp found by SSF.
                experiment.qubit_cfg['pi_amp'][QubitIndex] = float(best_ssf["pi_amp_used"])

                # Use the best SSF result downstream/save to h5.
                fid = best_ssf["fid"]
                angle = best_ssf["angle"]
                iq_list_g = best_ssf["iq_list_g"]
                iq_list_e = best_ssf["iq_list_e"]
                sys_config_ss = best_ssf["sys_config_ss"]
                meas_timestamp_ssge = best_ssf["meas_timestamp_ssge"]

                I_g = iq_list_g[QubitIndex][0].T[0]
                Q_g = iq_list_g[QubitIndex][0].T[1]
                I_e = iq_list_e[QubitIndex][0].T[0]
                Q_e = iq_list_e[QubitIndex][0].T[1]

                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"Q{QubitIndex + 1} best SSF pi_amp scan: "
                        f"Rabi pi_amp={base_pi_amp:.6f}, "
                        f"best pi_amp={best_ssf['pi_amp_used']:.6f}, "
                        f"offset={best_ssf['pi_amp_offset']:+.6f}, "
                        f"fid={best_ssf['fid']:.6f}\n"
                    )

            except Exception as e:
                if debug_mode:
                    raise e
                else:
                    rr_logger.exception(f'Got the following error in ge ssf, continuing: {e}')
                    if verbose:
                        print(f'Got the following error in ge ssf, continuing: {e}')
                    continue