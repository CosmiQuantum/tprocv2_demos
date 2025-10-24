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

sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
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
from analysis_020_gef_ssf_fstate_plots import GEF_SSF_ANALYSIS
from analysis_014_temp_calcsandplots_cosmiqgpvm import SSFTempCalcAndPlots
################################################ Run Configurations ####################################################
st = time.time()

n = 10000
pre_optimize = False
freq_offset_steps = 10
ssf_avgs_per_opt_pt = 5
save_r = 1  # how many rounds to save after
signal = 'None'  # 'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization
save_figs = True  # save plots for everything as you go along the RR script?
live_plot = False  # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True  # fit the data here and save or plot the fits?
save_data_h5 = True  # save all of the data to h5 files?
verbose = True  # print everything to the console in real time, good for debugging, bad for memory
qick_verbose = True  # qick verbose prints the progress bar for each qick experiment as it is happening (the red bar that fills out as more experiment rounds/reps are being done)
debug_mode = False  # if True, it disables the continuing function of RR if an error pops up in a class -- errors now stop the RR script
thresholding = False  # use internal QICK threshold for ratio of Binary values on y for rabi/t1/t2r/t2e, or analog avg when false

increase_qubit_reps_gerabi = True  # if you want to increase the reps for a qubit, set to True
increase_qubit_reps_t1 = False  # if you want to increase the reps for a qubit, set to True
increase_qubit_reps_t2r = False  # if you want to increase the reps for a qubit, set to True
increase_qubit_reps_t2e = False  # if you want to increase the reps for a qubit, set to True
increase_qubit_reps_efrabi = False  # if you want to increase the reps for a qubit, set to True
increase_qubit_reps_rpm = False  # if you want to increase the reps for a qubit, set to True

qubit_to_increase_reps_for = 3  # only has impact if previous line is True
multiply_qubit_reps_by = 2  # only has impact if the line above is True. MUST be an integer.

unmask = True  # Do you want to use the unmasking feature to increase resonator gain?
save_shots_gerabi = False  # save IQ shots instead of averaged IQ data? for ge rabi ?
save_shots_efrabi = False  # NOT implemented yet in this experiment. If you want to use this add code block to ef rabi experiment.
save_shots_fhrabi = False  # save IQ shots instead of averaged IQ data? for fh rabi

Qs_to_look_at = [0,1,2,3,4]  # only list the qubits you want to do the RR for

# Data saving info
run_name = 'run8'
device_name = '6transmon'
substudy_txt_notes = ('This data is post adding new channel on the qick box.\n') # Initial qubit checkouts quiet run 8

# set which of the following you'd like to run to 'True'

# run_flags = {"tof": False, "res_spec": True, "q_spec": True, "ss": True, "rabi": True, "ss_gef": False,
#              "t1": False, "t2r": False, "t2e": False, "ef_res_spec": False, "ef_q_spec": False,
#              "rabi_pop_meas": False, "ef_Rabi": False}

run_flags = {"tof": False, "res_spec": True, "q_spec": True, "ss": True, "rabi": True, "ss_gef": False,
             "t1": True, "t2r": True, "t2e": True, "ef_res_spec": True, "ef_q_spec": True,
             "rabi_pop_meas": True, "ef_Rabi": False}

#Updated 10/23, except for Q6 due to R6 double peak problem
res_leng_vals = [6.5, 9.0, 6.0, 8.5, 8.0, 8.0]
res_gain = [0.9, 0.9, 0.8, 0.5, 0.8, 0.92]
freq_offsets = [-0.3182, -0.1364, -0.5, 0.0, -0.4091, -0.0444]

qubit_freqs_ef = [None] * 6
increase_reps_to_ef = 5100
ef_res_sample_number = 1
number_of_qubits = 6
figure_quality = 200

ef_res_any = False
ef_qspec_any = False
rpm_any = False
################################################ Data Saving Setup ##################################################
# Folders
study = 'round_robin' #qubit_checkouts
sub_study = 'ABpaperdata3rdbatch_21dB_DACatten_Q1to5_t1shots_optional' #pre_AB_paper_data_still_optimizing, two_photon_peak_search, AB_Paper_Data_24hrs
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
            'Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num', 'Recycled QFreq',
              'Exp Config', 'Syst Config']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Ishots', 'Qshots', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t2r_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
t2e_keys = ['T2E', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
rabi_keys_ef_Qtemps = ['Dates', 'Qfreq_ge', 'I1', 'Q1', 'Gains1', 'Fit1', 'I2', 'Q2', 'Gains2', 'Fit2', 'Round Num',
                       'Batch Num', 'Exp Config', 'Syst Config']
ss_keys_gef = ['Fidelity', 'Angle_ef', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'I_f', 'Q_f', 'Round Num', 'Batch Num',
               'Exp Config', 'Syst Config']

# initialize a simple list to store the qspec values in incase a fit fails
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

ef_res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
ef_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
ef_rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
rabi_data_ef_Qtemps = create_data_dict(rabi_keys_ef_Qtemps, save_r, list_of_all_qubits)
ss_data_gef = create_data_dict(ss_keys_gef, save_r, list_of_all_qubits)

batch_num = 0
j = 0
angles = []
while j < n:
    j += 1
    for QubitIndex in Qs_to_look_at:
        recycled_qfreq = False
        ef_res_spec_survived = False
        ef_qspec_survived = False

        # Get the config for this qubit
        DAC_attenuator1 = 11
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
            try:
                res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs,
                                                 experiment=experiment, verbose=verbose, logger=rr_logger,
                                                 unmasking_resgain=unmask)
                res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
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

        # ################### Roll Signal into I (need to configure for recent updates) ################################
        # #get the average theta value, then use that to rotate the signal. Plug that value into system_config res_phase
        # leng=4
        # ss = SingleShotGE(QubitIndex, outerFolder, experiment, j, save_figs)
        # fid, angle, iq_list_g, iq_list_e = ss.run()
        # angles.append(angle)
        # #rr_logger.info(angles)
        # #rr_logger.info('avg theta: ', np.average(angles))
        # del ss

        ################################################## g-e Qubit spec ##################################################
        if run_flags["q_spec"]:
            try:
                increase_qubit_reps_qspec = False
                increase_qspec_rounds = False
                qspecge_increase_reps_to = None
                increase_qspec_rounds_to = None

                # if QubitIndex == 3: # Qubit 4 and 5
                #     increase_qubit_reps_qspec = True
                #     qspecge_increase_reps_to = 900
                if QubitIndex == 4 or QubitIndex == 3:
                    increase_qubit_reps_qspec = True
                    qspecge_increase_reps_to = 450
                    increase_qspec_rounds = True
                    increase_qspec_rounds_to = 3

                q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j,
                                           signal, save_figs, increase_reps = increase_qubit_reps_qspec, increase_rounds =increase_qspec_rounds,
                                           increase_reps_to = qspecge_increase_reps_to, increase_rounds_to = increase_qspec_rounds_to,
                                           plot_fit=True, experiment=experiment, live_plot=live_plot, verbose=verbose,
                                           logger=rr_logger, unmasking_resgain=unmask)
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
                else:
                    rr_logger.exception(f"RR g-e QSpec error on qubit {QubitIndex}: {e}")
                    if verbose:
                        print(f"g-e QSpec error on qubit {QubitIndex}: {e}")
                    continue

        ###################################################### g-e Rabi ####################################################
        if run_flags["rabi"]:
            try:
                rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                               save_figs=save_figs, save_shots=save_shots_gerabi,
                                               experiment=experiment, live_plot=live_plot,
                                               increase_qubit_reps=increase_qubit_reps_gerabi,
                                               qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                               multiply_qubit_reps_by=multiply_qubit_reps_by,
                                               verbose=verbose, logger=rr_logger, unmasking_resgain=unmask)
                (rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp,
                 sys_config_rabi) = rabi.run(thresholding=thresholding)

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

        ########################################## g-e Single Shot Measurements ############################################
        if run_flags["ss"]:
            try:
                ss = SingleShot(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs,
                                experiment=experiment,
                                verbose=verbose, logger=rr_logger, unmasking_resgain=unmask)
                fid, angle, iq_list_g, iq_list_e, sys_config_ss = ss.run()
                I_g = iq_list_g[QubitIndex][0].T[0]
                Q_g = iq_list_g[QubitIndex][0].T[1]
                I_e = iq_list_e[QubitIndex][0].T[0]
                Q_e = iq_list_e[QubitIndex][0].T[1]

            except Exception as e:
                if debug_mode:
                    raise e  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error in ge ssf, continuing: {e}')
                    if verbose: print(f'Got the following error in ge ssf, continuing: {e}')
                    continue  # skip the rest of this qubit

            #------------- get effective qubit temp via ssf method for the round that was just taken ---------------------
            try:
                ssf_qtemp = SSFTempCalcAndPlots(figure_quality, number_of_qubits, False)
                temp_mk = ssf_qtemp.get_ssf_qtemps_duringRR(QubitIndex, I_g, Q_g, I_e, Q_e, qubit_freq, None)
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"Q{QubitIndex + 1} Effective Temperature via SSF: {temp_mk} mK, using Qfreq: {qubit_freq} MHz\n")
            except Exception as e:
                rr_logger.exception(f"SSF Temp calc/log failed for Q{QubitIndex + 1}: {e}")
            # -------------------------------------------------------------------------------------------------------------

        ############################################## res spec ef ####################################################
        if run_flags["ef_res_spec"]:
            ef_res_freqs_samples = []
            for sample in range(ef_res_sample_number):
                try:
                    ef_res_spec = ResonanceSpectroscopyEF(QubitIndex, tot_num_of_qubits, studyDocumentationFolder,
                                                          sample,
                                                          save_figs, experiment=experiment, verbose=verbose,
                                                          logger=rr_logger, qick_verbose=qick_verbose,
                                                          unmasking_resgain=unmask)
                    ef_res_freqs, ef_freq_pts, ef_freq_center, ef_amps, sys_config_rspec_ef = ef_res_spec.run()
                    ef_res_freqs_samples.append(ef_res_freqs)
                    rr_logger.info(f"EF ResSpec sample {sample} for qubit {QubitIndex + 1}: {ef_res_freqs}")

                    del ef_res_spec

                except Exception as e:
                    if debug_mode:
                        raise e  # In debug mode, re-raise the exception immediately
                    rr_logger.exception(f"EF ResSpec error on qubit {QubitIndex + 1} sample {sample}: {e}")
                    # we don't skip the qubit if this throws an err because we want to save the rest of the data that was taken

            if ef_res_freqs_samples:
                # Average the ef resonator frequency values across samples
                avg_ef_res_freqs = np.mean(np.array(ef_res_freqs_samples), axis=0).tolist()
                ef_res_spec_survived = True

                experiment.readout_cfg['res_freq_ef'] = ef_res_freqs_samples[-1]  # use the last e-f res spec frequency to update the sys config

                rr_logger.info(f"Avg. EF resonator frequencies for qubit {QubitIndex + 1}: {avg_ef_res_freqs[QubitIndex]}")
                if verbose:
                    print(f"Avg. EF resonator frequencies for qubit {QubitIndex + 1}: {avg_ef_res_freqs}")

            else:
                rr_logger.error(f"No resonator spectroscopy data collected for qubit {QubitIndex + 1}.")

            ################################################ Qubit Spec EF ################################################
            if run_flags["ef_q_spec"]:
                if ef_res_spec_survived:
                    try:
                        # Qubit 4 usually needs more steps for e-f spec
                        increase_qubit_reps_ef = False
                        if QubitIndex == 3:
                            increase_qubit_reps_ef = True  # if you want to increase the steps for a qubit, set to True

                        ef_q_spec = EFQubitSpectroscopy(QubitIndex, number_of_qubits, studyDocumentationFolder, j, signal,
                                                        save_figs, experiment, live_plot, increase_reps = increase_qubit_reps_ef,
                                                        increase_reps_to = increase_reps_to_ef, unmasking_resgain=unmask)

                        efqspec_I, efqspec_Q, efqspec_freqs, sys_config_qspec_ef, efqspec_I_fit, efqspec_Q_fit, efqubit_freq = ef_q_spec.run()
                        qubit_freqs_ef[QubitIndex] = efqubit_freq
                        experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] = float(efqubit_freq)

                        ef_qspec_survived = True
                        rr_logger.info(f"EF Qubit {QubitIndex + 1} frequency: {efqubit_freq}")
                        if verbose:
                            print(f"EF Qubit {QubitIndex + 1} frequency: {efqubit_freq}")

                        del ef_q_spec

                    except Exception as e:
                        if debug_mode:
                            raise e  # In debug mode, re-raise the exception immediately
                        rr_logger.exception(f"EF qspec error on qubit {QubitIndex + 1}: {e}")
                        # we don't skip the qubit if this throws an err because we want to save the rest of the data that was taken

        ################################################ e-f rabi ################################################
        # NOT needed for rpm qubit temps, rpm itself is an ef rabi experiment
        if run_flags["ef_Rabi"]:
            if ef_qspec_survived and ef_res_spec_survived:
                try:
                    efrabi = EF_AmplitudeRabiExperiment(QubitIndex, number_of_qubits, studyDocumentationFolder, j,
                                                        signal, save_shots_efrabi, experiment=experiment,
                                                        live_plot=live_plot,
                                                        increase_qubit_reps=increase_qubit_reps_efrabi,
                                                        qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                                        multiply_qubit_reps_by=multiply_qubit_reps_by,
                                                        unmasking_resgain=unmask)
                    efrabi_I, efrabi_Q, efrabi_gains, efrabi_fit, efpi_amp, efsys_config_to_save = efrabi.run()
                    # if these are None, fit didnt work
                    if (efrabi_fit is None and efpi_amp is None):
                        print('EF Rabi fit didnt work for this qubit.')
                        continue  # skip the rest of this qubit

                    experiment.qubit_cfg['pi_ef_amp'][QubitIndex] = float(efpi_amp)
                    print('ef Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(efpi_amp))
                    del efrabi
                except Exception as e:
                    if debug_mode:
                        raise e
                    rr_logger.exception(f"ef rabi error on qubit {QubitIndex + 1}: {e}")
                    if verbose: print(f'Got the following error in ef rabi: {e}')
                    # we don't skip the qubit if this throws an err because we want to save the rest of the data that was taken

        ################################################ e-f amp rabi pop meas. ################################################
        if run_flags["rabi_pop_meas"]:
            if ef_qspec_survived and ef_res_spec_survived:
                t0 = time.perf_counter()
                try:
                    efAmprabi_Qtemps = Temps_EFAmpRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits,
                                                                 studyDocumentationFolder,
                                                                 j,
                                                                 signal, save_figs,
                                                                 experiment, live_plot,
                                                                 increase_qubit_reps_rpm, qubit_to_increase_reps_for,
                                                                 multiply_qubit_reps_by, unmasking_resgain=unmask)
                    (I1_qtemp, Q1_qtemp, gains1_qtemp, fit_cosine1_qtemp, pi_amp1_qtemp, A_amplitude1, amp_fit1,
                     I2_qtemp, Q2_qtemp, gains2_qtemp, fit_cosine2_qtemp, pi_amp2_qtemp, A_amplitude2, amp_fit2,
                     sysconfig_efrabi_Qtemps) = efAmprabi_Qtemps.run(experiment.soccfg, experiment.soc)

                    # Just to quickly output the qubit temperature live---------------------
                    # this ocasionally throws errs if something goes wrong with a fit
                    # date = data_set
                    # fit_saved = fit_data
                    # outerFolder_save_plots = ""
                    # unique_folder_path = ""
                    # outerFolder = ""
                    # qtempclass = PlotRR_noQick(date, figure_quality, save_figs, fit_saved, signal, run_name, number_of_qubits, outerFolder,
                    #  outerFolder_save_plots, unique_folder_path)
                    # T_K, T_mK, _, _ = qtempclass.Qubit_Temperature_Convert(A_amplitude1, A_amplitude2, qubit_freq)
                    # print(f"Q{QubitIndex + 1} temperature: {T_mK} mK")

                    try: #adding them to the notes text file instead
                        qtemp = PlotRR_noQick(data_set, figure_quality, False, False, signal, run_name, number_of_qubits,
                                              "", "", "")
                        T_K, T_mK, _, _ = qtemp.Qubit_Temperature_Convert(A_amplitude1, A_amplitude2, qubit_freq)
                        with open(file_path, "a", encoding="utf-8") as f:
                            f.write(
                                f"Q{QubitIndex + 1} Effective Temperature via RPM: {T_mK} mK using A1 = {float(A_amplitude1)}, A2 = {float(A_amplitude2)}, and Qfreq: {qubit_freq} MHz\n")
                    except Exception as e:
                        rr_logger.exception(f"RPM Temp calc/log failed for Q{QubitIndex + 1}: {e}")
                    # -------------------------------------------------------------------------

                    rr_logger.info(
                        f"RPM Amplitudes for qubit {QubitIndex + 1}: A1 = {float(A_amplitude1)}, A2 = {float(A_amplitude2)}")
                    if verbose:
                        print(
                            f"RPM Amplitudes for qubit {QubitIndex + 1}: A1 = {float(A_amplitude1)}, A2 = {float(A_amplitude2)}")

                    del efAmprabi_Qtemps

                except Exception as e:
                    if debug_mode:
                        raise e  # In debug mode, re-raise the exception immediately
                    rr_logger.exception(f"EF Rabi Population Measurements error on qubit {QubitIndex + 1}: {e}")
                    if verbose: print(f'Got the following error in ef rpm measurements: {e}')
                    # we don't skip the qubit if this throws an err because we want to save the rest of the data that was taken

                t1_time = time.perf_counter()
                print(f"RPM took {t1_time - t0:.4f} seconds")

        ###################################################### g-e T1 ######################################################
        if run_flags["t1"]:
            try:
                t1 = T1Measurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
                                   experiment=experiment,
                                   live_plot=live_plot, fit_data=fit_data,
                                   increase_qubit_reps=increase_qubit_reps_t1,
                                   qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                   multiply_qubit_reps_by=multiply_qubit_reps_by,
                                   verbose=verbose, logger=rr_logger, save_shots = False, unmasking_resgain=unmask)
                t1_est, t1_err, t1_I, t1_Q, t1_Ishots, t1_Qshots, t1_delay_times, q1_fit_exponential, sys_config_t1 = t1.run(
                    thresholding=thresholding)
                del t1

            except Exception as e:
                if debug_mode:
                    raise e  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error in ge T1, continuing: {e}')
                    if verbose: print(f'Got the following error in ge T1, continuing: {e}')
                    # we don't skip the qubit if this throws an err because we want to save the rest of the data that was taken

        ###################################################### g-e T2R #####################################################
        if run_flags["t2r"]:
            try:
                t2r = T2RMeasurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
                                     experiment=experiment, live_plot=live_plot, fit_data=fit_data,
                                     increase_qubit_reps=increase_qubit_reps_t2r,
                                     qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                     multiply_qubit_reps_by=multiply_qubit_reps_by,
                                     verbose=verbose, logger=rr_logger, unmasking_resgain=unmask)
                t2r_est, t2r_err, t2r_I, t2r_Q, t2r_delay_times, fit_ramsey, sys_config_t2r = t2r.run(
                    thresholding=thresholding)
                del t2r

            except Exception as e:
                if debug_mode:
                    raise e  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error in t2r, continuing: {e}')
                    if verbose: print(f'Got the following error in t2r, continuing: {e}')
                    # we don't skip the qubit if this throws an err because we want to save the rest of the data that was taken

        ##################################################### g-e T2E ######################################################
        if run_flags["t2e"]:
            try:
                t2e = T2EMeasurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
                                     experiment=experiment, live_plot=live_plot, fit_data=fit_data,
                                     increase_qubit_reps=increase_qubit_reps_t2e,
                                     qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                     multiply_qubit_reps_by=multiply_qubit_reps_by,
                                     verbose=verbose, logger=rr_logger, unmasking_resgain=unmask)
                (t2e_est, t2e_err, t2e_I, t2e_Q, t2e_delay_times,
                 fit_t2e, sys_config_t2e) = t2e.run(thresholding=thresholding)
                del t2e

            except Exception as e:
                if debug_mode:
                    raise e  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error in ge t2e: {e}')
                    if verbose: print(f'Got the following error in ge t2e: {e}')
                    # we don't skip the qubit if this throws an err because we want to save the rest of the data that was taken

        ########################################### g-e-f Single Shot Measurements ############################################
        if run_flags["ss_gef"]:
            ss = SingleShot_ef(QubitIndex, number_of_qubits, studyDocumentationFolder, j, save_figs, experiment)

            # iq_list_e, iq_list_f, ie_new, if_new,  theta_ef, threshold_ef, self.config
            iq_list_e, iq_list_f, ie_new, if_new, theta_ef, threshold_ef, sys_config_ss_gef = ss.run()
            # iq_list_g, iq_list_e, iq_list_f, ig_new, qg_new, ie_new, qe_new, if_new, qf_new, theta_ge, threshold_ge, sys_config_ss_gef
            I_g = iq_list_g[QubitIndex][0].T[0]
            Q_g = iq_list_g[QubitIndex][0].T[1]
            I_e = iq_list_e[QubitIndex][0].T[0]
            Q_e = iq_list_e[QubitIndex][0].T[1]
            I_f = iq_list_f[QubitIndex][0].T[0]
            Q_f = iq_list_f[QubitIndex][0].T[1]

            if run_flags["ss_gef"]:  # currently saves figs and h5 files every time this is run
                provided_sigma_num = None  # de state circle radius = sigma_num * sigma. Set as None if you want the code to choose an appropriate one for you.
                Analysis = False  # Keep as false, we are in RR mode here, not post-processing (analysis) mode
                RR = True  # Keep as true, we are in RR mode here
                date_analysis = None  # This only matters if you are in post-processing mode (for analysis purposes), keep as None here.
                round_num = j
                # analysis_gef_SSF = GEF_SSF_ANALYSIS(studyDocumentationFolder, QubitIndex, Analysis, RR,
                #                                     date_analysis, round_num)
                # (line_point1, line_point2, center_e, radius_e, T, v, f_outside, line_point1_rot, line_point2_rot,
                #  center_e_rot, radius_e_rot, T_rot, v_rot, f_outside_rot
                #  ) = analysis_gef_SSF.fstate_analysis_plot(I_g, Q_g, I_e, Q_e, I_f, Q_f, ig_new,  ie_new,
                #
                #                                            if_new,  theta_ef, threshold_ef, QubitIndex,
                #                                            provided_sigma_num)
                # # (line_point1, line_point2, center_e, radius_e, T, v, f_outside, line_point1_rot, line_point2_rot,
                #  center_e_rot, radius_e_rot, T_rot, v_rot, f_outside_rot
                #  ) = analysis_gef_SSF.fstate_analysis_plot(I_g, Q_g, I_e, Q_e, I_f, Q_f, ig_new, qg_new, ie_new,
                #                                            qe_new,
                #                                            if_new, qf_new, theta_ge, threshold_ge, QubitIndex,
                #                                            provided_sigma_num)
            del ss

        ############################################### Collect Results ################################################
        if save_data_h5:
            # ---------------------Collect g-e Res Spec Results----------------
            if run_flags["res_spec"]:
                res_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                res_data[QubitIndex]['freq_pts'][j - batch_num * save_r - 1] = freq_pts
                res_data[QubitIndex]['freq_center'][j - batch_num * save_r - 1] = freq_center
                res_data[QubitIndex]['Amps'][j - batch_num * save_r - 1] = amps
                res_data[QubitIndex]['Found Freqs'][j - batch_num * save_r - 1] = res_freqs
                res_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                res_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                res_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                res_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rspec

            # ---------------------Collect g-e QSpec Results----------------
            if run_flags["q_spec"]:
                qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
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

            # ---------------------Collect e-f res spec Results----------------
            if run_flags["ef_res_spec"] and ef_res_spec_survived:
                ef_res_data[QubitIndex]['Dates'][0] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                ef_res_data[QubitIndex]['freq_pts'][0] = ef_freq_pts
                ef_res_data[QubitIndex]['freq_center'][0] = ef_freq_center
                ef_res_data[QubitIndex]['Amps'][0] = ef_amps
                ef_res_data[QubitIndex]['Found Freqs'][0] = ef_res_freqs
                ef_res_data[QubitIndex]['Round Num'][0] = j
                ef_res_data[QubitIndex]['Batch Num'][0] = batch_num
                ef_res_data[QubitIndex]['Exp Config'][0] = expt_cfg
                ef_res_data[QubitIndex]['Syst Config'][0] = sys_config_rspec_ef
                ef_res_any = True

                # ---------------------Collect e-f qspec Results----------------
            if run_flags["ef_q_spec"] and ef_qspec_survived:
                ef_qspec_data[QubitIndex]['Dates'][0] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                ef_qspec_data[QubitIndex]['I'][0] = efqspec_I
                ef_qspec_data[QubitIndex]['Q'][0] = efqspec_Q
                ef_qspec_data[QubitIndex]['Frequencies'][0] = efqspec_freqs
                ef_qspec_data[QubitIndex]['I Fit'][0] = efqspec_I_fit
                ef_qspec_data[QubitIndex]['Q Fit'][0] = efqspec_Q_fit
                ef_qspec_data[QubitIndex]['Round Num'][0] = j
                ef_qspec_data[QubitIndex]['Batch Num'][0] = batch_num
                ef_qspec_data[QubitIndex]['Recycled QFreq'][0] = False  # no rr so no recycling here
                ef_qspec_data[QubitIndex]['Exp Config'][0] = expt_cfg
                ef_qspec_data[QubitIndex]['Syst Config'][0] = sys_config_qspec_ef
                ef_qspec_any = True

            # --------------------Collect rabi population measurements (qubit temperature data) ----------------
            if run_flags["rabi_pop_meas"] and ef_res_spec_survived and ef_qspec_survived:
                rabi_data_ef_Qtemps[QubitIndex]['Dates'][0] = (time.mktime(datetime.datetime.now().timetuple()))
                rabi_data_ef_Qtemps[QubitIndex]['Qfreq_ge'][
                    0] = qubit_freq  # save the g-e qubit freq too for this qubit
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

                rpm_any = True

                # ---------------------Collect g-e T1 Results----------------
            if run_flags["t1"]:
                t1_data[QubitIndex]['T1'][j - batch_num * save_r - 1] = t1_est
                t1_data[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t1_err
                t1_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t1_data[QubitIndex]['I'][j - batch_num * save_r - 1] = t1_I
                t1_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = t1_Q
                t1_data[QubitIndex]['Ishots'][j - batch_num * save_r - 1] = t1_Ishots
                t1_data[QubitIndex]['Qshots'][j - batch_num * save_r - 1] = t1_Qshots
                t1_data[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t1_delay_times
                t1_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = q1_fit_exponential
                t1_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t1_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t1_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                t1_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t1

            # ---------------------Collect g-e T2R Results----------------
            if run_flags["t2r"]:
                t2r_data[QubitIndex]['T2'][j - batch_num * save_r - 1] = t2r_est
                t2r_data[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t2r_err
                t2r_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t2r_data[QubitIndex]['I'][j - batch_num * save_r - 1] = t2r_I
                t2r_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = t2r_Q
                t2r_data[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t2r_delay_times
                t2r_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fit_ramsey
                t2r_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t2r_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t2r_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                t2r_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2r

            # ---------------------Collect g-e T2E Results----------------
            if run_flags["t2e"]:
                t2e_data[QubitIndex]['T2E'][j - batch_num * save_r - 1] = t2e_est
                t2e_data[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t2e_err
                t2e_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t2e_data[QubitIndex]['I'][j - batch_num * save_r - 1] = t2e_I
                t2e_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = t2e_Q
                t2e_data[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t2e_delay_times
                t2e_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fit_t2e
                t2e_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t2e_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t2e_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                t2e_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2e

            # ---------------------Collect g-e-f Single Shot Results----------------
            if run_flags["ss_gef"]:
                # ss_data[QubitIndex]['Fidelity'][j - batch_num * save_r - 1] = fid
                ss_data_gef[QubitIndex]['Angle_ef'][j - batch_num * save_r - 1] = theta_ef
                ss_data_gef[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                ss_data_gef[QubitIndex]['I_g'][j - batch_num * save_r - 1] = I_g
                ss_data_gef[QubitIndex]['Q_g'][j - batch_num * save_r - 1] = Q_g
                ss_data_gef[QubitIndex]['I_e'][j - batch_num * save_r - 1] = I_e
                ss_data_gef[QubitIndex]['Q_e'][j - batch_num * save_r - 1] = Q_e
                ss_data_gef[QubitIndex]['I_f'][j - batch_num * save_r - 1] = I_f
                ss_data_gef[QubitIndex]['Q_f'][j - batch_num * save_r - 1] = Q_f
                ss_data_gef[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                ss_data_gef[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                ss_data_gef[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                ss_data_gef[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_ss_gef

        del experiment

    ################################################## Potentially Save ################################################
    if save_data_h5:
        # Check if you are at the right round number
        # If so, then save all of the data and change the round num so you replace data starting next round
        if j % save_r == 0:
            batch_num += 1

            # --------------------------save g-e Res Spec-----------------------
            if run_flags["res_spec"]:
                saver_res = Data_H5(subStudyDataFolder, res_data, batch_num, save_r)
                saver_res.save_to_h5('res_ge')
                del saver_res
                del res_data

            # --------------------------save g-e QSpec-----------------------
            if run_flags["q_spec"]:
                saver_qspec = Data_H5(subStudyDataFolder, qspec_data, batch_num, save_r)
                saver_qspec.save_to_h5('qspec_ge')
                del saver_qspec
                del qspec_data

            # --------------------------save g-e Rabi-----------------------
            if run_flags["rabi"]:
                saver_rabi = Data_H5(subStudyDataFolder, rabi_data, batch_num, save_r)
                saver_rabi.save_to_h5('rabi_ge')
                del saver_rabi
                del rabi_data

            # --------------------------save g-e SS-----------------------
            if run_flags["ss"]:
                saver_ss = Data_H5(subStudyDataFolder, ss_data, batch_num, save_r)
                saver_ss.save_to_h5('ss_ge')
                del saver_ss
                del ss_data

            # --------------------------save e-f res spec-----------------------
            if run_flags["ef_res_spec"] and ef_res_any :
                saver_ef_res = Data_H5(subStudyDataFolder, ef_res_data, batch_num, save_r)  # save
                saver_ef_res.save_to_h5('res_ef')
                del saver_ef_res
                del ef_res_data
            # --------------------------save e-f qspec-----------------------
            if run_flags["ef_q_spec"] and ef_qspec_any:
                saver_ef_qspec = Data_H5(subStudyDataFolder, ef_qspec_data, batch_num, save_r)
                saver_ef_qspec.save_to_h5('qspec_ef')
                del saver_ef_qspec
                del ef_qspec_data
            # --------------------------save e-f Rabi-----------------------
            # if run_flags["ef_Rabi"]:
            #     saver_ef_rabi = Data_H5(subStudyDataFolder, ef_rabi_data, batch_num, save_r)
            #     saver_ef_rabi.save_to_h5('rabi_ef')
            #     del saver_ef_rabi
            #     del ef_rabi_data

            # --------save rabi population measurements (qubit temperature data) -----------------------
            if run_flags["rabi_pop_meas"] and rpm_any:
                saver_rabi_Qtemps = Data_H5(subStudyDataFolder, rabi_data_ef_Qtemps, batch_num, save_r)
                saver_rabi_Qtemps.save_to_h5('q_temperatures')
                del saver_rabi_Qtemps
                del rabi_data_ef_Qtemps

            # --------------------------save g-e t1-----------------------
            if run_flags["t1"]:
                saver_t1 = Data_H5(subStudyDataFolder, t1_data, batch_num, save_r)
                saver_t1.save_to_h5('t1_ge')
                del saver_t1
                del t1_data

            # --------------------------save g-e t2r-----------------------
            if run_flags["t2r"]:
                saver_t2r = Data_H5(subStudyDataFolder, t2r_data, batch_num, save_r)
                saver_t2r.save_to_h5('t2_ge')
                del saver_t2r
                del t2r_data

            # --------------------------save g-e t2e-----------------------
            if run_flags["t2e"]:
                saver_t2e = Data_H5(subStudyDataFolder, t2e_data, batch_num, save_r)
                saver_t2e.save_to_h5('t2e_ge')
                del saver_t2e
                del t2e_data

            # --------------------------save g-e-f SS-----------------------
            if run_flags["ss_gef"]:
                saver_ss = Data_H5(subStudyDataFolder, ss_data_gef, batch_num, save_r)
                saver_ss.save_to_h5('ss_gef')
                del saver_ss
                del ss_data_gef

    # reset all dictionaries to none for safety
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
    ef_res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    ef_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    rabi_data_ef_Qtemps = create_data_dict(rabi_keys_ef_Qtemps, save_r, list_of_all_qubits)
    t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
    t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
    t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
    ss_data_gef = create_data_dict(ss_keys_gef, save_r, list_of_all_qubits)

    ef_res_any = False
    ef_qspec_any = False
    rpm_any = False

en = time.time()
print('timetaken=', en - st)