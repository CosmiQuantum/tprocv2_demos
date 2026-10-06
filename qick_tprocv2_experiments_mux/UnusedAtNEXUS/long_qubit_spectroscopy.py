import copy
import sys
import os
import numpy as np

np.set_printoptions(threshold=int(1e15))  # need this so it saves absolutely everything returned from the classes
import datetime
import time
import logging
import visdom

sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_001_time_of_flight import TOFExperiment
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_004_qubit_spec_ef_V2 import EFQubitSpectroscopy
from section_004_qubit_spec_fh_V2 import FHQubitSpectroscopy
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
n = 70
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
debug_mode = True  # if True, it disables the continuing function of RR if an error pops up in a class -- errors now stop the RR script
thresholding = False  # use internal QICK threshold for ratio of Binary values on y for rabi/t1/t2r/t2e, or analog avg when false
increase_qubit_reps = False  # if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0  # only has impact if previous line is True
multiply_qubit_reps_by = 2  # only has impact if the line two above is True
Qs_to_look_at = [0, 1, 2, 3, 4, 5]  # only list the qubits you want to do the RR for

# Data saving info
run_name = 'run7'
device_name = '6transmon'

# Vd =
# Id =
# Vg =
# hemt_vd_id_vg = f'Vd_{Vd}__Id_{Id}__Vg_{Vg}'
# WarmAmpschain = ''
# MC_temp =
#
# substudy_txt_notes = ('Normal Round Robin during cooldown, now everything works properly, set debug to false to run '
#                       'overnight and running in terminal with repeater script' + f'The MC temperature is {MC_temp}mK' + f'HEMT bias is Vg={Vg}, Vd={Vd}, Id={Id}. The warm amp  chain is {WarmAmpschain}' + '\n ')

# set which of the following you'd like to run to 'True'
run_flags = {"ge_q_spec": False, "ef_q_spec": True, "fe_q_spec": True, }
# optimization outputs
res_leng_vals = [5.5, 12, 6.0, 6.5, 5.0, 6.0]
res_gain = [0.9, 0.95, 0.78, 0.58, 0.95, 0.57]
freq_offsets = [-0.1, 0.2, -0.1, -0.4, -0.1, -0.1]
################################################ Data Saving Setup ##################################################
# Folders
study = 'Parity'
sub_study = 'Junkyard'#'Repeated Qubit Spectroscopy'
data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # should have a new onoe for every optimization batch

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
# with open(file_path, "w", encoding="utf-8") as file:
#     file.write(substudy_txt_notes)


################################################## Configure logging ###################################################
# ''' We need to create a custom logger and disable propagation like this
# to remove the logs from the underlying qick from saving to the log file for RR'''

# log_file = os.path.join(studyDocumentationFolder, "RR_script.log")
# rr_logger = logging.getLogger("custom_logger_for_rr_only")
# rr_logger.setLevel(logging.DEBUG)

# file_handler = logging.FileHandler(log_file, mode='a')
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)

# rr_logger.addHandler(file_handler)
# rr_logger.propagate = False  #dont propagate logs from underlying qick package
##############################################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}


# Define what to save to h5 files

ge_qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num', 'Recycled QFreq',
                 'Exp Config', 'Syst Config', 'times']
ef_qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num', 'Recycled QFreq',
                 'Exp Config', 'Syst Config', 'times']
fh_qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num', 'Recycled QFreq',
                 'Exp Config', 'Syst Config', 'times']

ge_qspec_data = create_data_dict(ge_qspec_keys, save_r, list_of_all_qubits)
ef_qspec_data = create_data_dict(ef_qspec_keys, save_r, list_of_all_qubits)
fh_qspec_data = create_data_dict(fh_qspec_keys, save_r, list_of_all_qubits)

batch_num = 0
j = 0
angles = []

while j < 1:
    j += 1
    for QubitIndex in Qs_to_look_at:
        recycled_qfreq = False

        # Get the config for this qubit
        experiment = QICK_experiment(optimizationFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10,
                                     fridge=FRIDGE)
        experiment.create_folder_if_not_exists(optimizationFolder)

        # Mask out all other resonators except this one
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        ################################################## Qubit spec ##################################################
        if run_flags["ge_q_spec"]:
            try:
                q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j,
                                           signal, save_figs, plot_fit=False, experiment=experiment,
                                           live_plot=live_plot, verbose=verbose, logger=rr_logger)
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
                        if verbose:
                            print('No stored qubit spec value for qubit {QubitIndex}; skipping iteration.')
                        del q_spec

                        continue
                else:
                    experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
                    stored_qspec_list[QubitIndex] = float(qubit_freq)
                rr_logger.info(f"Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                if verbose:
                    print(f"Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                del q_spec

            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"RR QSpec error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"RR QSpec error on qubit {QubitIndex}: {e}")
                continue

            ################################################## ef Qubit spec ##################################################
        if run_flags["ef_q_spec"]:
            try:
                efq_spec = EFQubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j,
                                               signal, save_figs, plot_fit=False, experiment=experiment,
                                               live_plot=live_plot, verbose=verbose, logger=rr_logger)
                EFIs, EFQs, effreqs, times, ef_config = efq_spec.run_long()

                # if qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None:
                #     if stored_qspec_list[QubitIndex] is not None:
                #         experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] = stored_qspec_list[QubitIndex]
                #         rr_logger.warning(f"Using previous stored value: {stored_qspec_list[QubitIndex]}")
                #         recycled_qfreq = True
                #         qubit_freq = stored_qspec_list[QubitIndex]
                #         experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] = float(qubit_freq)
                #         stored_qspec_list[QubitIndex] = float(qubit_freq)
                #         if verbose:
                #             print(f"Using previous stored value: {qubit_freq}")
                #     else:
                #         rr_logger.warning(f"No stored qubit spec value for qubit {QubitIndex}; skipping iteration.")
                #         if verbose:
                #             print('No stored qubit spec value for qubit {QubitIndex}; skipping iteration.')
                #         del q_spec

                #         continue
                # else:
                #     experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
                #     stored_qspec_list[QubitIndex] = float(qubit_freq)
                # rr_logger.info(f"Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                # if verbose:
                #     print(f"Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                del efq_spec

            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"RR QSpec error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"RR QSpec error on qubit {QubitIndex}: {e}")
                continue

            ##################################################fh Qubit spec ##################################################
        if run_flags["fh_q_spec"]:
            try:
                fhq_spec = FHQubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j,
                                               signal, save_figs, plot_fit=False, experiment=experiment,
                                               live_plot=live_plot, verbose=verbose, logger=rr_logger)
                FHIs, FHQs, fhfreqs, eftimes, fh_config = fhq_spec.run_long()

                # if qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None:
                #     if stored_qspec_list[QubitIndex] is not None:
                #         experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] = stored_qspec_list[QubitIndex]
                #         rr_logger.warning(f"Using previous stored value: {stored_qspec_list[QubitIndex]}")
                #         recycled_qfreq = True
                #         qubit_freq = stored_qspec_list[QubitIndex]
                #         experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] = float(qubit_freq)
                #         stored_qspec_list[QubitIndex] = float(qubit_freq)
                #         if verbose:
                #             print(f"Using previous stored value: {qubit_freq}")
                #     else:
                #         rr_logger.warning(f"No stored qubit spec value for qubit {QubitIndex}; skipping iteration.")
                #         if verbose:
                #             print('No stored qubit spec value for qubit {QubitIndex}; skipping iteration.')
                #         del q_spec

                #         continue
                # else:
                #     experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
                #     stored_qspec_list[QubitIndex] = float(qubit_freq)
                # rr_logger.info(f"Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                # if verbose:
                #     print(f"Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                del fhq_spec

            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"RR QSpec error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"RR QSpec error on qubit {QubitIndex}: {e}")
                continue

                del experiment

            ############################################### Collect Results ################################################
        if save_data_h5:
            # ---------------------Collect QSpec Results----------------
            if run_flags["q_spec"]:
                qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                qspec_data[QubitIndex]['I'][j - batch_num * save_r - 1] = qspec_I
                qspec_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = qspec_Q
                qspec_data[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = qspec_freqs
                # qspec_data[QubitIndex]['I Fit'][j - batch_num * save_r - 1] = qspec_I_fit
                # qspec_data[QubitIndex]['Q Fit'][j - batch_num * save_r - 1] = qspec_Q_fit
                qspec_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                qspec_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                qspec_data[QubitIndex]['Recycled QFreq'][j - batch_num * save_r - 1] = recycled_qfreq
                qspec_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                qspec_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_qspec

            # ---------------------Collect EF QSpec Results----------------
            if run_flags["efq_spec"]:
                ef_qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                ef_qspec_data[QubitIndex]['I'][j - batch_num * save_r - 1] = EHIs
                ef_qspec_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = EHQs
                ef_qspec_data[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = effreqs
                ef_  # qspec_data[QubitIndex]['I Fit'][j - batch_num * save_r - 1] = efqspec_I_fit
                ef_  # qspec_data[QubitIndex]['Q Fit'][j - batch_num * save_r - 1] = efqspec_Q_fit
                ef_qspec_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                ef_qspec_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                ef_  # qspec_data[QubitIndex]['Recycled QFreq'][j - batch_num * save_r - 1] = recycled_qfreq
                ef_qspec_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                ef_qspec_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = ef_config
                ef_qspec_data[QubitIndex]['times'][j - batch_num * save_r - 1] = eftimes

        # ---------------------Collect FH QSpec Results----------------
        if run_flags["fhq_spec"]:
            fh_qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                time.mktime(datetime.datetime.now().timetuple()))
            fh_qspec_data[QubitIndex]['I'][j - batch_num * save_r - 1] = FHIs
            fh_qspec_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = FHQs
            fh_qspec_data[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = fhfreqs
            fh_  # qspec_data[QubitIndex]['I Fit'][j - batch_num * save_r - 1] = qspec_I_fit
            fh_  # qspec_data[QubitIndex]['Q Fit'][j - batch_num * save_r - 1] = qspec_Q_fit
            fh_qspec_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            fh_qspec_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            fh_  # qspec_data[QubitIndex]['Recycled QFreq'][j - batch_num * save_r - 1] = recycled_qfreq
            fh_qspec_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
            fh_qspec_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = fh_config
            fh_qspec_data[QubitIndex]['times'][j - batch_num * save_r - 1] = fhtimes

        qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
        ef_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
        fh_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)




