import sys
import os
from copy import deepcopy

import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
import logging
import visdom
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_001_time_of_flight import TOFExperiment
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_006p55_length_rabi_ge_qze import LengthRabiExperimentQZE
from section_005_single_shot_ge import SingleShot
from section_008_save_data_to_h5 import Data_H5
from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_002_res_spec_ef import ResonanceSpectroscopyEF
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_004_qubit_spec_ef import EFQubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_005_single_shot_gef import SingleShot_ef
from section_008_save_data_to_h5 import Data_H5
from section_006_amp_rabi_ef import EF_AmplitudeRabiExperiment
from section_007_T1_ef import EF_T1Measurement
from section_007_T1_ge import T1Measurement
from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from section_006p5_length_rabi_ge import LengthRabiExperiment
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement

################################################ Run Configurations ####################################################
zero_qubit_drive_gain = False
constant_zeno_pulse = True
adapt_starked_qubit_freq = False
wait_for_res_ring_up = True
n= 1
save_r = 1                           # how many rounds to save after
signal = 'None'                      # 'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization
save_figs = True                     # save plots for everything as you go along the RR script?
live_plot = False                    # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = False                     # fit the data here and save or plot the fits?
save_data_h5 = True                  # save all of the data to h5 files?
verbose = True                       # print everything to the console in real time, good for debugging, bad for memory
debug_mode = True                    # if True, it disables the continuing function of RR if an error pops up in a class -- errors now stop the RR script
thresholding = False                 # use internal QICK threshold for ratio of Binary values on y for rabi/t1/t2r/t2e, or analog avg when false
increase_qubit_reps = False          # if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0       # only has impact if previous line is True
multiply_qubit_reps_by = 2           # only has impact if the line two above is True
increase_qubit_steps_ef = False #if you want to increase the steps for all qubits, set to True, if you only want to set it to true for 1 qubit, see e-f qubit spec section
increase_steps_to_ef = 600
Qs_to_look_at = [1]        # only list the qubits you want to do the RR for
study = 'QZE'
sub_study = 'dephasing_from_higher_energy_levels'
substudy_txt_notes = ('Lets give this an initial test and make sure all of these experiments work well, on qubit 2')
# set which of the following you'd like to run to 'True'
run_flags = {"res_spec_ge": True, "q_spec_ge": True, "rabi_ge": True, "res_spec_ef": True, "q_spec_ef": True,
             "ss_gef": True, "ss_gef_fstate": True, "rabi_ef": True, "t1_ge": True, "t1_fg": True, "t1_fe": True,
             "t2r": True,"t2e": True}
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

formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
formatted_datetime = 'qze_experiment_with_starked_qfreq_old_readout_values'+formatted_datetime
dataSetFolder = os.path.join(subStudyFolder, formatted_datetime)
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

#Logging
log_file = os.path.join(studyDocumentationFolder, "QZE_rabi_script.log")
rr_logger = logging.getLogger("custom_logger_for_rr_only")
rr_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
rr_logger.addHandler(file_handler)
rr_logger.propagate = False

################################################ optimization outputs ##################################################
# Optimization parameters for resonator spectroscopy
res_leng_vals = [5.5, 7.5, 6.0, 6.5, 5.0, 6.0]
res_gain = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] #I dont need speed for this experiment, I can do many averages. So, I will do a very low gain to make sure im never punching out
freq_offsets = [0, 0, 0, 0, 0, 0]

####################################################### live plot ######################################################
if live_plot:
    # Check if visdom is connected right away, otherwise, throw an error
    if not (viz := visdom.Visdom()).check_connection(timeout_seconds=5):
        raise RuntimeError("Visdom server not connected!, Type \'visdom\' into the command line and go to "
                           "http://localhost:8097/ on firefox")

############################## Get first the res, qubit freqs #######################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}


# Define what to save to h5 files
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num','Recycled QFreq',
              'Exp Config', 'Syst Config']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']

res_keys_ef = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
qspec_keys_ef = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num','Recycled QFreq',
              'Exp Config', 'Syst Config']
rabi_keys_ef = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
ss_keys_gef = ['Fidelity', 'Angle_ge', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'I_f', 'Q_f', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t2r_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
t2e_keys = ['T2E', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']

#initialize a dictionary to store those values
res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
t1_data_eg = create_data_dict(t1_keys, save_r, list_of_all_qubits)
t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)


res_data_ef = create_data_dict(res_keys_ef, save_r, list_of_all_qubits)
qspec_data_ef = create_data_dict(qspec_keys_ef, save_r, list_of_all_qubits)
rabi_data_ef = create_data_dict(rabi_keys_ef, save_r, list_of_all_qubits)

t1_data_fg = create_data_dict(t1_keys, save_r, list_of_all_qubits)
t1_data_fe = create_data_dict(t1_keys, save_r, list_of_all_qubits)

ss_data_gef = create_data_dict(ss_keys_gef, save_r, list_of_all_qubits)

#initialize a simple list to store the qspec values in incase a fit fails
stored_qspec_list = [None] * tot_num_of_qubits
batch_num=0
j = 0
qubit_freqs_ge = np.zeros(6)
qubit_freqs_ef = np.zeros(6)
res_freq_ge = np.zeros(6)
for QubitIndex in Qs_to_look_at:
    experiment = QICK_experiment(optimizationFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10,
                                 fridge=FRIDGE)
    updated_qubit_gain = 0.05 # lets do a low gain to start so I dont have a broad linewidth for the qubit
    experiment.qubit_cfg['qubit_gain_ge'][
        QubitIndex] = updated_qubit_gain

    # Mask out all other resonators except this one
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

    ################################################# g-e Res spec ####################################################
    if run_flags["res_spec_ge"]:
        res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, subStudyDataFolder, j, save_figs,
                                         experiment)
        res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
        experiment.readout_cfg['res_freq_ge'] = res_freqs

        # incorporating offset (if you don't want to, then set all values inside freq_offsets to zero)
        offset = freq_offsets[QubitIndex]
        offset_res_freqs = [r + offset for r in res_freqs]
        experiment.readout_cfg['res_freq_ge'] = offset_res_freqs

        this_res_freq = offset_res_freqs[QubitIndex]
        res_freq_ge[QubitIndex] = float(this_res_freq)

        print('Qubit ', QubitIndex + 1, ' g-e res freq: ', this_res_freq)

        del res_spec

    ################################################### g-e Qubit spec ##################################################
    if run_flags["q_spec_ge"]:
        q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, subStudyDataFolder, j, signal, save_figs, experiment,
                                   live_plot, verbose=False, logger=None, qick_verbose=True, increase_reps=False,
                                   increase_reps_to=500)
        qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq, sys_config_qspec = q_spec.run()

        qubit_freqs_ge[QubitIndex] = qubit_freq
        experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
        print('Qubit ', QubitIndex + 1, ' g-e freq: ', float(qubit_freq))
        del q_spec

    ###################################################### g-e Rabi ####################################################
    if run_flags["rabi_ge"]:
        rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, subStudyDataFolder, j, signal, save_figs,
                                       experiment, live_plot,
                                       increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)

        rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_rabi = rabi.run()

        experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
        print('Qubit ', QubitIndex + 1, ' g-e Pi Amp: ', float(pi_amp))

        del rabi

    ################################################# e-f Res spec ####################################################
    if run_flags["res_spec_ef"]:
        res_specEF = ResonanceSpectroscopyEF(QubitIndex, tot_num_of_qubits, subStudyDataFolder, j, save_figs,
                                             experiment)
        res_freqs, freq_pts, freq_center, amps, sys_config_rspec_ef = res_specEF.run()
        experiment.readout_cfg['res_freq_ef'] = res_freqs
        print('Qubit ', QubitIndex + 1, ' e-f res freq: ', res_freqs[QubitIndex])

        del res_specEF

    ################################################## e-f Qubit spec ##################################################
    if run_flags["q_spec_ef"]:
        # Qubit 4 needs more steps for e-f spec
        if QubitIndex == 3:
            increase_qubit_steps_ef = True  # if you want to increase the steps for a qubit, set to True

        ef_q_spec = EFQubitSpectroscopy(QubitIndex, tot_num_of_qubits, list_of_all_qubits, subStudyDataFolder, j, signal,
                                        save_figs, experiment, live_plot, increase_qubit_steps_ef,
                                        increase_steps_to_ef)
        efqspec_I, efqspec_Q, efqspec_freqs, efqspec_I_fit, efqspec_Q_fit, efqubit_freq, sys_config_qspec_ef = ef_q_spec.run(
            experiment.soccfg,
            experiment.soc)
        qubit_freqs_ef[QubitIndex] = efqubit_freq
        experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] = float(efqubit_freq)
        print('Qubit ', QubitIndex + 1, ' e-f Freq: ', float(efqubit_freq))

        del ef_q_spec

    ###################################################### e-f Rabi ####################################################
    if run_flags["rabi_ef"]:
        efrabi = EF_AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, list_of_all_qubits, subStudyDataFolder, j, signal,
                                            save_figs,
                                            experiment, live_plot,
                                            increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
        efrabi_I, efrabi_Q, efrabi_gains, efrabi_fit, efpi_amp, sys_config_rabi_ef = efrabi.run(experiment.soccfg,
                                                                                                experiment.soc)

        experiment.qubit_cfg['pi_ef_amp'][QubitIndex] = float(efpi_amp)
        print('Qubit ', QubitIndex + 1, ' e-f pulse amp: ', float(efpi_amp))

        del efrabi

    # ############################################# modify for f-h Res spec ############################################
    # if run_flags["res_spec_fh"]:
    #     res_specEF = ResonanceSpectroscopyEF(QubitIndex, tot_num_of_qubits, subStudyDataFolder, j, save_figs,
    #                                          experiment)
    #     res_freqs, freq_pts, freq_center, amps, sys_config_rspec_ef = res_specEF.run()
    #     experiment.readout_cfg['res_freq_fh'] = res_freqs
    #     print('Qubit ', QubitIndex + 1, ' f-h res freq: ', res_freqs[QubitIndex])
    #
    #     del res_specEF

    # ######################################## modify for e-f Qubit spec ###############################################
    # if run_flags["q_spec_ef"]:
    #     # Qubit 4 needs more steps for e-f spec
    #     if QubitIndex == 3:
    #         increase_qubit_steps_ef = True  # if you want to increase the steps for a qubit, set to True
    #
    #     ef_q_spec = QubitSpectroscopyEF(QubitIndex, tot_num_of_qubits, list_of_all_qubits, subStudyDataFolder, j,
    #                                     signal,
    #                                     save_figs, experiment, live_plot, increase_qubit_steps_ef,
    #                                     increase_steps_to_ef)
    #     efqspec_I, efqspec_Q, efqspec_freqs, efqspec_I_fit, efqspec_Q_fit, efqubit_freq, sys_config_qspec_ef = ef_q_spec.run(
    #         experiment.soccfg,
    #         experiment.soc)
    #     qubit_freqs_ef[QubitIndex] = efqubit_freq
    #     experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] = float(efqubit_freq)
    #     print('Qubit ', QubitIndex + 1, ' e-f Freq: ', float(efqubit_freq))
    #
    #     del ef_q_spec

    ###################################################### g-e T1 ####################################################
    if run_flags["t1_ge"]:
        t1_ge = T1Measurement(QubitIndex, tot_num_of_qubits, subStudyDataFolder, j, signal, save_figs,
                              experiment=experiment,
                              live_plot=live_plot, fit_data=fit_data,
                              increase_qubit_reps=increase_qubit_reps,
                              qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                              multiply_qubit_reps_by=multiply_qubit_reps_by)
        t1_est_ge, t1_err_ge, t1_I_ge, t1_Q_ge, t1_delay_times_ge, q1_fit_exponential_ge, sys_config_t1_ge = t1_ge.run(
            thresholding=False)

        print('Qubit ', QubitIndex + 1, ' g-e T1: ', str(t1_est_ge))

        del t1_ge

    ###################################################### f-g T1 ####################################################
    if run_flags["t1_fg"]:
        t1_fg = EF_T1Measurement(QubitIndex, tot_num_of_qubits, subStudyDataFolder, j, signal, save_figs,
                                 experiment=experiment,
                                 live_plot=live_plot, fit_data=fit_data,
                                 increase_qubit_reps=increase_qubit_reps,
                                 qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                 multiply_qubit_reps_by=multiply_qubit_reps_by, expt_name='T1_fg')
        t1_est_fg, t1_err_fg, t1_I_fg, t1_Q_fg, t1_delay_times_fg, q1_fit_exponential_fg, sys_config_t1_fg = t1_fg.run(
            thresholding=False)

        print('Qubit ', QubitIndex + 1, ' f-g T1: ', str(t1_est_fg))

        del t1_fg

    ###################################################### f-e T1 ####################################################
    if run_flags["t1_fe"]:
        t1_fe = EF_T1Measurement(QubitIndex, tot_num_of_qubits, subStudyDataFolder, j, signal, save_figs,
                                 experiment=experiment,
                                 live_plot=live_plot, fit_data=fit_data,
                                 increase_qubit_reps=increase_qubit_reps,
                                 qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                 multiply_qubit_reps_by=multiply_qubit_reps_by, expt_name='T1_fe')
        t1_est_fe, t1_err_fe, t1_I_fe, t1_Q_fe, t1_delay_times_fe, q1_fit_exponential_fe, sys_config_t1_fe = t1_fe.run(
            thresholding=False)

        print('Qubit ', QubitIndex + 1, ' f-e T1: ', str(t1_est_fe))

        del t1_fe
    ###################################################### T2R #####################################################
    if run_flags["t2r"]:
        try:
            t2r = T2RMeasurement(QubitIndex, tot_num_of_qubits, subStudyDataFolder, j, signal, save_figs,
                                 experiment=experiment, live_plot=live_plot, fit_data=fit_data,
                                 increase_qubit_reps=increase_qubit_reps,
                                 qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                 multiply_qubit_reps_by=multiply_qubit_reps_by,
                                 verbose=verbose, logger=rr_logger)
            t2r_est, t2r_err, t2r_I, t2r_Q, t2r_delay_times, fit_ramsey, sys_config_t2r = t2r.run(
                thresholding=thresholding)
            del t2r

        except Exception as e:
            if debug_mode:
                raise e  # In debug mode, re-raise the exception immediately
            else:
                rr_logger.exception(f'Got the following error, continuing: {e}')
                if verbose: print(f'Got the following error, continuing: {e}')
                continue  # skip the rest of this qubit

    ##################################################### T2E ######################################################
    if run_flags["t2e"]:
        try:
            t2e = T2EMeasurement(QubitIndex, tot_num_of_qubits, subStudyDataFolder, j, signal, save_figs,
                                 experiment=experiment, live_plot=live_plot, fit_data=fit_data,
                                 increase_qubit_reps=increase_qubit_reps,
                                 qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                 multiply_qubit_reps_by=multiply_qubit_reps_by,
                                 verbose=verbose, logger=rr_logger)
            (t2e_est, t2e_err, t2e_I, t2e_Q, t2e_delay_times,
             fit_t2e, sys_config_t2e) = t2e.run(thresholding=thresholding)
            del t2e

        except Exception as e:
            if debug_mode:
                raise e  # In debug mode, re-raise the exception immediately
            else:
                rr_logger.exception(f'Got the following error, continuing: {e}')
                if verbose: print(f'Got the following error, continuing: {e}')
                continue  # skip the rest of this qubit

    ############################################### Collect Results ################################################
    if save_data_h5:
        # ---------------------Collect g-e Res Spec Results----------------
        if run_flags["res_spec_ge"]:
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

        # ---------------------Collect e-f Res Spec Results----------------
        if run_flags["res_spec_ef"]:
            res_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                time.mktime(datetime.datetime.now().timetuple()))
            res_data[QubitIndex]['freq_pts'][j - batch_num * save_r - 1] = freq_pts
            res_data[QubitIndex]['freq_center'][j - batch_num * save_r - 1] = freq_center
            res_data[QubitIndex]['Amps'][j - batch_num * save_r - 1] = amps
            res_data[QubitIndex]['Found Freqs'][j - batch_num * save_r - 1] = res_freqs
            res_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            res_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            res_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
            res_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rspec_ef

        # ---------------------Collect g-e QSpec Results----------------
        if run_flags["q_spec_ge"]:
            qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                time.mktime(datetime.datetime.now().timetuple()))
            qspec_data[QubitIndex]['I'][j - batch_num * save_r - 1] = qspec_I
            qspec_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = qspec_Q
            qspec_data[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = qspec_freqs
            qspec_data[QubitIndex]['I Fit'][j - batch_num * save_r - 1] = qspec_I_fit
            qspec_data[QubitIndex]['Q Fit'][j - batch_num * save_r - 1] = qspec_Q_fit
            qspec_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            qspec_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            # qspec_data[QubitIndex]['Recycled QFreq'][j - batch_num * save_r - 1] = recycled_qfreq
            qspec_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
            qspec_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_qspec

        # ---------------------Collect e-f QSpec Results----------------
        if run_flags["q_spec_ef"]:
            qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                time.mktime(datetime.datetime.now().timetuple()))
            qspec_data[QubitIndex]['I'][j - batch_num * save_r - 1] = qspec_I
            qspec_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = qspec_Q
            qspec_data[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = qspec_freqs
            qspec_data[QubitIndex]['I Fit'][j - batch_num * save_r - 1] = qspec_I_fit
            qspec_data[QubitIndex]['Q Fit'][j - batch_num * save_r - 1] = qspec_Q_fit
            qspec_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            qspec_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            # qspec_data[QubitIndex]['Recycled QFreq'][j - batch_num * save_r - 1] = recycled_qfreq
            qspec_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
            qspec_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_qspec_ef

        # ---------------------Collect g-e Rabi Results----------------
        if run_flags["rabi_ge"]:
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

        # ---------------------Collect e-f Rabi Results----------------
        if run_flags["rabi_ef"]:
            rabi_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                time.mktime(datetime.datetime.now().timetuple()))
            rabi_data[QubitIndex]['I'][j - batch_num * save_r - 1] = rabi_I
            rabi_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = rabi_Q
            rabi_data[QubitIndex]['Gains'][j - batch_num * save_r - 1] = rabi_gains
            rabi_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = rabi_fit
            rabi_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            rabi_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            rabi_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
            rabi_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rabi_ef

        # ---------------------Collect e-g T1 Results----------------
        if run_flags["t1_ge"]:
            t1_data_eg[QubitIndex]['T1'][j - batch_num * save_r - 1] = t1_est_ge
            t1_data_eg[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t1_err_ge
            t1_data_eg[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                time.mktime(datetime.datetime.now().timetuple()))
            t1_data_eg[QubitIndex]['I'][j - batch_num * save_r - 1] = t1_I_ge
            t1_data_eg[QubitIndex]['Q'][j - batch_num * save_r - 1] = t1_Q_ge
            t1_data_eg[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t1_delay_times_ge
            t1_data_eg[QubitIndex]['Fit'][j - batch_num * save_r - 1] = q1_fit_exponential_ge
            t1_data_eg[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            t1_data_eg[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            t1_data_eg[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
            t1_data_eg[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t1_ge

        # ---------------------Collect f-g T1 Results----------------
        if run_flags["t1_fg"]:
            t1_data_fg[QubitIndex]['T1'][j - batch_num * save_r - 1] = t1_est_fg
            t1_data_fg[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t1_err_fg
            t1_data_fg[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                time.mktime(datetime.datetime.now().timetuple()))
            t1_data_fg[QubitIndex]['I'][j - batch_num * save_r - 1] = t1_I_fg
            t1_data_fg[QubitIndex]['Q'][j - batch_num * save_r - 1] = t1_Q_fg
            t1_data_fg[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t1_delay_times_fg
            t1_data_fg[QubitIndex]['Fit'][j - batch_num * save_r - 1] = q1_fit_exponential_fg
            t1_data_fg[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            t1_data_fg[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            t1_data_fg[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
            t1_data_fg[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t1_fg

        # ---------------------Collect f-e T1 Results----------------
        if run_flags["t1_fe"]:
            t1_data_fe[QubitIndex]['T1'][j - batch_num * save_r - 1] = t1_est_fe
            t1_data_fe[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t1_err_fe
            t1_data_fe[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                time.mktime(datetime.datetime.now().timetuple()))
            t1_data_fe[QubitIndex]['I'][j - batch_num * save_r - 1] = t1_I_fe
            t1_data_fe[QubitIndex]['Q'][j - batch_num * save_r - 1] = t1_Q_fe
            t1_data_fe[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t1_delay_times_fe
            t1_data_fe[QubitIndex]['Fit'][j - batch_num * save_r - 1] = q1_fit_exponential_fe
            t1_data_fe[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            t1_data_fe[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            t1_data_fe[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
            t1_data_fe[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t1_fe

        # ---------------------Collect T2 Results----------------
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

        # ---------------------Collect T2E Results----------------
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

    del experiment

    ################################################## Potentially Save ################################################
if save_data_h5:
    # Check if you are at the right round number
    # If so, then save all of the data and change the round num so you replace data starting next round
    if j % save_r == 0:
        batch_num += 1

        # --------------------------save g-e Res Spec-----------------------
        if run_flags["res_spec_ge"]:
            saver_res = Data_H5(subStudyDataFolder, res_data, batch_num, save_r)
            saver_res.save_to_h5('Res_ge')
            del saver_res
            del res_data

        # --------------------------save e-f Res Spec-----------------------
        if run_flags["res_spec_ef"]:
            saver_res = Data_H5(subStudyDataFolder, res_data_ef, batch_num, save_r)
            saver_res.save_to_h5('Res_ef')
            del saver_res
            del res_data_ef

        # --------------------------save g-e QSpec-----------------------
        if run_flags["q_spec_ge"]:
            saver_qspec = Data_H5(subStudyDataFolder, qspec_data, batch_num, save_r)
            saver_qspec.save_to_h5('QSpec_ge')
            del saver_qspec
            del qspec_data

        # --------------------------save e-f QSpec-----------------------
        if run_flags["q_spec_ef"]:
            saver_qspec = Data_H5(subStudyDataFolder, qspec_data_ef, batch_num, save_r)
            saver_qspec.save_to_h5('QSpec_ef')
            del saver_qspec
            del qspec_data_ef

        # --------------------------save g-e Rabi-----------------------
        if run_flags["rabi_ge"]:
            saver_rabi = Data_H5(subStudyDataFolder, rabi_data, batch_num, save_r)
            saver_rabi.save_to_h5('Rabi_ge')
            del saver_rabi
            del rabi_data

        # --------------------------save e-f Rabi-----------------------
        if run_flags["rabi_ef"]:
            saver_rabi = Data_H5(subStudyDataFolder, rabi_data_ef, batch_num, save_r)
            saver_rabi.save_to_h5('Rabi_ef')
            del saver_rabi
            del rabi_data_ef

        # --------------------------save t1 e-g -----------------------
        if run_flags["t1_ge"]:
            saver_t1 = Data_H5(subStudyDataFolder, t1_data_eg, batch_num, save_r)
            saver_t1.save_to_h5('T1_ge')
            del saver_t1
            del t1_data_eg

        # --------------------------save t1 f-g -----------------------
        if run_flags["t1_fg"]:
            saver_t1 = Data_H5(subStudyDataFolder, t1_data_fg, batch_num, save_r)
            saver_t1.save_to_h5('T1_fg')
            del saver_t1
            del t1_data_fg

        # --------------------------save t1 f-e -----------------------
        if run_flags["t1_fe"]:
            saver_t1 = Data_H5(subStudyDataFolder, t1_data_fe, batch_num, save_r)
            saver_t1.save_to_h5('T1_fe')
            del saver_t1
            del t1_data_fe

        # --------------------------save t2r-----------------------
        if run_flags["t2r"]:
            saver_t2r = Data_H5(subStudyDataFolder, t2r_data, batch_num, save_r)
            saver_t2r.save_to_h5('T2_ge')
            del saver_t2r
            del t2r_data

        # --------------------------save t2e-----------------------
        if run_flags["t2e"]:
            saver_t2e = Data_H5(subStudyDataFolder, t2e_data, batch_num, save_r)
            saver_t2e.save_to_h5('T2E_ge')
            del saver_t2e
            del t2e_data

        # reset all dictionaries to none for safety
        res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
        res_data_ef = create_data_dict(res_keys_ef, save_r, list_of_all_qubits)

        qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
        qspec_data_ef = create_data_dict(qspec_keys_ef, save_r, list_of_all_qubits)

        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
        rabi_data_ef = create_data_dict(rabi_keys_ef, save_r, list_of_all_qubits)

        t1_data_eg = create_data_dict(t1_keys, save_r, list_of_all_qubits)
        t1_data_fg = create_data_dict(t1_keys, save_r, list_of_all_qubits)
        t1_data_fe = create_data_dict(t1_keys, save_r, list_of_all_qubits)

        t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
        t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)

