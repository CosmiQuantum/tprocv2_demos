import sys
import os
from copy import deepcopy

import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
import logging
import visdom
sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
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
from section_002_res_spec_fh import ResonanceSpectroscopyFH
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_004_qubit_spec_ef import EFQubitSpectroscopy
from section_004_qubit_spec_fh import FHQubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_005_single_shot_gef import SingleShot_ef
from section_008_save_data_to_h5 import Data_H5
from section_006_amp_rabi_ef import EF_AmplitudeRabiExperiment
from section_006_amp_rabi_fh import FH_AmplitudeRabiExperiment
from section_007_T1_ef import EF_T1Measurement
from section_007_T1_fh import FH_T1Measurement
from section_007_T1_ge import T1Measurement
from section_007_T1_fh_with_noise import FH_T1MeasurementWithNoise
from section_007p5_T1_ef_with_fh_noise import EF_T1MeasurementWithNoise
from section_007p5_T1_ge_with_fh_noise import T1MeasurementWithNoise
from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from section_006p5_length_rabi_ge import LengthRabiExperiment
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from section_009_T2R_ge_with_noise import T2RMeasurementWithNoise
from section_010_T2E_ge_with_noise import T2EMeasurementWithNoise
from section_014_dynamical_decoupling_ge import DephasingMeasurement
from section_015_dynamical_decoupling_with_fh_noise_ge import DephasingMeasurementWithEFNoise
from section_016_dynamical_decoupling_with_fh_noise_ge import DephasingMeasurementWithFHNoise
################################################ Run Configurations ####################################################
zero_qubit_drive_gain = False
constant_zeno_pulse = True
adapt_starked_qubit_freq = False
wait_for_res_ring_up = True
n= 3000
save_r = 1                           # how many rounds to save after
signal = 'None'                      # 'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization
save_figs = True                     # save plots for everything as you go along the RR script?
live_plot = False                    # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True                     # fit the data here and save or plot the fits?
save_data_h5 = True                  # save all of the data to h5 files?
verbose = True                       # print everything to the console in real time, good for debugging, bad for memory
debug_mode = True                    # if True, it disables the continuing function of RR if an error pops up in a class -- errors now stop the RR script
thresholding = False                 # use internal QICK threshold for ratio of Binary values on y for rabi/t1/t2r/t2e, or analog avg when false
increase_qubit_reps = False          # if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0       # only has impact if previous line is True
multiply_qubit_reps_by = 2           # only has impact if the line two above is True
increase_qubit_steps_ef = False #if you want to increase the steps for all qubits, set to True, if you only want to set it to true for 1 qubit, see e-f qubit spec section
increase_steps_to_ef = 600
Qs_to_look_at = [2]#0,1,2,3,4,5        # only list the qubits you want to do the RR for
study = 'QZE'
sub_study = 'dephasing_from_higher_energy_levels'
substudy_txt_notes = ('Lets give this an initial test and make sure all of these experiments work well, on qubit 2')
# set which of the following you'd like to run to 'True'
run_flags = {"res_spec_ge": True, "q_spec_ge": True, "rabi_ge": True, "res_spec_ef": True, "res_spec_fh": True,
             "q_spec_ef": True,"q_spec_fh": True, "rabi_ef": True,
             "rabi_fh": False, "t1_ge": False,  "t1_fe": False, "t1_fh": False, "t1_ge_w_noise": False,  "t1_fh_w_noise": False,
             "t2r": True,"t2e": True,"t2r_w_noise": True,"t2e_w_noise": True,
             "dephased": True,"dephased_with_fh_noise": True,"dephased_with_fh_noise": True}
#Folders
if not os.path.exists("/data/QICK_data/run6b/"):
    os.makedirs("/data/QICK_data/run6b/")
if not os.path.exists("/data/QICK_data/run6b/6transmon/"):
    os.makedirs("/data/QICK_data/run6b/6transmon/")
studyFolder = os.path.join("/data/QICK_data/run6b/6transmon/", study)
if not os.path.exists(studyFolder):
    os.makedirs(studyFolder)
subStudyFolder = os.path.join(studyFolder, sub_study)
if not os.path.exists(subStudyFolder):
    os.makedirs(subStudyFolder)

formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
log_file = os.path.join(studyDocumentationFolder, "QZE_script.log")
rr_logger = logging.getLogger("custom_logger_for_rr_only")
rr_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
rr_logger.addHandler(file_handler)
rr_logger.propagate = False

################################################ optimization outputs ##################################################
# Optimization parameters for resonator spectroscopy
res_leng_vals = [4, 14, 6, 10, 6, 7]
res_gain = [1,1,1,0.7,0.8,0.6]
freq_offsets = [0.1, -0.25, -0.2, 0.2, -0.1, -0.1]

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

t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
t2r_data_w_noise = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
t2e_data_w_noise = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
t2dephased_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
t2dephased_data_fh_noise = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
t2dephased_data_fh_noise = create_data_dict(t2e_keys, save_r, list_of_all_qubits)


res_data_ef = create_data_dict(res_keys_ef, save_r, list_of_all_qubits)
qspec_data_ef = create_data_dict(qspec_keys_ef, save_r, list_of_all_qubits)
rabi_data_ef = create_data_dict(rabi_keys_ef, save_r, list_of_all_qubits)

res_data_fh = create_data_dict(res_keys_ef, save_r, list_of_all_qubits)
qspec_data_fh = create_data_dict(qspec_keys_ef, save_r, list_of_all_qubits)
rabi_data_fh = create_data_dict(rabi_keys_ef, save_r, list_of_all_qubits)

t1_data_eg = create_data_dict(t1_keys, save_r, list_of_all_qubits)
t1_data_fg = create_data_dict(t1_keys, save_r, list_of_all_qubits)
t1_data_fe = create_data_dict(t1_keys, save_r, list_of_all_qubits)

t1_data_fh = create_data_dict(t1_keys, save_r, list_of_all_qubits)

t1_data_eg_w_noise = create_data_dict(t1_keys, save_r, list_of_all_qubits)
t1_data_fh_w_noise = create_data_dict(t1_keys, save_r, list_of_all_qubits)

ss_data_gef = create_data_dict(ss_keys_gef, save_r, list_of_all_qubits)


#initialize a simple list to store the qspec values in incase a fit fails
stored_qspec_list = [None] * tot_num_of_qubits
batch_num=0
j = 0
qubit_freqs_ge = np.zeros(6)
qubit_freqs_ef = np.zeros(6)
res_freq_ge = np.zeros(6)
while j < n:
    for QubitIndex in Qs_to_look_at:
        experiment = QICK_experiment(optimizationFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10,
                                     fridge=FRIDGE)
        updated_qubit_gain = 0.05 # lets do a low gain to start so I dont have a broad linewidth for the qubit
        experiment.qubit_cfg['qubit_gain_ge'][
            QubitIndex] = updated_qubit_gain

        # Mask out all other resonators except this one
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
        experiment.readout_cfg['res_gain_ge'] = res_gains

        res_gains_ef = experiment.mask_gain_res(QubitIndex, IndexGain=experiment.readout_cfg['res_gain_ef'][QubitIndex], num_qubits=tot_num_of_qubits)
        experiment.readout_cfg['res_gain_ef'] = res_gains_ef

        res_gains_fh = experiment.mask_gain_res(QubitIndex, IndexGain=experiment.readout_cfg['res_gain_fh'][QubitIndex],
                                                num_qubits=tot_num_of_qubits)
        experiment.readout_cfg['res_gain_fh'] = res_gains_fh

        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        ################################################# g-e Res spec ####################################################
        if run_flags["res_spec_ge"]:
            try:
                res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs,
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
            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"GE Res Spec error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"GE Res Spec error on qubit {QubitIndex}: {e}")
                continue

        ################################################### g-e Qubit spec ##################################################
        if run_flags["q_spec_ge"]:
            try:
                #experiment.qubit_cfg['qubit_gain_ge'][QubitIndex]=1
                q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs, experiment,
                                           live_plot, verbose=False, logger=None, qick_verbose=True, increase_reps=False,
                                           increase_reps_to=500)
                qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq, sys_config_qspec = q_spec.run()

                qubit_freqs_ge[QubitIndex] = qubit_freq
                experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
                print('Qubit ', QubitIndex + 1, ' g-e freq: ', float(qubit_freq))
                del q_spec
            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"GE Q Spec error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"GE Q Spec error on qubit {QubitIndex}: {e}")
                continue

        ###################################################### g-e Rabi ####################################################
        if run_flags["rabi_ge"]:
            try:
                rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs=save_figs,
                                               experiment=experiment, live_plot= live_plot,
                                               increase_qubit_reps=increase_qubit_reps, qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                               multiply_qubit_reps_by=multiply_qubit_reps_by)

                rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_rabi = rabi.run()

                experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
                print('Qubit ', QubitIndex + 1, ' g-e Pi Amp: ', float(pi_amp))

                del rabi
            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"GE rabi error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"GE rabi error on qubit {QubitIndex}: {e}")
                continue

        ################################################# e-f Res spec ####################################################
        if run_flags["res_spec_ef"]:
            try:
                res_specEF = ResonanceSpectroscopyEF(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs,
                                                     experiment)
                res_freqs_ef, freq_pts_ef, freq_center_ef, amps_ef, sys_config_rspec_ef = res_specEF.run()
                experiment.readout_cfg['res_freq_ef'] = res_freqs_ef
                print('Qubit ', QubitIndex + 1, ' e-f res freq: ', res_freqs[QubitIndex])

                del res_specEF
            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"ef Res Spec error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"ef Res Spec error on qubit {QubitIndex}: {e}")
                continue

        ################################################## e-f Qubit spec ##################################################
        if run_flags["q_spec_ef"]:
            try:
                # Qubit 4 needs more steps for e-f spec
                if QubitIndex == 3:
                    increase_qubit_steps_ef = True  # if you want to increase the steps for a qubit, set to True

                ef_q_spec = EFQubitSpectroscopy(QubitIndex, tot_num_of_qubits, list_of_all_qubits, studyDocumentationFolder, j, signal,
                                                save_figs, experiment, live_plot, increase_qubit_steps_ef,
                                                increase_steps_to_ef)
                efqspec_I, efqspec_Q, efqspec_freqs, efqspec_I_fit, efqspec_Q_fit, efqubit_freq, sys_config_qspec_ef = ef_q_spec.run(
                    experiment.soccfg,
                    experiment.soc)
                qubit_freqs_ef[QubitIndex] = efqubit_freq
                experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] = float(efqubit_freq)
                print('Qubit ', QubitIndex + 1, ' e-f Freq: ', float(efqubit_freq))

                del ef_q_spec

            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"ef Q Spec error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"ef Q Spec error on qubit {QubitIndex}: {e}")
                continue

        ###################################################### e-f Rabi ####################################################
        if run_flags["rabi_ef"]:
            try:
                efrabi = EF_AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, list_of_all_qubits, studyDocumentationFolder, j, signal,
                                                    save_figs=save_figs,
                                                    experiment=experiment, live_plot=live_plot,
                                                    increase_qubit_reps=increase_qubit_reps,
                                                    qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                                    multiply_qubit_reps_by=multiply_qubit_reps_by)
                efrabi_I, efrabi_Q, efrabi_gains, efrabi_fit, efpi_amp, sys_config_rabi_ef = efrabi.run(experiment.soccfg,
                                                                                                        experiment.soc)

                experiment.qubit_cfg['pi_ef_amp'][QubitIndex] = float(efpi_amp)
                print('Qubit ', QubitIndex + 1, ' e-f pulse amp: ', float(efpi_amp))

                del efrabi
            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"ef Rabi error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"ef Rabi error on qubit {QubitIndex}: {e}")
                continue

        ############################################# f-h Res spec ############################################
        if run_flags["res_spec_fh"]:
            try:
                res_specFH = ResonanceSpectroscopyFH(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs,
                                                     experiment)
                res_freqs_fh, freq_pts_fh, freq_center_fh, amps_fh, sys_config_rspec_fh = res_specFH.run()
                experiment.readout_cfg['res_freq_fh'] = res_freqs
                print('Qubit ', QubitIndex + 1, ' f-h res freq: ', res_freqs[QubitIndex])

                del res_specFH
            except Exception as e:
            if debug_mode:
                raise e
            rr_logger.exception(f"fh Res Spec error on qubit {QubitIndex}: {e}")
            if verbose:
                print(f"fh Res Spec error on qubit {QubitIndex}: {e}")
            continue

        ######################################## f-h Qubit spec ###############################################
        if run_flags["q_spec_fh"]:
            try:
                # Qubit 4 needs more steps for e-f spec
                experiment.qubit_cfg['pi_fh_amp'] = experiment.qubit_cfg['pi_fh_amp'][QubitIndex]
                if QubitIndex == 3:
                    increase_qubit_steps_ef = True  # if you want to increase the steps for a qubit, set to True

                fh_q_spec = FHQubitSpectroscopy(QubitIndex, tot_num_of_qubits, list_of_all_qubits, studyDocumentationFolder, j,
                                                signal,
                                                save_figs, experiment, live_plot, increase_qubit_steps_ef,
                                                increase_steps_to_ef)
                fhqspec_I, fhqspec_Q, fhqspec_freqs, fhqspec_I_fit, fhqspec_Q_fit, fhqubit_freq, sys_config_qspec_fh = fh_q_spec.run(
                    experiment.soccfg,
                    experiment.soc)
                experiment.qubit_cfg['qubit_freq_fh'][QubitIndex] = float(fhqubit_freq)
                print('Qubit ', QubitIndex + 1, ' f-h Freq: ', float(fhqubit_freq))

                del fh_q_spec
            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"fh Q Spec error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"fh Q Spec error on qubit {QubitIndex}: {e}")
                continue

        ###################################################### f-h Rabi ####################################################
        if run_flags["rabi_fh"]:
            try:
                fhrabi = FH_AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, list_of_all_qubits, studyDocumentationFolder,
                                                    j, signal,
                                                    save_figs=save_figs,
                                                    experiment=experiment, live_plot=live_plot,
                                                    increase_qubit_reps=increase_qubit_reps,
                                                    qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                                    multiply_qubit_reps_by=multiply_qubit_reps_by)
                fhrabi_I, fhrabi_Q, fhrabi_gains, fhrabi_fit, fhpi_amp, sys_config_rabi_fh = fhrabi.run(experiment.soccfg,
                                                                                                        experiment.soc)

                experiment.qubit_cfg['pi_fh_amp'][QubitIndex] = float(fhpi_amp)
                print('Qubit ', QubitIndex + 1, ' f-h pulse amp: ', float(fhpi_amp))

                del fhrabi
            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"fh rabi error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"fh rabi error on qubit {QubitIndex}: {e}")
                continue

        ###################################################### g-e T1 ####################################################
        if run_flags["t1_ge"]:
            try:
                t1_ge = T1Measurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
                                      experiment=experiment,
                                      live_plot=live_plot, fit_data=fit_data,
                                      increase_qubit_reps=increase_qubit_reps,
                                      qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                      multiply_qubit_reps_by=multiply_qubit_reps_by)
                t1_est_ge, t1_err_ge, t1_I_ge, t1_Q_ge, t1_delay_times_ge, q1_fit_exponential_ge, sys_config_t1_ge = t1_ge.run(
                    thresholding=False)

                print('Qubit ', QubitIndex + 1, ' g-e T1: ', str(t1_est_ge))

                del t1_ge
            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"GE t1 error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"GE t1 error on qubit {QubitIndex}: {e}")
                continue

        ################################################### h-f T1 w noise #############################################
        if run_flags["t1_hf"]:
            try:
                t1_fh = FH_T1Measurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
                                         experiment=experiment,
                                         live_plot=live_plot, fit_data=fit_data,
                                         increase_qubit_reps=increase_qubit_reps,
                                         qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                         multiply_qubit_reps_by=multiply_qubit_reps_by, expt_name='T1_fh')
                t1_est_fh, t1_err_fh, t1_I_fh, t1_Q_fh, t1_delay_times_fh, q1_fit_exponential_fh, sys_config_t1_fh = t1_fh.run(
                    thresholding=False)

                print('Qubit ', QubitIndex + 1, ' f-e T1: ', str(t1_est_fh))

                del t1_fh
            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"fh t1 error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"fh t1 error on qubit {QubitIndex}: {e}")
                continue
        ###################################################### T2R #####################################################
        if run_flags["t2r"]:
            try:
                t2r = T2RMeasurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
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
                    rr_logger.exception(f't2r error on qubit {QubitIndex}: {e}')
                    if verbose: print(f't2r error on qubit {QubitIndex}: {e}')
                    continue  # skip the rest of this qubit

        ##################################################### T2E ######################################################
        if run_flags["t2e"]:
            try:
                t2e = T2EMeasurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
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
                    rr_logger.exception(f't2e error on qubit {QubitIndex}: {e}')
                    if verbose: print(f't2e error on qubit {QubitIndex}: {e}')
                    continue  # skip the rest of this qubit

        ################################################## Dephasing ###################################################
        if run_flags["dephased"]:
            try:
                dephase = DephasingMeasurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
                                     experiment=experiment, live_plot=live_plot, fit_data=fit_data,
                                     increase_qubit_reps=increase_qubit_reps,
                                     qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                     multiply_qubit_reps_by=multiply_qubit_reps_by,
                                     verbose=verbose, logger=rr_logger)
                (t2dephased_est, t2dephased_err, t2dephased_I, t2dephased_Q, t2dephased_delay_times,
                 fit_t2dephased, sys_config_t2dephased) = dephase.run(thresholding=thresholding)
                del dephase

            except Exception as e:
                if debug_mode:
                    raise e  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'DD error on qubit {QubitIndex}: {e}')
                    if verbose: print(f'DD error on qubit {QubitIndex}: {e}')
                    continue  # skip the rest of this qubit

        ###################################################### g-e T1 with fh noise ##########################################
        if run_flags["t1_ge_w_noise"]:
            try:
                t1_ge_w_noise = T1MeasurementWithNoise(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
                                      experiment=experiment,
                                      live_plot=live_plot, fit_data=fit_data,
                                      increase_qubit_reps=increase_qubit_reps,
                                      qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                      multiply_qubit_reps_by=multiply_qubit_reps_by)
                t1_est_ge_w_noise, t1_err_ge_w_noise, t1_I_ge_w_noise, t1_Q_ge_w_noise, t1_delay_times_ge_w_noise, q1_fit_exponential_ge_w_noise, sys_config_t1_ge_w_noise = t1_ge_w_noise.run(
                    thresholding=False, noise_type='fh')

                print('Qubit ', QubitIndex + 1, ' g-e T1: ', str(t1_est_ge_w_noise))

                del t1_ge_w_noise
            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"ge t1 w fh noise error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"ge t1 w fh noise error on qubit {QubitIndex}: {e}")
                continue

        ###################################################### f-e T1 with fh noise #################################
        if run_flags["t1_fh_w_noise"]:
            try:
                t1_fh_w_noise = FH_T1MeasurementWithNoise(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                         save_figs,
                                         experiment=experiment,
                                         live_plot=live_plot, fit_data=fit_data,
                                         increase_qubit_reps=increase_qubit_reps,
                                         qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                         multiply_qubit_reps_by=multiply_qubit_reps_by, expt_name='T1_fh')
                (t1_est_fh_w_noise, t1_err_fh_w_noise, t1_I_fh_w_noise, t1_Q_fh_w_noise,
                 t1_delay_times_fh_w_noise, q1_fit_exponential_fh_w_noise, sys_config_t1_fh_w_noise) = t1_fh_w_noise.run(
                    thresholding=False)

                print('Qubit ', QubitIndex + 1, ' f-h T1: ', str(t1_est_fh_w_noise))

                del t1_fh_w_noise
            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"fh t1 w fh noise error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"fh t1 w fh noise error on qubit {QubitIndex}: {e}")
                continue

        ###################################################### T2R with fh noise ####################################
        if run_flags["t2r_w_noise"]:
            try:
                t2r_w_noise = T2RMeasurementWithNoise(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
                                     experiment=experiment, live_plot=live_plot, fit_data=fit_data,
                                     increase_qubit_reps=increase_qubit_reps,
                                     qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                     multiply_qubit_reps_by=multiply_qubit_reps_by,
                                     verbose=verbose, logger=rr_logger)
                (t2r_est_w_noise, t2r_err_w_noise, t2r_I_w_noise, t2r_Q_w_noise, t2r_delay_times_w_noise,
                 fit_ramsey_w_noise, sys_config_t2r_w_noise) = t2r_w_noise.run(
                    thresholding=thresholding, noise_type='fh')
                del t2r_w_noise

            except Exception as e:
                if debug_mode:
                    raise e  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f't2r w fh noise error on qubit {QubitIndex}: {e}')
                    if verbose: print(f't2r w fh noise error on qubit {QubitIndex}: {e}')
                    continue  # skip the rest of this qubit

        ##################################################### T2E with fh noise #########################################
        if run_flags["t2e_w_noise"]:
            try:
                t2e_w_noise = T2EMeasurementWithNoise(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
                                     experiment=experiment, live_plot=live_plot, fit_data=fit_data,
                                     increase_qubit_reps=increase_qubit_reps,
                                     qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                     multiply_qubit_reps_by=multiply_qubit_reps_by,
                                     verbose=verbose, logger=rr_logger)
                (t2e_est_w_noise, t2e_err_w_noise, t2e_I_w_noise, t2e_Q_w_noise, t2e_delay_times_w_noise,
                 fit_t2e_w_noise, sys_config_t2e_w_noise) = t2e_w_noise.run(thresholding=thresholding, noise_type='fh')
                del t2e_w_noise

            except Exception as e:
                if debug_mode:
                    raise e  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f't2e w fh noise error on qubit {QubitIndex}: {e}')
                    if verbose: print(f't2e w fh noise error on qubit {QubitIndex}: {e}')
                    continue  # skip the rest of this qubit

        ############################################## Dephasing with fh noise ###############################################
        if run_flags["dephased_with_fh_noise"]:
            try:
                dephase_fh_noise = DephasingMeasurementWithFHNoise(QubitIndex, tot_num_of_qubits,
                                                                   studyDocumentationFolder, j, signal,
                                                                   save_figs,
                                                                   experiment=experiment, live_plot=live_plot,
                                                                   fit_data=fit_data,
                                                                   increase_qubit_reps=increase_qubit_reps,
                                                                   qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                                                   multiply_qubit_reps_by=multiply_qubit_reps_by,
                                                                   verbose=verbose, logger=rr_logger)
                (t2dephased_fh_noise_est, t2dephased_fh_noise_err, t2dephased_fh_noise_I, t2dephased_fh_noise_Q,
                 t2dephased_fh_noise_delay_times,
                 fit_t2dephased_fh_noise, sys_config_t2dephased_fh_noise) = dephase_fh_noise.run(
                    thresholding=thresholding, gain=0.02, freq_offset=0)
                del dephase_fh_noise

            except Exception as e:
                if debug_mode:
                    raise e  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'DD w fh noise error on qubit {QubitIndex}: {e}')
                    if verbose: print(f'DD w fh noise error on qubit {QubitIndex}: {e}')
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
                res_data_ef[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                res_data_ef[QubitIndex]['freq_pts'][j - batch_num * save_r - 1] = freq_pts_ef
                res_data_ef[QubitIndex]['freq_center'][j - batch_num * save_r - 1] = freq_center_ef
                res_data_ef[QubitIndex]['Amps'][j - batch_num * save_r - 1] = amps_ef
                res_data_ef[QubitIndex]['Found Freqs'][j - batch_num * save_r - 1] = res_freqs_ef
                res_data_ef[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                res_data_ef[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                res_data_ef[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                res_data_ef[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rspec_ef

            # ---------------------Collect f-h Res Spec Results----------------
            if run_flags["res_spec_fh"]:
                res_data_fh[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                res_data_fh[QubitIndex]['freq_pts'][j - batch_num * save_r - 1] = freq_pts_fh
                res_data_fh[QubitIndex]['freq_center'][j - batch_num * save_r - 1] = freq_center_fh
                res_data_fh[QubitIndex]['Amps'][j - batch_num * save_r - 1] = amps_fh
                res_data_fh[QubitIndex]['Found Freqs'][j - batch_num * save_r - 1] = res_freqs_fh
                res_data_fh[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                res_data_fh[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                res_data_fh[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                res_data_fh[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rspec_fh

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
                qspec_data_ef[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                qspec_data_ef[QubitIndex]['I'][j - batch_num * save_r - 1] = efqspec_I
                qspec_data_ef[QubitIndex]['Q'][j - batch_num * save_r - 1] = efqspec_Q
                qspec_data_ef[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = efqspec_freqs
                qspec_data_ef[QubitIndex]['I Fit'][j - batch_num * save_r - 1] = efqspec_I_fit
                qspec_data_ef[QubitIndex]['Q Fit'][j - batch_num * save_r - 1] = efqspec_Q_fit
                qspec_data_ef[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                qspec_data_ef[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                # qspec_data[QubitIndex]['Recycled QFreq'][j - batch_num * save_r - 1] = recycled_qfreq
                qspec_data_ef[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                qspec_data_ef[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_qspec_ef

            # ---------------------Collect f-h QSpec Results----------------
            if run_flags["q_spec_fh"]:
                qspec_data_fh[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                qspec_data_fh[QubitIndex]['I'][j - batch_num * save_r - 1] = fhqspec_I
                qspec_data_fh[QubitIndex]['Q'][j - batch_num * save_r - 1] = fhqspec_Q
                qspec_data_fh[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = fhqspec_freqs
                qspec_data_fh[QubitIndex]['I Fit'][j - batch_num * save_r - 1] = fhqspec_I_fit
                qspec_data_fh[QubitIndex]['Q Fit'][j - batch_num * save_r - 1] = fhqspec_Q_fit
                qspec_data_fh[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                qspec_data_fh[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                # qspec_data[QubitIndex]['Recycled QFreq'][j - batch_num * save_r - 1] = recycled_qfreq
                qspec_data_fh[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                qspec_data_fh[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_qspec_fh

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
                rabi_data_ef[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                rabi_data_ef[QubitIndex]['I'][j - batch_num * save_r - 1] = efrabi_I
                rabi_data_ef[QubitIndex]['Q'][j - batch_num * save_r - 1] = efrabi_Q
                rabi_data_ef[QubitIndex]['Gains'][j - batch_num * save_r - 1] = efrabi_gains
                rabi_data_ef[QubitIndex]['Fit'][j - batch_num * save_r - 1] = efrabi_fit
                rabi_data_ef[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                rabi_data_ef[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                rabi_data_ef[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                rabi_data_ef[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rabi_ef

            # ---------------------Collect f-h Rabi Results----------------
            if run_flags["rabi_fh"]:
                rabi_data_fh[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                rabi_data_fh[QubitIndex]['I'][j - batch_num * save_r - 1] = fhrabi_I
                rabi_data_fh[QubitIndex]['Q'][j - batch_num * save_r - 1] = fhrabi_Q
                rabi_data_fh[QubitIndex]['Gains'][j - batch_num * save_r - 1] = fhrabi_gains
                rabi_data_fh[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fhrabi_fit
                rabi_data_fh[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                rabi_data_fh[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                rabi_data_fh[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                rabi_data_fh[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rabi_fh

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

            # ---------------------Collect f-h T1 Results----------------
            if run_flags["t1_fh"]:
                t1_data_fh[QubitIndex]['T1'][j - batch_num * save_r - 1] = t1_est_fh
                t1_data_fh[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t1_err_fh
                t1_data_fh[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t1_data_fh[QubitIndex]['I'][j - batch_num * save_r - 1] = t1_I_fh
                t1_data_fh[QubitIndex]['Q'][j - batch_num * save_r - 1] = t1_Q_fh
                t1_data_fh[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t1_delay_times_fh
                t1_data_fh[QubitIndex]['Fit'][j - batch_num * save_r - 1] = q1_fit_exponential_fh
                t1_data_fh[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t1_data_fh[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t1_data_fh[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                t1_data_fh[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t1_fh

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

            # ---------------------Collect Dephased Results----------------
            if run_flags["dephased"]:
                t2dephased_data[QubitIndex]['T2E'][j - batch_num * save_r - 1] = t2dephased_est
                t2dephased_data[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t2dephased_err
                t2dephased_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t2dephased_data[QubitIndex]['I'][j - batch_num * save_r - 1] = t2dephased_I
                t2dephased_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = t2dephased_Q
                t2dephased_data[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t2dephased_delay_times
                t2dephased_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fit_t2dephased
                t2dephased_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t2dephased_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t2dephased_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                t2dephased_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2dephased

            # ---------------------Collect e-g T1 Results w noise----------------
            if run_flags["t1_ge_w_noise"]:
                t1_data_eg_w_noise[QubitIndex]['T1'][j - batch_num * save_r - 1] = t1_est_ge_w_noise
                t1_data_eg_w_noise[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t1_err_ge_w_noise
                t1_data_eg_w_noise[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t1_data_eg_w_noise[QubitIndex]['I'][j - batch_num * save_r - 1] = t1_I_ge_w_noise
                t1_data_eg_w_noise[QubitIndex]['Q'][j - batch_num * save_r - 1] = t1_Q_ge_w_noise
                t1_data_eg_w_noise[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t1_delay_times_ge_w_noise
                t1_data_eg_w_noise[QubitIndex]['Fit'][j - batch_num * save_r - 1] = q1_fit_exponential_ge_w_noise
                t1_data_eg_w_noise[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t1_data_eg_w_noise[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t1_data_eg_w_noise[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                t1_data_eg_w_noise[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t1_ge_w_noise

            # ---------------------Collect f-h T1 Results w noise----------------
            if run_flags["t1_fh_w_noise"]:
                t1_data_fh_w_noise[QubitIndex]['T1'][j - batch_num * save_r - 1] = t1_est_fh_w_noise
                t1_data_fh_w_noise[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t1_err_fh_w_noise
                t1_data_fh_w_noise[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t1_data_fh_w_noise[QubitIndex]['I'][j - batch_num * save_r - 1] = t1_I_fh_w_noise
                t1_data_fh_w_noise[QubitIndex]['Q'][j - batch_num * save_r - 1] = t1_Q_fh_w_noise
                t1_data_fh_w_noise[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t1_delay_times_fh_w_noise
                t1_data_fh_w_noise[QubitIndex]['Fit'][j - batch_num * save_r - 1] = q1_fit_exponential_fh_w_noise
                t1_data_fh_w_noise[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t1_data_fh_w_noise[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t1_data_fh_w_noise[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                t1_data_fh_w_noise[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t1_fh_w_noise

            # ---------------------Collect T2 with noise Results----------------
            if run_flags["t2r_w_noise"]:
                t2r_data_w_noise[QubitIndex]['T2'][j - batch_num * save_r - 1] = t2r_est_w_noise
                t2r_data_w_noise[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t2r_err_w_noise
                t2r_data_w_noise[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t2r_data_w_noise[QubitIndex]['I'][j - batch_num * save_r - 1] = t2r_I_w_noise
                t2r_data_w_noise[QubitIndex]['Q'][j - batch_num * save_r - 1] = t2r_Q_w_noise
                t2r_data_w_noise[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t2r_delay_times_w_noise
                t2r_data_w_noise[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fit_ramsey_w_noise
                t2r_data_w_noise[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t2r_data_w_noise[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t2r_data_w_noise[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                t2r_data_w_noise[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2r_w_noise

            # ---------------------Collect T2E with noise Results----------------
            if run_flags["t2e_w_noise"]:
                t2e_data_w_noise[QubitIndex]['T2E'][j - batch_num * save_r - 1] = t2e_est_w_noise
                t2e_data_w_noise[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t2e_err_w_noise
                t2e_data_w_noise[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t2e_data_w_noise[QubitIndex]['I'][j - batch_num * save_r - 1] = t2e_I_w_noise
                t2e_data_w_noise[QubitIndex]['Q'][j - batch_num * save_r - 1] = t2e_Q_w_noise
                t2e_data_w_noise[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t2e_delay_times_w_noise
                t2e_data_w_noise[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fit_t2e_w_noise
                t2e_data_w_noise[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t2e_data_w_noise[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t2e_data_w_noise[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                t2e_data_w_noise[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2e_w_noise

            # ---------------------Collect Dephased with fh noise Results----------------
            if run_flags["dephased_with_fh_noise"]:
                t2dephased_data_fh_noise[QubitIndex]['T2E'][j - batch_num * save_r - 1] = t2dephased_fh_noise_est
                t2dephased_data_fh_noise[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t2dephased_fh_noise_err
                t2dephased_data_fh_noise[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t2dephased_data_fh_noise[QubitIndex]['I'][j - batch_num * save_r - 1] = t2dephased_fh_noise_I
                t2dephased_data_fh_noise[QubitIndex]['Q'][j - batch_num * save_r - 1] = t2dephased_fh_noise_Q
                t2dephased_data_fh_noise[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t2dephased_fh_noise_delay_times
                t2dephased_data_fh_noise[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fit_t2dephased_fh_noise
                t2dephased_data_fh_noise[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t2dephased_data_fh_noise[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t2dephased_data_fh_noise[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                t2dephased_data_fh_noise[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2dephased_fh_noise

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

            # --------------------------save f-h QSpec-----------------------
            if run_flags["q_spec_fh"]:
                saver_qspec = Data_H5(subStudyDataFolder, qspec_data_fh, batch_num, save_r)
                saver_qspec.save_to_h5('QSpec_ge')
                del saver_qspec
                del qspec_data_fh

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

            # --------------------------save f-h Rabi-----------------------
            if run_flags["rabi_fh"]:
                saver_rabi = Data_H5(subStudyDataFolder, rabi_data_fh, batch_num, save_r)
                saver_rabi.save_to_h5('Rabi_fh')
                del saver_rabi
                del rabi_data_fh

            # --------------------------save t1 e-g -----------------------
            if run_flags["t1_ge"]:
                saver_t1 = Data_H5(subStudyDataFolder, t1_data_eg, batch_num, save_r)
                saver_t1.save_to_h5('T1_ge')
                del saver_t1
                del t1_data_eg

            # --------------------------save t1 f-h -----------------------
            if run_flags["t1_fh"]:
                saver_t1 = Data_H5(subStudyDataFolder, t1_data_fh, batch_num, save_r)
                saver_t1.save_to_h5('T1_fh')
                del saver_t1
                del t1_data_fh

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

            # --------------------------save t2dephased-----------------------
            if run_flags["dephased"]:
                saver_t2dephased = Data_H5(subStudyDataFolder, t2dephased_data, batch_num, save_r)
                saver_t2dephased.save_to_h5('DD_ge')
                del saver_t2dephased
                del t2dephased_data

            # --------------------------save t1 e-g w noise-----------------------
            if run_flags["t1_ge_w_noise"]:
                saver_t1 = Data_H5(subStudyDataFolder, t1_data_eg_w_noise, batch_num, save_r)
                saver_t1.save_to_h5('T1_ge_w_noise')
                del saver_t1
                del t1_data_eg_w_noise

            # --------------------------save t1 f-e w noise-----------------------
            if run_flags["t1_fh_w_noise"]:
                saver_t1 = Data_H5(subStudyDataFolder, t1_data_fh_w_noise, batch_num, save_r)
                saver_t1.save_to_h5('t1_fh_w_noise')
                del saver_t1
                del t1_data_fh_w_noise

            # --------------------------save t2r w noise-----------------------
            if run_flags["t2r_w_noise"]:
                saver_t2r = Data_H5(subStudyDataFolder, t2r_data_w_noise, batch_num, save_r)
                saver_t2r.save_to_h5('T2_ge_w_noise')
                del saver_t2r
                del t2r_data_w_noise

            # --------------------------save t2e w noise-----------------------
            if run_flags["t2e_w_noise"]:
                saver_t2e = Data_H5(subStudyDataFolder, t2e_data_w_noise, batch_num, save_r)
                saver_t2e.save_to_h5('T2E_ge_w_noise')
                del saver_t2e
                del t2e_data_w_noise

            # --------------------------save t2dephased with noise-----------------------
            if run_flags["dephased_with_fh_noise"]:
                saver_t2dephased_fh_noise = Data_H5(subStudyDataFolder, t2dephased_data_fh_noise, batch_num, save_r)
                saver_t2dephased_fh_noise.save_to_h5('DD_ge_fh_noise')
                del saver_t2dephased_fh_noise
                del t2dephased_data_fh_noise


            # reset all dictionaries to none for safety
            res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
            res_data_ef = create_data_dict(res_keys_ef, save_r, list_of_all_qubits)
            res_data_fh = create_data_dict(res_keys, save_r, list_of_all_qubits)

            qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
            qspec_data_ef = create_data_dict(qspec_keys_ef, save_r, list_of_all_qubits)
            qspec_data_fh = create_data_dict(qspec_keys, save_r, list_of_all_qubits)

            rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
            rabi_data_ef = create_data_dict(rabi_keys_ef, save_r, list_of_all_qubits)
            rabi_data_fh = create_data_dict(rabi_keys_ef, save_r, list_of_all_qubits)

            t1_data_eg = create_data_dict(t1_keys, save_r, list_of_all_qubits)
            t1_data_fg = create_data_dict(t1_keys, save_r, list_of_all_qubits)
            t1_data_fe = create_data_dict(t1_keys, save_r, list_of_all_qubits)
            t1_data_fh = create_data_dict(t1_keys, save_r, list_of_all_qubits)

            t1_data_eg_w_noise = create_data_dict(t1_keys, save_r, list_of_all_qubits)
            t1_data_fh_w_noise = create_data_dict(t1_keys, save_r, list_of_all_qubits)


            t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
            t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
            t2r_data_w_noise = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
            t2e_data_w_noise = create_data_dict(t2e_keys, save_r, list_of_all_qubits)

            t2dephased_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
            t2dephased_data_fh_noise = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
            t2dephased_data_fh_noise = create_data_dict(t2e_keys, save_r, list_of_all_qubits)

    j+=1