import copy
import sys
import os
import numpy as np

# from tprocv2_demos.qick_tprocv2_experiments_mux.long_qubit_spectroscopy import fh_config

np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
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
from section_004_qubit_spec_fh_V2 import FHQubitSpectroscopy
from section_006_amp_rabi_ef import EF_AmplitudeRabiExperiment
from section_006_amp_fh import FH_AmplitudeRabiExperiment
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_011_qubit_temperatures_efRabipt3 import Temps_EFAmpRabiExperiment
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_005_single_shot_ef import SingleShot_ef
# from section_005_single_shot_gef import SingleShot_ef # Old way: Fix for example Unmask
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from system_config import QICK_experiment
from section_003_punch_out_ge_mux import PunchOut
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from test_active_reset import Active_Reset_test
from analysis_021_plot_allRR_noqick import PlotRR_noQick
from analysis_020_gef_ssf_fstate_plots import GEF_SSF_ANALYSIS
################################################ Run Configurations ####################################################
st = time.time()
#
n= 1000000
pre_optimize = False
freq_offset_steps = 10
ssf_avgs_per_opt_pt = 5
save_r = 1                           # how many rounds to save after
signal = 'None'                      # 'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization
save_figs = True                     # save plots for everything as you go along the RR script?
live_plot = False                     # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True                     # fit the data here and save or plot the fits?
save_data_h5 = True                  # save all of the data to h5 files?
verbose = False                      # print everything to the console in real time, good for debugging, bad for memory
qick_verbose = True                 # qick verbose prints the progress bar for each qick experiment as it is happening (the red bar that fills out as more experiment rounds/reps are being done)
debug_mode = False                   # if True, it disables the continuing function of RR if an error pops up in a class -- errors now stop the RR script
thresholding = False                 # use internal QICK threshold for ratio of Binary values on y for rabi/t1/t2r/t2e, or analog avg when false
increase_qubit_reps = False          # if you want to increase the reps for a qubit, set to True
unmask = True                          # Do you want to use the unmasking feature to increase resonator gain?
qubit_to_increase_reps_for = 0       # only has impact if previous line is True
multiply_qubit_reps_by = 2           # only has impact if the line two above is True

Qs_to_look_at = [0,1,2,3,5]#[0,1,2,5]#,1,2,3,4,5]       # only list the qubits you want to do the RR for

#Data saving info
run_name = 'run7'
device_name = '6transmon'
substudy_txt_notes = ('FH Spec Study. We will move the source about 50cm away from the fridge')#('Active Reset Test')##('round robin with relax delays of 1000us for T1 and T2 measurements.')#('Active Reset Test')#('Normal Round Robin during cooldown, now everything works properly, set debug to false to run '
                      # 'overFalsenight and running in terminal with repeater script')

# set which of the following you'd like to run to 'True'
run_flags = {"tof": False, "res_spec": True, "q_spec": False, "ss": False, "rabi":False, "ss_gef": False, "test_act":False, "fh_rabi":False,
             "t1": False, "t2r": False, "t2e": False, "ef_res_spec":False, "ef_q_spec": False, "fh_q_spec":False, "rabi_pop_meas": False, "ef_Rabi":False, "ef_ss": False}
# run_flags = {"tof": False, "res_spec": False, "q_spec": False, "ss": False, "rabi": False, "ss_gef": False, "test_act":False,
#              "t1": False, "t2r": False, "t2e": False, "ef_res_spec":False, "ef_q_spec": True, "fh_q_spec":True, "rabi_pop_meas": False, "ef_Rabi":False, "ef_ss": False}

# optimization outputs from qick board, unmasking set to true
res_leng_vals = [5.0,5.5,5.5,6.0,6.0,6.0]#[7.0, 5.1, 5.1, 5.6, 5.6, 5.6] # all updated on 7/29/2025
res_gain = [0.95,0.9,0.95,0.55,0.55,0.95]#[0.8, 0.9, 0.95, 0.51, 0.61, 0.95] # all updated on
# 7/29/2025 except R5, we need to debug res spec for that resonator
freq_offsets = [-0.2143, 0, -0.16, -0.16, -0.16, -0.16,]#[0.1190, 0.0238, -0.1190, 0.2143, -0.0714, 0.0238] # # all updated on 7/29/2025 except R5, we need to debug res spec for that resonator

qubit_freqs_ef = [None]*6
# increase_qubit_steps_ef = False #if you want to increase the steps for all qubits, set to True, if you only want to set it to true for 1 qubit, see e-f qubit spec section
increase_steps_to_ef = 600
ef_res_sample_number = 1
number_of_qubits = 6
figure_quality = 200
################################################ Data Saving Setup ##################################################
#Folders
study = 'warm_up_res_data'
sub_study = 'FH_Spec_Q2_Source_distance'#junkyard'#'AB_data_relax_delays_1000us'#'Active_Reset_Test'#'temperature_sweep'#'AB_tests_data'
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
rr_logger.propagate = False  #dont propagate logs from underlying qick package

####################################################### RR #############################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num','Recycled QFreq',
              'Exp Config', 'Syst Config']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
ss_ef_keys = ['Fidelity', 'Angle', 'Dates', 'I_e', 'Q_e', 'I_f', 'Q_f', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t2r_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
t2e_keys = ['T2E', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
rabi_keys_ef_Qtemps = ['Dates', 'Qfreq_ge', 'I1', 'Q1', 'Gains1', 'Fit1', 'I2', 'Q2', 'Gains2', 'Fit2', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
ss_keys_gef = ['Fidelity', 'Angle_ef', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'I_f', 'Q_f', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
act_keys = [ 'actI', 'actQ','noactI', 'noactQ', 'Syst Config']
#initialize a simple list to store the qspec values in incase a fit fails
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
ef_ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)

ef_res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
ef_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
ef_rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
fh_rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
fh_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
rabi_data_ef_Qtemps = create_data_dict(rabi_keys_ef_Qtemps, save_r, list_of_all_qubits)
ss_data_gef = create_data_dict(ss_keys_gef, save_r, list_of_all_qubits)
act_data = create_data_dict(act_keys, save_r, list_of_all_qubits)

if pre_optimize:
    ################################################## Simple Optimization ###############################################

    def sweep_frequency_offset(experiment_opt, QubitIndex_opt, offset_values, n_loops=10, number_of_qubits=6,
                               outerFolder="", studyDocumentationFolder_opt="", optimizationFolder_opt="", j=0):
        baseline_freq = experiment_opt.readout_cfg['res_freq_ge'][QubitIndex]
        ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
        ssf_dict = {}
        for offset in offset_values:
            fids = []
            # repeat n times for each offset
            for i in range(n_loops):
                exp_copy = copy.deepcopy(experiment_opt)  # python is python, doesnt overwrite things properly

                res_freqs = exp_copy.readout_cfg['res_freq_ge']
                res_freqs[QubitIndex] = baseline_freq + offset
                exp_copy.readout_cfg['res_freq_ge'] = res_freqs

                ss = SingleShot(QubitIndex, number_of_qubits, outerFolder, j, save_figs, exp_copy)
                fid, angle, iq_list_g, iq_list_e, ss_config = ss.run()
                fids.append(fid)
                del exp_copy

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
                ss_data[QubitIndex]['Syst Config'][0] = ss_config

                saver_ss = Data_H5(optimizationFolder_opt, ss_data, 0, save_r)
                saver_ss.save_to_h5('ss_ge')
                del saver_ss
                del ss_data

                ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)

            # find avg ssf
            avg_fid = np.mean(fids)
            ssf_dict[offset] = avg_fid
            if verbose:
                print(f"Offset: {offset} -> Average g-e SSF: {avg_fid:.4f}")
        import matplotlib.pyplot as plt
        plt.figure()
        offsets_sorted = sorted(ssf_dict.keys())
        ssf_values = [ssf_dict[offset] for offset in offsets_sorted]
        plt.plot(offsets_sorted, ssf_values, marker='o')
        plt.xlabel('Frequency Offset')
        plt.ylabel('Average g-e SSF')
        plt.title(f'g-e SSF vs Frequency Offset for Qubit {QubitIndex + 1}')
        plt.grid(True)
        if outerFolder:
            os.makedirs(outerFolder, exist_ok=True)
            plot_path = os.path.join(studyDocumentationFolder, f"ge_SSF_vs_offset_Q{QubitIndex + 1}.png")
            plt.savefig(plot_path)
            if verbose:
                print(f"Plot saved to {plot_path}")
        plt.close()

        # Determine the offset value that yielded the best (highest) average SSF.
        optimal_offset = max(ssf_dict, key=ssf_dict.get)
        if verbose:
            print(
                f"Optimal frequency offset for Qubit {QubitIndex}: {optimal_offset} (Avg g-e SSF: {ssf_dict[optimal_offset]:.4f})")

        return optimal_offset, ssf_dict
    ################################################## Simple Optimization ###############################################
    for Q in Qs_to_look_at:
        try:
            experiment = QICK_experiment(
                optimizationFolder,
                DAC_attenuator1=15,
                DAC_attenuator2=10,
                ADC_attenuator=17,
                fridge=FRIDGE
            )
            # Set resonator configuration for this qubit
            res_gains = experiment.mask_gain_res(Q, IndexGain=res_gain[Q], num_qubits=tot_num_of_qubits)
            experiment.readout_cfg['res_gain_ge'] = res_gains
            experiment.readout_cfg['res_length'] = res_leng_vals[Q]

            res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
            res_spec = ResonanceSpectroscopy(Q, tot_num_of_qubits, optimizationFolder, 0,
                                             save_figs=True, experiment=experiment, verbose=verbose,
                                             logger=rr_logger, qick_verbose=True)
            res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
            experiment.readout_cfg['res_freq_ge'] = res_freqs
            rr_logger.info(f"g-e ResSpec for qubit {Q}: {res_freqs}")

            res_data[Q]['Dates'][0] = (
                time.mktime(datetime.datetime.now().timetuple()))
            res_data[Q]['freq_pts'][0] = freq_pts
            res_data[Q]['freq_center'][0] = freq_center
            res_data[Q]['Amps'][0] = amps
            res_data[Q]['Found Freqs'][0] = res_freqs
            res_data[Q]['Batch Num'][0] = 0
            res_data[Q]['Exp Config'][0] = expt_cfg
            res_data[Q]['Syst Config'][0] = sys_config_rspec

            saver_res = Data_H5(optimizationFolder, res_data, 0, save_r)  # save
            saver_res.save_to_h5('res_ge')
            del saver_res
            del res_data
            res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)  # initialize again to a blank for saftey
            del res_spec
        except Exception as e:
            rr_logger.exception(f"g-e ResSpec error on qubit {Q} : {e}")
            continue
        ############ g-e Qubit Spec ##############
        qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
        try:
            q_spec = QubitSpectroscopy(Q, tot_num_of_qubits, optimizationFolder, 0,
                                       signal, plot_fit=False, save_figs=True, experiment=experiment,
                                       live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                       qick_verbose=True, increase_reps=True, increase_reps_to=500)
            (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit,
             qubit_freq, sys_config_qspec) = q_spec.run()
            experiment.qubit_cfg['qubit_freq_ge'][Q] = float(qubit_freq)
            stored_qspec = float(qubit_freq)
            rr_logger.info(f"Tune-up: g-e Qubit {Q + 1} frequency: {stored_qspec}")
            del q_spec

            qspec_data[Q]['Dates'][0] = (
                time.mktime(datetime.datetime.now().timetuple()))
            qspec_data[Q]['I'][0] = qspec_I
            qspec_data[Q]['Q'][0] = qspec_Q
            qspec_data[Q]['Frequencies'][0] = qspec_freqs
            qspec_data[Q]['I Fit'][0] = qspec_I_fit
            qspec_data[Q]['Q Fit'][0] = qspec_Q_fit
            qspec_data[Q]['Round Num'][0] = 0
            qspec_data[Q]['Batch Num'][0] = 0
            qspec_data[Q]['Recycled QFreq'][0] = False  # no rr so no recycling here
            qspec_data[Q]['Exp Config'][0] = expt_cfg
            qspec_data[Q]['Syst Config'][0] = sys_config_qspec

            saver_qspec = Data_H5(optimizationFolder, qspec_data, 0, save_r)
            saver_qspec.save_to_h5('qspec_ge')
            del saver_qspec
            del qspec_data

        except Exception as e:
            rr_logger.exception(f"g-e QubitSpectroscopyGE error on qubit {Q}: {e}")
            continue

        ################### g-e amp rabi ################
        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
        try:
            rabi = AmplitudeRabiExperiment(Q, tot_num_of_qubits, optimizationFolder, 0,
                                           signal, save_figs=True, experiment=experiment,
                                           live_plot=live_plot,
                                           increase_qubit_reps=increase_qubit_reps,
                                           qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                           multiply_qubit_reps_by=multiply_qubit_reps_by,
                                           verbose=verbose, logger=rr_logger,
                                           qick_verbose=True)
            (rabi_I, rabi_Q, rabi_gains, rabi_fit, stored_pi_amp, sys_config_rabi) = rabi.run()
            experiment.qubit_cfg['pi_amp'][Q] = float(stored_pi_amp)
            rr_logger.info(f"Tune-up: g-e Pi amplitude for qubit {Q + 1}: {float(stored_pi_amp)}")
            with open(log_file, "a", encoding="utf-8") as file:
                file.write("\n" + f'g-e Pi Amplitude Used for optimization: {float(stored_pi_amp)}')


            rabi_data[Q]['Dates'][0] = (
                time.mktime(datetime.datetime.now().timetuple()))
            rabi_data[Q]['I'][0] = rabi_I
            rabi_data[Q]['Q'][0] = rabi_Q
            rabi_data[Q]['Gains'][0] = rabi_gains
            rabi_data[Q]['Fit'][0] = rabi_fit
            rabi_data[Q]['Round Num'][0] = 0
            rabi_data[Q]['Batch Num'][0] = 0
            rabi_data[Q]['Exp Config'][0] = expt_cfg
            rabi_data[Q]['Syst Config'][0] = sys_config_rabi
            saver_rabi = Data_H5(optimizationFolder, rabi_data, 0, save_r)
            saver_rabi.save_to_h5('rabi_ge')
            del rabi
            del saver_rabi
            del rabi_data
        except Exception as e:
            rr_logger.exception(f"g-e Rabi error on qubit {Q}: {e}")
            continue

        # ################### length rabi test ################
        # from section_006p5_length_rabi_ge import LengthRabiExperiment
        # len_rabi = LengthRabiExperiment(Q, tot_num_of_qubits, '/data/QICK_data/run6/6transmon/test/', 0,
        #                                signal, save_figs=True, experiment=experiment,
        #                                live_plot=live_plot,
        #                                increase_qubit_reps=increase_qubit_reps,
        #                                qubit_to_increase_reps_for=qubit_to_increase_reps_for,
        #                                multiply_qubit_reps_by=multiply_qubit_reps_by,
        #                                verbose=verbose, logger=rr_logger,
        #                                qick_verbose=True)
        # (rabi_I, rabi_Q, rabi_gains, rabi_fit, stored_pi_amp, sys_config_rabi) = len_rabi.run()


        ################################################ optimize ################################################

        freq_range = np.linspace(-0.5, 0.5, freq_offset_steps)

        optimal_offset, ssf_dict = sweep_frequency_offset(experiment, Q, freq_range, n_loops=ssf_avgs_per_opt_pt, number_of_qubits=6,
                                                          outerFolder=optimizationFolder, studyDocumentationFolder_opt=studyDocumentationFolder,
                                                          optimizationFolder_opt=optimizationFolder, j=0)

        freq_offsets[Q] = optimal_offset

        with open(log_file, "a", encoding="utf-8") as file:
            file.write("\n" + f'g-e Offset Frequency used for study: {optimal_offset}')

        del experiment


    # initialize a dictionary to store those values
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
    t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
    t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
    t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)

batch_num=0
j = 0
angles=[]
while j < n:
    j += 1
    for QubitIndex in Qs_to_look_at:
        recycled_qfreq = False

        #Get the config for this qubit
        experiment = QICK_experiment(optimizationFolder, DAC_attenuator1 = 10, DAC_attenuator2 = 15, qubit_DAC_attenuator1 = 5,
                                     qubit_DAC_attenuator2 = 4, ADC_attenuator = 17, fridge=FRIDGE) # ADC_attenuator MUST be above 16dB
        experiment.create_folder_if_not_exists(optimizationFolder)

        #Mask out all other resonators except this one
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_gain_ef'] = res_gains
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        ###################################################### TOF #####################################################
        if run_flags["tof"]:
            tof        = TOFExperiment(QubitIndex, studyDocumentationFolder, experiment, j, save_figs, unmasking_resgain = unmask)
            tof.run()
            del tof

        ################################################# g-e Res spec ####################################################
        if run_flags["res_spec"]:
            try:
                res_spec   = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs,
                                                   experiment = experiment, verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
                offset = freq_offsets[QubitIndex] #use optimized offset values or whats set at top of script based on pre_optimize flag
                offset_res_freqs = [r + offset for r in res_freqs]
                experiment.readout_cfg['res_freq_ge'] = offset_res_freqs
                del res_spec

            except Exception as e:
                if debug_mode:
                    raise e # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue #skip the rest of this qubit

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
                q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j,
                                           signal, save_figs, plot_fit=True,experiment=experiment,
                                           live_plot=live_plot, verbose=verbose, logger=rr_logger, unmasking_resgain = unmask)
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
                rr_logger.exception(f"RR g-e QSpec error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"RR g-e QSpec error on qubit {QubitIndex}: {e}")
                continue
        ###################################################### g-e Rabi ####################################################
        if run_flags["rabi"]:
            # try:
            rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs=save_figs,save_shots=False,
                                           experiment = experiment, live_plot = live_plot,
                                           increase_qubit_reps = increase_qubit_reps,
                                           qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                           multiply_qubit_reps_by = multiply_qubit_reps_by,
                                           verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
            (rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp,
             sys_config_rabi)  = rabi.run(thresholding=thresholding)

            # if these are None, fit didnt work
            if (rabi_fit is None and pi_amp is None):
                rr_logger.info('g-e Rabi fit didnt work, skipping the rest of this qubit')
                if verbose: print('g-e Rabi fit didnt work, skipping the rest of this qubit')
                continue  # skip the rest of this qubit

            experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
            rr_logger.info(f'g-e Pi amplitude for qubit {QubitIndex + 1} is: {float(pi_amp)}')
            if verbose: print('g-e Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
            del rabi

            # except Exception as e:
            #     if debug_mode:
            #         raise e # In debug mode, re-raise the exception immediately
            #     else:
            #         rr_logger.exception(f'Got the following error, continuing: {e}')
            #         if verbose: print(f'Got the following error, continuing: {e}')
            #         continue #skip the rest of this qubit

        ########################################## g-e Single Shot Measurements ############################################
        if run_flags["ss"]:
            # try:
            ss = SingleShot(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs, experiment = experiment,
                            verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
            fid, angle, iq_list_g, iq_list_e, sys_config_ss = ss.run()
            I_g = iq_list_g[QubitIndex][0].T[0]
            Q_g = iq_list_g[QubitIndex][0].T[1]
            I_e = iq_list_e[QubitIndex][0].T[0]
            Q_e = iq_list_e[QubitIndex][0].T[1]

            # fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(
            #     data=[I_g, Q_g, I_e, Q_e], cfg=ss.config, plot=save_figs)
            print(sys_config_ss)
            # except Exception as e:
            #     if debug_mode:
            #         raise  # In debug mode, re-raise the exception immediately
            #     else:
            #         rr_logger.exception(f'Got the following error, continuing: {e}')
            #         if verbose: print(f'Got the following error, continuing: {e}')
            #         continue #skip the rest of this qubit

        ############################################## res spec ef ####################################################
        if run_flags["ef_res_spec"]:
            rr_logger.info("----------------- Starting Res Spec EF  -----------------")
            if verbose:
                print("----------------- Starting Res Spec EF  -----------------")

            ef_res_freqs_samples = []
            for sample in range(ef_res_sample_number):
                try:
                    ef_res_spec = ResonanceSpectroscopyEF(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, sample,
                                                          save_figs, experiment=experiment, verbose=verbose,
                                                          logger=rr_logger, qick_verbose=qick_verbose, unmasking_resgain = unmask)
                    ef_res_freqs, ef_freq_pts, ef_freq_center, ef_amps, sys_config_rspec_ef = ef_res_spec.run()
                    ef_res_freqs_samples.append(ef_res_freqs)
                    rr_logger.info(f"EF ResSpec sample {sample} for qubit {QubitIndex + 1}: {ef_res_freqs}")

                    del ef_res_spec

                except Exception as e:
                    if debug_mode:
                        raise  # In debug mode, re-raise the exception immediately
                    rr_logger.exception(f"EF ResSpec error on qubit {QubitIndex + 1} sample {sample}: {e}")


            if ef_res_freqs_samples:
                # Average the resonator frequency values across samples
                avg_ef_res_freqs = np.mean(np.array(ef_res_freqs_samples), axis=0).tolist()
            else:
                rr_logger.error(f"No resonator spectroscopy data collected for qubit {QubitIndex + 1}.")

            experiment.readout_cfg['res_freq_ef'] = ef_res_freqs_samples[-1]  # use the last e-f res spec frequency to update the sys config

            rr_logger.info(f"Avg. EF resonator frequencies for qubit {QubitIndex + 1}: {avg_ef_res_freqs[QubitIndex]}")
            if verbose:
                print(f"Avg. EF resonator frequencies for qubit {QubitIndex + 1}: {avg_ef_res_freqs}")

        ################################################ Qubit Spec EF ################################################
        if run_flags["ef_q_spec"]:
            rr_logger.info("----------------- Starting Qubit Spec EF  -----------------")
            if verbose:
                print("----------------- Starting Qubit Spec EF  -----------------")

            # try:
            increase_qubit_steps_ef = False
            # Qubit 4 needs more steps for e-f spec
            if QubitIndex == 3:
                increase_qubit_steps_ef = True  # if you want to increase the steps for a qubit, set to True

            ef_q_spec = EFQubitSpectroscopy(QubitIndex, number_of_qubits, studyDocumentationFolder, j, signal,
                           True, experiment, live_plot, unmasking_resgain = unmask)

            efqspec_I, efqspec_Q, efqspec_freqs, sys_config_qspec_ef, efqspec_I_fit, efqspec_Q_fit, efqubit_freq = ef_q_spec.run()
            # efqspec_I, efqspec_Q, efqspec_freqs, sys_config_qspec_ef,  efqubit_freq = ef_q_spec.run()
            qubit_freqs_ef[QubitIndex] = efqubit_freq
            # experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] = float(efqubit_freq)

            # rr_logger.info(f"EF Qubit {QubitIndex + 1} frequency: {efqubit_freq}")
            if verbose:
                print(f"EF Qubit {QubitIndex + 1} frequency: {efqubit_freq}")

            del ef_q_spec

            # except Exception as e:
            #     if debug_mode:
            #         raise  # In debug mode, re-raise the exception immediately
            #     rr_logger.exception(f"EF QubitSpectroscopyGE error on qubit {QubitIndex + 1}: {e}")

        ################################################ e-f amp rabi pop meas. ################################################
        if run_flags["rabi_pop_meas"]:
            t0 = time.perf_counter()
            rr_logger.info(
                "----------------- Starting EF Amplitude Rabi Population Measurements -----------------")
            if verbose:
                print("----------------- EF Amplitude Rabi Population Measurements  -----------------")

            try:
                efAmprabi_Qtemps = Temps_EFAmpRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits,
                                                             studyDocumentationFolder,
                                                             j,
                                                             signal, save_figs,
                                                             experiment, live_plot,
                                                             increase_qubit_reps, qubit_to_increase_reps_for,
                                                             multiply_qubit_reps_by, unmasking_resgain = unmask)
                (I1_qtemp, Q1_qtemp, gains1_qtemp, fit_cosine1_qtemp, pi_amp1_qtemp, A_amplitude1, amp_fit1,
                 I2_qtemp, Q2_qtemp, gains2_qtemp, fit_cosine2_qtemp, pi_amp2_qtemp, A_amplitude2, amp_fit2,
                 sysconfig_efrabi_Qtemps) = efAmprabi_Qtemps.run(experiment.soccfg, experiment.soc)

                # Just to quickly output the qubit temperature ---------------------
                # date = data_set
                # fit_saved = fit_data
                # outerFolder_save_plots = ""
                # unique_folder_path = ""
                # outerFolder = ""
                # qtempclass = PlotRR_noQick(date, figure_quality, save_figs, fit_saved, signal, run_name, number_of_qubits, outerFolder,
                #  outerFolder_save_plots, unique_folder_path)
                # T_K, T_mK, _, _ = qtempclass.Qubit_Temperature_Convert(A_amplitude1, A_amplitude2, qubit_freq)
                # print(f"Q{QubitIndex + 1} temperature: {T_mK} mK")
                #-------------------------------------------------------------------------

                rr_logger.info(f"RPM Amplitudes for qubit {QubitIndex + 1}: A1 = {float(A_amplitude1)}, A2 = {float(A_amplitude2)}")
                if verbose:
                    print(f"RPM Amplitudes for qubit {QubitIndex + 1}: A1 = {float(A_amplitude1)}, A2 = {float(A_amplitude2)}")

                del efAmprabi_Qtemps

            except Exception as e:
                if debug_mode:
                    raise  # In debug mode, re-raise the exception immediately
                rr_logger.exception(f"EF Rabi Population Measurements error on qubit {QubitIndex + 1}: {e}")

            t1 = time.perf_counter()
            print(f"RPM took {t1 - t0:.4f} seconds")

        ###################################################### g-e T1 ######################################################
        if run_flags["t1"]:
            try:
                t1 = T1Measurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
                                   experiment = experiment,
                                   live_plot = live_plot, fit_data = fit_data,
                                   increase_qubit_reps = increase_qubit_reps,
                                   qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                   multiply_qubit_reps_by = multiply_qubit_reps_by,
                                   verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                t1_est, t1_err, t1_I, t1_Q, t1_delay_times, q1_fit_exponential, sys_config_t1 = t1.run(
                    thresholding=thresholding)
                del t1

            except Exception as e:
                if debug_mode:
                    raise e # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue #skip the rest of this qubit

        ###################################################### g-e T2R #####################################################
        if run_flags["t2r"]:
            try:
                t2r = T2RMeasurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
                                     experiment = experiment, live_plot = live_plot, fit_data = fit_data,
                                     increase_qubit_reps = increase_qubit_reps,
                                     qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                     multiply_qubit_reps_by = multiply_qubit_reps_by,
                                     verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                t2r_est, t2r_err, t2r_I, t2r_Q, t2r_delay_times, fit_ramsey, sys_config_t2r = t2r.run(
                    thresholding=thresholding)
                del t2r

            except Exception as e:
                if debug_mode:
                    raise e # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue #skip the rest of this qubit

        ##################################################### g-e T2E ######################################################
        if run_flags["t2e"]:
            try:
                t2e = T2EMeasurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
                                     experiment = experiment, live_plot = live_plot, fit_data = fit_data,
                                     increase_qubit_reps = increase_qubit_reps,
                                     qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                     multiply_qubit_reps_by = multiply_qubit_reps_by,
                                     verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                (t2e_est, t2e_err, t2e_I, t2e_Q, t2e_delay_times,
                 fit_t2e, sys_config_t2e) = t2e.run(thresholding=thresholding)
                del t2e

            except Exception as e:
                if debug_mode:
                    raise e # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue #skip the rest of this qubit

        ################################################ e-f rabi ################################################
        if run_flags["ef_Rabi"]:
            increase_qubit_reps = False  # if you want to increase the reps for a qubit, set to True
            qubit_to_increase_reps_for = 0  # only has impact if previous line is True
            multiply_qubit_reps_by = 2  # only has impact if the line two above is True
            print('ef rabi')
            # try:
            # thresholding = False

            efrabi = EF_AmplitudeRabiExperiment(QubitIndex, number_of_qubits, studyDocumentationFolder, j, signal,
                                                False, experiment=experiment, live_plot=live_plot,
                                                increase_qubit_reps=increase_qubit_reps,
                                                qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                                multiply_qubit_reps_by=multiply_qubit_reps_by,
                                                unmasking_resgain=unmask)
            efrabi_I, efrabi_Q, efrabi_gains, efrabi_fit, efpi_amp, efsys_config_to_save = efrabi.run()
            # experiment.soccfg
            # if these are None, fit didnt work
            if (efrabi_fit is None and efpi_amp is None):
                print('Rabi fit didnt work, skipping the rest of this qubit')
                continue  # skip the rest of this qubit

            experiment.qubit_cfg['pi_ef_amp'][QubitIndex] = float(efpi_amp)
            print('ef Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(efpi_amp))
            del efrabi
        #
        # ########################################## e-f Single Shot Measurements ############################################
        # if run_flags["ef_ss"]:
        #     try:
        #
        #         ss_ef = SingleShot_ef(QubitIndex, number_of_qubits, studyDocumentationFolder, j,
        #                                          save_figs, experiment=experiment, unmasking_resgain=unmask)
        #         iq_list_e, iq_list_f, ie_new, qe_new, if_new, qf_new, theta_ef, threshold_ef, sys_config_ss_ef = ss_ef.run()
        #         # iq_list_g, iq_list_e, iq_list_f, ie_new, qe_new, if_new, qf_new, theta_ef, threshold_ef, self.config
        #         I_e = iq_list_e[QubitIndex][0].T[0]
        #         Q_e = iq_list_e[QubitIndex][0].T[1]
        #         I_f = iq_list_f[QubitIndex][0].T[0]
        #         Q_f = iq_list_f[QubitIndex][0].T[1]
        #
        #         # fid_ef, theta_ef, ie_new, if_new, threshold_ef = ss_ef.hist_ssf(
        #         #     data=[I_e, Q_e, I_f, Q_f], cfg=sys_config_ss_ef, plot=save_figs)
        #         fid_ef, theta_ef, ie_new, qe_new, if_new, qf_new = ss_ef.fidelity_test()
        #         # fid, theta_ef, ie_new, qe_new, if_new, qf_new, threshold_ef
        #         print(sys_config_ss_ef)
        #     except Exception as e:
        #         if debug_mode:
        #             raise  # In debug mode, re-raise the exception immediately
        #         else:
        #             rr_logger.exception(f'Got the following error, continuing: {e}')
        #             if verbose: print(f'Got the following error, continuing: {e}')
        #             continue  # skip the rest of this qubit
            ################################################ Qubit Spec FH ################################################
        if run_flags["fh_q_spec"]:
            rr_logger.info("----------------- Starting Qubit Spec FH  -----------------")
            if verbose:
                print("----------------- Starting Qubit Spec FH  -----------------")

            # try:
            increase_qubit_steps_fh = False
            # Qubit 4 needs more steps for e-f spec
            # if QubitIndex == 3:
            #     increase_qubit_steps_fh = True  # if you want to increase the steps for a qubit, set to True

            fh_q_spec = FHQubitSpectroscopy(QubitIndex, number_of_qubits, studyDocumentationFolder, j,
                                            signal,
                                            True, experiment, live_plot, unmasking_resgain=unmask)

            fhqspec_I, fhqspec_Q, fhqspec_freqs,  sys_config_qspec_fh = fh_q_spec.run()
            # fhqspec_I_fit, fhqspec_Q_fit, fhqubit_freq,
            # experiment.soccfg,
            # experiment.soc)
            # qubit_freqs_fh[QubitIndex] = fhqubit_freq
            # experiment.qubit_cfg['qubit_freq_fh'][QubitIndex] = float(fhqubit_freq)

            # rr_logger.info(f"FH Qubit {QubitIndex + 1} frequency: {fhqubit_freq}")
            # if verbose:
            #     print(f"FH Qubit {QubitIndex + 1} frequency: {fhqubit_freq}")

            del fh_q_spec

            # except Exception as e:
            #     if debug_mode:
            #         raise  # In debug mode, re-raise the exception immediately
            #     rr_logger.exception(f"FH QubitSpectroscopyGE error on qubit {QubitIndex + 1}: {e}")

            ################################################ f-h rabi ################################################
        if run_flags["fh_rabi"]:
            increase_qubit_reps = False  # if you want to increase the reps for a qubit, set to True
            qubit_to_increase_reps_for = 0  # only has impact if previous line is True
            multiply_qubit_reps_by = 2  # only has impact if the line two above is True
            print('fh rabi')
            # try:
            # thresholding = False

            fhrabi = FH_AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs=save_figs,save_shots=False,
                                           experiment = experiment, live_plot = live_plot,
                                           increase_qubit_reps = increase_qubit_reps,
                                           qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                           multiply_qubit_reps_by = multiply_qubit_reps_by,
                                           verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
            fhrabi_I, fhrabi_Q, fhrabi_gains, fhrabi_fit, fhpi_amp, fhsys_config_to_save = fhrabi.run()
            # experiment.soccfg
            # if these are None, fit didnt work
            if (fhrabi_fit is None and fhpi_amp is None):
                print('Rabi fit didnt work, skipping the rest of this qubit')
                continue  # skip the rest of this qubit

            experiment.qubit_cfg['pi_fh_amp'][QubitIndex] = float(fhpi_amp)
            print('fh Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(fhpi_amp))
            del fhrabi
        ########################################### g-e-f Single Shot Measurements ############################################

        if run_flags["ss_gef"]:
            ss = SingleShot_ef(QubitIndex, number_of_qubits, studyDocumentationFolder, j, save_figs, experiment)

            # iq_list_e, iq_list_f, ie_new, if_new,  theta_ef, threshold_ef, self.config
            iq_list_e, iq_list_f, ie_new,  if_new,  theta_ef, threshold_ef, sys_config_ss_gef = ss.run()
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
        ########################################## e-f Single Shot Measurements ############################################
        if run_flags["test_act"]:
            # try:

            act = Active_Reset_test(QubitIndex, number_of_qubits, studyDocumentationFolder, j,
                                             save_figs, experiment=experiment, unmasking_resgain=unmask)
            act_idata, act_qdata, no_act_idata, no_act_qdata, act_cfg = act.run()
                # iq_list_g, iq_list_e, iq_list_f, ie_new, qe_new, if_new, qf_new, theta_ef, threshold_ef, self.config
                # I_e = iq_list_e[QubitIndex][0].T[0]
                # Q_e = iq_list_e[QubitIndex][0].T[1]
                # I_f = iq_list_f[QubitIndex][0].T[0]
                # Q_f = iq_list_f[QubitIndex][0].T[1]

                # fid_ef, theta_ef, ie_new, if_new, threshold_ef = ss_ef.hist_ssf(
                #     data=[I_e, Q_e, I_f, Q_f], cfg=sys_config_ss_ef, plot=save_figs)
                # idata, qdata = act.run()
                # fid, theta_ef, ie_new, qe_new, if_new, qf_new, threshold_ef
                # print(sys_config_ss_ef)
            # except Exception as e:
            #     if debug_mode:
            #         raise  # In debug mode, re-raise the exception immediately
            #     else:
            #         rr_logger.exception(f'Got the following error, continuing: {e}')
            #         if verbose: print(f'Got the following error, continuing: {e}')
            #         continue  # skip the rest of this qubit
            del act

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
                qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1]=(
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
            if run_flags["ef_res_spec"]:
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

            # ---------------------Collect e-f qspec Results----------------
            if run_flags["ef_q_spec"]:
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

            if run_flags["fh_q_spec"]:
                fh_qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                fh_qspec_data[QubitIndex]['I'][j - batch_num * save_r - 1] = fhqspec_I
                fh_qspec_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = fhqspec_Q
                fh_qspec_data[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = fhqspec_freqs
                fh_qspec_data[QubitIndex]['I Fit'][j - batch_num * save_r - 1] = None #fhqspec_I_fit
                fh_qspec_data[QubitIndex]['Q Fit'][j - batch_num * save_r - 1] = None #fhqspec_Q_fit
                fh_qspec_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                fh_qspec_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                fh_qspec_data[QubitIndex]['Recycled QFreq'][j - batch_num * save_r - 1] = False  # no rr so no recycling here
                fh_qspec_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                fh_qspec_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_qspec_fh

            # ----------Collect rabi population measurements (qubit temperature data) ----------------
            if run_flags["rabi_pop_meas"]:
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

            #---------------------Collect g-e T1 Results----------------
            if run_flags["t1"]:
                t1_data[QubitIndex]['T1'][j - batch_num*save_r - 1] = t1_est
                t1_data[QubitIndex]['Errors'][j - batch_num*save_r - 1] = t1_err
                t1_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t1_data[QubitIndex]['I'][j - batch_num*save_r - 1] = t1_I
                t1_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = t1_Q
                t1_data[QubitIndex]['Delay Times'][j - batch_num*save_r - 1] = t1_delay_times
                t1_data[QubitIndex]['Fit'][j - batch_num*save_r - 1] = q1_fit_exponential
                t1_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t1_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t1_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                t1_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t1

            #---------------------Collect g-e T2 Results----------------
            if run_flags["t2r"]:
                t2r_data[QubitIndex]['T2'][j - batch_num*save_r - 1] = t2r_est
                t2r_data[QubitIndex]['Errors'][j - batch_num*save_r - 1] = t2r_err
                t2r_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t2r_data[QubitIndex]['I'][j - batch_num*save_r - 1] = t2r_I
                t2r_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = t2r_Q
                t2r_data[QubitIndex]['Delay Times'][j - batch_num*save_r - 1] = t2r_delay_times
                t2r_data[QubitIndex]['Fit'][j - batch_num*save_r - 1] = fit_ramsey
                t2r_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t2r_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t2r_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                t2r_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2r

            #---------------------Collect g-e T2E Results----------------
            if run_flags["t2e"]:
                t2e_data[QubitIndex]['T2E'][j - batch_num*save_r - 1] = t2e_est
                t2e_data[QubitIndex]['Errors'][j - batch_num*save_r - 1] = t2e_err
                t2e_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t2e_data[QubitIndex]['I'][j - batch_num*save_r - 1] = t2e_I
                t2e_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = t2e_Q
                t2e_data[QubitIndex]['Delay Times'][j - batch_num*save_r - 1] = t2e_delay_times
                t2e_data[QubitIndex]['Fit'][j - batch_num*save_r - 1] = fit_t2e
                t2e_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t2e_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t2e_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                t2e_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2e

            # ---------------------Collect e-f Single Shot Results----------------
            # if run_flags["ef_ss"]:
            #     ef_ss_data[QubitIndex]['Fidelity'][j - batch_num * save_r - 1] = fid_ef
            #     ef_ss_data[QubitIndex]['Angle'][j - batch_num * save_r - 1] = theta_ef
            #     ef_ss_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
            #         time.mktime(datetime.datetime.now().timetuple()))
            #     ef_ss_data[QubitIndex]['I_g'][j - batch_num * save_r - 1] = I_e
            #     ef_ss_data[QubitIndex]['Q_g'][j - batch_num * save_r - 1] = Q_e
            #     ef_ss_data[QubitIndex]['I_e'][j - batch_num * save_r - 1] = I_f
            #     ef_ss_data[QubitIndex]['Q_e'][j - batch_num * save_r - 1] = Q_f
            #     ef_ss_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            #     ef_ss_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            #     ef_ss_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
            #     ef_ss_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_ss_ef
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

            # ---------------------Collect g-e Single Shot Results----------------
            if run_flags["test_act"]:

                act_data[QubitIndex]['actI'][j - batch_num * save_r - 1] = act_idata
                act_data[QubitIndex]['actQ'][j - batch_num * save_r - 1] = act_qdata
                act_data[QubitIndex]['noactI'][j - batch_num * save_r - 1] = no_act_idata
                act_data[QubitIndex]['noactQ'][j - batch_num * save_r - 1] = no_act_qdata
                act_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = act_cfg

            # ---------------------Collect g-e Rabi Results----------------
            if run_flags["fh_rabi"]:
                rabi_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                rabi_data[QubitIndex]['I'][j - batch_num * save_r - 1] = fhrabi_I
                rabi_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = fhrabi_Q
                rabi_data[QubitIndex]['Gains'][j - batch_num * save_r - 1] = fhrabi_gains
                rabi_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fhrabi_fit
                rabi_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                rabi_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                rabi_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                rabi_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = fhsys_config_to_save

        del experiment

    ################################################## Potentially Save ################################################
    if save_data_h5:
        # Check if you are at the right round number
        # If so, then save all of the data and change the round num so you replace data starting next round
        if j % save_r == 0:
            batch_num+=1

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
            if run_flags["ef_res_spec"]:
                saver_ef_res = Data_H5(subStudyDataFolder, ef_res_data, batch_num, save_r)  # save
                saver_ef_res.save_to_h5('res_ef')
                del saver_ef_res
                del ef_res_data
            # --------------------------save e-f qspec-----------------------
            if run_flags["ef_q_spec"]:
                saver_ef_qspec = Data_H5(subStudyDataFolder, ef_qspec_data, batch_num, save_r)
                saver_ef_qspec.save_to_h5('qspec_ef')
                del saver_ef_qspec
                del ef_qspec_data
            # --------------------------save e-f Rabi-----------------------
            if run_flags["ef_Rabi"]:
                saver_ef_rabi = Data_H5(subStudyDataFolder, ef_rabi_data, batch_num, save_r)
                saver_ef_rabi.save_to_h5('rabi_ef')
                del saver_ef_rabi
                del ef_rabi_data

            # --------------------------save e-f Rabi-----------------------
            if run_flags["fh_rabi"]:
                saver_fh_rabi = Data_H5(subStudyDataFolder, fh_rabi_data, batch_num, save_r)
                saver_fh_rabi.save_to_h5('rabi_fh')
                del saver_fh_rabi
                del fh_rabi_data
            # --------------------------save f-h qspec-----------------------
            if run_flags["fh_q_spec"]:
                saver_ef_qspec = Data_H5(subStudyDataFolder, fh_qspec_data, batch_num, save_r)
                saver_ef_qspec.save_to_h5('qspec_fh')
                del saver_ef_qspec
                del fh_qspec_data

            # --------save rabi population measurements (qubit temperature data) -----------------------
            if run_flags["rabi_pop_meas"]:
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

            #--------------------------save g-e t2r-----------------------
            if run_flags["t2r"]:
                saver_t2r = Data_H5(subStudyDataFolder, t2r_data, batch_num, save_r)
                saver_t2r.save_to_h5('t2_ge')
                del saver_t2r
                del t2r_data

            #--------------------------save g-e t2e-----------------------
            if run_flags["t2e"]:
                saver_t2e = Data_H5(subStudyDataFolder, t2e_data, batch_num, save_r)
                saver_t2e.save_to_h5('t2e_ge')
                del saver_t2e
                del t2e_data

            # --------------------------save e-f SS-----------------------
            if run_flags["ef_ss"]:
                saver_ss_ef = Data_H5(subStudyDataFolder, ef_ss_data, batch_num, save_r)
                saver_ss_ef.save_to_h5('ef_ss')
                del saver_ss_ef
                del ef_ss_data

            # --------------------------save g-e-f SS-----------------------
            if run_flags["ss_gef"]:
                saver_ss = Data_H5(subStudyDataFolder, ss_data_gef, batch_num, save_r)
                saver_ss.save_to_h5('SS_gef')
                del saver_ss
                del ss_data_gef

            # --------------------------save g-e SS-----------------------
            if run_flags["test_act"]:
                saver_act = Data_H5(subStudyDataFolder, act_data, batch_num, save_r)
                saver_act.save_to_h5('test_act')
                del saver_act
                del act_data

    # reset all dictionaries to none for safety
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    fh_rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
    ss_data_ef = create_data_dict(ss_keys, save_r, list_of_all_qubits)
    ef_ss_data = create_data_dict(ss_ef_keys, save_r, list_of_all_qubits)
    ef_res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    ef_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    fh_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    rabi_data_ef_Qtemps = create_data_dict(rabi_keys_ef_Qtemps, save_r, list_of_all_qubits)
    t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
    t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
    t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
    ss_data_gef = create_data_dict(ss_keys_gef, save_r, list_of_all_qubits)
    act_data = create_data_dict(act_keys, save_r, list_of_all_qubits)

en=time.time()
print('timetaken=',en-st)