import copy
import sys
import os
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
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from starkshift import ResStarkShiftSpec
from starkshift import StarkShiftSpec
from system_config import QICK_experiment
from section_003_punch_out_ge_mux import PunchOut
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE

################################################ Run Configurations ####################################################
n=10
pre_optimize = False
freq_offset_steps = 15
ssf_avgs_per_opt_pt = 5
save_r = 1                           # how many rounds to save after
signal = 'None'                      # 'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization
save_figs = False                     # save plots for everything as you go along the RR script?
live_plot = False                     # for live plotting do "visdom" in command line and then open http://localhost:8097/ on firefox
fit_data = False                     # fit the data here and save or plot the fits?
save_data_h5 = True                  # save all of the data to h5 files?
verbose = False                      # print everything to the console in real time, good for debugging, bad for memory
debug_mode = True                   # if True, it disables the continuing function of RR if an error pops up in a class -- errors now stop the RR script
thresholding = False                 # use internal QICK threshold for ratio of Binary values on y for rabi/t1/t2r/t2e, or analog avg when false
increase_qubit_reps = False          # if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0       # only has impact if previous line is True
multiply_qubit_reps_by = 2           # only has impact if the line two above is True
Qs_to_look_at = [4]#0, 1, 2, 3, 4, 5        # only list the qubits you want to do the RR for

# set which of the following you'd like to run to 'True'
run_flags = {"tof": False, "res_spec": False, "q_spec": True, "ss": True, "rabi": True,
             "t1": True, "t2r": False, "t2e": False, "med_gain_q_spec": True, "high_gain_q_spec": True, "stark_spec": True}

topFolder = "/data/QICK_data/run6/6transmon/Round_Robin_Benchmark_with_stark/"#Round_Robin_Benchmark
if not os.path.exists('/data/QICK_data/run6/6transmon/Round_Robin_Benchmark_with_stark/'): os.makedirs('/data/QICK_data/run6/6transmon/Round_Robin_Benchmark_with_stark/')
dataFolder = os.path.join(topFolder, 'Data')
optimizationFolder = os.path.join(topFolder, 'Optimization')
documentationFolder = os.path.join(topFolder, 'Documentation')
outerFolder = os.path.join(dataFolder, str(datetime.date.today()))

optDocumentationFolder = os.path.join(optimizationFolder, 'opt_documentation')
opt_notes_path = os.path.join(optDocumentationFolder, 'opt_notes.txt')
studyDocumentationFolder = os.path.join(outerFolder, 'study_documentation')
study_notes_path = os.path.join(studyDocumentationFolder, 'study_notes.txt')

if not os.path.exists(topFolder): os.makedirs(topFolder)
if not os.path.exists(dataFolder): os.makedirs(dataFolder)
if not os.path.exists(optimizationFolder): os.makedirs(optimizationFolder)
if not os.path.exists(documentationFolder): os.makedirs(documentationFolder)
if not os.path.exists(outerFolder): os.makedirs(outerFolder)
if not os.path.exists(studyDocumentationFolder): os.makedirs(studyDocumentationFolder)
if not os.path.exists(optDocumentationFolder): os.makedirs(optDocumentationFolder)

################################################ optimization outputs ##################################################
# Optimization parameters for resonator spectroscopy
# res_leng_vals = [4.3, 5, 5, 4, 4.5, 9]
# res_gain = [1, 1, 0.6, 0.6, 1, 0.6]
# freq_offsets = [0]*6

# from 3/27
res_leng_vals = [4.3, 6, 5, 6.1, 5.8, 7]
res_gain = [0.9600, 1, 0.7200, 0.5733, 0.8, 0.55]
freq_offsets = [-0.05, -0.19, -0.19, -0.15, -0.2, -0.05]

################################################## Configure logging ###################################################
''' We need to create a custom logger and disable propagation like this 
to remove the logs from the underlying qick from saving to the log file for RR'''

log_file = os.path.join(outerFolder, "RR_script.log")
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
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t2r_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
t2e_keys = ['T2E', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
starkspec_keys = ['Dates', 'I', 'Q', 'P', 'shots','Gain Sweep','Round Num', 'Batch Num', 'Exp Config', 'Syst Config']

#initialize a simple list to store the qspec values in incase a fit fails
stored_qspec_list = [None] * tot_num_of_qubits

if live_plot:
    # Check if visdom is connected right away, otherwise, throw an error
    if not (viz := visdom.Visdom()).check_connection(timeout_seconds=5):
        raise RuntimeError("Visdom server not connected!, Type \'visdom\' into the command line and go to "
                           "http://localhost:8097/ on firefox")


# initialize a dictionary to store those values
res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
high_gain_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
med_gain_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
starkspec_data = create_data_dict(starkspec_keys, save_r, list_of_all_qubits)
res_starkspec_data = create_data_dict(starkspec_keys, save_r, list_of_all_qubits)

if pre_optimize:
    ################################################## Simple Optimization ###############################################

    def sweep_frequency_offset(experiment, QubitIndex, offset_values, n_loops=10, number_of_qubits=6,
                               outerFolder="", studyDocumentationFolder="",optimizationFolder="", j=0):
        import matplotlib.pyplot as plt
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
                    ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
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

    ################################################## Simple Optimization ###############################################
    for Q in Qs_to_look_at:
        try:
            experiment = QICK_experiment(
                dataFolder,
                DAC_attenuator1=5,
                DAC_attenuator2=10,
                ADC_attenuator=10,
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
            rr_logger.info(f"ResSpec for qubit {Q}: {res_freqs}")

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
            saver_res.save_to_h5('Res')
            del saver_res
            del res_data
            res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)  # initialize again to a blank for saftey
            del res_spec
        except Exception as e:
            rr_logger.exception(f"ResSpec error on qubit {Q} : {e}")
            continue
        ############ Qubit Spec ##############
        qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
        try:
            q_spec = QubitSpectroscopy(Q, tot_num_of_qubits, documentationFolder, 0,
                                       signal, plot_fit=False, save_figs=True, experiment=experiment,
                                       live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                       qick_verbose=True, increase_reps=True, increase_reps_to=500)
            (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit,
             qubit_freq, sys_config_qspec) = q_spec.run()
            experiment.qubit_cfg['qubit_freq_ge'][Q] = float(qubit_freq)
            stored_qspec = float(qubit_freq)
            rr_logger.info(f"Tune-up: Qubit {Q + 1} frequency: {stored_qspec}")
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
            saver_qspec.save_to_h5('QSpec')
            del saver_qspec
            del qspec_data

        except Exception as e:
            rr_logger.exception(f"QubitSpectroscopyGE error on qubit {Q}: {e}")
            continue

        ################### amp rabi ################
        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
        try:
            rabi = AmplitudeRabiExperiment(Q, tot_num_of_qubits, documentationFolder, 0,
                                           signal, save_figs=True, experiment=experiment,
                                           live_plot=live_plot,
                                           increase_qubit_reps=increase_qubit_reps,
                                           qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                           multiply_qubit_reps_by=multiply_qubit_reps_by,
                                           verbose=verbose, logger=rr_logger,
                                           qick_verbose=True)
            (rabi_I, rabi_Q, rabi_gains, rabi_fit, stored_pi_amp, sys_config_rabi) = rabi.run()
            experiment.qubit_cfg['pi_amp'][Q] = float(stored_pi_amp)
            rr_logger.info(f"Tune-up: Pi amplitude for qubit {Q + 1}: {float(stored_pi_amp)}")
            with open(opt_notes_path, "a", encoding="utf-8") as file:
                file.write("\n" + f'Pi Amplitude Used for optimization: {float(stored_pi_amp)}')


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
            saver_rabi.save_to_h5('Rabi')
            del rabi
            del saver_rabi
            del rabi_data
        except Exception as e:
            rr_logger.exception(f"Rabi error on qubit {Q}: {e}")
            continue

        ################################################ optimize ################################################

        freq_range = np.linspace(-0.5, 0.5, freq_offset_steps)

        optimal_offset, ssf_dict = sweep_frequency_offset(experiment, Q, freq_range, n_loops=ssf_avgs_per_opt_pt, number_of_qubits=6,
                               outerFolder=optimizationFolder, studyDocumentationFolder=studyDocumentationFolder,
                               optimizationFolder=optimizationFolder, j=0)

        freq_offsets[Q] = optimal_offset

        with open(opt_notes_path, "a", encoding="utf-8") as file:
            file.write("\n" + f'Offset Frequency used for study: {optimal_offset}')

        del experiment


    # initialize a dictionary to store those values
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    high_gain_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    med_gain_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
    t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
    t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
    t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
    starkspec_data = create_data_dict(starkspec_keys, save_r, list_of_all_qubits)
    res_starkspec_data = create_data_dict(starkspec_keys, save_r, list_of_all_qubits)



batch_num=0
j = 0
angles=[]
while j < n:
    j += 1
    for QubitIndex in Qs_to_look_at:
        recycled_qfreq = False

        #Get the config for this qubit
        experiment = QICK_experiment(outerFolder, DAC_attenuator1 = 5, DAC_attenuator2 = 10, ADC_attenuator = 10,
                                     fridge=FRIDGE)
        experiment.create_folder_if_not_exists(outerFolder)

        #Mask out all other resonators except this one
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        ###################################################### TOF #####################################################
        if run_flags["tof"]:
            tof        = TOFExperiment(QubitIndex, outerFolder, experiment, j, save_figs)
            tof.run()
            del tof

        ################################################## Res spec ####################################################
        if run_flags["res_spec"]:
            try:
                res_spec   = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, outerFolder, j, save_figs,
                                                   experiment = experiment, verbose = verbose, logger = rr_logger)
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

        ################################################## Qubit spec ##################################################
        if run_flags["q_spec"]:
            try:
                q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, outerFolder, j,
                                           signal, save_figs, plot_fit=False,experiment=experiment,
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

        ###################################################### Rabi ####################################################
        if run_flags["rabi"]:
            try:
                rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, outerFolder, j, signal, save_figs,
                                               experiment = experiment, live_plot = live_plot,
                                               increase_qubit_reps = increase_qubit_reps,
                                               qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                               multiply_qubit_reps_by = multiply_qubit_reps_by,
                                               verbose = verbose, logger = rr_logger)
                (rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp,
                 sys_config_rabi)  = rabi.run(thresholding=thresholding)

                # if these are None, fit didnt work
                if (rabi_fit is None and pi_amp is None):
                    rr_logger.info('Rabi fit didnt work, skipping the rest of this qubit')
                    if verbose: print('Rabi fit didnt work, skipping the rest of this qubit')
                    continue  # skip the rest of this qubit

                experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
                rr_logger.info(f'Pi amplitude for qubit {QubitIndex + 1} is: {float(pi_amp)}')
                if verbose: print('Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
                del rabi

            except Exception as e:
                if debug_mode:
                    raise e # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue #skip the rest of this qubit

        ########################################## Single Shot Measurements ############################################
        if run_flags["ss"]:
            try:
                ss = SingleShot(QubitIndex, tot_num_of_qubits, outerFolder, j, save_figs, experiment = experiment,
                                verbose = verbose, logger = rr_logger)
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
                    continue #skip the rest of this qubit

        ###################################################### T1 ######################################################
        if run_flags["t1"]:
            try:
                t1 = T1Measurement(QubitIndex, tot_num_of_qubits, outerFolder, j, signal, save_figs,
                                   experiment = experiment,
                                   live_plot = live_plot, fit_data = fit_data,
                                   increase_qubit_reps = increase_qubit_reps,
                                   qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                   multiply_qubit_reps_by = multiply_qubit_reps_by,
                                   verbose = verbose, logger = rr_logger)
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

        ###################################################### T2R #####################################################
        if run_flags["t2r"]:
            try:
                t2r = T2RMeasurement(QubitIndex, tot_num_of_qubits, outerFolder, j, signal, save_figs,
                                     experiment = experiment, live_plot = live_plot, fit_data = fit_data,
                                     increase_qubit_reps = increase_qubit_reps,
                                     qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                     multiply_qubit_reps_by = multiply_qubit_reps_by,
                                     verbose = verbose, logger = rr_logger)
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

        ##################################################### T2E ######################################################
        if run_flags["t2e"]:
            try:
                t2e = T2EMeasurement(QubitIndex, tot_num_of_qubits, outerFolder, j, signal, save_figs,
                                     experiment = experiment, live_plot = live_plot, fit_data = fit_data,
                                     increase_qubit_reps = increase_qubit_reps,
                                     qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                     multiply_qubit_reps_by = multiply_qubit_reps_by,
                                     verbose = verbose, logger = rr_logger)
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

        ################################################## Medium Gain Qubit spec ##################################################
        if run_flags["med_gain_q_spec"]:
            try:
                qubit_gain_temp = experiment.qubit_cfg['qubit_gain_ge']  # save current config parameters
                experiment.qubit_cfg['qubit_gain_ge'] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] #* len(qubit_gain_temp)  # set medium gain

                med_gain_q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, documentationFolder, 0,
                                                     signal, plot_fit=False, save_figs=True, experiment=experiment,
                                                     live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                                     qick_verbose=True, increase_reps=True, increase_reps_to=500)
                (mgqspec_I, mgqspec_Q, mgqspec_freqs, mgqspec_I_fit, mgqspec_Q_fit,
                 mgqubit_freq, sys_config_mgqspec) = med_gain_q_spec.run()

                # do not update config with the fitted qubit freq, that is job of regular qspec!

                rr_logger.info(f"Medium gain qspec on qubit {QubitIndex + 1} frequency: {qubit_freq}")
                del med_gain_q_spec

                experiment.qubit_cfg['qubit_gain_ge'] = qubit_gain_temp  # restore parameters for regular qspec

            except Exception as e:
                rr_logger.exception(f"medium gain QubitSpectroscopyGE error on qubit {QubitIndex}: {e}")
                continue



        ################################################## High Gain Qubit spec ##################################################
        if run_flags["high_gain_q_spec"]:
            try:
                qubit_gain_temp = experiment.qubit_cfg['qubit_gain_ge']  # save current config parameters
                experiment.qubit_cfg['qubit_gain_ge'] = np.ones(len(qubit_gain_temp))  # set high gain

                high_gain_q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, documentationFolder, 0,
                                                     signal, plot_fit=False, save_figs=True, experiment=experiment,
                                                     live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                                     qick_verbose=True, increase_reps=True, increase_reps_to=500)
                (hgqspec_I, hgqspec_Q, hgqspec_freqs, hgqspec_I_fit, hgqspec_Q_fit,
                 hgqubit_freq, sys_config_hgqspec) = high_gain_q_spec.run()

                # do not update config with the fitted qubit freq, that is job of regular qspec!

                rr_logger.info(f"High gain qspec on qubit {QubitIndex + 1} frequency: {qubit_freq}")
                del high_gain_q_spec

                experiment.qubit_cfg['qubit_gain_ge'] = qubit_gain_temp  # restore parameters for regular qspec

            except Exception as e:
                rr_logger.exception(f"high gain QubitSpectroscopyGE error on qubit {QubitIndex}: {e}")
                continue

        ##################################################### stark shift spectroscopy ######################################################
        if run_flags["stark_spec"]:
            try:
                res_freq_stark = copy.deepcopy(experiment.readout_cfg['res_freq_ge'])
                res_freq_stark.append(res_freq_stark[QubitIndex])
                res_phase_stark = copy.deepcopy(experiment.readout_cfg['res_phase'])
                res_phase_stark.append(res_phase_stark[QubitIndex])

                res_stark_shift_spec = ResStarkShiftSpec(QubitIndex, tot_num_of_qubits, outerFolder, res_freq_stark, res_phase_stark, save_figs,
                                                      experiment=experiment)
                res_starkspec_I, res_starkspec_Q, res_starkspec_P, res_starkspec_shots, res_starkspec_gain_sweep, sys_config_res_starkspec = res_stark_shift_spec.run()
                del res_stark_shift_spec

            except Exception as e:
                if debug_mode:
                    raise e  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue  # skip the rest of this qubit

        ##################################################### stark shift spectroscopy ######################################################
        if run_flags["stark_spec"]:
            try:
                stark_shift_spec = StarkShiftSpec(QubitIndex, tot_num_of_qubits, outerFolder, save_figs,
                                                             experiment=experiment)
                starkspec_I, starkspec_Q, starkspec_P, starkspec_shots, starkspec_gain_sweep, sys_config_starkspec = stark_shift_spec.run_with_qick_sweep()
                del stark_shift_spec

            except Exception as e:
                if debug_mode:
                    raise e  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue  # skip the rest of this qubit



        ############################################### Collect Results ################################################
        if save_data_h5:
            # ---------------------Collect Res Spec Results----------------
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

            # ---------------------Collect QSpec Results----------------
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

            # ---------------------Collect Rabi Results----------------
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

            # ---------------------Collect Single Shot Results----------------
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

            #---------------------Collect T1 Results----------------
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

            #---------------------Collect T2 Results----------------
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

            #---------------------Collect T2E Results----------------
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

                # ---------------------Collect medium gain QSpec Results----------------
            if run_flags["med_gain_q_spec"]:
                med_gain_qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                        time.mktime(datetime.datetime.now().timetuple()))
                med_gain_qspec_data[QubitIndex]['I'][j - batch_num * save_r - 1] = mgqspec_I
                med_gain_qspec_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = mgqspec_Q
                med_gain_qspec_data[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = mgqspec_freqs
                med_gain_qspec_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                med_gain_qspec_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                med_gain_qspec_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                med_gain_qspec_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_mgqspec

            # ---------------------Collect high gain QSpec Results----------------
            if run_flags["high_gain_q_spec"]:
                high_gain_qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                        time.mktime(datetime.datetime.now().timetuple()))
                high_gain_qspec_data[QubitIndex]['I'][j - batch_num * save_r - 1] = hgqspec_I
                high_gain_qspec_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = hgqspec_Q
                high_gain_qspec_data[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = hgqspec_freqs
                high_gain_qspec_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                high_gain_qspec_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                high_gain_qspec_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                high_gain_qspec_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_hgqspec

            # ---------------------Collect starkSpec Results----------------
            if run_flags["stark_spec"]:
                starkspec_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = time.mktime(datetime.datetime.now().timetuple())
                starkspec_data[QubitIndex]['I'][j - batch_num*save_r - 1] = starkspec_I
                starkspec_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = starkspec_Q
                starkspec_data[QubitIndex]['P'][j - batch_num*save_r - 1] = starkspec_P
                starkspec_data[QubitIndex]['shots'][j - batch_num*save_r - 1] = starkspec_shots
                starkspec_data[QubitIndex]['Gain Sweep'][j - batch_num*save_r - 1] = starkspec_gain_sweep
                starkspec_data[QubitIndex]['Round Num'][j - batch_num*save_r - 1] = j
                starkspec_data[QubitIndex]['Batch Num'][j - batch_num*save_r - 1] = batch_num
                starkspec_data[QubitIndex]['Exp Config'][j - batch_num*save_r - 1] = expt_cfg
                starkspec_data[QubitIndex]['Syst Config'][j - batch_num*save_r - 1] = sys_config_starkspec

            if run_flags["stark_spec"]:
                res_starkspec_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = time.mktime(datetime.datetime.now().timetuple())
                res_starkspec_data[QubitIndex]['I'][j - batch_num*save_r - 1] = res_starkspec_I
                res_starkspec_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = res_starkspec_Q
                res_starkspec_data[QubitIndex]['P'][j - batch_num*save_r - 1] = res_starkspec_P
                res_starkspec_data[QubitIndex]['shots'][j - batch_num*save_r - 1] = res_starkspec_shots
                res_starkspec_data[QubitIndex]['Gain Sweep'][j - batch_num*save_r - 1] = res_starkspec_gain_sweep
                res_starkspec_data[QubitIndex]['Round Num'][j - batch_num*save_r - 1] = j
                res_starkspec_data[QubitIndex]['Batch Num'][j - batch_num*save_r - 1] = batch_num
                res_starkspec_data[QubitIndex]['Exp Config'][j - batch_num*save_r - 1] = expt_cfg
                res_starkspec_data[QubitIndex]['Syst Config'][j - batch_num*save_r - 1] = sys_config_res_starkspec


        del experiment

    ################################################## Potentially Save ################################################
    if save_data_h5:
        # Check if you are at the right round number
        # If so, then save all of the data and change the round num so you replace data starting next round
        if j % save_r == 0:
            batch_num+=1

            # --------------------------save Res Spec-----------------------
            if run_flags["res_spec"]:
                saver_res = Data_H5(outerFolder, res_data, batch_num, save_r)
                saver_res.save_to_h5('Res_ge')
                del saver_res
                del res_data

            # --------------------------save QSpec-----------------------
            if run_flags["q_spec"]:
                saver_qspec = Data_H5(outerFolder, qspec_data, batch_num, save_r)
                saver_qspec.save_to_h5('QSpec_ge')
                del saver_qspec
                del qspec_data

            # --------------------------save Rabi-----------------------
            if run_flags["rabi"]:
                saver_rabi = Data_H5(outerFolder, rabi_data, batch_num, save_r)
                saver_rabi.save_to_h5('Rabi_ge')
                del saver_rabi
                del rabi_data

            # --------------------------save SS-----------------------
            if run_flags["ss"]:
                saver_ss = Data_H5(outerFolder, ss_data, batch_num, save_r)
                saver_ss.save_to_h5('SS_ge')
                del saver_ss
                del ss_data

            # --------------------------save t1-----------------------
            if run_flags["t1"]:
                saver_t1 = Data_H5(outerFolder, t1_data, batch_num, save_r)
                saver_t1.save_to_h5('T1_ge')
                del saver_t1
                del t1_data

            #--------------------------save t2r-----------------------
            if run_flags["t2r"]:
                saver_t2r = Data_H5(outerFolder, t2r_data, batch_num, save_r)
                saver_t2r.save_to_h5('T2_ge')
                del saver_t2r
                del t2r_data

            #--------------------------save t2e-----------------------
            if run_flags["t2e"]:
                saver_t2e = Data_H5(outerFolder, t2e_data, batch_num, save_r)
                saver_t2e.save_to_h5('T2E_ge')
                del saver_t2e
                del t2e_data

            # --------------------------save medium  gain QSpec-----------------------
            if run_flags["med_gain_q_spec"]:
                saver_med_gain_qspec = Data_H5(outerFolder, med_gain_qspec_data, batch_num, save_r)
                saver_med_gain_qspec.save_to_h5('med_gain_QSpec_ge')
                del saver_med_gain_qspec
                del med_gain_qspec_data


            # --------------------------save high gain QSpec-----------------------
            if run_flags["high_gain_q_spec"]:
                saver_high_gain_qspec = Data_H5(outerFolder, high_gain_qspec_data, batch_num, save_r)
                saver_high_gain_qspec.save_to_h5('high_gain_QSpec_ge')
                del saver_high_gain_qspec
                del high_gain_qspec_data

            #--------------------------save starkSpec-----------------------
            if run_flags["stark_spec"]:
                saver_starkspec = Data_H5(outerFolder, starkspec_data, batch_num, save_r)
                saver_starkspec.save_to_h5('starkSpec')
                del saver_starkspec
                del starkspec_data

            if run_flags["stark_spec"]:
                saver_res_starkspec = Data_H5(outerFolder, res_starkspec_data, batch_num, save_r)
                saver_res_starkspec.save_to_h5('res_starkSpec')
                del saver_res_starkspec
                del res_starkspec_data

            # reset all dictionaries to none for safety
            res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
            qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
            high_gain_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
            med_gain_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
            rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
            ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
            t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
            t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
            t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
            starkspec_data = create_data_dict(starkspec_keys, save_r, list_of_all_qubits)
            res_starkspec_data = create_data_dict(starkspec_keys, save_r, list_of_all_qubits)