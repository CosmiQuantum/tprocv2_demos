import sys
import os
from copy import deepcopy
import copy
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
Qs_to_look_at = [1]        # only list the qubits you want to do the RR for
study = 'QZE'
sub_study = 'rabi'
substudy_txt_notes = ('try without a rabi drive to see if brightness thing stays')
# substudy_txt_notes = ('making a qubit frequency vs qubit pulse length sweep for a given zeno pulse gain. This will tell me '
#                       'if i can use a singe value for the starked qubit frequency for every rabi value, or if i need to '
#                       'update it for every point. My prediction is that as long as I wait for the resonator ring up before '
#                       'doing the qubit pulse this should be constant on average across the x axis. ill try for a few different zeno gains.')

# set which of the following you'd like to run to 'True'
run_flags = {"tof": False, "res_spec": True, "q_spec": True, "ss": True, "rabi": True,
             "t1": True, "t2r": True, "t2e": True}

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
formatted_datetime = 'no_qubit_rabi_drive'+formatted_datetime
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
res_leng_vals = [5.5, 12, 6.0, 6.5, 5.0, 6.0]
res_gain = [0.9, 0.95, 0.78, 0.58, 0.95, 0.57]
#freq_offsets = [-0.1, -0.2, 0.1, -0.4, -0.1, -0.1]
freq_offsets = [-0.1, 0.2, -0.1, -0.4, -0.1, -0.1]

####################################################### RR #############################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num','Recycled QFreq',
              'Exp Config', 'Syst Config']
rabi_keys = ['Dates', 'I', 'Q', 'Mag', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t2r_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
t2e_keys = ['T2E', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
#initialize a simple list to store the qspec values in incase a fit fails
stored_qspec_list = [None] * tot_num_of_qubits

if live_plot:
    # Check if visdom is connected right away, otherwise, throw an error
    if not (viz := visdom.Visdom()).check_connection(timeout_seconds=5):
        raise RuntimeError("Visdom server not connected!, Type \'visdom\' into the command line and go to "
                           "http://localhost:8097/ on firefox")



############################## Get first the res, qubit freqs #######################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}


res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)


def sweep_frequency_offset(experiment_opt, QubitIndex_opt, offset_values, n_loops=10, number_of_qubits=6,
                           outerFolder="", studyDocumentationFolder_opt="",optimizationFolder_opt="", j=0):
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
            saver_ss.save_to_h5('SS')
            del saver_ss
            del ss_data

            ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)


        # find avg ssf
        avg_fid = np.mean(fids)
        ssf_dict[offset] = avg_fid
        if verbose:
            print(f"Offset: {offset} -> Average SSF: {avg_fid:.4f}")
    import matplotlib.pyplot as plt
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

#Vary the readout pulse height, (number of photons int he resonator) and look at rabi (fig 2 here
# https://iopscience.iop.org/article/10.1088/1367-2630/17/6/063035/pdf)
batch_num=0
j = 0

for QubitIndex in Qs_to_look_at:
    experiment = QICK_experiment(optimizationFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10,
                                 fridge=FRIDGE)
    updated_qubit_gain = 0.05 #experiment.qubit_cfg['qubit_gain_ge'][QubitIndex] * 20
    experiment.qubit_cfg['qubit_gain_ge'][
        QubitIndex] = updated_qubit_gain

    # Mask out all other resonators except this one
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

    ################################ Do Res spec once per qubit and store the value ####################################
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, optimizationFolder, 0,
                                     save_figs=True, experiment=experiment, verbose=verbose,
                                     logger=rr_logger, qick_verbose=True)

    res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()

    experiment.readout_cfg['res_freq_ge'] = res_freqs
    rr_logger.info(f"ResSpec for qubit {QubitIndex}: {res_freqs}")

    res_data[QubitIndex]['Dates'][0] = (
        time.mktime(datetime.datetime.now().timetuple()))
    res_data[QubitIndex]['freq_pts'][0] = freq_pts
    res_data[QubitIndex]['freq_center'][0] = freq_center
    res_data[QubitIndex]['Amps'][0] = amps
    res_data[QubitIndex]['Found Freqs'][0] = res_freqs
    res_data[QubitIndex]['Batch Num'][0] = 0
    res_data[QubitIndex]['Exp Config'][0] = expt_cfg
    res_data[QubitIndex]['Syst Config'][0] = sys_config_rspec

    saver_res = Data_H5(optimizationFolder, res_data, 0, save_r)  # save
    saver_res.save_to_h5('Res')
    del saver_res
    del res_data
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)  # initialize again to a blank for saftey
    del res_spec

    ############################################################ Qubit Spec ############################################
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)

    q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, optimizationFolder, 0,
                               signal, save_figs=True, experiment=experiment,
                               live_plot=live_plot, verbose=verbose, logger=rr_logger,
                               qick_verbose=True, increase_reps=True, increase_reps_to=500)
    (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit,
     qubit_freq, sys_config_qspec, fwhm) = q_spec.run(return_fwhm=True)
    experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
    experiment.qubit_cfg['qubit_freq_ge_starked'][QubitIndex] = float(
        qubit_freq)  # update this one too incase it isnt updated later
    experiment.qubit_cfg['fwhm_w01'] = fwhm
    stored_qspec = float(qubit_freq)
    rr_logger.info(f"Tune-up: Qubit {QubitIndex + 1} frequency: {stored_qspec}")
    del q_spec

    qspec_data[QubitIndex]['Dates'][0] = (
        time.mktime(datetime.datetime.now().timetuple()))
    qspec_data[QubitIndex]['I'][0] = qspec_I
    qspec_data[QubitIndex]['Q'][0] = qspec_Q
    qspec_data[QubitIndex]['Frequencies'][0] = qspec_freqs
    qspec_data[QubitIndex]['I Fit'][0] = qspec_I_fit
    qspec_data[QubitIndex]['Q Fit'][0] = qspec_Q_fit
    qspec_data[QubitIndex]['Round Num'][0] = 0
    qspec_data[QubitIndex]['Batch Num'][0] = 0
    qspec_data[QubitIndex]['Recycled QFreq'][0] = False  # no rr so no recycling here
    qspec_data[QubitIndex]['Exp Config'][0] = expt_cfg
    qspec_data[QubitIndex]['Syst Config'][0] = sys_config_qspec

    saver_qspec = Data_H5(optimizationFolder, qspec_data, 0, save_r)
    saver_qspec.save_to_h5('QSpec')
    del saver_qspec
    del qspec_data

    # reinitialize
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)

    ###################################### amp rabi for single shot optimization #########################################
    rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, optimizationFolder, j, signal,
                                   True, experiment, live_plot,
                                   increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
    rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_to_save = rabi.run()

    experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)

    ################## length rabi to grab the pi length value using regular qubit frequency and no QZE pulse ################
    from section_006p5_length_rabi_ge import LengthRabiExperiment

    get_pi_len_experiment = deepcopy(experiment)
    get_pi_len_experiment.qubit_cfg['qubit_gain_ge'][
        QubitIndex] = updated_qubit_gain
    # get_pi_len_experiment.qubit_cfg['qubit_len_ge'] = 0.2 #us, make really small so you can fit to pi length
    len_rabi = LengthRabiExperiment(QubitIndex, tot_num_of_qubits, optimizationFolder+'/length_rabi_test_no_qze/', 0,
                                    signal, save_figs=True, experiment=get_pi_len_experiment,
                                    live_plot=live_plot,
                                    increase_qubit_reps=increase_qubit_reps,
                                    qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                    multiply_qubit_reps_by=multiply_qubit_reps_by,
                                    verbose=verbose, logger=rr_logger,
                                    qick_verbose=True, bare_pi_len_qze_experiment=True)
    (rabi_I, rabi_Q, rabi_gains, rabi_fit, stored_pi_len_no_qze, sys_config_rabi) = len_rabi.run()

    experiment.qubit_cfg['qubit_pi_len'] = stored_pi_len_no_qze #this is just a single value that will be used in the QZE experiment to turn on the zeno pulse
    gain_to_print=get_pi_len_experiment.qubit_cfg['qubit_gain_ge'][
        QubitIndex]
    len_to_print = get_pi_len_experiment.qubit_cfg['qubit_length_ge']
    print(f'unstarked rabi pi len using qubit pulse gain of {gain_to_print} and pulse length of {len_to_print} is {stored_pi_len_no_qze}')

    #################################################### Optimization ##################################################
    freq_offset_steps = 5
    ssf_avgs_per_opt_pt = 5
    freq_range = np.linspace(-0.5, 0.5, freq_offset_steps)

    optimal_offset, ssf_dict = sweep_frequency_offset(experiment, QubitIndex, freq_range, n_loops=ssf_avgs_per_opt_pt,
                                                      number_of_qubits=6,
                                                      outerFolder=optimizationFolder,
                                                      studyDocumentationFolder_opt=studyDocumentationFolder,
                                                      optimizationFolder_opt=optimizationFolder, j=0)

    offset_res_freqs = [r + optimal_offset for r in experiment.readout_cfg['res_freq_ge']]
    experiment.readout_cfg['res_freq_ge'] = offset_res_freqs  # update with ofset added

    # ############################################### rabi chevron after pi pulse ########################################
    # qubit_freq_offset_mhz = np.linspace(0, 3, 150)
    # for row_idx, freq_offset in enumerate(qubit_freq_offset_mhz):
    #     updated_exp = deepcopy(experiment)
    #     updated_exp.qubit_cfg['qubit_freq_chevron_detuned_ge'] = freq_offset +updated_exp.qubit_cfg['qubit_freq_ge'][QubitIndex]
    #     rabi = LengthRabiExperimentQZE(QubitIndex, tot_num_of_qubits, subStudyDataFolder, j, signal, save_figs,
    #                                    experiment=updated_exp, live_plot=live_plot,
    #                                    increase_qubit_reps=increase_qubit_reps,
    #                                    qubit_to_increase_reps_for=qubit_to_increase_reps_for,
    #                                    multiply_qubit_reps_by=multiply_qubit_reps_by,
    #                                    verbose=True, QZE=True, zeno_pulse_gain=0)
    #     (rabi_I_QZE_chev, rabi_Q_QZE_chev, rabi_magnitude_QZE_chev, rabi_gains_QZE_chev, rabi_fit_QZE_chev, pi_amp_QZE_chev,
    #      sys_config_rabi_QZE_chev) = rabi.run_rabi_chevron_after_pi_pulse(wait_for_res_ring_up=wait_for_res_ring_up,
    #                                                            exp=updated_exp, optimizationFolder=optimizationFolder,
    #                                                            subStudyFolder=subStudyDataFolder,
    #                                                            three_pulse_binary=False)
    #
    #     # ---------------------Collect Rabi Results----------------
    #     rabi_data[QubitIndex]['Dates'][0] = (
    #         time.mktime(datetime.datetime.now().timetuple()))
    #     rabi_data[QubitIndex]['I'][0] = rabi_I_QZE_chev
    #     rabi_data[QubitIndex]['Q'][0] = rabi_Q_QZE_chev
    #     rabi_data[QubitIndex]['Mag'][0] = rabi_magnitude_QZE_chev
    #     rabi_data[QubitIndex]['Gains'][0] = rabi_gains_QZE_chev
    #     rabi_data[QubitIndex]['Fit'][0] = rabi_fit_QZE_chev
    #     rabi_data[QubitIndex]['Round Num'][0] = j
    #     rabi_data[QubitIndex]['Batch Num'][0] = batch_num
    #     rabi_data[QubitIndex]['Exp Config'][0] = expt_cfg
    #     rabi_data[QubitIndex]['Syst Config'][0] = sys_config_rabi_QZE_chev
    #
    #     # --------------------------save Rabi-----------------------
    #     saver_rabi = Data_H5(subStudyDataFolder, rabi_data, 0, save_r)
    #     saver_rabi.save_to_h5('Rabi_QZE')
    #     del saver_rabi
    #     del rabi_data
    #
    #     # reset all dictionaries to none for safety
    #     rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)

    #################################################### Rabi QZE ######################################################
    qze_pulse_gains=np.linspace(0.001, 0.1, 20)
    #qze_pulse_gains = [0.01,0.1, 0.3, 0.6, 0.8, 1]

    I=[]
    Q=[]
    stark_res_qu_freq_sweep=None
    starked_freqs=[]
    for gain in qze_pulse_gains:
        #################################### try starking qubit freqency but qze style ####################################
        from section_004_qubit_spec_ge import QZEStyleResStarkShift2D
        import copy

        stark2D_keys = ['Dates', 'I', 'Q', 'Qu Frequency Sweep', 'Res Gain Sweep', 'Round Num', 'Batch Num',
                        'Exp Config',
                        'Syst Config']
        res_stark_data = create_data_dict(stark2D_keys, save_r, list_of_all_qubits)
        res_freq_stark = copy.deepcopy(experiment.readout_cfg['res_freq_ge'])
        res_freq_stark.append(res_freq_stark[QubitIndex])

        res_phase_stark = copy.deepcopy(experiment.readout_cfg['res_phase'])
        res_phase_stark.append(res_phase_stark[QubitIndex])
        res_stark_shift_2D = QZEStyleResStarkShift2D(QubitIndex, tot_num_of_qubits, optimizationFolder, res_freq_stark,
                                                     res_phase_stark, save_figs, experiment=experiment, zeno_stark_pulse_gain=gain)
        stark_res_I, stark_res_Q, stark_res_qu_freq_sweep, starked_freq, fwhm_starked, sys_config_stark_res = res_stark_shift_2D.run()

        # from section_004_qubit_spec_ge import ResStarkShift2DAdapted
        # res_stark_shift_2D = ResStarkShift2DAdapted(QubitIndex, tot_num_of_qubits, optimizationFolder, res_freq_stark,
        #                                              res_phase_stark, save_figs, experiment=experiment,
        #                                              )
        # stark_res_I, stark_res_Q, stark_res_qu_freq_sweep, sys_config_stark_res = res_stark_shift_2D.run()


        I.append(stark_res_I)
        Q.append(stark_res_Q)
        starked_freqs.append(starked_freq)
        print(fwhm_starked)
        res_stark_data[QubitIndex]['Dates'][0] = time.mktime(datetime.datetime.now().timetuple())
        res_stark_data[QubitIndex]['I'][0] = stark_res_I
        res_stark_data[QubitIndex]['Q'][0] = stark_res_Q
        res_stark_data[QubitIndex]['Qu Frequency Sweep'][0] = stark_res_qu_freq_sweep
        res_stark_data[QubitIndex]['Res Gain Sweep'][0] = None
        res_stark_data[QubitIndex]['Round Num'][0] = 0
        res_stark_data[QubitIndex]['Batch Num'][0] = 0
        res_stark_data[QubitIndex]['Exp Config'][0] = expt_cfg
        res_stark_data[QubitIndex]['Syst Config'][0] = sys_config_stark_res

        saver_stark_res = Data_H5(optimizationFolder, res_stark_data, 0, save_r)
        saver_stark_res.save_to_h5('stark_res_calibration')

        del saver_stark_res
        del res_stark_shift_2D
        del res_stark_data

        experiment.qubit_cfg['fwhm_w01_starked'] = fwhm_starked
        experiment.qubit_cfg['qubit_freq_ge_starked'][QubitIndex] = starked_freq #save the found starked freq

        ##################### start rabi experiment
        j += 1
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        # tof = TOFExperiment(QubitIndex, outerFolder, experiment, j, save_figs)
        # tof.run()

        ################################### Do Rabi and store the value #####################################
        exp = deepcopy(experiment) #before updating for qze on ch 7, use for qspec
        # exp.qubit_cfg['qubit_gain_ge'][
        #     QubitIndex] = 0
        rabi = LengthRabiExperimentQZE(QubitIndex, tot_num_of_qubits, subStudyDataFolder, j, signal, save_figs,
                                    experiment=exp, live_plot=live_plot,
                                    increase_qubit_reps=increase_qubit_reps,
                                    qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                    multiply_qubit_reps_by=multiply_qubit_reps_by,
                                    verbose=True, QZE=True, zeno_pulse_gain=gain)


        # # (rabi_I_QZE, rabi_Q_QZE, rabi_magnitude_QZE, rabi_gains_QZE, rabi_fit_QZE, pi_amp_QZE,
        # #  sys_config_rabi_QZE) = (
        # rabi.run_QZE_one_starked_qfreq(wait_for_res_ring_up=wait_for_res_ring_up,
        #                                                        exp=exp, optimizationFolder=optimizationFolder, subStudyFolder=subStudyDataFolder,
        #                                                         three_pulse_binary=False)

        (rabi_I_QZE, rabi_Q_QZE, rabi_magnitude_QZE, rabi_gains_QZE, rabi_fit_QZE, pi_amp_QZE,
         sys_config_rabi_QZE) = rabi.run_QZE_one_starked_qfreq(wait_for_res_ring_up=wait_for_res_ring_up,
                                                               exp=exp, optimizationFolder=optimizationFolder, subStudyFolder=subStudyDataFolder,
                                                                three_pulse_binary=False)

        # (rabi_I_QZE, rabi_Q_QZE, rabi_magnitude_QZE, rabi_gains_QZE, rabi_fit_QZE, pi_amp_QZE,
        #  sys_config_rabi_QZE) = rabi.run_QZE_starked_qfreq_test(wait_for_res_ring_up=wait_for_res_ring_up,
        #                                                        exp=exp, optimizationFolder=optimizationFolder,
        #                                                        subStudyFolder=subStudyDataFolder,
        #                                                        three_pulse_binary=False)

        print('Moving to next zeno gain loop')
        # rabi.run_QZE(thresholding=thresholding)
        # rabi.run_oscilliscope_zeno(thresholding=thresholding)

        del rabi

        ############################################### Collect Results ################################################

        # ---------------------Collect Rabi Results----------------
        rabi_data[QubitIndex]['Dates'][0] = (
            time.mktime(datetime.datetime.now().timetuple()))
        rabi_data[QubitIndex]['I'][0] = rabi_I_QZE
        rabi_data[QubitIndex]['Q'][0] = rabi_Q_QZE
        rabi_data[QubitIndex]['Mag'][0] = rabi_magnitude_QZE
        rabi_data[QubitIndex]['Gains'][0] = rabi_gains_QZE
        rabi_data[QubitIndex]['Fit'][0] = rabi_fit_QZE
        rabi_data[QubitIndex]['Round Num'][0] = j
        rabi_data[QubitIndex]['Batch Num'][0] = batch_num
        rabi_data[QubitIndex]['Exp Config'][0] = expt_cfg
        rabi_data[QubitIndex]['Syst Config'][0] = sys_config_rabi_QZE

        # --------------------------save Rabi-----------------------
        saver_rabi = Data_H5(subStudyDataFolder, rabi_data, 0, save_r)
        saver_rabi.save_to_h5('Rabi_QZE')
        del saver_rabi
        del rabi_data

        # reset all dictionaries to none for safety
        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)

    del experiment


    # import matplotlib.pyplot as plt
    #
    # ###check that im finding starked freq
    # plt.figure()
    # plt.plot(qze_pulse_gains,starked_freqs)
    # plt.xlabel('qze_pulse_gains')
    # plt.ylabel('starked_freq MHz')
    # #plt.show()
    # #### plot starked qubit spectroscopy if experiment was done above
    # fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    #
    # if type(qze_pulse_gains) is list:
    #     qze_pulse_gains=np.array(qze_pulse_gains)
    #
    # plot = axes[0]
    # plot.set_box_aspect(1)
    # plt.colorbar(plot.pcolormesh(stark_res_qu_freq_sweep, qze_pulse_gains, I, cmap="viridis"), ax=plot, shrink=0.7)
    # plot.set_title("I [a.u.]")
    # plot.set_ylabel("stark tone power [a.u.]")
    # plot.set_xlabel("qubit pulse frequency [MHz]")
    #
    # plot = axes[1]
    # plot.set_box_aspect(1)
    # plt.colorbar(plot.pcolormesh(stark_res_qu_freq_sweep, qze_pulse_gains, Q, cmap='viridis'), ax=plot, shrink=0.7)
    # plot.set_title("Q [a.u.]")
    # plot.set_ylabel("stark tone power [a.u.]")
    # plot.set_xlabel("qubit pulse frequency [MHz]")
    #
    # plot = axes[2]
    # plot.set_box_aspect(1)
    # plt.colorbar(plot.pcolormesh(stark_res_qu_freq_sweep, qze_pulse_gains, np.sqrt(np.square(I) + np.square(Q)), cmap='viridis'), ax=plot,
    #              shrink=0.7)
    # plot.set_title("magnitude")
    # plot.set_ylabel("stark tone power [a.u.]")
    # plot.set_xlabel("qubit pulse frequency [MHz]")
    #
    # plt.show()
    #
    #
    # now = datetime.datetime.now()
    # formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
    # file_name = os.path.join(optimizationFolder, f"{formatted_datetime}_"  + f"_q{QubitIndex}.png")
    # fig.savefig(file_name, dpi=100, bbox_inches='tight')
    # plt.close(fig)




