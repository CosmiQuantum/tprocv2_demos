import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
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
from f_to_res_swap_spec import FtoResQubitSpectroscopy
from analysis_020_gef_ssf_fstate_plots import GEF_SSF_ANALYSIS
from section_005_single_shot_ge import SingleShot
import copy
################################################ Run Configurations ####################################################
n= 1
save_r = 1            # how many rounds to save after
signal = 'None'       #'I', or 'Q' depending on where the signal is (after optimization). Put'None' if no optimization
save_figs = True    # save plots for RR experiments? Note; this is NOT asking whether to plot rabi chevron exp, that will always plot in this script
fig_quality = 200
live_plot = False      # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True      # fit the data here and save or plot the fits?
save_data_h5 = True   # save the data of the measurements you are taking to h5 files?
number_of_qubits = 6 # 4 for nexus, 6 for quiet
freq_offset_steps = 15
ssf_avgs_per_opt_pt = 3
ss_sample_number = 1
verbose=False
rr_logger=None
qick_verbose=False

Qs_to_look_at = [4] #only list the qubits you want to do the RR for

increase_qubit_reps = False #if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0 #only has impact if previous line is True
multiply_qubit_reps_by = 2 #only has impact if the line two above is True

#----For Rabi Chevron---
sigma_multiplier = 5
multiply_sigmas = True
save_shots_chev = True
#----------------------

#Folders
study = 'gain_rabi_ge_qfreq_study'
sub_study = 'rabi_Qfreq_2Dsweeps_wshots_quintupled_sigmas'

if not os.path.exists("/data/QICK_data/run6/"):
    os.makedirs("/data/QICK_data/run6/")
if not os.path.exists("/data/QICK_data/run6/6transmon/"):
    os.makedirs("/data/QICK_data/run6/6transmon/")
studyFolder = os.path.join("/data/QICK_data/run6/6transmon/", study)
subStudyFolder = os.path.join(studyFolder, sub_study)

formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
dataSetFolder = os.path.join(subStudyFolder, formatted_datetime)
optimizationFolder = os.path.join(dataSetFolder, 'optimization')
studyDocumentationFolder = os.path.join(dataSetFolder, 'documentation')
studyFolder = os.path.join(dataSetFolder, 'study_data')

path_saveplots = "/home/quietuser/acolonce/run6/6transmon/plots/"
path_saveplots_chev = os.path.join(path_saveplots, f'rabi_chev_plots/{formatted_datetime}')
path_saveplotsRR = os.path.join(path_saveplots, f'RR_plots/{formatted_datetime}')

if not os.path.exists(studyFolder): os.makedirs(studyFolder)
if not os.path.exists(subStudyFolder): os.makedirs(subStudyFolder)

if not os.path.exists(path_saveplots): os.makedirs(path_saveplots)
if not os.path.exists(path_saveplots_chev): os.makedirs(path_saveplots_chev)
if not os.path.exists(path_saveplotsRR): os.makedirs(path_saveplotsRR)

if not os.path.exists(studyDocumentationFolder): os.makedirs(studyDocumentationFolder)

# set which of the following measurements you would like to take
run_flags = {"res_spec_ge": True, "q_spec_ge": True, "rabi_ge_chevron": True, "ge_rabi": True,
             "repeated_ssf": True, "offset_opt": True}

##################################################### Additional Functions #####################################################################
def sweep_frequency_offset(experiment, QubitIndex, offset_values, n_loops=10, number_of_qubits=6,
                           outerFolder="", studyDocumentationFolder="",optimizationFolder="", j=0):
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

            ss = SingleShot(QubitIndex, number_of_qubits, outerFolder, 0, False, exp_copy)
            fid, angle, iq_list_g, iq_list_e, ss_config = ss.run()
            fids.append(fid)
            del exp_copy

            try:
                offset_data = create_data_dict(offset_keys, save_r, list_of_all_qubits)

                I_g = iq_list_g[QubitIndex][0].T[0]
                Q_g = iq_list_g[QubitIndex][0].T[1]
                I_e = iq_list_e[QubitIndex][0].T[0]
                Q_e = iq_list_e[QubitIndex][0].T[1]

                offset_data[QubitIndex]['Res Frequency'][0] = offset
                offset_data[QubitIndex]['Fidelity'][0] = fid
                offset_data[QubitIndex]['Angle'][0] = angle
                offset_data[QubitIndex]['Dates'][0] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                offset_data[QubitIndex]['I_g'][0] = I_g
                offset_data[QubitIndex]['Q_g'][0] = Q_g
                offset_data[QubitIndex]['I_e'][0] = I_e
                offset_data[QubitIndex]['Q_e'][0] = Q_e
                offset_data[QubitIndex]['Round Num'][0] = i
                offset_data[QubitIndex]['Batch Num'][0] = j
                offset_data[QubitIndex]['Exp Config'][0] = expt_cfg
                offset_data[QubitIndex]['Syst Config'][0] = ss_config

                saver_offset = Data_H5(outerFolder, offset_data, 0, save_r)
                saver_offset.save_to_h5('offset')
                del saver_offset
                del offset_data
                offset_data = create_data_dict(offset_keys, save_r, list_of_all_qubits)

            except Exception as e:
                raise e

        # find avg ssf
        avg_fid = np.mean(fids)
        ssf_dict[offset] = avg_fid

        print(f"Offset: {offset} -> Average SSF: {avg_fid:.4f}")

    # plt.figure()
    # offsets_sorted = sorted(ssf_dict.keys())
    # ssf_values = [ssf_dict[offset] for offset in offsets_sorted]
    # plt.plot(offsets_sorted, ssf_values, marker='o')
    # plt.xlabel('Frequency Offset')
    # plt.ylabel('Average SSF')
    # plt.title(f'SSF vs Frequency Offset for Qubit {QubitIndex + 1}')
    # plt.grid(True)
    # if outerFolder:
    #     os.makedirs(outerFolder, exist_ok=True)
    #     plot_path = os.path.join(studyDocumentationFolder, f"SSF_vs_offset_Q{QubitIndex + 1}.png")
    #     plt.savefig(plot_path)
    #     if verbose:
    #         print(f"Plot saved to {plot_path}")
    # plt.close()

    # Determine the offset value that yielded the best (highest) average SSF.
    optimal_offset = max(ssf_dict, key=ssf_dict.get)
    print(f"Optimal frequency offset for Qubit {QubitIndex + 1}: {optimal_offset} (Avg SSF: {ssf_dict[optimal_offset]:.4f})")

    return optimal_offset, ssf_dict

################################################ readout optimization vals ##################################################
res_leng_vals = [5.5, 7.5, 6.0, 6.5, 5.0, 6.0]
res_gain = [0.9, 0.95, 0.78, 0.58, 0.95, 0.57]

default_qubit_freqs = [ 4189.8656, 3820.4723, 4161.3726, 4463.15226, 4471.446, 4997.86] # from 3/11, qfreqs to fall back on if qspec is set to False
default_res_freqs = [6216.9331, 6275.9373, 6335, 6407.0338, 6476.1256, 6538] #04/07, res freqs to fall back on if res spec is set to False
default_ge_rabis = [0.6748, 0.634499, 0.76542, 0.7754, 0.6446, 0.9] # from 3/11, pi amps to fall back on if g-e rabi is set to False

####################################################### RR ##################################################################
def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num','Recycled QFreq',
              'Exp Config', 'Syst Config']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']

chev_keys_noshots = ['I', 'Q','Gains', 'Freqs_MHz', 'q_center_freq_MHz', 'res_freq_ge_MHz']
chev_keys_wshots = ['I', 'Q', 'Ishots', 'Qshots','Gains', 'Freqs_MHz', 'q_center_freq_MHz', 'res_freq_ge_MHz']

ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
offset_keys = ['Res Frequency', 'Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']

batch_num=0
j = 0
angles=[]

qubit_freqs_ge = np.zeros(6)
res_freq_ge = np.zeros(6)

for QubitIndex in Qs_to_look_at:
    #Get the config for this qubit
    experiment = QICK_experiment(path_saveplotsRR, DAC_attenuator1 = 5, DAC_attenuator2 = 10, ADC_attenuator = 10, fridge=FRIDGE)
    #Mask out all other resonators except this one
    res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex])
    experiment.readout_cfg['res_gain_ge'] = res_gains
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

    ################################################# g-e Res spec ####################################################
    if run_flags["res_spec_ge"]: #If set to false, it will fall back on res freqs already in the system config
        res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)

        res_spec = ResonanceSpectroscopy(QubitIndex, number_of_qubits, path_saveplotsRR, j, save_figs,
                                         experiment)
        res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
        experiment.readout_cfg['res_freq_ge'] = res_freqs

        this_res_freq = res_freqs[QubitIndex]
        res_freq_ge[QubitIndex] = float(this_res_freq)

        print('Qubit ', QubitIndex + 1, ' g-e res freq: ', this_res_freq)
        del res_spec

        res_data[QubitIndex]['Dates'][0] = (
            time.mktime(datetime.datetime.now().timetuple()))
        res_data[QubitIndex]['freq_pts'][0] = freq_pts
        res_data[QubitIndex]['freq_center'][0] = freq_center
        res_data[QubitIndex]['Amps'][0] = amps
        res_data[QubitIndex]['Found Freqs'][0] = res_freqs
        res_data[QubitIndex]['Round Num'][0] = j
        res_data[QubitIndex]['Batch Num'][0] = 0
        res_data[QubitIndex]['Exp Config'][0] = expt_cfg
        res_data[QubitIndex]['Syst Config'][0] = sys_config_rspec

        saver_res = Data_H5(optimizationFolder, res_data, j, save_r)  # save
        saver_res.save_to_h5('res_ge')
        del saver_res
        del res_data

    else:
        this_res_freq = default_res_freqs[QubitIndex]
        experiment.readout_cfg['res_freq_ge'][QubitIndex] = float(this_res_freq)
        res_freq_ge[QubitIndex] = float(this_res_freq)

    ################################################### g-e Qubit spec #############################################################
    if run_flags["q_spec_ge"]:
        qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
        q_spec = QubitSpectroscopy(QubitIndex, number_of_qubits, path_saveplotsRR, j, signal, save_figs, experiment, live_plot, verbose = False, logger = None, qick_verbose = True, increase_reps = False, increase_reps_to = 500)
        qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq, sys_config_qspec = q_spec.run()

        qubit_freqs_ge[QubitIndex] = qubit_freq
        experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
        print('Qubit ', QubitIndex + 1, ' g-e freq: ', float(qubit_freq))
        del q_spec

        qspec_data[QubitIndex]['Dates'][0] = (time.mktime(datetime.datetime.now().timetuple()))
        qspec_data[QubitIndex]['I'][0] = qspec_I
        qspec_data[QubitIndex]['Q'][0] = qspec_Q
        qspec_data[QubitIndex]['Frequencies'][0] = qspec_freqs
        qspec_data[QubitIndex]['I Fit'][0] = qspec_I_fit
        qspec_data[QubitIndex]['Q Fit'][0] = qspec_Q_fit
        qspec_data[QubitIndex]['Round Num'][0] = j
        qspec_data[QubitIndex]['Batch Num'][0] = 0
        qspec_data[QubitIndex]['Recycled QFreq'][0] = False  # no rr so no recycling here
        qspec_data[QubitIndex]['Exp Config'][0] = expt_cfg
        qspec_data[QubitIndex]['Syst Config'][0] = sys_config_qspec

        saver_qspec = Data_H5(optimizationFolder, qspec_data, 0, save_r)
        saver_qspec.save_to_h5('qspec_ge')
        del saver_qspec
        del qspec_data

    else:
        qubit_freq = default_qubit_freqs[QubitIndex]
        print(f"Q_spec disabled → using default g-e freq for Q{QubitIndex + 1}: {qubit_freq:.4f} MHz")
        experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)

    ################################################ g-e amp rabi #############################################################
    if run_flags["ge_rabi"]:
        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
        rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, path_saveplotsRR, j,
                                       signal, save_figs=save_figs, experiment=experiment,
                                       live_plot=live_plot,
                                       increase_qubit_reps=increase_qubit_reps,
                                       qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                       multiply_qubit_reps_by=multiply_qubit_reps_by,
                                       verbose=verbose, logger=rr_logger,
                                       qick_verbose=qick_verbose)
        (rabi_I, rabi_Q, rabi_gains, rabi_fit, stored_pi_amp, sys_config_rabi) = rabi.run()
        experiment.qubit_cfg['pi_amp'][QubitIndex] = float(stored_pi_amp)
        print(f"Pi amplitude for qubit {QubitIndex + 1}: {float(stored_pi_amp)}")

        rabi_data[QubitIndex]['Dates'][0] = (
            time.mktime(datetime.datetime.now().timetuple()))
        rabi_data[QubitIndex]['I'][0] = rabi_I
        rabi_data[QubitIndex]['Q'][0] = rabi_Q
        rabi_data[QubitIndex]['Gains'][0] = rabi_gains
        rabi_data[QubitIndex]['Fit'][0] = rabi_fit
        rabi_data[QubitIndex]['Round Num'][0] = j
        rabi_data[QubitIndex]['Batch Num'][0] = 0
        rabi_data[QubitIndex]['Exp Config'][0] = expt_cfg
        rabi_data[QubitIndex]['Syst Config'][0] = sys_config_rabi
        saver_rabi = Data_H5(optimizationFolder, rabi_data, 0, save_r)
        saver_rabi.save_to_h5('rabi_ge')
        del rabi
        del saver_rabi
        del rabi_data
    else:
        ge_amp_rabi = default_ge_rabis[QubitIndex]
        print(f"g-e Rabi disabled → using default pi amps for Q{QubitIndex + 1}: {ge_amp_rabi:.4f} MHz")
        experiment.qubit_cfg['pi_amp'][QubitIndex] = float(ge_amp_rabi)

    ################################################ optimize freq offset ################################################
    if run_flags["offset_opt"]:
        reference_frequency = float(res_freq_ge[QubitIndex])
        freq_range = np.linspace(-1, 1, freq_offset_steps)

        optimal_offset, ssf_dict = sweep_frequency_offset(experiment, QubitIndex, freq_range,
                                                          n_loops=ssf_avgs_per_opt_pt, number_of_qubits=6,
                                                          outerFolder=optimizationFolder,
                                                          studyDocumentationFolder=studyDocumentationFolder, j=0)

        offset_res_freqs = [r + optimal_offset for r in res_freq_ge]
        experiment.readout_cfg['res_freq_ge'][QubitIndex] = offset_res_freqs[QubitIndex]  # update with offset added
        print(experiment.readout_cfg)

    ################################################ repeated ss ################################################
    if run_flags["repeated_ssf"]:
        ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
        angles = []
        thresholds = []
        for ss_round in range(ss_sample_number):
            try:
                ss = SingleShot(QubitIndex, tot_num_of_qubits, path_saveplotsRR, 0, False, experiment=experiment,
                                verbose=verbose, logger=rr_logger)
                fid, angle, iq_list_g, iq_list_e, sys_config_ss = ss.run()
                I_g = iq_list_g[QubitIndex][0].T[0]
                Q_g = iq_list_g[QubitIndex][0].T[1]
                I_e = iq_list_e[QubitIndex][0].T[0]
                Q_e = iq_list_e[QubitIndex][0].T[1]

                fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(
                    data=[I_g, Q_g, I_e, Q_e], cfg=ss.config, plot=save_figs)
                angles.append(angle)
                thresholds.append(threshold)

                ss_data[QubitIndex]['Fidelity'][0] = fid
                ss_data[QubitIndex]['Angle'][0] = angle
                ss_data[QubitIndex]['Dates'][0] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                ss_data[QubitIndex]['I_g'][0] = I_g
                ss_data[QubitIndex]['Q_g'][0] = Q_g
                ss_data[QubitIndex]['I_e'][0] = I_e
                ss_data[QubitIndex]['Q_e'][0] = Q_e
                ss_data[QubitIndex]['Round Num'][0] = ss_round
                ss_data[QubitIndex]['Batch Num'][0] = 0
                ss_data[QubitIndex]['Exp Config'][0] = expt_cfg
                ss_data[QubitIndex]['Syst Config'][0] = sys_config_ss

                saver_ss = Data_H5(optimizationFolder, ss_data, 0, save_r)
                saver_ss.save_to_h5('ss_ge')
                del saver_ss
                del ss_data

                ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
            except Exception as e:
                print(f'Got the following error, continuing: {e}')
                continue  # skip the rest of this

        # average angles and thresholds to use
        avg_angle = np.mean(angles)
        avg_thresh = np.mean(thresholds)
        experiment.readout_cfg['res_phase'][
            QubitIndex] = avg_angle * 180 / np.pi  # need it to be a list of 6, the other qubits dont matter so just amke them the same val
        experiment.readout_cfg['threshold'][QubitIndex] = avg_thresh

    ############################################# Make a copies of og experiment: IMPORTANT #############################################
    experiment_template = copy.deepcopy(experiment) #optimized master copy
    chevron_template = copy.deepcopy(experiment_template) #separate chevron‐specific template

    # multiply (increase) sigmas only in the chevron copy
    if multiply_sigmas:
        for q in Qs_to_look_at:
            chevron_template.qubit_cfg['sigma'][q] *= sigma_multiplier

    del experiment
    ####################################################### 2D sweep (Rabi Chevron) ######################################################
    if run_flags["rabi_ge_chevron"]:
        """
        rabi_I and rabi_Q are each a Python list, of length = number of rabi gain points, containing the averaged I- and Q-values at each gain.
        rabi_Ishots and rabi_Qshots (when save_shots=True) are lists of length = number of rabi gain points, where each element is itself a list or array of the individual shot values for that gain.
        rabi_gains is usually a NumPy array (or at least a list) of the rabi gain amplitudes you swept over.
        
        all_rabi_I is a 2-D numpy array of shape ( # freq steps, # gain points ), where each element is a single float (the averaged I value for that qubit frequency and rabi gain).
        all_rabi_Q is a 2-D numpy array of shape ( # freq steps, # gain points ), where each element is a single float (the averaged Q value for that qubit frequency and rabi gain).
        example of all_rabi_I and all_rabi_Q structure:
            array([[0.10, 0.20, 0.15],
            [0.12, 0.22, 0.18]])
        """
        # frequency grid around optimized qubit freq center
        freq_steps = 45
        freqs_mhz = np.linspace(qubit_freq - 5, qubit_freq + 5, freq_steps)

        signal_map = []  # will have shape (len(freqs), len(gains))
        all_rabi_I = [] # to save I data in h5 files
        all_rabi_Q = [] # to save Q data in h5 files
        all_rabi_Ishots = [] # to save Ishots data in h5 files (only used if save_shots = True)
        all_rabi_Qshots = [] # to save Qshots data in h5 files (only used if save_shots = True)

        for f in freqs_mhz:
            print('Rabi 2D sweep ongoing...')
            # get the “optimized” experiment defined above. If you set multiply_sigmas = True, this template includes the updated sigmas.
            experiment = copy.deepcopy(chevron_template)

            # change the qubit drive freq
            experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(f)

            # run the gain‐sweep Rabi
            rabi = AmplitudeRabiExperiment(QubitIndex, number_of_qubits, path_saveplotsRR,0, signal, save_shots = save_shots_chev,
                    save_figs=False, experiment=experiment, live_plot=live_plot, increase_qubit_reps=increase_qubit_reps,
                    qubit_to_increase_reps_for=qubit_to_increase_reps_for, multiply_qubit_reps_by=multiply_qubit_reps_by, verbose=verbose,
                    logger=rr_logger, qick_verbose=qick_verbose)
            if save_shots_chev:
                rabi_I, rabi_Q, rabi_Ishots, rabi_Qshots, rabi_gains, *_ = rabi.run()
                all_rabi_Ishots.append(rabi_Ishots)
                all_rabi_Qshots.append(rabi_Qshots)
            else:
                rabi_I, rabi_Q, rabi_gains, *_ = rabi.run()

            all_rabi_I.append(rabi_I) # all_rabi_I is a 2-D numpy array of shape ( # freq steps, # gain points )
            all_rabi_Q.append(rabi_Q) # all_rabi_Q is a 2-D numpy array of shape ( # freq steps, # gain points )

            # compute signal magnitude √(I²+Q²)
            mag = np.sqrt(np.array(rabi_I) ** 2 + np.array(rabi_Q) ** 2)
            signal_map.append(mag)

            del rabi
            del experiment

        signal_map = np.vstack(signal_map)  # shape (freq_steps, len(rabi_gains))

        if save_data_h5:
            if save_shots_chev:
                chev_data = create_data_dict(chev_keys_wshots, save_r, list_of_all_qubits)
                # also stack I and Q data into arrays of shape (freq_steps, n_gain_points)
                all_rabi_I = np.vstack(all_rabi_I)
                all_rabi_Q = np.vstack(all_rabi_Q)

                chev_data[QubitIndex]['I'][0] = all_rabi_I  # [0] is the round number, always zero since we don't use that parameter in this script
                chev_data[QubitIndex]['Q'][0] = all_rabi_Q

                chev_data[QubitIndex]['Ishots'][0] = all_rabi_Ishots
                chev_data[QubitIndex]['Qshots'][0] = all_rabi_Qshots

                chev_data[QubitIndex]['Gains'][0] = rabi_gains
                chev_data[QubitIndex]['Freqs_MHz'][0] = freqs_mhz
                chev_data[QubitIndex]['q_center_freq_MHz'][0] = qubit_freq
                chev_data[QubitIndex]['res_freq_ge_MHz'][0] = this_res_freq

                saver_chev = Data_H5(studyFolder, chev_data, 0, save_r)
                saver_chev.save_to_h5('rabi_ge_chevron_wshots')
            else: # not saving shots
                chev_data = create_data_dict(chev_keys_noshots, save_r, list_of_all_qubits)
                # also stack I and Q data into arrays of shape (freq_steps, n_gain_points)
                all_rabi_I = np.vstack(all_rabi_I)
                all_rabi_Q = np.vstack(all_rabi_Q)

                chev_data[QubitIndex]['I'][0] = all_rabi_I #shape (n_rounds, n_freqs, n_gains). [0] is the round number, always zero since we don't use that parameter in this script.
                chev_data[QubitIndex]['Q'][0] = all_rabi_Q #shape (n_rounds, n_freqs, n_gains)
                chev_data[QubitIndex]['Gains'][0] = rabi_gains
                chev_data[QubitIndex]['Freqs_MHz'][0] = freqs_mhz
                chev_data[QubitIndex]['q_center_freq_MHz'][0] = qubit_freq
                chev_data[QubitIndex]['res_freq_ge_MHz'][0] = this_res_freq

                saver_chev = Data_H5(studyFolder, chev_data, 0, save_r)
                saver_chev.save_to_h5('rabi_ge_chevron_noshots')
            del saver_chev

        # Plot chevron
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(signal_map, aspect='auto', origin='lower',
            extent=[rabi_gains[0],  # gain min
                    rabi_gains[-1],  # gain max
                    freqs_mhz[0],  # freq min (MHz)
                    freqs_mhz[-1]  # freq max (MHz)
                    ])
        ax.set_xlabel('Gain (amplitude)')
        ax.set_ylabel('Qubit Drive frequency (MHz)')
        ax.set_title(f'Rabi Chevron: Qubit {QubitIndex + 1}; g-e Qfreq = {qubit_freq:.4f}')
        plt.colorbar(im, ax=ax, label='IQ Signal Mag (a. u.)')
        plt.tight_layout()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rabichevron_Q{QubitIndex + 1}_{timestamp}.png"
        fig.savefig(os.path.join(path_saveplots_chev, filename), dpi=fig_quality)
        plt.close(fig)

        del experiment_template
        del chevron_template