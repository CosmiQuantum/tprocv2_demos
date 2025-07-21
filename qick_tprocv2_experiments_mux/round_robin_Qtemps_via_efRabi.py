import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pprint as pp
from scipy.constants import hbar, k, pi
from qualang_tools.plot import Fit
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
from section_005_single_shot_ge import SingleShot
from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from fitting import fitdecaysin
from f_to_res_swap_spec import FtoResQubitSpectroscopy
from analysis_020_gef_ssf_fstate_plots import GEF_SSF_ANALYSIS
from section_011_qubit_temperatures_efRabipt2 import LengthRabiExperiment
from section_011_qubit_temperatures_efRabipt3 import Temps_EFAmpRabiExperiment
################################################ Run Configurations ####################################################
n= 1
save_r = 1            # how many rounds to save after
signal = 'None'       #'I', or 'Q' depending on where the signal is (after optimization). Put'None' if no optimization
save_figs = True    # save plots for everything as you go along the RR script?
fig_quality = 200
live_plot = False      # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True      # fit the data here and save or plot the fits?
save_data_h5 = False   # save the data of the measurements you are taking to h5 files?
SS = False  # True if using single-shot normalization
ef = True
IS_VISDOM = False

number_of_qubits = 6 # 4 for nexus, 6 for quiet

Qs_to_look_at = [4] #only list the qubits you want to do the RR for

increase_qubit_reps = False #if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0 #only has impact if previous line is True
multiply_qubit_reps_by = 2 #only has impact if the line two above is True

increase_qubit_steps_ef = False #if you want to increase the steps for all qubits, set to True, if you only want to set it to true for 1 qubit, see e-f qubit spec section
increase_steps_to_ef = 600

outerFolder = os.path.join("/data/QICK_data/run6/6transmon/ef_studies/QubitTemps_efRabi_method/", str(datetime.date.today()), "Optimization/Round_Robin_mode")
outerFolder_qtemps = os.path.join("/data/QICK_data/run6/6transmon/ef_studies/QubitTemps_efRabi_method/", str(datetime.date.today()), "Study_Data")

if not os.path.exists(outerFolder): os.makedirs(outerFolder)
if not os.path.exists(outerFolder_qtemps): os.makedirs(outerFolder_qtemps)


# # set which of the following measurements you would like to take
run_flags = {"res_spec_ge": True, "q_spec_ge": True, "rabi_ge": True, "res_spec_ef": True, "q_spec_ef": True,
             "ss_gef": False, "ss_ge": False, "ss_gef_fstate": False, "rabi_ef": False, "t1_ge": False, "t1_fg": False, "t1_fe": False,
             "FtoRes_Spec": False, "Qtemps": True}

################################################ optimization outputs ##################################################
res_leng_vals = [4.3, 6, 5, 6.1, 5.8, 7]
res_gain = [0.9600, 1, 0.7200, 0.5733, 0.8, 0.55]
freq_offsets = [-0.05, -0.19, -0.19, -0.15, -0.2, -0.05]
####################################################### RR #############################################################

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

rabi_keys_ef_Qtemps = ['Dates', 'Qfreq_ge', 'I1', 'Q1', 'Gains1', 'Fit1', 'I2', 'Q2', 'Gains2', 'Fit2', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']

ss_keys_gef = ['Fidelity', 'Angle_ge', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'I_f', 'Q_f', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
#initialize a dictionary to store those values
res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
t1_data_eg = create_data_dict(t1_keys, save_r, list_of_all_qubits)
res_data_ef = create_data_dict(res_keys_ef, save_r, list_of_all_qubits)
qspec_data_ef = create_data_dict(qspec_keys_ef, save_r, list_of_all_qubits)
rabi_data_ef = create_data_dict(rabi_keys_ef, save_r, list_of_all_qubits)
rabi_data_ef_Qtemps = create_data_dict(rabi_keys_ef_Qtemps, save_r, list_of_all_qubits)
t1_data_fg = create_data_dict(t1_keys, save_r, list_of_all_qubits)
t1_data_fe = create_data_dict(t1_keys, save_r, list_of_all_qubits)
ss_data_gef = create_data_dict(ss_keys_gef, save_r, list_of_all_qubits)
ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)

batch_num=0
j = 0
angles=[]

qubit_freqs_ge = [None]*6
qubit_freqs_ef = np.zeros(6)
res_freq_ge = np.zeros(6)

while j < n:
    j += 1
    for QubitIndex in Qs_to_look_at:
        #Get the config for this qubit
        experiment = QICK_experiment(outerFolder, DAC_attenuator1 = 5, DAC_attenuator2 = 10, ADC_attenuator = 10, fridge=FRIDGE)
        #Mask out all other resonators except this one
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex])
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_gain_ef'] = res_gains
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        ################################################# g-e Res spec ####################################################
        if run_flags["res_spec_ge"]:
            while True:
                try:
                    res_spec = ResonanceSpectroscopy(QubitIndex, number_of_qubits, outerFolder, j, save_figs,
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
                    break  # Exit the retry loop if successful
                except Exception as e:
                    print(f"Retrying g-e Res Spec for Qubit {QubitIndex + 1} due to error: {e}")

        ################################################### g-e Qubit spec ##################################################
        if run_flags["q_spec_ge"]:
            while True:
                try:
                    q_spec = QubitSpectroscopy(QubitIndex, number_of_qubits, outerFolder, j, signal, save_figs, experiment,
                                               live_plot, verbose=False, logger=None, qick_verbose=True, increase_reps=False,
                                               increase_reps_to=500)
                    qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq, sys_config_qspec = q_spec.run()

                    qubit_freqs_ge[QubitIndex] = qubit_freq
                    experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
                    print('Qubit ', QubitIndex + 1, ' g-e freq: ', float(qubit_freq))
                    del q_spec
                    break  # Exit the retry loop if successful
                except Exception as e:
                    print(f"Retrying g-e QSpec for Qubit {QubitIndex + 1} due to error: {e}")

        ###################################################### g-e Rabi ####################################################
        if run_flags["rabi_ge"]:
            while True:
                try:
                    rabi = AmplitudeRabiExperiment(QubitIndex, number_of_qubits, outerFolder, j, signal, save_figs,
                                                   experiment, live_plot,
                                                   increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)

                    rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_rabi = rabi.run()

                    experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
                    print('Qubit ', QubitIndex + 1, ' g-e Pi Amp: ', float(pi_amp))

                    del rabi
                    break  # Exit the retry loop if successful
                except Exception as e:
                    print(f"Retrying g-e Rabi for Qubit {QubitIndex + 1} due to error: {e}")

        ################################################# e-f Res spec ####################################################
        if run_flags["res_spec_ef"]:
            while True:
                try:
                    res_specEF = ResonanceSpectroscopyEF(QubitIndex, number_of_qubits, outerFolder, j, save_figs,
                                                         experiment)
                    res_freqs, freq_pts, freq_center, amps, sys_config_rspec_ef = res_specEF.run()
                    experiment.readout_cfg['res_freq_ef'] = res_freqs
                    print('Qubit ', QubitIndex + 1, ' e-f res freq: ', res_freqs[QubitIndex])

                    del res_specEF
                    break  # Exit the retry loop if successful
                except Exception as e:
                    print(f"Retrying e-f Res Spec for Qubit {QubitIndex + 1} due to error: {e}")

        ################################################## e-f Qubit spec ##################################################
        if run_flags["q_spec_ef"]:
            while True:
                try:
                    # Qubit 4 needs more steps for e-f spec
                    if QubitIndex == 3:
                        increase_qubit_steps_ef = True  # if you want to increase the steps for a qubit, set to True

                    ef_q_spec = EFQubitSpectroscopy(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, signal,
                                                    save_figs, experiment, live_plot, increase_qubit_steps_ef,
                                                    increase_steps_to_ef)
                    efqspec_I, efqspec_Q, efqspec_freqs, efqspec_I_fit, efqspec_Q_fit, efqubit_freq, sys_config_qspec_ef = ef_q_spec.run(
                        experiment.soccfg,
                        experiment.soc)
                    qubit_freqs_ef[QubitIndex] = efqubit_freq
                    experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] = float(efqubit_freq)
                    print('Qubit ', QubitIndex + 1, ' e-f Freq: ', float(efqubit_freq))

                    del ef_q_spec
                    break  # Exit the retry loop if successful
                except Exception as e:
                    print(f"Retrying e-f Qspec for Qubit {QubitIndex + 1} due to error: {e}")

        ###################################################### e-f Rabi ####################################################
        if run_flags["rabi_ef"]:
            while True:
                try:
                    efrabi = EF_AmplitudeRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j,
                                                        signal, save_figs,
                                                        experiment, live_plot,
                                                        increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
                    efrabi_I, efrabi_Q, efrabi_gains, efrabi_fit, efpi_amp, sys_config_rabi_ef = efrabi.run(experiment.soccfg,
                                                                                                            experiment.soc)

                    experiment.qubit_cfg['pi_ef_amp'][QubitIndex] = float(efpi_amp)
                    print('Qubit ', QubitIndex + 1, ' e-f pulse amp: ', float(efpi_amp))

                    del efrabi
                    break  # Exit the retry loop if successful
                except Exception as e:
                    print(f"Retrying e-f Rabi for Qubit {QubitIndex + 1} due to error: {e}")
        ###################################################### g-e SSF measurements ######################################################
        if run_flags["ss_ge"]:
            while True:
                try:
                    rr_logger = False
                    verbose = False
                    debug_mode = False
                    ss = SingleShot(QubitIndex, tot_num_of_qubits, outerFolder, j, save_figs, experiment = experiment,
                                    verbose = verbose, logger = rr_logger)
                    fid, angle, iq_list_g, iq_list_e, sys_config_ss = ss.run()
                    I_g = iq_list_g[QubitIndex][0].T[0]
                    Q_g = iq_list_g[QubitIndex][0].T[1]
                    I_e = iq_list_e[QubitIndex][0].T[0]
                    Q_e = iq_list_e[QubitIndex][0].T[1]

                    fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(data=[I_g, Q_g, I_e, Q_e], cfg=ss.config, plot=save_figs)
                    del ss
                    break  # Exit the retry loop if successful
                except Exception as e:
                    print(f"Retrying g-e SSF for Qubit {QubitIndex + 1} due to error: {e}")
        #-------------------------------------------------Qubit Temps: Rabi Population Measurements------------------------------------------------------------------------
        if run_flags["Qtemps"]:
            while True:
                try:
                    efAmprabi_Qtemps = Temps_EFAmpRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder_qtemps, j,
                                                        signal, save_figs,
                                                        experiment, live_plot,
                                                        increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
                    (I1_qtemp, Q1_qtemp, gains1_qtemp, fit_cosine1_qtemp, pi_amp1_qtemp, A_amplitude1, amp_fit1,
                     I2_qtemp, Q2_qtemp, gains2_qtemp, fit_cosine2_qtemp, pi_amp2_qtemp, A_amplitude2, amp_fit2, sysconfig_efrabi_Qtemps) = efAmprabi_Qtemps.run(experiment.soccfg, experiment.soc)
                    del efAmprabi_Qtemps
                    break  # Exit the retry loop if successful
                except Exception as e:
                    print(f"Retrying Qubit Temps RPMs for Qubit {QubitIndex + 1} due to error: {e}")
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

            # ---------------------Collect Single Shot Results----------------
            if run_flags["ss_ge"]:
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

            # ---------------------Collect Results for Qubit Temperature Rabis ----------------
            if run_flags["Qtemps"]:
                rabi_data_ef_Qtemps[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                rabi_data_ef_Qtemps[QubitIndex]['Qfreq_ge'][j - batch_num * save_r - 1] = qubit_freqs_ge[QubitIndex] #save the qubit freq too for this qubit

                rabi_data_ef_Qtemps[QubitIndex]['I1'][j - batch_num * save_r - 1] = I1_qtemp
                rabi_data_ef_Qtemps[QubitIndex]['Q1'][j - batch_num * save_r - 1] = Q1_qtemp
                rabi_data_ef_Qtemps[QubitIndex]['Gains1'][j - batch_num * save_r - 1] = gains1_qtemp
                rabi_data_ef_Qtemps[QubitIndex]['Fit1'][j - batch_num * save_r - 1] = fit_cosine1_qtemp

                rabi_data_ef_Qtemps[QubitIndex]['I2'][j - batch_num * save_r - 1] = I2_qtemp
                rabi_data_ef_Qtemps[QubitIndex]['Q2'][j - batch_num * save_r - 1] = Q2_qtemp
                rabi_data_ef_Qtemps[QubitIndex]['Gains2'][j - batch_num * save_r - 1] = gains2_qtemp
                rabi_data_ef_Qtemps[QubitIndex]['Fit2'][j - batch_num * save_r - 1] = fit_cosine2_qtemp

                rabi_data_ef_Qtemps[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                rabi_data_ef_Qtemps[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                rabi_data_ef_Qtemps[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                rabi_data_ef_Qtemps[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sysconfig_efrabi_Qtemps

        del experiment

    ################################################## Potentially Save ################################################
    if save_data_h5:
        # Check if you are at the right round number
        # If so, then save all of the data and change the round num so you replace data starting next round
        if j % save_r == 0:
            batch_num += 1

            # --------------------------save g-e Res Spec-----------------------
            if run_flags["res_spec_ge"]:
                saver_res = Data_H5(outerFolder, res_data, batch_num, save_r)
                saver_res.save_to_h5('Res_ge')
                del saver_res
                del res_data

            # --------------------------save e-f Res Spec-----------------------
            if run_flags["res_spec_ef"]:
                saver_res = Data_H5(outerFolder, res_data_ef, batch_num, save_r)
                saver_res.save_to_h5('Res_ef')
                del saver_res
                del res_data_ef

            # --------------------------save g-e QSpec-----------------------
            if run_flags["q_spec_ge"]:
                saver_qspec = Data_H5(outerFolder, qspec_data, batch_num, save_r)
                saver_qspec.save_to_h5('QSpec_ge')
                del saver_qspec
                del qspec_data

            # --------------------------save e-f QSpec-----------------------
            if run_flags["q_spec_ef"]:
                saver_qspec = Data_H5(outerFolder, qspec_data_ef, batch_num, save_r)
                saver_qspec.save_to_h5('QSpec_ef')
                del saver_qspec
                del qspec_data_ef

            # --------------------------save g-e Rabi-----------------------
            if run_flags["rabi_ge"]:
                saver_rabi = Data_H5(outerFolder, rabi_data, batch_num, save_r)
                saver_rabi.save_to_h5('Rabi_ge')
                del saver_rabi
                del rabi_data

            # --------------------------save e-f Rabi-----------------------
            if run_flags["rabi_ef"]:
                saver_rabi = Data_H5(outerFolder, rabi_data_ef, batch_num, save_r)
                saver_rabi.save_to_h5('Rabi_ef')
                del saver_rabi
                del rabi_data_ef

            # --------------------------save SS-----------------------
            if run_flags["ss_ge"]:
                saver_ss = Data_H5(outerFolder, ss_data, batch_num, save_r)
                saver_ss.save_to_h5('SS_ge')
                del saver_ss
                del ss_data

            # --------------------------save Qubit Temperatures-----------------------
            if run_flags["Qtemps"]:
                saver_rabi_Qtemps = Data_H5(outerFolder_qtemps, rabi_data_ef_Qtemps, batch_num, save_r)
                saver_rabi_Qtemps.save_to_h5('Qtemps')
                del saver_rabi_Qtemps
                del rabi_data_ef_Qtemps

            # reset all dictionaries to none for safety
            res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
            res_data_ef = create_data_dict(res_keys_ef, save_r, list_of_all_qubits)

            qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
            qspec_data_ef = create_data_dict(qspec_keys_ef, save_r, list_of_all_qubits)

            rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
            rabi_data_ef = create_data_dict(rabi_keys_ef, save_r, list_of_all_qubits)

            rabi_data_ef_Qtemps = create_data_dict(rabi_keys_ef_Qtemps, save_r, list_of_all_qubits)

            ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)