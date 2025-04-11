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
sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
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
save_data_h5 = True   # save the data of the measurements you are taking to h5 files?
SS = False  # True if using single-shot normalization
ef = True
IS_VISDOM = False

number_of_qubits = 6 # 4 for nexus, 6 for quiet

Qs_to_look_at = [0] #only list the qubits you want to do the RR for

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
             "ss_gef": True, "ss_gef_fstate": True, "rabi_ef": True, "t1_ge": False, "t1_fg": False, "t1_fe": False,
             "FtoRes_Spec": False, "Qtemps": True}

# run_flags = {"res_spec_ge": False, "q_spec_ge": False, "rabi_ge": False, "res_spec_ef": False, "q_spec_ef": False,
#              "ss_gef": False, "ss_gef_fstate": False, "rabi_ef": False, "t1_ge": False, "t1_fg": False, "t1_fe": False,
#              "FtoRes_Spec": False, "Qtemps": False}

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

rabi_keys_ef_Qtemps = ['Dates', 'I1', 'Q1', 'Gains1', 'Fit1', 'I2', 'Q2', 'Gains2', 'Fit2', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']

ss_keys_gef = ['Fidelity', 'Angle_ge', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'I_f', 'Q_f', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
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

batch_num=0
j = 0
angles=[]

qubit_freqs_ge = np.zeros(6)
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

        ################################################### g-e Qubit spec ##################################################
        if run_flags["q_spec_ge"]:
            q_spec = QubitSpectroscopy(QubitIndex, number_of_qubits, outerFolder, j, signal, save_figs, experiment,
                                       live_plot, verbose=False, logger=None, qick_verbose=True, increase_reps=False,
                                       increase_reps_to=500)
            qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq, sys_config_qspec = q_spec.run()

            qubit_freqs_ge[QubitIndex] = qubit_freq
            experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
            print('Qubit ', QubitIndex + 1, ' g-e freq: ', float(qubit_freq))
            del q_spec

        ###################################################### g-e Rabi ####################################################
        if run_flags["rabi_ge"]:
            rabi = AmplitudeRabiExperiment(QubitIndex, number_of_qubits, outerFolder, j, signal, save_figs,
                                           experiment, live_plot,
                                           increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)

            rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_rabi = rabi.run()

            experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
            print('Qubit ', QubitIndex + 1, ' g-e Pi Amp: ', float(pi_amp))

            del rabi

        ################################################# e-f Res spec ####################################################
        if run_flags["res_spec_ef"]:
            res_specEF = ResonanceSpectroscopyEF(QubitIndex, number_of_qubits, outerFolder, j, save_figs,
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

        ###################################################### e-f Rabi ####################################################
        if run_flags["rabi_ef"]:
            efrabi = EF_AmplitudeRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j,
                                                signal, save_figs,
                                                experiment, live_plot,
                                                increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
            efrabi_I, efrabi_Q, efrabi_gains, efrabi_fit, efpi_amp, sys_config_rabi_ef = efrabi.run(experiment.soccfg,
                                                                                                    experiment.soc)

            experiment.qubit_cfg['pi_ef_amp'][QubitIndex] = float(efpi_amp)
            print('Qubit ', QubitIndex + 1, ' e-f pulse amp: ', float(efpi_amp))

            del efrabi
        #-------------------------------------------------Qubit Temps: Rabi Population Measurements------------------------------------------------------------------------
        if run_flags["Qtemps"]:
            # QTEMPS = EFRabiQubitTempsExperiment(QubitIndex, number_of_qubits, experiment, outerFolder_qtemps, j)
            # fit_result, fit_result_ref = QTEMPS.run(experiment.soccfg, experiment.soc, expt_cfg, outerFolder_qtemps, IS_VISDOM, save_data_h5, SS)
            # print(fit_result)

            # len_rabi = LengthRabiExperiment(QubitIndex, tot_num_of_qubits, outerFolder, j,
            #                                 signal, save_figs=True, experiment=experiment,
            #                                 live_plot=live_plot,
            #                                 increase_qubit_reps=increase_qubit_reps,
            #                                 qubit_to_increase_reps_for=qubit_to_increase_reps_for,
            #                                 multiply_qubit_reps_by=multiply_qubit_reps_by,
            #                                 verbose=False, logger=None,
            #                                 qick_verbose=True)
            # (I1_Lrabi, Q1_Lrabi, lengths1_Lrabi, q1_fit_cosine_1_Lrabi, pi_len_1_Lrabi, I2_Lrabi, Q2_Lrabi, lengths2_Lrabi, q1_fit_cosine_2_Lrabi, pi_len_2_Lrabi, sys_config_Lrabi) = len_rabi.run_rabiforQtemps()

            efAmprabi_Qtemps = Temps_EFAmpRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder_qtemps, j,
                                                signal, save_figs,
                                                experiment, live_plot,
                                                increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
            (I1_qtemp, Q1_qtemp, gains1_qtemp, fit_cosine1_qtemp, pi_amp1_qtemp, A_amplitude1, amp_fit1,
             I2_qtemp, Q2_qtemp, gains2_qtemp, fit_cosine2_qtemp, pi_amp2_qtemp, A_amplitude2, amp_fit2, sysconfig_efrabi_Qtemps) = efAmprabi_Qtemps.run(experiment.soccfg, experiment.soc)

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

            # ---------------------Collect Results for Qubit Temperature Rabis ----------------
            if run_flags["Qtemps"]:
                rabi_data_ef_Qtemps[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))

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

            t1_data_eg = create_data_dict(t1_keys, save_r, list_of_all_qubits)
            t1_data_fg = create_data_dict(t1_keys, save_r, list_of_all_qubits)
            t1_data_fe = create_data_dict(t1_keys, save_r, list_of_all_qubits)

            ss_data_gef = create_data_dict(ss_keys_gef, save_r, list_of_all_qubits)

# P_e = A_e / (A_e + A_g)
A_e = A_amplitude1
A_g = A_amplitude2
P_e = np.abs(A_e / (A_e + A_g))
print('Excited state population (leakage): ',P_e)

def Qubit_Temperature_Convert(pop_e, Omega_q):
    s = pop_e
    Omega_q = Omega_q * 2*np.pi* 1e9 # Omega_q in the unit Hz
    k_B = 1.38 * 10**-23
    hbar = 1.05 * 10**-34
    T = hbar * Omega_q/(k_B * np.log((1-s)/s)) # Temperature in the unit Kelvin
    return T

qubit_freqs = [4.1898656, 3.156504, 3.098868, 3.285196, 3.25550, 3.294782]
T= Qubit_Temperature_Convert(P_e, qubit_freqs[0])
print("Temperature Q1: ", T, " Kelvin")

temperature_mK = T * 1000  # Convert to millikelvin

print(f"Temperature: {temperature_mK:.1f} mK")

# exit()
# #---------------------------------------Plotting from a file----------------------------------------------------------
# if not os.path.exists("/data/QICK_data/run6/6transmon/ef_studies/QubitTemps_efRabi_method/2025-04-09/Study_Data/plots"): os.makedirs("/data/QICK_data/run6/6transmon/ef_studies/QubitTemps_efRabi_method/2025-04-09/Study_Data/plots")
# timestamp = datetime.datetime.now().strftime("%H%M%S")
# data = os.path.join("/data/QICK_data/run6/6transmon/ef_studies/QubitTemps_efRabi_method/2025-04-09/Study_Data/Data_h5", "Q1_temperature_efRabi_round1_150025.h5")
#
# with h5py.File(data, 'r') as f:
#     # Read datasets
#     lengths = np.array(f['lengths'])
#     I = np.array(f['avgi'])
#     Q = np.array(f['avgq'])
#     amps = np.array(f['amps'])
#     I_ref = np.array(f['avgi_ref'])
#     Q_ref = np.array(f['avgq_ref'])
#     amps_ref = np.array(f['amps_ref'])
#     # Read and decode configuration
#     config = f.attrs['config']
#     config = json.loads(config)
#
#     if SS:
#         Ig = np.array(f['I_g'])
#         Qg = np.array(f['Q_g'])
#         Ie = np.array(f['I_e'])
#         Qe = np.array(f['Q_e'])
#         if ef:
#             If = np.array(f['I_f'])
#             Qf = np.array(f['Q_f'])
#
# # Process data: if using single-shot normalization, compute the normalized populations;
# # otherwise, just use the amplitude data.
# if SS:
#     # Compute average complex signals for calibration
#     e_val = np.mean(Ie + 1j * Qe)
#     g_val = np.mean(Ig + 1j * Qg)
#     # Initial discrimination using g and e
#     pop_norm = np.abs(((I + 1j * Q) - g_val) * (e_val - g_val) / np.abs(e_val - g_val) ** 2)
#     pop_norm_ref = np.abs(((I_ref + 1j * Q_ref) - g_val) * (e_val - g_val) / np.abs(e_val - g_val) ** 2)
#
#     if ef:
#         f_val = np.mean(If + 1j * Qf)
#         # For e-f discrimination: shift the reference so that the amplitude relates directly to population
#         pop_norm = np.abs(((I + 1j * Q) - e_val) * (f_val - e_val) / np.abs(f_val - e_val) ** 2)
#         pop_norm_ref = np.abs(((I_ref + 1j * Q_ref) - e_val) * (f_val - e_val) / np.abs(f_val - e_val) ** 2)
#     ydata = pop_norm
#     ydata_ref = pop_norm_ref
# else:
#     ydata = amps
#     ydata_ref = amps_ref
#
# # Plot the main (data) oscillation
# plt.figure(figsize=(12, 6))
# plt.plot(lengths, ydata, marker='o', label='Data')
# plt.xlim(20, 22)
# plt.ylabel("Qubit Population" if SS else "a.u.")
# plt.xlabel(r"Pulse Length ($\mu s$)")
# plt.title('Length Rabi Oscillations')
#
# # Fit the Rabi oscillation using the Fit class
# fit = Fit()
# fit_results = fit.rabi(lengths, ydata, plot=True)
# pp.pprint(fit_results)
#
# # Compute a "peak" value from the fit (this is an example based on your snippet)
# peak = 0.5 / fit_results['f'][0] - (fit_results['phase'][0] / (360 / np.pi))
# print('Peak:', peak)
# plt.axvline(peak, color='red', linestyle='--', label=f"x180 length = {peak:.3e} s")
#
# plt.legend()
# save_path = os.path.join("/data/QICK_data/run6/6transmon/ef_studies/QubitTemps_efRabi_method/2025-04-09/Study_Data/plots", f"test_1_{timestamp}.png")
# plt.savefig(save_path)
#
# #####################
# fig, ax0 = plt.subplots(1, figsize=(12,6))
# fit = Fit()
#
# # Plot main (data) oscillation
# ax0.plot(lengths, ydata, marker='o', label='Data')
# fit_func = True  # set to True to use fit.rabi(), False to use fitdecaysin()
#
# if fit_func:
#     fit_results = fit.rabi(lengths, ydata, plot=True)
#     pp.pprint(fit_results)
#     # Compute the peak as a characteristic pulse length from the fit parameters
#     peak = 0.5 / fit_results['f'][0] - (fit_results['phase'][0] / (360/np.pi))
# else:
#     fit_results = fitdecaysin(lengths, ydata, showfit=True)
#     pp.pprint(fit_results)
#     peak = 0.5 / fit_results[1] - (fit_results[2] / (180 * np.pi * fit_results[1]))
#
# print('Peak:', peak)
# ax0.axvline(peak, color='red', linestyle='--', label=f"x180 length = {peak:.3e} s")
#
# if SS:
#     ax0.set_ylabel("Qubit Population")
# else:
#     ax0.set_ylabel("a.u.")
# ax0.set_xlabel(r"Pulse Length ($\mu s$)")
# ax0.set_title('Length Rabi Oscillations')
# ax0.legend()
# save_path = os.path.join("/data/QICK_data/run6/6transmon/ef_studies/QubitTemps_efRabi_method/2025-04-09/Study_Data/plots", f"test_2_{timestamp}.png")
# plt.savefig(save_path)
#
# # Plot reference (data_ref) oscillation in a new figure
# fig2, ax1 = plt.subplots(1, figsize=(12,6))
# ax1.plot(lengths, ydata_ref, marker='o', label='Data_ref')
# fit_func = False  # set to False to use fitdecaysin() for reference data
#
# if fit_func:
#     fit_results_ref = fit.rabi(lengths, ydata_ref, plot=True)
#     pp.pprint(fit_results_ref)
#     peak_ref = 0.5 / fit_results_ref['f'][0] - (fit_results_ref['phase'][0] / (360/np.pi))
# else:
#     fit_results_ref = fitdecaysin(lengths, ydata_ref, showfit=True)
#     pp.pprint(fit_results_ref)
#     peak_ref = 0.5 / fit_results_ref[1]  # adjust if needed
#
# print('Peak_ref:', peak_ref)
# ax1.axvline(peak_ref, color='green', linestyle='--', label=f"x180 length (ref) = {peak_ref:.3e} s")
#
# if SS:
#     ax1.set_ylabel("Qubit Population")
# else:
#     ax1.set_ylabel("a.u.")
# ax1.set_xlabel(r"Pulse Length ($\mu s$)")
# ax1.set_title('Length Rabi Oscillations (Reference)')
# ax1.legend()
# save_path2 = os.path.join("/data/QICK_data/run6/6transmon/ef_studies/QubitTemps_efRabi_method/2025-04-09/Study_Data/plots", f"test_3_{timestamp}.png")
# fig2.savefig(save_path2)


#--------------------Temp calcs-------------------
# Ag = fit_results['amp']   # amplitude from the main fit
# Ae = fit_results_ref['amp']  # amplitude from the reference fit
#
# # 1. Compute excited-state population (P_e) from two amplitudes:
# #    P_e = A_e / (A_e + A_g)
# Pe = Ae / (Ae + Ag)
# print("Estimated Excited-State Population:", Pe)
#
# # 2. Define a function to convert the population into a temperature.
# #    The formula below uses the relation
# #       T = ? * O / [k_B * ln((1 - P_e)/P_e)]
#
# def qubit_temperature_convert(pop_e, freq_hz):
#     """
#     Convert excited-state population (pop_e) to temperature (K)
#     given the qubit transition frequency in Hz.
#     """
#     Omega = 2.0 * pi * freq_hz   # angular frequency (rad/s)
#     # Avoid math error if pop_e is 0 or 1
#     if pop_e <= 0.0 or pop_e >= 1.0:
#         return np.nan
#     return (hbar * Omega) / (k * np.log((1.0 - pop_e)/pop_e))
#
# # 3. Calculate the qubit temperature.
# freq_hz = config['qubit_freq_ef']  # e-f frequency in Hz
# T = qubit_temperature_convert(Pe, freq_hz)
# print(f"Estimated Qubit Temperature: {T:.4f} K")