import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes, copy pasted from QUIET
import datetime
import time
import visdom
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus/"))
from section_001_time_of_flight import TOFExperiment
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from system_config import QICK_experiment
from section_003_punch_out_ge_mux import PunchOut
from expt_config import expt_cfg
from NetDrivers import E36300

n= 1000
save_r = 1            # how many rounds to save after
signal = 'None'       #'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization has happened
save_figs = True    # save plots for everything as you go along the RR script?
live_plot = True      # for live plotting open http://localhost:8097/ on firefox
fit_data = True      # fit the data here and save or plot the fits? Should be false! we fit in post processing. But spec and rabi always fit no matter what the flag is
save_data_h5 = True   # save all of the data to h5 files?

increase_qubit_reps = False #if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0 #only has impact if previous line is True
multiply_qubit_reps_by = 2 #only has impact if the line two above is True

outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30/", str(datetime.date.today())) #Change run number in each new run
custom_Ramsey=False

dictionary_qs = [0, 1, 2, 3] #needs to be the total number of qubits that you have
Qs_to_look_at = [0, 1, 2, 3] #only list the qubits you want to do the RR for


# optimization outputs
res_leng_vals = [5.6, 5.85, 6.35, 3.35] #from 1/31-2/1 optimization
res_gain = [0.34, 0.3, 0.34, 0.34] #from 1/31-2/1 optimization, after implementing punchout threshold
freq_offsets = [-0.32, -0.16, -0.08, -0.08] #from 1/31-2/1 optimization, after implementing punchout threshold


#NOTE: Everything below this line was copy pasted from QUIET, to update this RR script

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num']
t2r_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num']
t2e_keys = ['T2E', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num']

#initialize a dictionary to store those values
res_data = create_data_dict(res_keys, save_r, dictionary_qs)
qspec_data = create_data_dict(qspec_keys, save_r, dictionary_qs)
rabi_data = create_data_dict(rabi_keys, save_r, dictionary_qs)
ss_data = create_data_dict(ss_keys, save_r, dictionary_qs)
t1_data = create_data_dict(t1_keys, save_r, dictionary_qs)
t2r_data = create_data_dict(t2r_keys, save_r, dictionary_qs)
t2e_data = create_data_dict(t2e_keys, save_r, dictionary_qs)

batch_num=0
j = 0
angles=[]
while j < n:
    j += 1
    for QubitIndex in Qs_to_look_at:
        #Get the config for this qubit
        experiment = QICK_experiment(outerFolder, DAC_attenuator1 = 5, DAC_attenuator2 = 10, ADC_attenuator = 10)

        #Mask out all other resonators except this one
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex])
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        ###################################################### TOF #####################################################
        #tof        = TOFExperiment(QubitIndex, outerFolder, experiment, j, save_figs)
        #tof.run(experiment.soccfg, experiment.soc)
        #del tof

        ################################################## Res spec ####################################################
        try:
            res_spec = ResonanceSpectroscopy(QubitIndex, outerFolder, j, save_figs, experiment)
            res_freqs, freq_pts, freq_center, amps = res_spec.run(experiment.soccfg, experiment.soc)
            experiment.readout_cfg['res_freq_ge'] = res_freqs
            offset = freq_offsets[QubitIndex]  # use optimized offset values
            offset_res_freqs = [r + offset for r in res_freqs]
            experiment.readout_cfg['res_freq_ge'] = offset_res_freqs
            del res_spec
        except Exception as e:
            print(f'Got the following error, continuing: {e}')
            continue  # skip the rest of this qubit
        #
        # # # ############################################ Roll Signal into I ##############################################
        # # # #get the average theta value, then use that to rotate the signal. Plug that value into system_config res_phase
        # # # leng=4
        # # # ss = SingleShot(QubitIndex, outerFolder, experiment, j, save_figs)
        # # # fid, angle, iq_list_g, iq_list_e = ss.run(experiment.soccfg, experiment.soc)
        # # # angles.append(angle)
        # # # #print(angles)
        # # # #print('avg theta: ', np.average(angles))
        # # # del ss
        # #
        # # ################################################## Qubit spec ##################################################
        try:
            q_spec = QubitSpectroscopy(QubitIndex, outerFolder, j, signal, save_figs, experiment, live_plot)
            qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq = q_spec.run(experiment.soccfg,
                                                                                             experiment.soc)
            # if these are None, fit didnt work
            if (qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None):
                print('QSpec fit didnt work, skipping the rest of this qubit')
                continue  # skip the rest of this qubit

            experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
            print('Qubit freq for qubit ', QubitIndex + 1, ' is: ', float(qubit_freq))
            del q_spec

        except Exception as e:
            print(f'Got the following error, continuing: {e}')
            continue  # skip the rest of this qubit

        ###################################################### Rabi ####################################################
        try:


            rabi = AmplitudeRabiExperiment(QubitIndex, outerFolder, j, signal, save_figs, experiment, live_plot,
                                           increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
            rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_to_save = rabi.run(experiment.soccfg,
                                                                                        experiment.soc)

            # if these are None, fit didnt work
            if (rabi_fit is None and pi_amp is None):
                print('Rabi fit didnt work, skipping the rest of this qubit')
                continue  # skip the rest of this qubit

            experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
            print('Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
            del rabi

        except Exception as e:
            print(f'Got the following error, continuing: {e}')
            continue  # skip the rest of this qubit

        # ########################################## Single Shot Measurements ############################################
        try:


            ss = SingleShot(QubitIndex, outerFolder, j, save_figs, experiment)
            fid, angle, iq_list_g, iq_list_e = ss.run(experiment.soccfg, experiment.soc)



            I_g = iq_list_g[QubitIndex][0].T[0]
            Q_g = iq_list_g[QubitIndex][0].T[1]
            I_e = iq_list_e[QubitIndex][0].T[0]
            Q_e = iq_list_e[QubitIndex][0].T[1]

            fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(
                data=[I_g, Q_g, I_e, Q_e], cfg=ss.config, plot=save_figs)

        except Exception as e:
            print(f'Got the following error, continuing: {e}')
            continue  # skip the rest of this qubit
        #
        # # # ###################################################### T1 ######################################################
        try:
            t1 = T1Measurement(QubitIndex, outerFolder, j, signal, save_figs, experiment, live_plot, fit_data,
                               increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
            t1_est, t1_err, t1_I, t1_Q, t1_delay_times, q1_fit_exponential = t1.run(experiment.soccfg, experiment.soc)
            del t1

        except Exception as e:
            print(f'Got the following error, continuing: {e}')
            continue  # skip the rest of this qubit

        ###################################################### T2R #####################################################
        try:

            t2r = T2RMeasurement(QubitIndex, outerFolder, j, signal, save_figs, experiment, live_plot, fit_data,
                                 increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
            t2r_est, t2r_err, t2r_I, t2r_Q, t2r_delay_times, fit_ramsey = t2r.run(experiment.soccfg, experiment.soc)

            #np.savez(os.path.join(outerFolder, f'T2_Q{QubitIndex+1}_{timestamp}_volts.npz'), I=t2r_I, Q=t2r_Q, time=t2r_delay_times)
            del t2r

        except Exception as e:
            print(f'Got the following error, continuing: {e}')
            continue  # skip the rest of this qubit

        ##################################################### T2E ######################################################
        try:
            t2e = T2EMeasurement(QubitIndex, outerFolder, j, signal, save_figs, experiment, live_plot, fit_data,
                                 increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
            t2e_est, t2e_err, t2e_I, t2e_Q, t2e_delay_times, fit_ramsey_t2e, sys_config_to_save = t2e.run(
                experiment.soccfg,
                experiment.soc)
            del t2e

        except Exception as e:
            print(f'Got the following error, continuing: {e}')
            continue  # skip the rest of this qubit

        ############################################### Collect Results ################################################
        if save_data_h5:
            #---------------------Collect Res Spec Results----------------
            res_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = time.mktime(datetime.datetime.now().timetuple())
            res_data[QubitIndex]['freq_pts'][j - batch_num * save_r - 1] = freq_pts
            res_data[QubitIndex]['freq_center'][j - batch_num * save_r - 1] = freq_center
            res_data[QubitIndex]['Amps'][j - batch_num * save_r - 1] = amps
            res_data[QubitIndex]['Found Freqs'][j - batch_num * save_r - 1] = res_freqs
            res_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            res_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num

            # ---------------------Collect QSpec Results----------------
            qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1]=time.mktime(datetime.datetime.now().timetuple())
            qspec_data[QubitIndex]['I'][j - batch_num * save_r - 1] = qspec_I
            qspec_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = qspec_Q
            qspec_data[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = qspec_freqs
            qspec_data[QubitIndex]['I Fit'][j - batch_num * save_r - 1] = qspec_I_fit
            qspec_data[QubitIndex]['Q Fit'][j - batch_num * save_r - 1] = qspec_Q_fit
            qspec_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            qspec_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num

            #---------------------Collect Rabi Results----------------
            rabi_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = time.mktime(datetime.datetime.now().timetuple())
            rabi_data[QubitIndex]['I'][j - batch_num * save_r - 1] = rabi_I
            rabi_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = rabi_Q
            rabi_data[QubitIndex]['Gains'][j - batch_num * save_r - 1] = rabi_gains
            rabi_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = rabi_fit
            rabi_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            rabi_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            #
            # ---------------------Collect Single Shot Results----------------
            ss_data[QubitIndex]['Fidelity'][j - batch_num * save_r - 1] = fid
            ss_data[QubitIndex]['Angle'][j - batch_num * save_r - 1] = angle
            ss_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = time.mktime(datetime.datetime.now().timetuple())
            ss_data[QubitIndex]['I_g'][j - batch_num * save_r - 1] = I_g
            ss_data[QubitIndex]['Q_g'][j - batch_num * save_r - 1] = Q_g
            ss_data[QubitIndex]['I_e'][j - batch_num * save_r - 1] = I_e
            ss_data[QubitIndex]['Q_e'][j - batch_num * save_r - 1] = Q_e
            ss_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            ss_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num

            #---------------------Collect T1 Results----------------
            t1_data[QubitIndex]['T1'][j - batch_num*save_r - 1] = t1_est
            t1_data[QubitIndex]['Errors'][j - batch_num*save_r - 1] = t1_err
            t1_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = time.mktime(datetime.datetime.now().timetuple())
            t1_data[QubitIndex]['I'][j - batch_num*save_r - 1] = t1_I
            t1_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = t1_Q
            t1_data[QubitIndex]['Delay Times'][j - batch_num*save_r - 1] = t1_delay_times
            t1_data[QubitIndex]['Fit'][j - batch_num*save_r - 1] = q1_fit_exponential
            t1_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            t1_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num

            ##--------------------Collect T2 Results----------------
            t2r_data[QubitIndex]['T2'][j - batch_num*save_r - 1] = t2r_est
            t2r_data[QubitIndex]['Errors'][j - batch_num*save_r - 1] = t2r_err
            t2r_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = time.mktime(datetime.datetime.now().timetuple())
            t2r_data[QubitIndex]['I'][j - batch_num*save_r - 1] = t2r_I
            t2r_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = t2r_Q
            t2r_data[QubitIndex]['Delay Times'][j - batch_num*save_r - 1] = t2r_delay_times
            t2r_data[QubitIndex]['Fit'][j - batch_num*save_r - 1] = fit_ramsey
            t2r_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            t2r_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num

            #---------------------Collect T2E Results----------------
            t2e_data[QubitIndex]['T2E'][j - batch_num*save_r - 1] = t2e_est
            t2e_data[QubitIndex]['Errors'][j - batch_num*save_r - 1] = t2e_err
            t2e_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = time.mktime(datetime.datetime.now().timetuple())
            t2e_data[QubitIndex]['I'][j - batch_num*save_r - 1] = t2e_I
            t2e_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = t2e_Q
            t2e_data[QubitIndex]['Delay Times'][j - batch_num*save_r - 1] = t2e_delay_times
            t2e_data[QubitIndex]['Fit'][j - batch_num*save_r - 1] = fit_ramsey_t2e
            t2e_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            t2e_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num

            # ---------------------save last system config and expt_cfg---------------------
            saver_config = Data_H5(outerFolder)
            saver_config.save_config(sys_config_to_save, expt_cfg)
            del saver_config

        del experiment

    ################################################## Potentially Save ################################################
    if save_data_h5:
        # Check if you are at the right round number
        # If so, then save all of the data and change the round num so you replace data starting next round
        if j % save_r == 0:
            batch_num+=1

            # --------------------------save Res Spec-----------------------
            saver_res = Data_H5(outerFolder, res_data, batch_num, save_r)
            saver_res.save_to_h5('Res')
            del saver_res
            del res_data

            # --------------------------save QSpec-----------------------
            saver_qspec = Data_H5(outerFolder, qspec_data, batch_num, save_r)
            saver_qspec.save_to_h5('QSpec')
            del saver_qspec
            del qspec_data

            # --------------------------save Rabi-----------------------
            saver_rabi = Data_H5(outerFolder, rabi_data, batch_num, save_r)
            saver_rabi.save_to_h5('Rabi')
            del saver_rabi
            del rabi_data

            # --------------------------save SS-----------------------
            saver_ss = Data_H5(outerFolder, ss_data, batch_num, save_r)
            saver_ss.save_to_h5('SS')
            del saver_ss
            del ss_data

            # --------------------------save t1-----------------------
            saver_t1 = Data_H5(outerFolder, t1_data, batch_num, save_r)
            saver_t1.save_to_h5('T1')
            del saver_t1
            del t1_data

            #--------------------------save t2r-----------------------
            saver_t2r = Data_H5(outerFolder, t2r_data, batch_num, save_r)
            saver_t2r.save_to_h5('T2')
            del saver_t2r
            del t2r_data

            #--------------------------save t2e-----------------------
            saver_t2e = Data_H5(outerFolder, t2e_data, batch_num, save_r)
            saver_t2e.save_to_h5('T2E')
            del saver_t2e
            del t2e_data

            # reset all dictionaries to none for safety
            res_data = create_data_dict(res_keys, save_r, dictionary_qs)
            qspec_data = create_data_dict(qspec_keys, save_r, dictionary_qs)
            rabi_data = create_data_dict(rabi_keys, save_r, dictionary_qs)
            ss_data = create_data_dict(ss_keys, save_r, dictionary_qs)
            t1_data = create_data_dict(t1_keys, save_r, dictionary_qs)
            t2r_data = create_data_dict(t2r_keys, save_r, dictionary_qs)
            t2e_data = create_data_dict(t2e_keys, save_r, dictionary_qs)





