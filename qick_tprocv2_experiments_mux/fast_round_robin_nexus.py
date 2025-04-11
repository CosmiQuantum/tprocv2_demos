import time
import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
import copy
from windfreak import SynthHD
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_008_save_data_to_h5 import Data_H5
from NetDrivers import E36300
from system_config_nexus import QICK_experiment
from expt_config_nexus import expt_cfg, list_of_all_qubits


################################################ Run Configurations ####################################################
save_r = 1            # how many rounds to save after
hourly_save_r = 1     # for hourly res spec, qubit spec, and rabi measurements
signal = 'None'       #'I', or 'Q' depending on where the signal is (after optimization). Put'None' if no optimization
save_figs = True    # save plots for everything as you go along the RR script?
live_plot = False      # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True      # fit the data here and save or plot the fits?
save_data_h5 = True   # save all of the data to h5 files?

number_of_qubits = 4 # currently 4 for NEXUS, 6 for QUIET
Qs_to_look_at = [0,1,2,3] #only list the qubits you want to do the RR for

increase_qubit_reps = False #if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0 #only has impact if previous line is True
multiply_qubit_reps_by = 2 #only has impact if the line two above is True

def create_folder_if_not_exists(folder_path):
    """Creates a folder at the given path if it doesn't already exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30/Fast_RR/", str(datetime.date.today()))

create_folder_if_not_exists(outerFolder)
################################################ optimizated parameters ##################################################
# For NEXUS
res_leng_vals = [5.8, 3.8, 4, 4.6] #[6.15, 5.85, 6.45, 5.7] # from 2/19/2025 optimization, after punchout test
res_gain = [0.38, 0.26, 0.28, 0.31]#[0.3143, 0.1857, 0.1429, 0.1857] # from 2/19/2025 optimization, after punchout test
freq_offsets = [0, 0, 0, 0] #[0.0, -0.0667, -0.2667, -0.400] # from 2/19/2025 optimization, after punchout test

#TWPA
synth = SynthHD('/dev/ttyACM1')
synth[0].power =     -12.85
synth[0].frequency = 7.826e9
synth[0].enable = True
time.sleep(5)

#Turning off HEMT power supply channels for bias lines
start_voltage = 0
Bias_PS_ip = ['192.168.0.44', '192.168.0.44', '192.168.0.44',
                  '192.168.0.41']  # IP address of bias PS (qubits 1-3 are the same PS)
Bias_ch = [1, 2, 3, 1]  # Channel number of qubit 1-4 on associated PS
for Q in range(4):
    BiasPS = E36300(Bias_PS_ip[Q], server_port=5025)

    BiasPS.setVoltage(start_voltage, Bias_ch[Q])
    BiasPS.enable(Bias_ch[Q])
####################################################### RR #############################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num']

#initialize a dictionary to store those values
res_data = create_data_dict(res_keys, hourly_save_r, list_of_all_qubits)
qspec_data = create_data_dict(qspec_keys, hourly_save_r, list_of_all_qubits)
rabi_data = create_data_dict(rabi_keys, hourly_save_r, list_of_all_qubits)

t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)

#initialize a simple list to store the qspec values in incase a fit fails
max_index = max(Qs_to_look_at)
stored_qspec_list = [None] * (max_index + 1)

#################################### Total Run Time & Timer Setup ##############################################
num_hours_runtime = 1 # Total hours that you want this program to run for
update_interval = 3600  # Seconds. How often you want to do res spec, qubit spec and rabi. For normal operation (1 hour) use 3600, which is the number of seconds in an hour
total_runtime = num_hours_runtime * update_interval
start_total = time.time()
last_hourly_update = start_total - update_interval  # Force the first hourly update immediately

batch_num = 0
hourly_batch_num = 0

j = 0
hourly_j = 0

og_experiment = QICK_experiment(outerFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10)
experiment_dict = {}

######################################## Main Loop #############################################################
while time.time() - start_total < total_runtime:
    current_time = time.time()

    # --------------------------Hourly Update: Run this "top part" once per hour ------------------------------------------
    if current_time - last_hourly_update >= update_interval:
        print("Performing hourly update...")
        hourly_j += 1

        for QubitIndex in Qs_to_look_at:
            # Create a fresh experiment for each qubit.
            experiment_dict[QubitIndex] = copy.deepcopy(og_experiment)
            experiment = experiment_dict[QubitIndex]

            res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex])
            experiment.readout_cfg['res_gain_ge'] = res_gains
            experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

            #-----------------------Resonance Spectroscopy---------------------------------------
            while True:
                try:
                    res_spec = ResonanceSpectroscopy(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, 0,
                                                     save_figs, experiment)
                    res_freqs, freq_pts, freq_center, amps = res_spec.run(experiment.soccfg, experiment.soc)
                    offset = freq_offsets[QubitIndex]
                    offset_res_freqs = [r + offset for r in res_freqs]
                    experiment.readout_cfg['res_freq_ge'] = offset_res_freqs
                    del res_spec
                    break
                except Exception as e:
                    print(f"Resonance spectroscopy failed for qubit {QubitIndex + 1}: {e}. Retrying...")

            #------------------------Qubit Spectroscopy----------------------------------------------------------------
            while True:
                try:
                    q_spec = QubitSpectroscopy(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, 0, signal,
                                               save_figs, experiment, live_plot)
                    qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq = q_spec.run(experiment.soccfg,
                                                                                                     experiment.soc)
                    experiment.qubit_cfg['qubit_freq_ge'][QubitIndex] = float(qubit_freq)
                    stored_qspec_list[QubitIndex] = float(qubit_freq)
                    print('Hourly update - Qubit freq for qubit', QubitIndex + 1, 'is:', float(qubit_freq))
                    del q_spec
                    break
                except Exception as e:
                    print(f"Qubit spectroscopy failed for qubit {QubitIndex + 1}: {e}. Retrying...")

            #------------------------Amplitude Rabi Experiment-----------------------------------------------------------
            while True:
                try:
                    rabi = AmplitudeRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, 0, signal,
                                                   save_figs, experiment, live_plot,
                                                   increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
                    rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_to_save = rabi.run(experiment.soccfg,
                                                                                                experiment.soc)
                    if float(pi_amp) < 0.2:
                        print(f"Rabi amplitude {pi_amp} for qubit {QubitIndex + 1} is below threshold. Retrying...")
                        del rabi
                        continue

                    experiment.qubit_cfg['pi_amp'][QubitIndex] = float(pi_amp)
                    print('Hourly update - Pi amplitude for qubit', QubitIndex + 1, 'is:', float(pi_amp))
                    del rabi
                    break
                except Exception as e:
                    print(f"Rabi measurement failed for qubit {QubitIndex + 1}: {e}. Retrying...")

            if save_data_h5:
                hourly_idx = hourly_j - hourly_batch_num * hourly_save_r - 1
                # ---------------------Collect Res Spec Results----------------
                res_data[QubitIndex]['Dates'][hourly_idx] = time.mktime(datetime.datetime.now().timetuple())
                res_data[QubitIndex]['freq_pts'][hourly_idx] = freq_pts
                res_data[QubitIndex]['freq_center'][hourly_idx] = freq_center
                res_data[QubitIndex]['Amps'][hourly_idx] = amps
                res_data[QubitIndex]['Found Freqs'][hourly_idx] = res_freqs
                res_data[QubitIndex]['Round Num'][hourly_idx] = hourly_j
                res_data[QubitIndex]['Batch Num'][hourly_idx] = hourly_batch_num

                # ---------------------Collect QSpec Results----------------
                qspec_data[QubitIndex]['Dates'][hourly_idx] = time.mktime(datetime.datetime.now().timetuple())
                qspec_data[QubitIndex]['I'][hourly_idx] = qspec_I
                qspec_data[QubitIndex]['Q'][hourly_idx] = qspec_Q
                qspec_data[QubitIndex]['Frequencies'][hourly_idx] = qspec_freqs
                qspec_data[QubitIndex]['I Fit'][hourly_idx] = qspec_I_fit
                qspec_data[QubitIndex]['Q Fit'][hourly_idx] = qspec_Q_fit
                qspec_data[QubitIndex]['Round Num'][hourly_idx] = hourly_j
                qspec_data[QubitIndex]['Batch Num'][hourly_idx] = hourly_batch_num

                # ---------------------Collect Rabi Results----------------
                rabi_data[QubitIndex]['Dates'][hourly_idx] = time.mktime(datetime.datetime.now().timetuple())
                rabi_data[QubitIndex]['I'][hourly_idx] = rabi_I
                rabi_data[QubitIndex]['Q'][hourly_idx] = rabi_Q
                rabi_data[QubitIndex]['Gains'][hourly_idx] = rabi_gains
                rabi_data[QubitIndex]['Fit'][hourly_idx] = rabi_fit
                rabi_data[QubitIndex]['Round Num'][hourly_idx] = hourly_j
                rabi_data[QubitIndex]['Batch Num'][hourly_idx] = hourly_batch_num

        #------------------------Save data------------------------------------------------------------------
        if hourly_j % hourly_save_r == 0:
            hourly_batch_num += 1

            # --------------------------save Res Spec-----------------------
            saver_res = Data_H5(outerFolder, res_data, hourly_batch_num, hourly_save_r)
            saver_res.save_to_h5('Res')
            del saver_res
            del res_data

            # --------------------------save QSpec-----------------------
            saver_qspec = Data_H5(outerFolder, qspec_data, hourly_batch_num, hourly_save_r)
            saver_qspec.save_to_h5('QSpec')
            del saver_qspec
            del qspec_data

            # --------------------------save Rabi-----------------------
            saver_rabi = Data_H5(outerFolder, rabi_data, hourly_batch_num, hourly_save_r)
            saver_rabi.save_to_h5('Rabi')
            del saver_rabi
            del rabi_data

            # reset all dictionaries to none for safety
            res_data = create_data_dict(res_keys, hourly_save_r, list_of_all_qubits)
            qspec_data = create_data_dict(qspec_keys, hourly_save_r, list_of_all_qubits)
            rabi_data = create_data_dict(rabi_keys, hourly_save_r, list_of_all_qubits)


        last_hourly_update = current_time

    # -------------------------------------------Fast Loop: T1 Measurements----------------------------------------------
    j += 1
    for QubitIndex in Qs_to_look_at:
        # Retrieve the experiment instance for this qubit from the dictionary
        experiment = experiment_dict.get(QubitIndex)
        # Update readout configuration if needed
        res_gains = experiment.mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex])
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        while True:
            try:
                t1 = T1Measurement(QubitIndex, number_of_qubits, list_of_all_qubits, outerFolder, j, signal, save_figs,
                                   experiment, live_plot, fit_data,
                                   increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
                t1_est, t1_err, t1_I, t1_Q, t1_delay_times, q1_fit_exponential = t1.run(experiment.soccfg, experiment.soc)
                del t1
                break
            except Exception as e:
                print(f"Error during T1 measurement for qubit {QubitIndex + 1}: {e}. Retrying...")


        #-----------------------------------------------Collect T1 results-------------------------------------------------
        if save_data_h5:
            # ---------------------Collect T1 Results----------------
            idx = j - batch_num * save_r - 1
            t1_data[QubitIndex]['T1'][idx] = t1_est
            t1_data[QubitIndex]['Errors'][idx] = t1_err
            t1_data[QubitIndex]['Dates'][idx] = time.mktime(datetime.datetime.now().timetuple())
            t1_data[QubitIndex]['I'][idx] = t1_I
            t1_data[QubitIndex]['Q'][idx] = t1_Q
            t1_data[QubitIndex]['Delay Times'][idx] = t1_delay_times
            t1_data[QubitIndex]['Fit'][idx] = q1_fit_exponential
            t1_data[QubitIndex]['Round Num'][idx] = j
            t1_data[QubitIndex]['Batch Num'][idx] = batch_num

            # Save current system config and expt_cfg
            saver_config = Data_H5(outerFolder)
            saver_config.save_config(sys_config_to_save, expt_cfg)
            del saver_config

    #-------------------------------------------------------------Save data-------------------------------------------------------
    if save_data_h5 and (j % save_r == 0):
        batch_num += 1
        saver_t1 = Data_H5(outerFolder, t1_data, batch_num, save_r)
        saver_t1.save_to_h5('T1')
        del saver_t1
        t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)

print(f"Total run time of {num_hours_runtime} hours reached. Exiting.")