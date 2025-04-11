import sys
import os

from NetDrivers import E36300
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))  ## Change for quiet vs nexus
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus"))
#sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus/"))
from system_config import QICK_experiment
from parity_geV2 import AllQubitParityMeasurement#ParityMeasurement
from parity_geV2 import ParityMeasurement
import time
import datetime
import numpy as np
from tprocv2_demos.qick_tprocv2_experiments_mux_nexus.parity_geV2 import ParityProgram
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from windfreak import SynthHD
from system_config import QICK_experiment  ## Change for quiet vs nexus
# from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits  ## Change for quiet vs nexus
# from expt_config import expt_cfg, list_of_all_qubits
################################################ Run Configurations ####################################################

save_r = 1            # how many rounds to save after
signal = 'None'       #'I', or 'Q' depending on where the signal is (after optimization). Put'None' if no optimization
save_figs = True    # save plots for everything as you go along the RR script?
live_plot = False      # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True      # fit the data here and save or plot the fits?
save_data_h5 = True   # save all of the data to h5 files?
number_of_qubits = 4 # 4 for nexus, 6 for quiet
Qs_to_look_at = [0, 1, 2, 3] #only list the qubits you want to do the RR for

increase_qubit_reps = False #if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0 #only has impact if previous line is True
multiply_qubit_reps_by = 2 #only has impact if the line two above is True


outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30", str(datetime.date.today()))
################################################ optimization outputs ##################################################

Bias_PS_ip = ['192.168.0.44', '192.168.0.44', '192.168.0.44',
                      '192.168.0.41']  # IP address of bias PS (qubits 1-3 are the same PS)

res_leng_vals = [5.8, 3.8, 4, 4.6] #[6.15, 5.85, 6.45, 5.7] # from 2/19/2025 optimization, after punchout test
res_gain = [0.38, 0.26, 0.28, 0.31]

synth = SynthHD('/dev/ttyACM1')

synth[0].power =     -12.85
synth[0].frequency = 7.826e9
synth[0].enable = True
time.sleep(5)

Qs=[0]#,1,2,3]
number_of_qubits = 4

start_voltage = 0 #V
stop_voltage  = 0 #V
voltage_pts = 1100

Bias_ch = [1, 2, 3, 1]  # Channel number of qubit 1-4 on associated PS



rabiGs=np.zeros(4)
resPhases=np.zeros(4)
fids=np.zeros(4)

#resGs=np.linspace(0.1,0.5,5)
#resFs=np.linspace(5959.553-1.5, 5959.553+1.5, 4)
#f = 5858.673

#for g in resGs:
#for f in resFs:
# wts=np.linspace(1/2.7/32 , 1/2.7/1, 20)
# for i in range(len(wts)):
# expt_cfg["Parity_ge_q4"]["wait_time"]=wts[i]
n=1
j=0
while j < n:
    j += 1
    ssf_I_g = []
    ssf_I_e = []
    ssf_Q_g = []
    ssf_Q_e = []
    for Q in Qs:
        #
        # start_voltage = 0.1
        # BiasPS = E36300(Bias_PS_ip[Q], server_port=5025)
        #
        # BiasPS.setVoltage(start_voltage, Bias_ch[Q])
        # BiasPS.enable(Bias_ch[Q])

        experiment = QICK_experiment(outerFolder, DAC_attenuator1=5, DAC_attenuator2=10, ADC_attenuator=10)

        experiment.readout_cfg['res_gain_ge'] = res_gain
        experiment.readout_cfg['res_length'] = res_leng_vals[Q]

        rabi = AmplitudeRabiExperiment(Q, number_of_qubits, list_of_all_qubits, outerFolder, j, signal, save_figs,
                                       experiment, live_plot,
                                       increase_qubit_reps, qubit_to_increase_reps_for, multiply_qubit_reps_by)
        rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_to_save = rabi.run(experiment.soccfg, experiment.soc)

        # if these are None, fit didnt work
        if (rabi_fit is None and pi_amp is None):
            # logging.info('Rabi fit didnt work, skipping the rest of this qubit')
            print('Rabi fit didnt work, skipping the rest of this qubit')
            continue  # skip the rest of this qubit

        experiment.qubit_cfg['pi_amp'][Q] = float(pi_amp)

        rabiGs[Q] = float(pi_amp)
        # logging.info('Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
        print('Pi amplitude for qubit ', Q + 1, ' is: ', float(pi_amp))
        del rabi

        ss = SingleShot(Q, number_of_qubits, list_of_all_qubits, outerFolder, j, save_figs, experiment)
        fid, angle, iq_list_g, iq_list_e = ss.run(experiment.soccfg, experiment.soc)
        ssf_I_g.append(iq_list_g[Q][0].T[0])
        ssf_Q_g.append(iq_list_g[Q][0].T[1])
        ssf_I_e.append(iq_list_e[Q][0].T[0])
        ssf_Q_e.append(iq_list_e[Q][0].T[1])

        fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(
            data=[ssf_I_g[Q], ssf_Q_g[Q], ssf_I_e[Q], ssf_Q_e[Q]], cfg=ss.config, plot=save_figs)
        print('fid', fid)
        fids[Q] = fid

        ## Get num of rounds to use by total time you want, or just set manually below:
    QubitIndex=0
    run_time = 3  # hrs, 11pm to 730am, ~8.5 hrs
    round_time = 1  # min, actually more like 1.5 min but want to leave extra time
    round_num = int(run_time * 60 / round_time)
    start_time = time.time()
    resPhases[Q] = angle * 180 / np.pi
    experiment.readout_cfg['res_phase'][Q] = angle * 180 / np.pi

    #parityMeas = AllQubitParityMeasurement( outerFolder, number_of_qubits, experiment,  fids, ssf_I_g, ssf_I_e, ssf_Q_g, ssf_Q_e )
    # QubitIndex, outerFolder, number_of_qubits, experiment
    parityMeas = ParityMeasurement(QubitIndex, outerFolder, number_of_qubits, experiment, fids, ssf_I_g, ssf_I_e, ssf_Q_g,
                                           ssf_Q_e)
    print(experiment.soccfg)


    parityMeas.run_Parity(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken : {elapsed_time:.2f} seconds")
    print('experiment.soccfg',experiment.soccfg)
    #BiasPS.setVoltage(0, Bias_ch[int(Q)])

    del parityMeas
    del experiment