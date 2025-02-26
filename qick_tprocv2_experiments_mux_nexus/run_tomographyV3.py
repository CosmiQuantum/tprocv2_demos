import sys
import os
import numpy as np
np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus"))
from system_config import QICK_experiment
from tomography import TomographyMeasurement
from tomography import AllQubitTomographyMeasurement
import scripts_for_long_tomography_datataking

#sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_001_time_of_flight import TOFExperiment
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement

# For NEXUS
from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits

import datetime

################################################ Run Configurations ####################################################
n= 100000
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
# For NEXUS
res_leng_vals = [3.0, 3.0, 3.0, 3.0] # from 2/19/2025 optimization, after punchout test
res_gain = [0.3143, 0.1857, 0.1429, 0.1857] # from 2/19/2025 optimization, after punchout test
freq_offsets = [0.0, -0.0667, -0.2667, -0.400] # from 2/19/2025 optimization, after punchout test
####################################################### RR #############################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

#

#initialize a simple list to store the qspec values in incase a fit fails
max_index = max(Qs_to_look_at)
stored_qspec_list = [None] * (max_index + 1)


batch_num=0
j = 0
angles=[]
rfreqs=np.zeros(4)
qfreqs=np.zeros(4)
rabiGs=np.zeros(4)
while j < n:
    j += 1
    #####################Res, Qu specs and Rabi #####################################
    Rfreqs, Qfreqs, RabiGs=scripts_for_long_tomography_datataking.specs_rabi()
    rfreqs = Rfreqs
    qfreqs = Qfreqs
    rabiGs = RabiGs

    ######################## Readout Optimization ########################################
    gain_range = [0.1, 0.4]  # Gain range in a.u.
    freq_steps = 15
    gain_steps = 6
    lengs = np.arange(2, 5.7, 0.1)


    ######################  Tomography  ##################################################

        ## Repeated All Qubit Tomography
    start_voltage = 0  # V
    stop_voltage = 0.15  # 0.1 #V
    voltage_pts = 45

    ## Get num of rounds to use by total time you want, or just set manually below:
    run_time = 3  # hrs, 11pm to 730am, ~8.5 hrs

    scripts_for_long_tomography_datataking.runTomography(start_voltage, stop_voltage, voltage_pts, run_time)






## Unblock one of the follow block to do single qubit tomography or repeated tomography for all qubits

# ## Single Qubit Tomography
#
# qubit = 3  #Qubit to Run
# start_voltage = 0 #V
# stop_voltage = 0.1 #V
# voltage_pts = 30
#
# experiment = QICK_experiment(outerFolder)
# tomography = TomographyMeasurement(qubit-1, outerFolder, experiment)
# tomography.run_tomography(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts, plot=True, save = False)
#
# del tomography
# del experiment



## Repeated All Qubit Tomography
# start_voltage = 0 #V
# stop_voltage = 0.1#0.1 #V
# voltage_pts = 30
#
# ## Get num of rounds to use by total time you want, or just set manually below:
# run_time = 3 # hrs, 11pm to 730am, ~8.5 hrs
# round_time = 2 #min, actually more like 1.5 min but want to leave extra time
# round_num = int(run_time*60/round_time)
#
# rounds = 25
#
# experiment = QICK_experiment(outerFolder)
# qs_tomography = AllQubitTomographyMeasurement(outerFolder, experiment)
# qs_tomography.allq_run_tomography(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts, round_num, plot=False, save=True)
#
# del qs_tomography
# del experiment