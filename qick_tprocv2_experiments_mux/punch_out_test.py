import sys
import os
import numpy as np
# sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/")) # for QUIET
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus/")) # for NEXUS
# from system_config import QICK_experiment # for QUIET
from system_config_nexus import QICK_experiment  # for NEXUS
from section_003_punch_out_ge_mux import PunchOut
import datetime

number_of_qubits = 4  #currently 4 for NEXUS, 6 for QUIET

sweep_DAC_attenuator1 =[5] #np.linspace(5,20, 4)
sweep_DAC_attenuator2 =[10]#[15,20,25,30] #np.linspace(5,20,4)

# outerFolder = "/data/QICK_data/6transmon_run5/" + str(datetime.date.today()) + "/" # for QUIET
outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30/", str(datetime.date.today())) # for NEXUS
# for att_1 in sweep_DAC_attenuator1:
#     for att_2 in sweep_DAC_attenuator2:
#         att_1 = round(att_1, 3)
#         att_2 = round(att_2, 3)
#         experiment = QICK_experiment(outerFolder, DAC_attenuator1 = att_1, DAC_attenuator2 = att_2)
#         punch_out   = PunchOut(outerFolder, experiment)
#
#         start_gain, stop_gain, num_points = 0.1, 1, 10
#         punch_out.run(experiment.soccfg, experiment.soc, start_gain, stop_gain, num_points, att_1, att_2, plot_Center_shift = True, plot_res_sweeps = True)
#
#         del punch_out
#         del experiment
#

att_1=999
att_2=999

experiment = QICK_experiment(outerFolder)
punch_out   = PunchOut(number_of_qubits, outerFolder, experiment)

# start_gain, stop_gain, num_points = 0.0, 1.0, 10 # for QUIET
start_gain, stop_gain, num_points = 0.0, 0.8, 10 # for NEXUS
punch_out.run(experiment.soccfg, experiment.soc, start_gain, stop_gain, num_points, att_1, att_2, plot_Center_shift = False, plot_res_sweeps = True)

del punch_out
del experiment
