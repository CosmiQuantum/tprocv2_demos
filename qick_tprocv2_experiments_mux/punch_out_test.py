import sys
import os
import numpy as np
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/")) # for QUIET
# sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus/")) # for NEXUS
from system_config import QICK_experiment
from expt_config import FRIDGE
from section_003_punch_out_ge_mux import PunchOut
import datetime

number_of_qubits = 6  #currently 4 for NEXUS, 6 for QUIET

# sweep_DAC_attenuator1 =[] #np.linspace(5,20, 4)
# sweep_DAC_attenuator2 =[10]#[15,20,25,30] #np.linspace(5,20,4)

substudy = 'initial_punchout'
outerFolder = os.path.join(f"/data/QICK_data/run8/6transmon/readout_optimization/{substudy}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")
outerfolder_plots = outerFolder + "/documentation/"
#outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30/", str(datetime.date.today())) # for NEXUS
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
Unmask = True
DAC_att_1=14
DAC_att_2=10
DAC_att=DAC_att_1+DAC_att_2
ADC_att=17

experiment = QICK_experiment(outerfolder_plots, DAC_attenuator1 = DAC_att_1, DAC_attenuator2 = DAC_att_2, qubit_DAC_attenuator1 = 5 , qubit_DAC_attenuator2 = 4 ,ADC_attenuator = ADC_att, fridge=FRIDGE)

start_gain, stop_gain, num_points =  0.3, 0.5, 5  # for QUIET 0.55, 0.775, 5 #

# To look at only one resonator at a time:
# Qubit_index= 0 #starts at 0
# punch_out   = PunchOut(Qubit_index, number_of_qubits, outerfolder_plots, experiment, Unmask)
# punch_out.run(experiment.soccfg, experiment.soc, start_gain, stop_gain, num_points, DAC_att, ADC_att, plot_Center_shift = False, plot_res_sweeps = True)

# For resonators back to back:
Qs = [1]#[0,1,2,3,4,5] #starts at 0
for QubitIndex in Qs:
    punch_out  = PunchOut(QubitIndex, number_of_qubits, outerfolder_plots, experiment, Unmask)
    punch_out.run(experiment.soccfg, experiment.soc, start_gain, stop_gain, num_points, DAC_att, ADC_att, plot_Center_shift = False, plot_res_sweeps = True)

del punch_out
del experiment
