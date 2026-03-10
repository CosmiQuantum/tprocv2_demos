import sys
import os
import numpy as np
# sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/")) # for QUIET
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/")) # for NEXUS
from system_config import QICK_experiment
from section_003_punch_out_ge_mux import PunchOut
import datetime

number_of_qubits = 4  #currently 4 for NEXUS, 6 for QUIET

substudy = 'Punchout_mux'
outerFolder = os.path.join(f"/home/nexusadmin/Documents/Data/run35/4charge/Punchout Study/{substudy}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")
outerfolder_plots = outerFolder + "/documentation/"


DAC_att_1=10
DAC_att_2=15
DAC_att=DAC_att_1+DAC_att_2
ADC_att=17
from expt_config import FRIDGE
experiment = QICK_experiment(outerfolder_plots, DAC_attenuator1 = DAC_att_1, DAC_attenuator2 = DAC_att_2, qubit_DAC_attenuator1 = 5 , qubit_DAC_attenuator2 = 4 ,ADC_attenuator = ADC_att, fridge=FRIDGE)
qubits = [1] #[0, 1, 2, 3]
Unmask = False  #True is single, False is muxed
for Q in qubits:
    punch_out   = PunchOut(Q, number_of_qubits, outerfolder_plots, experiment, Unmask)
    start_gain, stop_gain, num_points = 0.1, 1, 5
    punch_out.run(experiment.soccfg, experiment.soc, start_gain, stop_gain, num_points, DAC_att, ADC_att, plot_Center_shift = True, plot_res_sweeps = True, plot_2d = True)

    del punch_out
del experiment
