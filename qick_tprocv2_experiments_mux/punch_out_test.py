import sys
import os
import numpy as np
#sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/")) # for QUIET
# sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus/")) # for NEXUS
from system_config import QICK_experiment
from expt_config import FRIDGE
from section_003_punch_out_ge_mux import PunchOut
import datetime

number_of_qubits = 6  #currently 4 for NEXUS, 6 for QUIET

print('Fridge: ', FRIDGE)

Unmask = True
DAC_att_1=15
DAC_att_2=10
DAC_att=DAC_att_1+DAC_att_2
print('DAC atten: ', DAC_att)
ADC_att=17
print('ADC atten: ', ADC_att)

substudy = f'punchout_test'

outerFolder = os.path.join(f"/data/QICK_data/run10/entangling_chip/readout_optimization/{substudy}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")
outerfolder_plots = outerFolder + "/documentation/"

os.makedirs(outerfolder_plots, exist_ok=True)

experiment = QICK_experiment(outerfolder_plots, DAC_attenuator1 = DAC_att_1, DAC_attenuator2 = DAC_att_2, qubit_DAC_attenuator1 = 5 , qubit_DAC_attenuator2 = 4 ,ADC_attenuator = ADC_att, fridge=FRIDGE)

# ------------------------------ Runs for 1 resonator (specified by QubitIndex) -------------------------------------
# start_gain, stop_gain, num_points =  0.4, 0.65, 4 # for QUIET 0.55, 0.775, 5 #
# Qubit_index= 0 #starts at 0
# punch_out   = PunchOut(Qubit_index, number_of_qubits, outerfolder_plots, experiment, Unmask)
# punch_out.run(experiment.soccfg, experiment.soc, start_gain, stop_gain, num_points, DAC_att, ADC_att, plot_Center_shift = False, plot_res_sweeps = True)

#punch_out_vals = [0.95, 0.95, 1.0, 0.65, 1.0, 1.0] from 10/17
# [1.0, 0.9, 1.0, 0.65, 0.95, 1.0]  from 10/22/2025

# ------------------------------------ Loops over desired resonators ---------------------------------------------------
Qs = [0] #starts at 0

for QubitIndex in Qs:
    increase_geres_reps = False
    increase_geres_reps_to = None

    start_gain, stop_gain, num_points = 0.2, 0.3, 5

    punch_out  = PunchOut(QubitIndex, number_of_qubits, outerfolder_plots, experiment, Unmask, increase_geres_reps, increase_geres_reps_to)
    punch_out.run(experiment.soccfg, experiment.soc, start_gain, stop_gain, num_points, DAC_att, ADC_att, plot_Center_shift = False, plot_res_sweeps = True)

del punch_out
del experiment
