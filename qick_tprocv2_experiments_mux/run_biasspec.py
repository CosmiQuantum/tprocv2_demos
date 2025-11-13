import sys
import os
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux"))
from system_config import QICK_experiment
from bias_qubit_spec import BiasQubitSpectroscopy
from expt_config import tot_num_of_qubits, FRIDGE
import datetime
import numpy as np

#number_of_qubits = 4

outerFolder = os.path.join(f"/home/nexusadmin/Documents/Data/run33d/4charge/Initial Checkout/Q2_BiasSpec", str(datetime.date.today()))

experiment = QICK_experiment(outerFolder, fridge=FRIDGE)
#resGs=np.linspace(0.1,0.5,11)
#resFs=np.linspace(5958.673-1.5, 5958.673+1.5, 11)
#num_qubits = 4
qubit = [2] #[1, 2, 3, 4] #Qubit to Run, 1-4
start_voltage = [0.0] #[0.0]*4 #0.06 #V
stop_voltage = [0.1] #[0.15]*4 #0.08 #0.15 max!!! #V
voltage_pts = [10] #*4

Unmask = True

for i,q in enumerate(qubit):
    bias_spec = BiasQubitSpectroscopy(q-1, tot_num_of_qubits, outerFolder, experiment, Unmask)
    bias_spec.run(experiment.soccfg, experiment.soc, start_voltage[i], stop_voltage[i], voltage_pts[i], plot_sweeps=True, plot_2d=True, plot_2dbacksub=True)
    print(bias_spec.config)

del bias_spec
del experiment