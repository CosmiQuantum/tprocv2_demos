import sys
import os
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux"))
from system_config import QICK_experiment
from bias_qubit_spec import BiasQubitSpectroscopy
from expt_config import tot_num_of_qubits, FRIDGE
import datetime
import numpy as np

#number_of_qubits = 4
qubit = [4] #[1, 2, 3, 4] #Qubit to Run, 1-4
start_voltage = [0.03] #[0.0]*4 #0.06 #V
stop_voltage = [0.05] #[0.15]*4 #0.08 #0.15 max!!! #V
voltage_pts = [10] #*]4

run = "run36"
study = "Initial Checkout"
substudy = f"Q{qubit[0]}_BiasSpec" #"AllQ_BiasSpec" #"Q{qubit[0]}_BiasSpec"

outerFolder = os.path.join(f"/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}", str(datetime.date.today()))

experiment = QICK_experiment(outerFolder, fridge=FRIDGE)
#resGs=np.linspace(0.1,0.5,11)
#resFs=np.linspace(5958.673-1.5, 5958.673+1.5, 11)
#num_qubits = 4

PS = 'Keithley' #'Keithley' #'Keysight'

Unmask = True

for i,q in enumerate(qubit):
    bias_spec = BiasQubitSpectroscopy(q-1, tot_num_of_qubits, outerFolder, experiment, Unmask)
    bias_spec.run(experiment.soccfg, experiment.soc, PS, start_voltage[0], stop_voltage[0], voltage_pts[0], plot_sweeps=True, plot_2d=True, plot_2dbacksub=True)
    print(bias_spec.config)

del bias_spec
del experiment