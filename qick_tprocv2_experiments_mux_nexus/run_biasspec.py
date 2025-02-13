import sys
import os
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus"))
from system_config import QICK_experiment
from bias_qubit_spec import BiasQubitSpectroscopy

import datetime
import numpy as np
outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30", str(datetime.date.today()))



qubit = 3  #Qubit to Run
start_voltage = 0 #V
stop_voltage = 0.15 #V
voltage_pts = 30


#resGs=np.linspace(0.1,0.5,5)
#resFs=np.linspace(5959.553-1.5, 5959.553+1.5, 4)
#f = 5858.673

#for g in resGs:
#for f in resFs:
experiment = QICK_experiment(outerFolder)
bias_spec = BiasQubitSpectroscopy(qubit-1, outerFolder, experiment)
# bias_spec.config['res_freq_ge'][qubit-1]=6187.411
# bias_spec.config['res_gain_ge'][qubit - 1] =0.46
# bias_spec.config['res_length'] =5.50214
# experiment.qconfig["qubit_freq_ge"] =4902
# bias_spec.config["qubit_gain_ge"] =0.1
# bias_spec.config['start']= 4902 - 5
# bias_spec.config['stop'] = 4902 + 5
# bias_spec.config['step'] = 100
# bias_spec.config['reps'] = 700


#print(bias_spec.config)

bias_spec.run(experiment.soccfg, experiment.soc, start_voltage, stop_voltage, voltage_pts, plot_sweeps=True, plot_3d=True, plot_3d_backsub=True)


del bias_spec
del experiment