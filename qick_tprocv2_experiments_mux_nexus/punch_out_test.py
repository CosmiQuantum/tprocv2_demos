import sys
import os
import numpy as np
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux_nexus/"))
from system_config import QICK_experiment
from section_003_punch_out_ge_mux import PunchOut
import datetime
from expt_config import expt_cfg #must change res spec expt to the pounchout one (it is commented out inside expt_config)

att_1=999
att_2=999

number_of_qubits = 4 #for QUIET 6, for NEXUS 4

outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30/", str(datetime.date.today()))

experiment = QICK_experiment(outerFolder)
punch_out   = PunchOut(outerFolder, experiment, expt_cfg, number_of_qubits)

# start_gain, stop_gain, num_points = 0.0, 0.8, 10
start_gain, stop_gain, num_points = 0.2, 0.45, 4
punch_out.run(experiment.soccfg, experiment.soc, start_gain, stop_gain, num_points, att_1, att_2, plot_Center_shift = False, plot_res_sweeps = True)

del punch_out
del experiment
