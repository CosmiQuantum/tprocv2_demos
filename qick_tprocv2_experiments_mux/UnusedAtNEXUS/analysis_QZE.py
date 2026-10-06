from syspurpose.files import three_way_merge
from qick_tprocv2_experiments_mux.UnusedAtNEXUS.analysis_004_pi_amp_vs_time_plots import PiAmpsVsTime
# from datetime import datetime
import datetime
import pytz
import os
from expt_config import tot_num_of_qubits
import numpy as np
import json
import h5py
from qualang_tools.plot import Fit
import visdom

###################################################### Set These #######################################################
save_figs = True
fit_saved = False
show_legends = False
signal = 'None'
run_number = 3 #starting from first run with qubits. Run 1 = run4a at quiet, run 2 = run5a at quiet, etc
figure_quality = 100 #ramp this up to like 500 for presentation plots
final_figure_quality = 100

FRIDGE = "QUIET"
run_notes = ('Added IR shielding, better cryo terminators, thermalizing with 0dB attenuator ') #please make it brief for the plot

date='no_qubit_rabi_drive2025-05-20_06-55-16'

outerFolder = '/data/QICK_data/run6/6transmon/QZE/rabi/study_data'
outerFolder_systematics = f'/data/QICK_data/run6/6transmon/QZE/rabi/{date}/study_data'

outerFolder_save_plots = os.path.join(f'/data/QICK_data/run6/6transmon/QZE/rabi/{date}/study_data/', "QZE_plot")
if not os.path.exists(outerFolder_save_plots):
    os.makedirs(outerFolder_save_plots)

################################################ 01: Get all data ######################################################

pi_amps_vs_time = PiAmpsVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, [date], save_figs,
                              fit_saved,signal,'/data/QICK_data/run6/6transmon/QZE/')

pi_amps_vs_time.runQZE(outerFolder_systematics, outerFolder_save_plots, fit=False, expt_name = "length_rabi_ge_qze",
                       old_format=False, filter_amp_above=0, mark_w01s=True, plot_detuned_amps=True,plot_detuning=True,
                       pi_line_label_left='No\nZeno\nPulse',pi_line_label_right='Zeno Pulse', qubit_index=1,
                       z_limit=None,z_limit_lower=None, gain_max=0.075)#5.5
# pi_amps_vs_time.runQZE_systematics_subtraction(outerFolder, outerFolder_systematics, outerFolder_save_plots,
#                                                fit=False, expt_name = "length_rabi_ge_qze")
#
# pi_amps_vs_time.plot_chevron_qze(outerFolder_systematics, outerFolder_save_plots, fit=False, expt_name = "length_rabi_ge_qze",
#                        old_format=False, filter_amp_above=0, mark_w01s=True, plot_detuned_amps=True,plot_detuning=True,
#                        pi_line_label_left='Actual\nQfreq',pi_line_label_right='Shifted Qfreq', qubit_index=1,z_limit=5.5)
