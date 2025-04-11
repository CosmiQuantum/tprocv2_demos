from analysis_014_temperature_calcsandplots import TempCalcAndPlots
from expt_config import tot_num_of_qubits

###################################################### Set These #######################################################
save_figs = True
fit_saved = False
show_legends = False
signal = 'None'
run_number = 2 #starting from first run with qubits. Run 1 = run4a at quiet, run 2 = run5a at quiet, etc
figure_quality = 100 #ramp this up to like 500 for presentation plots
final_figure_quality = 200
run_name = '6transmon_run6'
FRIDGE = "QUIET"
top_folder_dates = ['2025-03-02']
date = '2025-03-02'
outerFolder = f"/data/QICK_data/{run_name}/" + date + "/"

# ########################################## Angles vs Time Plots ###########################################
temps_class_obj = TempCalcAndPlots(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
                                   save_figs, fit_saved, signal, run_name,outerFolder, fridge = FRIDGE)

ssf, qubit_ssf_dates, angles,thresholds = temps_class_obj.load_ss_data(outerFolder)
temps_class_obj.plot_hist(thresholds, show_legends, data_name='Threshold', x_label='Threshold (a.u.)', save_name='thresh', bin_num=10)
temps_class_obj.plot_vs_time(qubit_ssf_dates, thresholds, show_legends,ylabel='Thresholds',
                             save_name='thresh_vs_time', title = 'Thresholds vs Time')

# temps_class_obj.plot_vs_time(qubit_ssf_dates, angles, show_legends,ylabel='SS angle (Rad)',
#                              save_name='ss_angle_vs_time', title = 'SS Angle vs Time')
# temps_class_obj.plot_vs_time(qubit_ssf_dates, ssf, show_legends,ylabel='SSF',
#                              save_name='ssf_vs_time', title = 'SSF vs Time')
# temps_class_obj.plot_hist(angles, show_legends, data_name='SS Angle', x_label='SS angle (Rad)', save_name='ss_angle', bin_num=10)
# temps_class_obj.plot_hist(ssf, show_legends, data_name='SSF', x_label='SSF', save_name='ssf', bin_num=10)