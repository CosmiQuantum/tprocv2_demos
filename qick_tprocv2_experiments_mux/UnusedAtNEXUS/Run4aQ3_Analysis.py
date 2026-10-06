from syspurpose.files import three_way_merge

from qick_tprocv2_experiments_mux.UnusedAtNEXUS.analysis_015_plot_all_run_stats import CompareRuns
from analysis_000_load_configs import LoadConfigs
from qick_tprocv2_experiments_mux.UnusedAtNEXUS.analysis_017_plot_metric_dependencies import PlotMetricDependencies

###################################################### Set These #######################################################
save_figs = True
fit_saved = False
show_legends = False
signal = 'None'
number_of_qubits = 6
run_number = 2 #starting from first run with qubits
figure_quality = 100 #ramp this up to like 500 for presentation plots
final_figure_quality = 200
run_name = '6transmon_run4'
# run_notes = ('Added more eccosorb filters and a lpf on mxc before and after the device. Added thermometry '
#              'next to the device') #please make it brief for the plot
top_folder_dates = ['2024-11-21', '2024-11-23', '2024-11-24']

date = '2024-11-23'  #only plot all of the data for one date at a time because there is a lot
outerFolder = f"/data/QICK_data/{run_name}/" + date + "/"
config_loader = LoadConfigs(outerFolder)
sys_config, exp_config = config_loader.run()
########################################################################################################################

#--------------------------------------- Load ALl data for this run ----------------------------------------------------
r = 1 #Will be analyzing data for this run. Note: run 1 = run 4a at QUIET and run 2 = run 5a at QUIET
run_number_list = [1]
run_stats_folder = f"run_stats/run{r}/"
filename = run_stats_folder + 'experiment_data.h5'
compare_runs = CompareRuns(run_number_list) #class instance
data = compare_runs.load_from_h5(filename)
#print(data.keys()) #if you want to see the type of data that is contained inside the file

#-----------------------------------Extracting relevant Q3 data for our plots-------------------------------------------

# Extract date/times and T1 values for qubit 3
date_times_t1_q3 = data['date_times_t1']['2']
t1_vals_q3       = data['t1_vals']['2']

date_times_pi_amps_q3 = data['date_times_pi_amps']['2']
pi_amps_q3 = data['pi_amps']['2']

q_freqs_q3 = data['q_freqs']['2']
date_times_q_spec_q3 = data['date_times_q_spec']['2']

#----------------------------------Q3 metrics vs time in one plot-------------------------------
plotter = PlotMetricDependencies(run_name, number_of_qubits, final_figure_quality)

#function originally was set up for qubit 1, ignore that it says q1, just provide the right data.
plotter.plot_q1_temp_and_t1(q1_temp_times= None, q1_temps= None, q1_t1_times=date_times_t1_q3, q1_t1_vals=t1_vals_q3, temp_label="Q3 Qubit Temp (mK)", t1_label="T1 (µs)",
                            magcan_dates = None, magcan_temps = None, magcan_label = "Mag Can Temp (mK)",
                            mcp2_dates = None, mcp2_temps = None, mcp2_label = "MCP2 Temp (mK)",
                            Q1_freqs = q_freqs_q3, Q1_dates_spec = date_times_q_spec_q3, qspec_label = "Frequency (MHz)",
                            date_times_pi_amps_Q1 = date_times_pi_amps_q3, pi_amps_Q1 = pi_amps_q3, pi_amps_label = "Pi Amp (a.u.)")
