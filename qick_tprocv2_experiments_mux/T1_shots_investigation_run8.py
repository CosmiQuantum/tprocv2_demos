import sys
import os
sys.path.append(os.path.abspath("/home/quietuser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from analysis_006_T1_vs_time_plots import T1VsTime
from analysis_009_T1_hist_cumul_err_plots import T1HistCumulErrPlots
from analysis_001_plot_all_RR_h5 import PlotAllRR
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from analysis_t1_qicklab_vs_offlineQick_funcs import comp_t1_methods_allQs_offline_vs_qicklab, run_qicklab_t1_all_qubits
###################################################### Set These #######################################################
save_figs = True
fit_saved = True
show_legends = False
signal = 'None'
run_number = 8
num_of_qubits = 6
figure_quality = 100 #ramp this up to like 500 for presentation plots
final_figure_quality = 200
saved_shots_t1ge = True # used for non-Qicklab analysis, where we can decide to process shots offline or just use QICK avg IQ arrays
per_pt_errs = True

t1_analysis_flags = {"load_t1_data_tprocv2": True, "Qicklab_T1_processing_allQs": True, "plot_RR_data": False, "t1_vs_time_plots": False, "t1_hists": False,
                     "t1_qicklab_vs_offline_shots": True}

if run_number == 8:
    run_name = "run8\ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional"
        # 'run8/6transmon/round_robin/AB_paper_datadump_for_analysis' # AB_paper_datadump_T1_Analysis
    data_path = fr"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\{run_name}"
        #f'/data/QICK_data/{run_name}'
    plots_path = data_path

    save_bad_t1plts_dir = r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional\t1_ge_analysis"

    top_folder_dates = ["2025-10-27_22-04-57"]

    # all of run 8 thus far, located in AB_paper_datadump_for_analysis
    # top_folder_dates = [
    #     "2025-10-19_11-09-32",
    #     "2025-10-19_12-05-25",
    #     "2025-10-19_19-43-00",
    #     "2025-10-19_20-25-18",
    #     "2025-10-20_12-10-19",
    #     "2025-10-23_00-49-28",
    #     "2025-10-23_14-47-22",
    #     "2025-10-24_13-58-37",
    #     "2025-10-27_14-15-40",
    #     "2025-10-27_14-24-29",
    #     "2025-10-27_22-04-57",
    #     "2025-10-28_21-57-47",
    #     "2025-10-29_18-38-25",
    #     "2025-10-29_23-48-45",
    #     "2025-10-31_01-54-57",
    #     "2025-10-31_20-40-11",
    #     "2025-11-01_12-54-55"
    # ]

    # # when saving t1 shots + avg IQ data started
    # top_folder_dates = [
    #                     "2025-10-24_13-58-37",
    #                     "2025-10-27_14-15-40",
    #                     "2025-10-27_14-24-29",
    #                     "2025-10-27_22-04-57"]

elif run_number == 7:
    run_name = 'run7/6transmon/round_robin_benchmark/AB_paper_data'
    data_path = f'/data/QICK_data/{run_name}'
    plots_path = data_path

    # all dates:
    top_folder_dates = ["2025-07-19_08-34-39",
                        "2025-07-19_16-16-14",
                        "2025-07-19_16-56-45",
                        "2025-07-19_23-11-39",
                        "2025-07-20_06-33-03" ]

elif run_number == 6:
    run_name = 'run6/6transmon'
    data_path = f'/exp/cosmiq/data/QUIET/QICK_data/{run_name}'
    plots_path = data_path

    # all dates:
    top_folder_dates = [
    "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-21",
    "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-22",
    "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-23",
    "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-24",
    "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-26",
    "ge_round_robin_presciencerun_data/ge_coherence_data/2025-02-28",
    "ge_round_robin_presciencerun_data/ge_coherence_data/2025-03-01",
    "ge_round_robin_presciencerun_data/ge_coherence_data/2025-03-02"]


    # add TLS_Comprehensive_Study/ before each of the ones below
    # "source_off_detuning_17MHz_Q1_substudy1/2025-05-15_14-47-38",
    # "source_off_detuning_17MHz_Q1_substudy1/2025-05-15_18-08-15",
    # "source_off_detuning_17MHz_Q1_substudy1/2025-05-15_22-02-12",
    # "source_off_detuning_17MHz_Q1_substudy1/2025-05-16_01-28-20",
    # "source_off_detuning_17MHz_Q1_substudy1/2025-05-16_04-49-59",
    # "source_off_detuning_17MHz_Q1_substudy1/2025-05-16_08-13-47",
    #
    # "source_off_detuning_24MHz_Q1_substudy1/2025-05-15_11-19-50",
    # "source_off_detuning_24MHz_Q1_substudy1/2025-05-15_18-35-56",
    #
    # "source_off_post_temperature_sweep_substudy1/2025-05-14_19-25-55",
    # "source_off_post_temperature_sweep_substudy1/2025-05-14_22-50-51",
    # "source_off_post_temperature_sweep_substudy1/2025-05-15_02-29-34",
    # "source_off_post_temperature_sweep_substudy1/2025-05-15_05-50-12",
    # "source_off_post_temperature_sweep_substudy1/2025-05-15_09-13-30",
    #
    # "source_off_substudy1/2025-04-15_21-24-46",
    #
    # "source_off_substudy2/2025-04-16_11-47-09",
    # "source_off_substudy2/2025-04-16_12-51-09",
    # "source_off_substudy2/2025-04-16_17-50-00",
    # "source_off_substudy2/2025-04-16_22-47-49",
    # "source_off_substudy2/2025-04-17_03-42-36",
    # "source_off_substudy2/2025-04-17_08-42-24",
    #
    # "source_off_substudy3/2025-04-17_12-28-37",
    # "source_off_substudy3/2025-04-17_17-22-46",
    # "source_off_substudy3/2025-04-17_22-16-39",
    # "source_off_substudy3/2025-04-18_01-45-53",
    # "source_off_substudy3/2025-04-18_06-40-55",
    #
    # "source_off_substudy4/2025-04-18_11-59-33",
    # "source_off_substudy4/2025-04-18_16-56-58",
    # "source_off_substudy4/2025-04-18_21-51-13",
    # "source_off_substudy4/2025-04-19_02-45-41",
    # "source_off_substudy4/2025-04-19_07-39-57",
    # "source_off_substudy4/2025-04-19_12-34-26",
    # "source_off_substudy4/2025-04-19_17-48-44",
    # "source_off_substudy4/2025-04-19_22-43-02",
    # "source_off_substudy4/2025-04-20_03-37-50",
    # "source_off_substudy4/2025-04-20_08-32-36",
    # "source_off_substudy4/2025-04-20_13-26-47",
    # "source_off_substudy4/2025-04-20_18-25-13",
    # "source_off_substudy4/2025-04-20_23-25-04",
    # "source_off_substudy4/2025-04-21_04-23-31",
    #
    # "source_off_substudy5/2025-05-04_20-56-05",
    # "source_off_substudy5/2025-05-04_23-28-05",
    # "source_off_substudy5/2025-05-05_03-03-40",
    # "source_off_substudy5/2025-05-05_06-40-15",
    # "source_off_substudy5/2025-05-05_10-18-53",
    # "source_off_substudy5/2025-05-05_13-57-22",
    # "source_off_substudy5/2025-05-05_17-34-21",
    # "source_off_substudy5/2025-05-05_21-18-14",
    # "source_off_substudy5/2025-05-06_02-18-57",
    #
    # "source_off_substudy6/2025-05-06_11-30-17",
    # "source_off_substudy6/2025-05-06_14-50-55",
    # "source_off_substudy6/2025-05-06_18-14-29",
    # "source_off_substudy6/2025-05-06_21-35-26",
    # "source_off_substudy6/2025-05-07_01-00-14",
    # "source_off_substudy6/2025-05-07_04-23-45",
    # "source_off_substudy6/2025-05-07_07-46-44",
    # "source_off_substudy6/2025-05-07_11-09-17",
    # "source_off_substudy6/2025-05-07_14-30-29",
    # "source_off_substudy6/2025-05-07_17-50-59",
    # "source_off_substudy6/2025-05-07_21-13-50",
    # "source_off_substudy6/2025-05-08_00-36-15",
    # "source_off_substudy6/2025-05-08_03-56-41",
    # "source_off_substudy6/2025-05-08_07-19-10",
    # "source_off_substudy6/2025-05-08_11-53-46"]

elif run_number == 5:
    run_name = 'run5/6transmon/Official_Round_Robin_Data_run5/CoolDown_Dec9_to_Dec20'
    data_path = f'/data/QICK_data/{run_name}'
    plots_path = data_path

    # all dates:
    top_folder_dates = [ # Condensing started 12/8/2024
                        "2024-12-09",
                        "2024-12-10",
                        "2024-12-11",
                        "2024-12-12",
                        "2024-12-13",
                        "2024-12-14",
                        "2024-12-15",
                        "2024-12-16",
                        "2024-12-17",
                        "2024-12-18",
                        "2024-12-19",
                        "2024-12-20"]
elif run_number == 4:
    run_name = 'run4/6transmon/Official_run4_RR_Data_which_started_Nov21'
    data_path = f'/data/QICK_data/{run_name}'
    plots_path = data_path

    # all dates:
    top_folder_dates = [
                        "2024-11-21",
                        "2024-11-23",
                        "2024-11-24",
                        "2024-11-25",
                        "2024-12-09",
                        "2024-12-10"]

FRIDGE = "QUIET"
run_notes = ('Added IR shielding, better cryo terminators, thermalizing with 0dB attenuator ') #please make it brief for the plot

############################################### Get all data #########################################################
if t1_analysis_flags["load_t1_data_tprocv2"]:
    t1_vs_time = T1VsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                     signal, run_name, FRIDGE, run_number, per_pt_errs = per_pt_errs)
    if saved_shots_t1ge:
        if per_pt_errs:
            print(f'Analyzing T1 data using tprocv2 functions. Processing shots and returning per-point errs too.')
            date_times_t1, t1_vals, t1_fit_err, I_per_pt_errs, Q_per_pt_errs = t1_vs_time.run(return_errs=True, exp_extension = '_ge', saved_shots = saved_shots_t1ge)
            offline_tuple = (date_times_t1, t1_vals, t1_fit_err, I_per_pt_errs, Q_per_pt_errs)
            # print('t1 vals', t1_vals[0])
        else:
            print(f'Analyzing T1 data using tprocv2 functions. Processing shots but not returning per-point errs.')
            date_times_t1, t1_vals, t1_fit_err, _, _ = t1_vs_time.run(return_errs=True, exp_extension='_ge', saved_shots=saved_shots_t1ge)
            I_per_pt_errs = None
            Q_per_pt_errs = None
            offline_tuple = (date_times_t1, t1_vals, t1_fit_err, I_per_pt_errs, Q_per_pt_errs)
    else:
        print(f'Analyzing T1 data using tprocv2 functions. Processing QICK Avg IQ data, not shots.')
        date_times_t1, t1_vals, t1_fit_err= t1_vs_time.run(return_errs=True,exp_extension='_ge',saved_shots=saved_shots_t1ge)

########################################  Plot All Individual Data Plots ###########################################
# from tprocv2_demos.qick_tprocv2_experiments_mux.socProxy import makeProxy
# soc, soccfg = makeProxy()
# print(soccfg)

if t1_analysis_flags["plot_RR_data"]:
    date = "2025-10-27_22-04-57"  #only plot all of the data for one date at a time because there is a lot
    unique_folder_path = r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional/replotted_RR_data/2025-10-27_22-04-57/t1_ge/shots_method/"
        # f"/data/QICK_data/run8/6transmon/replotted_RR_data/{date}/shots_method/" # only used when plot_rabis_Qtemps = True or for load_t1_shots_vs_avgIQ_arrays()
    outerFolder = r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional/2025-10-27_22-04-57/study_data"
        #f"/data/QICK_data/run8/6transmon/round_robin/ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional/{date}/study_data"
    #outerFolder_save_plots = f"/data/QICK_data/run8/6transmon/replotted_RR_data/{date}/"
    outerFolder_save_plots = r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional/replotted_RR_data/2025-10-27_22-04-57/t1_ge/avg_IQ_method/"
        #f"/data/QICK_data/run8/6transmon/replotted_RR_data/{date}/avg_IQ_method/"
    saved_shots_t1 = True # for plot_t1. plot_t1_shots_analysis does both methods regardless of this flag (purpose if to compare them).
    per_pt_errs_T1 = True
    plotter = PlotAllRR(date, figure_quality, save_figs, fit_saved, signal, run_name, run_number, tot_num_of_qubits, outerFolder,
                      outerFolder_save_plots, unique_folder_path, saved_shots = saved_shots_t1, per_pt_errs = per_pt_errs_T1)
    plotter.run(plot_res_spec = False, plot_q_spec = False, plot_rabi = False, rabi_rolling_avg=False, plot_ss = False,
                plot_ss_hist_only=False,ss_plot_title = None, ss_plot_gef = False, plot_t1 = False,
                plot_t2r = False, plot_t2e = False, plot_rabis_Qtemps = False, plot_t1_shots_analysis = True)

################################################# T1 vs Time Plots #################################################
if t1_analysis_flags["t1_vs_time_plots"]:
    # t1_vs_time.plot_without_errs(date_times_t1, t1_vals, show_legends)
    t1_vs_time.plot_with_errs(date_times_t1, t1_vals, t1_fit_err, show_legends)
    # t1_vs_time.plot_with_errs_single_plot(date_times_t1, t1_vals, t1_fit_err, show_legends=True)

############################################### T1 hist/cumul/err Plots #############################################
if t1_analysis_flags["t1_hists"]:
    t1_distribution_plots = T1HistCumulErrPlots(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates,
                                                save_figs, fit_saved, signal, data_path, plots_path, run_name, run_notes, run_number, fridge=FRIDGE)
    dates, t1_vals, t1_errs = t1_distribution_plots.run(exp_extension="_ge", saved_shots = saved_shots_t1ge)
    t1_std_values, t1_mean_values = t1_distribution_plots.plot(dates, t1_vals, t1_errs, show_legends)

###############################################  QICKLab shot-based T1 (with thresholding option) ############################
if t1_analysis_flags["Qicklab_T1_processing_allQs"]:
    study_dir = r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8" # on Arianna's local pc
        #"/data/QICK_data/run8/6transmon/round_robin" # on qubituser-daq01
    substudy = 'ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional'
        #'AB_paper_datadump_T1_Analysis' # on qubituser-daq01
    data_dir = os.path.join(study_dir, substudy)
    dataset = '2025-10-27_22-04-57'
    res_phase = [0, 0, 0, 0, 0, 0]  # can be pulled from system config of optimization rspec or qspec (any measurement before SSF overwrites it)
    ro_length = [249, 345, 230, 326, 307, 384] # from QICK us2cycles conversion. QICK calculates it as: ro_length_cycles = trunc(res_length_us * decimated_MHz)
    iminuit_method_t1fit = True # Instead of the default Curvefit() T1 fitting, do you want to use iminuit?
    ssf_numbins = 55
    method_ssf = "max_contrast" # "gauss2" and "max_contrast" are the two options. This defines how the thresh and fid are calc in ssf
    do_thresholding = True # thresholding for T1 analysis?
    verbose = False 
    plot_threshold = False 
    plot_t1_round = True
    save_rejected_plots = False
    max_t1_keep = 300
    rounds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

    save_good_t1plts_dir = save_bad_t1plts_dir
    
    qicklab_out = run_qicklab_t1_all_qubits(
        data_dir=data_dir,
        dataset=dataset,
        qubits_to_analyze=num_of_qubits,                 # or however many are in the H5 files
        res_phase=res_phase,                # list, length = n_qubits
        ro_length=ro_length,                # list, length = n_qubits
        method_ssf=method_ssf,              # e.g. "max_contrast" or "gauss2"
        ssf_numbins=ssf_numbins,
        do_thresholding=do_thresholding,   
        iminuit_method_t1fit=iminuit_method_t1fit,
        selected_rounds=rounds,
        per_pt_errs = per_pt_errs,
        plot_threshold=plot_threshold,               # or True if you want SSF/auto-thresh plots
        save_plot_t1_round=plot_t1_round,     # save plots for ACCEPTED rounds (no show)
        save_rejected_plots=save_rejected_plots,  # save plots for REJECTED rounds above max_t1_keep
        save_plt_dir = save_good_t1plts_dir,
        rejected_plots_dir=save_bad_t1plts_dir,
        max_t1_keep=max_t1_keep,
        verbose=verbose,
    )
################################## Comparing Offline T1 shots vs Qicklab Processed Shots with Thresholding #############
if t1_analysis_flags["t1_qicklab_vs_offline_shots"]:
    comp = comp_t1_methods_allQs_offline_vs_qicklab(
        qicklab_out=qicklab_out,  # from your QICKLab runner (shots+thresholding)
        offline_tuple=offline_tuple,  # from your offline processing run()
        out_dir=r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional\t1_ge_analysis",
        save_comp_results_plot=save_figs,
        max_t1 = max_t1_keep
    )