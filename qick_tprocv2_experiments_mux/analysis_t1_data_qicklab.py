import numpy as np
import matplotlib.pyplot as plt
import os

from qicklab.analysis.t1 import AnaT1
from qicklab.analysis.ssf import AnaSSF
from qicklab.analysis.auto_threshold import AnaAutoThreshold
from qicklab.utils import get_abs_min
import random

random.seed(1001)
np.random.seed(1001)
############### set values here ###################
study_dir = r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8" # on Arianna's local pc
    #"/data/QICK_data/run8/6transmon/round_robin" # on qubituser-daq01

substudy = 'ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional'
    #'AB_paper_datadump_T1_Analysis' # on qubituser-daq01

data_dir = os.path.join(study_dir, substudy)
dataset = '2025-10-27_22-04-57'

save_plts = True # saves plots
plot = True # uses plt.show()
save_plts_dir = r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional\t1_ge_analysis"
QubitIndex = 0  # zero indexed
analysis_flags = {"get_threshold": True, "load_all_data": True, "plot_t1_round": True}
selected_round = [0] # file you want to make plots for (we've only saved 1 round per h5 file in recent QUIET runs)
threshold = 0  # overwritten when get_threshold flag is set to True
theta = 0  # overwritten when get_threshold flag is set to True
sz = 7  # fontsize for plots
method_ssf = "max_contrast" # "gauss2" and "max_contrast" are the two options. This defines how the thresh and fid are calc in ssf
do_thresholding = True # thresholding for T1 analysis?
verbose = False
ssf_numbins = 55
iminuit_method_t1fit = True # Instead of the default Curvefit() T1 fitting, do you want to use iminuit?

############### experimental constants ###############
res_phase = [0, 0, 0, 0, 0, 0]  # can be pulled from system config of optimization rspec or qspec (any measurement before SSF overwrites it)
ro_length = [249, 345, 230, 326, 307, 384] # from QICK us2cycles conversion. QICK calculates it as: ro_length_cycles = trunc(res_length_us * decimated_MHz)

if analysis_flags["get_threshold"]:
    print("Determining threshold...")

    ana_params = {
        "idx": 0,
        "plot": plot,
        "method": "from_ssf",
        "ssf_theta": None,
        "ssf_threshold": None,
        "res_phase": res_phase[QubitIndex],
        "ro_length": ro_length[QubitIndex],
    }

    if ana_params["method"] == "from_ssf":
        ## load ssf data from study_data. Pass data and result to the AnaAutoThreshold class
        print("Loading SSF data...")

        ssf_ana_params = {
            "method": method_ssf,
            "numbins": ssf_numbins,
        }

        opt_ssf_ge = AnaSSF(data_dir, dataset, QubitIndex, folder="study_data", ana_params=ssf_ana_params)
        ssf_data = opt_ssf_ge.load_all(verbose=verbose)
        ssf_result = opt_ssf_ge.run_analysis(verbose=verbose)

        print('done')

        ana_params["ssf_theta"] = ssf_result["thetas"][0]
        ana_params["ssf_threshold"] = ssf_result["thresholds"][0]

        print('SSF threshold before being passed to AnaAutoThreshold: ', ana_params["ssf_threshold"])
        print('SSF theta before being passed to AnaAutoThreshold: ', ana_params["ssf_theta"])

    auto = AnaAutoThreshold(data_dir, dataset, QubitIndex, expt_name="t1_ge", datagroup="T1", ana_params=ana_params)

    data = auto.load_all()
    result = auto.run_analysis(verbose=verbose)

    theta = result["theta"]
    threshold = result["threshold"]

    auto.cleanup()
    del auto

if analysis_flags["load_all_data"]:

    ## =============================== SSF =============================== ##
    print("Loading SSF data...")

    ana_params = {
        "method": method_ssf,
        "numbins": ssf_numbins
    }

    ssf_ge = AnaSSF(data_dir, dataset, QubitIndex, ana_params=ana_params)
    data = ssf_ge.load_all(verbose=verbose)
    result = ssf_ge.run_analysis(verbose=verbose)

    ssf_dates = data["dates"]
    start_time = data["dates"][0]
    ssf_n = data["n"]
    I_g = data["I_g"]
    Q_g = data["Q_g"]
    I_e = data["I_e"]
    Q_e = data["Q_e"]
    fids = result["fids"]
    # angles = result["thetas"]
    # thresholds = result["thresholds"]

    # ssf_ge.cleanup()
    print('done')

    ## =============================== T1 =============================== ##
    print("Loading T1 data...")

    ana_params = {
        "theta": theta,
        "threshold": threshold,
        "thresholding": do_thresholding,
        "iminuit_fitting": iminuit_method_t1fit,
        "per_pt_errs": True
    }

    t1_ge = AnaT1(data_dir, dataset, QubitIndex, ana_params=ana_params)
    data = t1_ge.load_all(verbose=verbose)
    result = t1_ge.run_analysis(verbose=verbose)

    t1_dates = data["dates"]
    delay_times = data["delay_times"]
    t1_n = data["n"]
    t1_p_excited = result["p_excited"]
    t1s = result["t1s"]
    t1_errs = result["t1_errs"]
    print('done')
    # t1_ge.cleanup()

if analysis_flags["plot_t1_round"]:
    for r in selected_round:
        print(f"Plotting round {r} T1 data...")

        q1_fit_exponential, T1_err, T1_est = t1_ge.get_round(r, plot=plot, save_fig = save_plts, save_plt_dir = save_plts_dir,iminuit_method = iminuit_method_t1fit, verbose = True)
