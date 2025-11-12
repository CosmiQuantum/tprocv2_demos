import numpy as np
import matplotlib.pyplot as plt
import os

from qicklab.analysis.t1 import AnaT1
from qicklab.analysis.ssf import AnaSSF
from qicklab.analysis.auto_threshold import AnaAutoThreshold
from qicklab.utils import get_abs_min

np.random.seed(1001)
############### set values here ###################
study_dir = "/data/QICK_data/run8/6transmon/round_robin"

substudy = 'AB_paper_datadump_T1_Analysis'
data_dir = os.path.join(study_dir, substudy)
dataset = '2025-10-27_22-04-57'

QubitIndex = 0  # zero indexed
analysis_flags = {"get_threshold": False, "load_all_data": True, "timestream": False, "plot_RR_t1": True}
selected_round = [0]
threshold = 0  # overwritten when get_threshold flag is set to True
theta = 0  # overwritten when get_threshold flag is set to True
sz = 10  # fontsize for plots

############### experimental constants ###############
res_phase = [1.748, 0, 0, 0, 2.61,0]  # can be pulled from system config of optimization rspec or qspec (any measurement before SSF overwrites it)
ro_length = [249, 345, 230, 326, 307, 384] # from QICK us2cycles conversion. Depends on the qubit's readout length

verbose = True

if analysis_flags["get_threshold"]:
    print("Determining threshold...")

    ana_params = {
        "idx": 0,
        "plot": True,
        "method": "from_ssf",
        "ssf_theta": None,
        "ssf_threshold": None,
        "res_phase": res_phase[QubitIndex],
        "ro_length": ro_length[QubitIndex],
    }

    if ana_params["method"] == "from_ssf":
        ## load ssf data from optimization. Pass data and result to the AnaAutoThreshold class
        print("Loading optimization SSF data...")

        ssf_ana_params = {
            "method": "gauss2",
        }

        opt_ssf_ge = AnaSSF(data_dir, dataset, QubitIndex, folder="optimization", ana_params=ssf_ana_params)
        ssf_data = opt_ssf_ge.load_all(verbose=verbose)
        ssf_result = opt_ssf_ge.run_analysis(verbose=verbose)
        ana_params["ssf_theta"] = ssf_result["thetas"][0]
        ana_params["ssf_threshold"] = ssf_result["thresholds"][0]

    auto = AnaAutoThreshold(data_dir, dataset, QubitIndex, ana_params=ana_params)

    data = auto.load_all()
    result = auto.run_analysis(verbose=verbose)
    plt.show()

    theta = result["theta"]
    threshold = result["threshold"]

    auto.cleanup()
    del auto

if analysis_flags["load_all_data"]:

    ## =============================== SSF =============================== ##
    print("Loading SSF data...")

    ana_params = {
        "method": "gauss2",
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

    ## =============================== T1 =============================== ##
    print("Loading T1 data...")

    ana_params = {
        "theta": theta,
        "threshold": threshold,
        "thresholding": True
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
if analysis_flags["plot_RR_t1"]:
    fig, ax = plt.subplots(3, 2, layout='constrained')


if analysis_flags["timestream"]:
    fig, ax = plt.subplots(3, 2, layout='constrained')

    ##### single-shot ##########
    plot = ax[1][0]
    try:
        plot.errorbar(get_abs_min(start_time, ssf_dates), np.array(fids) * 100, fmt='o')
        plot.set_xlabel('time [min]')
        plot.set_ylabel('single-shot fidelity [%]')
        for i in selected_round:
            plot.scatter((ssf_dates[i] - start_time).total_seconds() / 60, fids[i] * 100, marker="o", s=200, alpha=0.5)
    except Exception:
        plot.set_title("ssf_ge data error")

    ##### t1 data #####
    plot = ax[2][0]
    try:
        plot.errorbar(get_abs_min(start_time, t1_dates), t1s, t1_errs, fmt='o')
        plot.set_xlabel('time [min]')
        plot.set_ylabel('t1_ge [us]')
        for i in selected_round:
            plot.scatter((t1_dates[i] - start_time).total_seconds() / 60, t1s[i], marker="o", s=200, alpha=0.5)
    except Exception:
        plot.title("t1_ge data error")


    ##### t1 data #####
    q1_fit_exponential, T1_err, T1_est = t1_ge.get_round(round, plot=False)
    plot = ax[0][2]

    try:
        plot.plot(delay_times, t1_p_excited[round],
                  label=f'round {round + 1} T1 = {T1_est:.2f} +/- {T1_err:.2f} us')
        plot.plot(delay_times, q1_fit_exponential, 'k:')
        plot.set_title('t1_ge', fontsize=sz)
        plot.set_ylabel('P(e)', fontsize=sz)
        plot.set_xlabel('delay time [us]', fontsize=sz)
        plot.legend(fontsize=sz)
    except Exception:
        plot.set_title("t1_ge data error", fontsize=sz)

    ##### ssf data ####
    theta0, threshold0, fid0, ig_new, qg_new, ie_new, qe_new, xg, yg, xe, ye = ssf_ge.get_round(round)
    plot = ax[1][2]
    try:
        plot.scatter(ig_new, qg_new, c='b', label='g', s=2)
        plot.scatter(ie_new, qe_new, c='r', label='e', s=2)
        plot.set_title(f'ssf_ge: theta = {np.round(theta0, 3)}, threshold = {np.round(threshold0, 2)}', fontsize=sz)
        plot.scatter(xg, yg, c='k', s=6)
        plot.scatter(xe, ye, c='k', s=6)
        plot.set_xlabel('I [a.u.]', fontsize=sz)
        plot.set_ylabel('Q [a.u.]', fontsize=sz)
        plot.plot([threshold0, threshold0], [np.min(qe_new), np.max(qe_new)], 'k:', linewidth=2)
        plot.legend(fontsize=sz)
        plot.set_aspect("equal")
    except Exception:
        plot.set_title("ssf_ge data error", fontsize=sz)

    fig.suptitle(f'{substudy} dataset {dataset} qubit {QubitIndex + 1}')

plt.show()