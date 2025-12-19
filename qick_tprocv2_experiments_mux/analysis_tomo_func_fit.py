import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os
import re
import math as ma
from datetime import datetime

from sympy.concrete.guess import guess


def fit_func(n_g, d, nu, Vconv, phi):
    P = d + nu*np.cos(np.pi*np.cos(2*np.pi*Vconv*n_g - phi))
    return P



def guessFit_func(vsweep, data):
    nvals = 5
    fit_params = {}

    sort_data = np.sort(data)
    avg_min = np.mean(sort_data[:nvals])
    avg_max = np.mean(sort_data[-nvals:])
    #fit_params["g_d"] = (avg_max + avg_min)/2
    #fit_params["g_nu"] = (avg_max - avg_min)/2
    g_d = (avg_max + avg_min) / 2
    g_nu = (avg_max - avg_min)/2
    # fit_params["g_Vconv"] = (1/80) #1 e / period in mV
    #
    # fit_params["g_phi"] = np.pi
    return g_d, g_nu

def curveFit_func(fit_params, vsweep, data, r, backsub = False, plot = True, saveFolder = None):

    g_d = fit_params["g_d"]
    g_nu = fit_params["g_nu"]
    g_Vconv = fit_params["g_Vconv"]
    g_phi = fit_params["g_phi"]

    bounds = ((g_d-0.5, 1.5*g_nu, 0.5* g_Vconv, 0), (g_d+0.5, 0.2*g_nu, 1.2*g_Vconv, (2*np.pi)))
    popt, pcov = curve_fit(fit_func, vsweep, data, p0 = (g_d, g_nu, g_Vconv, g_phi), bounds = bounds)

    fit_params["f_d"] = popt[0]
    fit_params["f_nu"] = popt[1]
    fit_params["f_Vconv"] = popt[2]
    fit_params["f_phi"] = popt[3]
    fit_params["f_err"] = np.sqrt(np.diag(pcov))

    #print(fit_params)

    if plot:
        plt.plot(vsweep, data, '.', label = 'Data')
        plt.plot(vsweep, fit_func(vsweep, fit_params["f_d"], fit_params["f_nu"], fit_params["f_Vconv"], fit_params["f_phi"]), label = 'FIt')
        plt.legend()
        plt.xlabel("Bias (mV)")
        plt.ylabel("Amplitude (a.u.)")
        title = f'Fit Charge Tomography, Q4'
        if backsub:
            title += ', Bkgd Sub'
        title += f'\n d = {fit_params["f_d"]:.2f}, nu = {fit_params["f_nu"]:.2f}, Vconv = {fit_params["f_Vconv"]:.4f}, phi = {fit_params["f_phi"]:.2f}'
        plt.title(title)
        save_name = f'Fit_Charge_Tomography_Q4_R{r}'
        if backsub:
            save_name += '_Bkgdsub'
        save_name += '.png'
        fig_path = os.path.join(saveFolder, save_name)
        plt.savefig(fig_path)
        plt.close()
    return fit_params

# Simply, single file loader

def load_extractQ_data(studyFolder):
    metafile = glob.glob(os.path.join(studyFolder, "Tomography_Metadata*.npz"))
    if len(metafile) == 0:
        raise FileNotFoundError("No metadata file found in given folder")
    if len(metafile) > 1:
        raise ValueError("Multiple metadata files found. Specify one.")
    meta_path = metafile[0]

    datafiles = glob.glob(os.path.join(studyFolder, "Tomography_All*"))  # AllQs*"))
    if len(datafiles) == 0:
        raise ValueError(f"No tomography files found in given folder, {studyFolder}")

    meta = np.load(meta_path, allow_pickle=True)
    metadata = {k: meta[k].item() if meta[k].shape == () else meta[k] for k in meta}
    q_list = [3]  # [0, 1, 2, 3]

    data = {
        q: {
            # "vsweep": vsweep,
            "xi": [],
            "xq": []
        } for q in q_list
    }
    cycle_rounds = []
    cycle_timestamps = []

    for file in datafiles:
        base = os.path.basename(file).replace(".npz", "")

        find_round = re.search(r"_R(\d+)_", base)
        n_round = int(find_round.group(1)) if find_round else None

        find_time = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", base)
        if find_time:
            date_str = find_time.group(1)
            time_str = find_time.group(2).replace("-", ":")
            timestamp = datetime.fromisoformat(f"{date_str} {time_str}")
        else:
            timestamp = None
        cycle_rounds.append(n_round)
        cycle_timestamps.append(timestamp)
        #print(file)
        data_arrs = np.load(file, allow_pickle=True)["all_xi_xq"]

        for qi, q in enumerate(q_list):
            I = data_arrs[2 * qi]
            Q = data_arrs[2 * qi + 1]
            data[q]["xi"].append(I)
            data[q]["xq"].append(Q)

        data["_rounds"] = np.array(cycle_rounds)
        data["_timestamps"] = np.array(cycle_timestamps, dtype=object)

    return metadata, data


# Run single timestamp folder

run = 'run33e'
study = 'Cs_TimeStudy_Tomography' #SC_Tomography' #DDoff_SC_HoleOpen' #'DDon_SC_HoleClosed' #'DDon_SC_HoleOpen' #'Longtime_Study
substudy = 'AllQ_Tomography' #AllQ_Tomography'

qubit = 3

date = '2025-12-04'
timestamp = ('2025-12-04_13-36-05')
studyFolder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/study_data'
if not os.path.exists(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots/fit_plots'):
    os.makedirs(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots/fit_plots')
plotFolder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots/fit_plots'

metadata, data = load_extractQ_data(studyFolder)
vsweep = np.array(metadata["vsweep"])*1000 #in mV now
xi = np.array(data[qubit]["xi"][:]) #shape is (rounds, v_points)
xi_analysis = xi[0:30,]

guess_params = {"g_d": 0, #3.8,
                "g_nu": -0.5, #-1.35,
                "g_Vconv": 0.008,
                "g_phi": ma.pi+0.8}
guess_d = []
guess_nu = []

fit_d = []
fit_nu = []
fit_Vconv = []
fit_phi = []

xi_val = xi.astype(float)
for i in range(0, 30):
    xi_val[i] -= np.mean(xi[i,])
xi = xi_val

for r in range(0, 30):
    g_d, g_nu = guessFit_func(vsweep, xi[r,])
    guess_d.append(g_d)
    guess_nu.append(g_nu)
    fit_params = curveFit_func(guess_params, vsweep, xi[r,], r, backsub = True, plot = True, saveFolder = plotFolder)

    fit_d.append(fit_params["f_d"])
    fit_nu.append(fit_params["f_nu"])
    fit_Vconv.append(fit_params["f_Vconv"])
    fit_phi.append(fit_params["f_phi"])

print("Guesses")
print(np.mean(guess_d))
print(np.mean(guess_nu))

print("Fits")
print(np.mean(fit_d))
print(np.mean(fit_nu))
print(np.mean(fit_Vconv))
print(np.mean(fit_phi))

saveg_d = np.array(guess_d)
saveg_nu = np.array(guess_nu)

save_d = np.array(fit_d)
save_nu = np.array(fit_nu)
save_Vconv = np.array(fit_Vconv)
save_phi = np.array(fit_phi)

save_name = os.path.join(plotFolder, "fit_parameters_bkgdsub.npz")
np.savez(save_name, fit_d = save_d, fit_nu = save_nu, fit_Vonv = save_Vconv, fit_phi = save_phi)

save_name_g = os.path.join(plotFolder, "guess_parameters_bkgdsub.npz")
np.savez(save_name_g, guess_d = saveg_d, guess_nu = saveg_nu)

##############################
## Try plotting and fitting yourself

# plt.plot(vsweep, xi[0,], '.', label = 'Data')
# #f_d, f_nu, f_Vconv, f_phi
# plt.plot(vsweep, fit_func(vsweep, 3.75, -0.917, 0.0063, 3.11), label = 'Fit')
# plt.legend()
# plt.xlabel("Bias (mV)")
# plt.ylabel("Amplitude (a.u.)")
# plt.title("Fit Charge Tomography, Q4")
# plt.show()

#3.75, 0.917