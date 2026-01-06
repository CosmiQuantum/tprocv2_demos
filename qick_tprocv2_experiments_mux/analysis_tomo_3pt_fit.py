import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar
import glob
import os
from datetime import datetime
import re

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

def model(n_g, d, nu, mv2e, phi):
    P = d + nu*np.cos(np.pi*np.cos(2*np.pi*mv2e*n_g - phi))
    return P

def phi_minimization(vsweep, data, d, nu, mv2e):
    vsweep = np.asarray(vsweep)
    data = np.asarray(data)

    def cost(phi):
        data_pred = model(vsweep, d, nu, mv2e, phi)
        return np.sum((data - data_pred)**2)

    res = minimize_scalar(cost,bounds = (0, 2*np.pi), method = 'bounded')

    return res.x, res.fun, res.success

def phi_curvefit(vsweep, data, d, nu, mv2e):
    def model_phi(n_g, phi):
        return d + nu*np.cos(np.pi*np.cos(2*np.pi*mv2e*n_g - phi))

    popt, pcov = curve_fit(model_phi, vsweep, data, p0 = [0.0], bounds = (0, 2*np.pi))
    phi = popt[0]
    return phi, pcov

def phi_minimization_normalized(frac_T, data, d, nu):
    x = (np.asarray(data) - d) / abs(nu)

    def cost(phi):
        return np.sum(
            (x + np.cos(np.pi * np.cos(2 * np.pi * frac_T - phi)))**2
        )

    res = minimize_scalar(cost, bounds = (0, 2*np.pi), method = 'bounded')
    return res.x, res.fun, res.success

# unwrap phi: np.unwrap(phi)

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


#### Run one scan ####

round = 0

metadata, data = load_extractQ_data(studyFolder)
vsweep = np.array(metadata["vsweep"])*1000  #in mV
vsweep_ng = np.array([0, 3/25, 6/25]) #vsweep values - don't need Vconv if inputted this way
xi = np.array(data[qubit]["xi"][:])
xq = np.array(data[qubit]["xq"][:])

round_data = xi[round,] #pick xi or xq data, or add line to get magnitude

d = 3.6309
nu = -0.88719
mv2e = 0.0062051
phi, chi2, success = phi_minimization(vsweep, data, d, nu, mv2e)

#### Run multiple scans ####