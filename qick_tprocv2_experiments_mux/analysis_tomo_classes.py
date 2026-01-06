import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import h5py
import json
import glob
import os
import re
from datetime import datetime

class GetData:
    def __init__(self, studyFolder, num_of_qubits):
        self.studyFolder = studyFolder
        self.num_of_qubits = num_of_qubits

    def getNPZmetadata(self):
        ## Structure: q_cfg: Q0, etc (indiv configs for each qubit), vsweep (voltage array (V)),
        ##      tot_rounds (# of rounds set (usually more than # of actual rounds)), vsweep_mV (voltage array (mV))
        metafile = glob.glob(os.path.join(self.studyFolder, "Tomography_Metadata*.npz"))
        if len(metafile) == 0:
            raise FileNotFoundError(f"No metadata file found in given folder, {self.studyFolder}")
        if len(metafile) > 1:
            raise ValueError("Multiple metadata files found. Specify one.")
        meta_path = metafile[0]
        meta = np.load(meta_path, allow_pickle=True)
        metadata = {k: meta[k].item() if meta[k].shape == () else meta[k] for k in meta}
        metadata["vsweep_mV"] = metadata["vsweep"] * 1000
        return metadata

    def getNPZdata_fromallQ(self, qubits, rounds=(None, None)):
        ## Structure: qubit #: {xi (array of values for each round), xq (array of values for each round) - rounds x 30 shape},
        ##      _rounds (array of rounds numbers), _timestamps (array of dt object timestamps for each round)
        tot_qubits = np.arange(self.num_of_qubits)
        all_files = glob.glob(os.path.join(self.studyFolder, "Tomography_AllQs_R*_*.npz"))
        if len(all_files) == 0:
            raise ValueError(f"No tomography files found in given folder, {self.studyFolder}")
        selected_files = []
        for f in all_files:
            base = os.path.basename(f)
            try:
                r = int(base.split("_R")[1].split("_", 1)[0])
            except (IndexError, ValueError):
                continue

            if rounds[0] is not None and r < rounds[0]:
                continue
            if rounds[1] is not None and r > rounds[1]:
                continue

            selected_files.append((r,f))

        if not selected_files:
            raise ValueError("No files found in requested round range")

        # Sort by round number
        selected_files.sort(key=lambda x: x[0])

        # Prepare data containers
        data = {
            q: {
                "xi": [],
                "xq": [],
            } for q in qubits
        }

        cycle_rounds = []
        cycle_timestamps = []

        # Load selected files
        for r, file in selected_files:
            base = os.path.basename(file)

            date, time = base.rsplit("_", 2)[1:]
            timestamp = datetime.fromisoformat(date + " " + time.replace("-", ":").replace(".npz", ""))

            cycle_rounds.append(r)
            cycle_timestamps.append(timestamp)

            data_arrs = np.load(file, allow_pickle=True)["all_xi_xq"]

            for qi, q in enumerate(tot_qubits):
                if q in qubits:
                    I = data_arrs[2 * qi]
                    Q = data_arrs[2 * qi + 1]
                    data[q]["xi"].append(np.array(I))
                    data[q]["xq"].append(np.array(Q))
            data["_rounds"] = np.array(cycle_rounds)
            data["_timestamps"] = np.array(cycle_timestamps, dtype=object)
        for q in qubits:
            data[q]["xi"] = np.array(data[q]["xi"])
            data[q]["xq"] = np.array(data[q]["xq"])
        return data

    def getNPZdata_fromsingleQ(self, qubits, rounds=(None, None)):
        ## Structure: qubit #: {xi (array of values for each round), xq (array of values for each round) - rounds x 3 shape},
        ##      _rounds (array of rounds numbers), _timestamps (array of dt object timestamps for each round)
        if len(qubits) > 1:
            print('Only input 1 qubit')
        qubit = qubits[0]
        all_files = glob.glob(os.path.join(self.studyFolder, f"Tomography_Q{qubit+1}_R*_*.npz"))
        if len(all_files) == 0:
            raise ValueError(f"No qubit {qubit+1} tomography files found in given folder, {self.studyFolder}")
        selected_files = []
        for f in all_files:
            base = os.path.basename(f)
            try:
                r = int(base.split("_R")[1].split("_", 1)[0])
            except (IndexError, ValueError):
                continue

            if rounds[0] is not None and r < rounds[0]:
                continue
            if rounds[1] is not None and r > rounds[1]:
                continue

            selected_files.append((r, f))

        if not selected_files:
            raise ValueError("No files found in requested round range")

        # Sort by round number
        selected_files.sort(key=lambda x: x[0])

        # Prepare data containers
        data = {
            q: {
                "xi": [],
                "xq": [],
            } for q in qubits
        }

        cycle_rounds = []
        cycle_timestamps = []

        # Load selected files
        for r, file in selected_files:
            base = os.path.basename(file)

            date, time = base.rsplit("_", 2)[1:]
            timestamp = datetime.fromisoformat(date + " " + time.replace("-", ":").replace(".npz", ""))

            cycle_rounds.append(r)
            cycle_timestamps.append(timestamp)

            data_arrs = np.load(file, allow_pickle=True)["xi_xq"]
            for q in qubits:
                I = data_arrs[0]
                Q = data_arrs[1]
                data[q]["xi"].append(I)
                data[q]["xq"].append(Q)
            data["_rounds"] = np.array(cycle_rounds)
            data["_timestamps"] = np.array(cycle_timestamps, dtype=object)
        for q in qubits:
            data[q]["xi"] = np.array(data[q]["xi"])
            data[q]["xq"] = np.array(data[q]["xq"])
        return data

    def getH5metadata(self, file):
        ## Structure: file_timestamp ('%Y-%m-%d_%H-%M-%S'), tot_rounds (# of rows written, int),
        ##      vsweep_mV (voltage array (mV)), config (single qubit config)
        metafile = os.path.join(self.studyFolder, file)
        match = re.search(r'Tomography_Q\d+_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', file)
        file_timestamp = match.group(1) if match else "Unknown"

        with h5py.File(metafile, "r") as f:
            rows = int(f.attrs["rows_written"])
            vsweep_mV = f["vsweep"][:] * 1000
            config_raw = f["config"][()]

        if isinstance(config_raw, bytes):
            config_raw = config_raw.decode('ascii')
        config = json.loads(config_raw)

        metadata = {
            "file_timestamp": file_timestamp,
            "tot_rounds": rows,
            "vsweep_mV": vsweep_mV,
            "config": config,
        }
        return metadata

    def getH5data(self, file, qubit, rounds=(None, None)):
        ## Structure: qubit #: {xi (array of values for each round), xq (array of values for each round) - rounds x 3 shape},
        ##      _rounds (array of rounds numbers), _timestamps (array of dt object timestamps for each round)
        # Currently only written for data on one qubit (since all H5 data only has one qubit
        datafile = os.path.join(self.studyFolder, file)
        with h5py.File(datafile,'r') as f:
            rows = int(f.attrs["rows_written"])

            start = 0 if rounds[0] is None else rounds[0]
            stop = rows if rounds[1] is None else min(rounds[1]+1, rows)

            if start >= stop:
                raise ValueError("Invalid round range")

            qdata = f["qdata"][start:stop]
            timestamps = f["timestamps"][start:stop]
            voltage_check = f["voltage_check"][start:stop]

        timestamps_dt = np.array([datetime.strptime(t.decode('ascii'), "%Y-%m-%d_%H-%M-%S") for t in timestamps])

        n_rounds = stop - start
        rounds_arr = np.arange(start, stop)

        data = {
            q: {
                "xi": [],
                "xq": [],
            } for q in qubit
        }

        for q in qubit:
            data[q]["xi"] = qdata[:, 0, :]
            data[q]["xq"] = qdata[:, 1, :]

        data["_rounds"] = rounds_arr
        data["_timestamps"] = timestamps_dt
        data["_voltage_check"] = voltage_check
        return data

# class FuncFit:
#     def __init__(self, plotFolder, vsweep, data, qubit, data_type):
#         self.plotFolder = plotFolder
#         self.vsweep = vsweep # array in mV, length is 1 scan length (3 or 30)
#         #self.full_data = data # full data dict
#         self.qubit = qubit
#         self.data_type = data_type
#         if data_type == 'I':
#             self.data = data[qubit]["xi"][:] #shape is (rounds, v_points)
#         elif data_type == 'Q':
#             self.data = data[qubit]["xq"][:] #shape is (rounds, v_points)
#         elif data_type == 'Amp':
#             xi = data[qubit]["xi"][:]
#             xq = data[qubit]["xq"][:]
#             self.data = np.sqrt(xi**2 + xq**2)
#         else:
#             raise ValueError(f"Invalid data type: {data_type}. Must be 'I', 'Q', or 'Amp'")
#
#         self.rounds = data["_rounds"]
#         self.timestamps = data["_timestamps"]
#
#     def analytic_form(self, n_g, d, nu, mv2e, phi):
#         P = d + nu*np.cos(np.pi*np.cos(2*np.pi*mv2e*n_g - phi))
#         return P
#
#     def guess_analytic_form(self, fit_data, n_vals):
#         if n_vals == 1:
#             min = np.min(data)
#             max = np.max(data)
#         elif n_vals > 1:
#             sort_data = np.sort(data)
#             min = np.mean(sort_data[:n_vals])
#             max = np.mean(sort_data[-n_vals:])
#         else:
#             raise ValueError(f"Input n_vals must be positive and less than data length. {n_vals} is not allowed")
#
#         g_d = (max + min) / 2
#         g_nu = -((max - min) / 2)
#         return g_d, g_nu
#
#     def curve_analytical_func(self, fit_params, scan_data, round, plot = True, save = False):
#         g
#### Get the data ####
run = 'run33e'
study = 'Cs_TimeStudy_Tomography' #SC_Tomography' #DDoff_SC_HoleOpen' #'DDon_SC_HoleClosed' #'DDon_SC_HoleOpen' #'Longtime_Study
substudy = 'SingleQ4_Tomography' #'SingleQ4_Tomography' #'AllQ_Tomography' #AllQ_Tomography'

qubit = [3]
num_of_qubits = 4

date = '2025-12-08' #'2025-12-14' #'2025-12-08' #'2025-12-04'
timestamp = ('2025-12-08_14-48-43')  #('2025-12-14_20-56-26') #('2025-12-08_14-48-43') #('2025-12-04_13-36-05')
studyFolder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/study_data'
if not os.path.exists(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots/fit_plots'):
    os.makedirs(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots/fit_plots')
plotFolder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots/fit_plots'

## if h5 file
file = f'Tomography_Q{qubit[0]+1}_{timestamp}.h5'

data_acq = GetData(studyFolder, num_of_qubits)

# Get metadata - pick one
metadata = data_acq.getNPZmetadata()
#metadata = data_acq.getH5metadata(file)
#
# # Get data - pick one
#data = data_acq.getNPZdata_fromallQ(qubit, rounds = (10, 20))
data = data_acq.getNPZdata_fromsingleQ(qubit, rounds = (0, 100))
#data = data_acq.getH5data(file, qubit, rounds = (10, 20))

###### Fit 3 pt data

class Fit_3pt:
    def __init__(self, vsweep, data, plotFolder, data_type):
        self.plotFolder = plotFolder
        self.vsweep = vsweep # array in mV, length is 1 scan length (3 or 30)
        self.qubit = qubit
        self.data_type = data_type
        if data_type == 'I':
            self.data = data[qubit]["xi"][:] #shape is (rounds, v_points)
        elif data_type == 'Q':
            self.data = data[qubit]["xq"][:] #shape is (rounds, v_points)
        elif data_type == 'Amp':
            xi = data[qubit]["xi"][:]
            xq = data[qubit]["xq"][:]
            self.data = np.sqrt(xi**2 + xq**2)
        else:
            raise ValueError(f"Invalid data type: {data_type}. Must be 'I', 'Q', or 'Amp'")

        self.rounds = data["_rounds"]
        self.timestamps = data["_timestamps"]

    def model(self, n_g, d, nu, mv2e, phi):
        P = d + nu*np.cos(np.pi*np.cos(2*np.pi*mv2e*n_g - phi))
        return P

    def get_d_nu_fromchunk(self, start, stop, nvals):
        scans_together = self.data[start:stop].reshape(-1)

        if nvals == 1:
            min = np.min(scans_together)
            max = np.max(scans_together)
        elif nvals > 1:
            sort_data = np.sort(scans_together)
            min = np.mean(sort_data[:nvals])
            max = np.mean(sort_data[-nvals:])
        else:
            raise ValueError(f"nvals input {nvals} is not allowed")

        g_d = (max + min) / 2
        g_nu = (max - min) / 2
        return g_d, g_nu

    def fit_phi_mv2e(self, d, nu, g_mv2e, g_phi):
        # per 3 pt scan

        def cost(x):
            mv2e, phi = x
            data_pred = model(self.vsweep, d, nu, mv2e, phi)
            return np.sum((data - data_pred) ** 2)

        x0 = [g_mv2e, g_phi]

        bounds = [(0.005, 0.0075),
                  (0, 2 * np.pi)]
        res = minimize(cost, x0=x0, bounds=bounds, method='L-BFGS-B')

        mv2e_fit, phi_fit = res.x
        return mv2e_fit, phi_fit, res.fun, res.success




def get_d_nu(data, nvals):
    if nvals == 1:
        min = np.min(data)
        max = np.max(data)
    elif nvals > 1:
        sort_data = np.sort(data)
        min = np.mean(sort_data[:nvals])
        max = np.mean(sort_data[-nvals:])
    else:
        raise ValueError(f"nvals input {nvals} is not allowed")

    g_d = (max + min) / 2
    g_nu = (max - min) / 2
    return g_d, g_nu

def PutScansTogether(data, start, stop):
    scans_together = data[start:stop].reshape(-1)
    return scans_together

def fit_phi_mv2e(vsweep, data, d, nu, g_mv2e, g_phi):
    #per 3 pt scan

    def cost(x):
        mv2e, phi = x
        data_pred = model(vsweep, d, nu, mv2e, phi)
        return np.sum((data - data_pred)**2)

    x0 = [g_mv2e, g_phi]

    bounds = [(0.005, 0.0075),
              (0, 2*np.pi)]
    res = minimize(cost, x0 = x0, bounds = bounds, method = 'L-BFGS-B')

    mv2e_fit, phi_fit = res.x
    return mv2e_fit, phi_fit, res.fun, res.success

def plot_fit(vsweep, data, round, d, nu, mv2e, phi, plotFolder):
    vsweep_morepts = np.linspace(vsweep.min(), vsweep.max(), 400)
    P_morepts = model(vsweep_morepts, d, nu, mv2e, phi)

    plt.plot(vsweep_morepts, P_morepts, '-', label = 'Fit')
    plt.plot(vsweep, data, '.', label = 'Data')

    plt.xlabel("Votlage (mV)")
    plt.ylabel("I Amplitude")
    plt.legend()
    plt.title(f'Q4 3pt Tomography R{round} Fit \n d = {d:.2f}, nu = {nu:.2f}, mv2e = {mv2e:.5f}, phi = {phi:.2f}')
    plt.tight_layout()

    save_name = f'Q4_3pt_Fit_R{round},png'
    fig_path = os.path.join(plotFolder, save_name)
    plt.savefig(fig_path)
    plt.close()
    return

#print(data[qubit[0]]['xi'])
data_for_fit = data[qubit[0]]["xi"][:] #shape is (rounds, v_points)
vsweep = metadata["vsweep_mV"]

rounds = data_for_fit.shape[0]

# Put together x scans and get d and nu
scan_chunk = PutScansTogether(data_for_fit, 0, 50)
g_d, g_nu = get_d_nu(scan_chunk, 5)

# Do fitting
for round in range(0, 10):
    f_mv2e, f_phi, fit_output, success = fit_phi_mv2e(vsweep, data_for_fit[round,], g_d, g_nu, 0.0065, np.pi)
    print(f_mv2e, f_phi, success)
    plot_fit(vsweep, data_for_fit[round,], round, g_d, g_nu, f_mv2e, f_phi, plotFolder)

