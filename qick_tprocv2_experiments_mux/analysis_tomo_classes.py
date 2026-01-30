import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import h5py
import json
import glob
import os
import re
from datetime import datetime


################ NOTES ON THE CODE ##########################
# mv2e value is currently hard coded from by-eye looks at the 30 pt data fits. Need to look into a better value and how much it affects the analysis


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
        base = os.path.basename(meta_path)

        #get timestamp from file name
        date, time = base.rsplit("_", 2)[1:]
        time = time.replace(".npz", "").replace("-", ":")
        file_timestamp = datetime.fromisoformat(f"{date} {time}")

        meta = np.load(meta_path, allow_pickle=True)
        metadata = {k: meta[k].item() if meta[k].shape == () else meta[k] for k in meta}
        metadata["vsweep_mV"] = metadata["vsweep"] * 1000
        metadata["file_timestamp"] =  file_timestamp # datetime.datetime, '%Y-%m-%d_%H-%M-%S'
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

def plot_2dtomo(vsweep, data, qubit, start, stop, v1 = None, v2 = None, saveFolder = None):
    xi = np.array(data[qubit]["xi"][:])
    xq = np.array(data[qubit]["xq"][:])
    times = data["_timestamps"][:]

    t0 = times[0]
    elapsed_min = np.array([(dt - t0).total_seconds() / 60.0 for dt in times])
    tot_min = elapsed_min[-1] - elapsed_min[0]

    fig, (axI, axQ) = plt.subplots(2, 1, figsize = (9,7), sharex = True)

    imI = axI.imshow(xi.T, aspect = 'auto', origin = 'lower',
                     extent = [times[0], times[-1], vsweep[0], vsweep[-1]])

    axI.set_ylabel('Voltage Bias (mV)')
    plt.colorbar(imI, ax=axI, label='I amp')

    imQ = axQ.imshow(xq.T, aspect = 'auto', origin = 'lower',
                     extent = [times[0], times[-1], vsweep[0], vsweep[-1]])
    axQ.set_ylabel('Voltage Bias (mV)')
    plt.colorbar(imQ, ax=axQ, label='Q amp')

    if v1 is not None:
        axI.axvline(v1, color = 'r', linestyle = '--', alpha = 0.5)
        axQ.axvline(v1, color = 'r', linestyle = '--', alpha = 0.5)
    if v2 is not None:
        axI.axvline(v2, color='r', linestyle='--', alpha=0.5)
        axQ.axvline(v2, color='r', linestyle='--', alpha=0.5)

    title = f"Qubit {qubit + 1} I/Q Tomography {start}-{stop}, {tot_min} min"
    fig.suptitle(title)

    axI.xaxis_date()
    axQ.xaxis_date()
    fmt = mdates.DateFormatter("%m-%d\n%H:%M")
    axI.xaxis.set_major_formatter(fmt)
    axQ.xaxis.set_major_formatter(fmt)

    plt.tight_layout()

    if saveFolder is not None:
        save_name = f'Q{qubit + 1}_Tomography_{start}-{stop}_newplot_{times[0].strftime("%Y-%m-%d_%H-%M-%S")}.png'
        fig_path = os.path.join(saveFolder, save_name)
        plt.savefig(fig_path)
        plt.close()
    else:
        plt.show()
    return


class Fit_3pt:
    def __init__(self, metadata, data, qubit, d_group, a_group, plotFolder, data_type):
        self.plotFolder = plotFolder
        self.vsweep = metadata["vsweep_mV"] # array in mV, length is 1 scan length (3 or 30)
        self.dataset_timestamp = metadata["file_timestamp"] # datetime.datetime of dataset file/folder
        self.qubit = qubit
        self.data_type = data_type
        if data_type == 'I':
            data_full = data[qubit]["xi"][:]  # shape is (rounds, v_points)
        elif data_type == 'Q':
            data_full = data[qubit]["xq"][:]  # shape is (rounds, v_points)
        elif data_type == 'Amp':
            xi = data[qubit]["xi"][:]
            xq = data[qubit]["xq"][:]
            data_full = np.sqrt(xi ** 2 + xq ** 2)
        else:
            raise ValueError(f"Invalid data type: {data_type}. Must be 'I', 'Q', or 'Amp'")

        self.d_group_data = data_full[d_group[0]:d_group[1]]
        self.a_group_data = data_full[a_group[0]:a_group[1]]

        rounds_full = data["_rounds"]
        timestamps_full = data["_timestamps"]
        self.d_group_rounds = rounds_full[d_group[0]:d_group[1]]
        self.a_group_rounds = rounds_full[a_group[0]:a_group[1]]
        self.a_group_timestamps = timestamps_full[a_group[0]:a_group[1]]

    def model(self, n_g, d, nu, mv2e, phi):
        P = d + nu * np.cos(np.pi * np.cos(2 * np.pi * mv2e * n_g - phi))
        return P

    def get_d_nu_fromchunk(self, nvals):
        scans_together = self.a_group_data.reshape(-1)

        if nvals == 1:
            min = np.min(scans_together)
            max = np.max(scans_together)
        elif nvals > 1:
            sort_data = np.sort(scans_together)
            min = np.median(sort_data[:nvals])
            max = np.median(sort_data[-nvals:])
        else:
            raise ValueError(f"nvals input {nvals} is not allowed")

        g_d = (max + min) / 2
        g_nu = (max - min) / 2
        return g_d, g_nu

    def fit_phi_mv2e(self, r_index, d, nu, g_mv2e, g_phi):
        # per 3 pt scan

        def cost(x):
            mv2e, phi = x
            data_pred = self.model(self.vsweep, d, nu, mv2e, phi)
            return np.sum((self.a_group_data[r_index,] - data_pred) ** 2)

        x0 = [g_mv2e, g_phi]

        bounds = [(0.005, 0.0075),
                  (-2 * np.pi, 2 * np.pi)]
        res = minimize(cost, x0=x0, bounds=bounds, method='L-BFGS-B')

        mv2e_fit, phi_fit = res.x
        return mv2e_fit, phi_fit, res.fun, res.success

    def fit_phi(self, r_index, d, nu, mv2e):
        # per 3 pt scan

        def cost(phi):
            data_pred = self.model(self.vsweep, d, nu, mv2e, phi)
            return np.sum((self.a_group_data[r_index,] - data_pred) ** 2)

        bounds = (0, 2 * np.pi)
        res = minimize_scalar(cost, bounds=bounds, method='bounded')

        phi_fit = res.x
        return phi_fit, res.fun, res.success

    def plot_fit(self, r_index, r_num, d, nu, mv2e, phi, single_fit = True, save_name_add = None):
        vsweep_morepts = np.linspace(self.vsweep.min(), self.vsweep.max(), 400)
        P_morepts = self.model(vsweep_morepts, d, nu, mv2e, phi)


        plt.plot(vsweep_morepts, P_morepts, '-', label='Fit')
        plt.plot(self.vsweep, self.a_group_data[r_index,], '.', label='Data')

        plt.xlabel("Votlage (mV)")
        plt.ylabel(f"{self.data_type} Amplitude")
        plt.legend()
        if single_fit:
            plt.title(f'Q4 3pt Tomography R{r_num} Fit {self.dataset_timestamp} (R{self.d_group_rounds[0]}-R{self.d_group_rounds[-1]}) \n d = {d:.2f}, nu = {nu:.2f}, mv2e = {mv2e:.5f}, phi fit = {phi:.2f}')
        else:
            plt.title(f'Q4 3pt Tomography R{r_num} Fit {self.dataset_timestamp} (R{self.d_group_rounds[0]}-R{self.d_group_rounds[-1]}) \n d = {d:.2f}, nu = {nu:.2f}, mv2e fit = {mv2e:.5f}, phi fit = {phi:.2f}')
        plt.tight_layout()

        if single_fit:
            save_name = f'Q4_3pt_PhiFit_R{r_num}_Rstart{self.d_group_rounds[0]}-Rstop{self.d_group_rounds[-1]}'
        else:
            save_name = f'Q4_3pt_2Fit_R{r_num}_Rstart{self.d_group_rounds[0]}-Rstop{self.d_group_rounds[-1]}'
        if save_name_add is not None:
            save_name += f'_{save_name_add}'

        fig_path = os.path.join(self.plotFolder, save_name)
        plt.savefig(fig_path)
        plt.close()
        return

    def run_3pt_2fit(self, nvals, plot = False, save_name_add = None):
        # Put together x scans and get d and nu
        g_d, g_nu = self.get_d_nu_fromchunk(nvals)

        # Do fitting
        allf_mv2e = []
        allf_phi = []
        allfit_output = []
        allsuccess = []


        for r_index, r_num in enumerate(self.a_group_rounds): # Need round index in a_group for fit and actually round # for plot
            f_mv2e, f_phi, fit_output, success = self.fit_phi_mv2e(r_index, g_d, g_nu, 0.0065, np.pi)
            #print(f_mv2e, f_phi, success)
            if plot:
                self.plot_fit(r_index, r_num, g_d, g_nu, f_mv2e, f_phi, single_fit=False, save_name_add = save_name_add)
            allf_mv2e.append(f_mv2e)
            allf_phi.append(f_phi)
            allfit_output.append(fit_output)
            allsuccess.append(success)
        return allf_mv2e, allf_phi, allfit_output, allsuccess

    def run_3pt_1fit(self,nvals, mv2e, plot = False, save_name_add = None):
        # Put together x scans and get d and nu
        g_d, g_nu = self.get_d_nu_fromchunk(nvals)
        g_mv2e = mv2e

        # Do fitting
        allf_phi = []
        allfit_output = []
        allsuccess = []
        for r_index, r_num in enumerate(self.a_group_rounds): # Need round index in a_group for fit and actually round # for plot
            f_phi, fit_output, success = self.fit_phi(r_index, g_d, g_nu, g_mv2e)
            #print(f_phi, success)
            if plot:
                self.plot_fit(r_index, r_num, g_d, g_nu, g_mv2e, f_phi, single_fit = True, save_name_add = save_name_add)
            allf_phi.append(f_phi)
            allfit_output.append(fit_output)
            allsuccess.append(success)
        return allf_phi, allfit_output, allsuccess

    def get_single_outliers(self, allf_phi, same_threshold):
        # check each scan (i) to see if i-1 and i+1 have the same phi (within a threshold) and then throw it out if bad fit
        single_outlier = np.full_like(self.a_group_rounds, -10) #-10 is the placeholder value
        for r_index, r_num in enumerate(self.a_group_rounds):
            if r_num == self.a_group_rounds[0] or r_num == self.a_group_rounds[-1]:
                ## skip first and last rounds of data since you can't look before and after
                single_outlier[r_index] = 0 #0 value indicates 'good' scan (these not checked)
                continue
            i_before = allf_phi[r_index-1]
            i = allf_phi[r_index]
            i_after = allf_phi[r_index+1]

            if abs(i-i_before) > same_threshold and abs(i_before - i_after) <= same_threshold:
                single_outlier[r_index] = 1 #1 value indicates bad/outlier scan, to be skipped during jump checking
            else:
                single_outlier[r_index] = 0 #good scan
        return single_outlier

    def get_double_outliers(self, allf_phi, same_threshold):
        # check every 2 scans (i, i+1) to see if i-1 and i+1 have the same phi (within a threshold) and then throw them out if both bad fit
        double_outlier = np.full_like(self.a_group_rounds, -10) #-10 is the placeholder value
        for r_index, r_num in enumerate(self.a_group_rounds):
            if r_num == self.a_group_rounds[0] or r_num == self.a_group_rounds[-2] or r_num == self.a_group_rounds[-1]:
                ## skip first, second to last, and last rounds of data since you can't look before and after the 2
                double_outlier[r_index] = 0 #0 value indicates 'good' scan (these not checked)
                continue
            if double_outlier[r_index] == 1:
                # means it was already found in previous loop and doesn't need to be rechecked.
                continue
            i_before = allf_phi[r_index-1]
            i1 = allf_phi[r_index]
            i2 = allf_phi[r_index +1]
            i_after = allf_phi[r_index +2]

            if abs(i1 - i_before) > same_threshold and abs(i_before - i_after) <= same_threshold:
                double_outlier[r_index] = 1 #1 value indicates bad/outlier scan, to be skipped during jump checking
                double_outlier[r_index + 1] = 1
            else:
                double_outlier[r_index] = 0
        return double_outlier

    def mask_outliers(self, single_outlier = None, double_outlier = None):
        if single_outlier is not None and double_outlier is None:
            outlier_mask = single_outlier
        elif double_outlier is not None and single_outlier is None:
            outlier_mask = double_outlier
        elif single_outlier is not None and double_outlier is not None:
            outlier_mask = np.full_like(single_outlier, -20) #-20 is initialize value, to be overwritten
            for i in range(0, len(outlier_mask)):
                if single_outlier[i] == 1 or double_outlier[i] == 1:
                    outlier_mask[i] = 1 #if scan is bad from either check it's bad
                elif single_outlier[i] == 0 and double_outlier[i] == 0:
                    outlier_mask[i] = 0 #if scan is good in both checks it's good
                elif single_outlier[i] == -10 or double_outlier[i] == -10:
                    print(f'Issue with index {i} in outlier check (R{self.a_group_rounds[i]})')
                    outlier_mask[i] = -10
        else:
            print('Neither single_outlier and double_outlier lists given')
            return
        return outlier_mask

    def find_jumps(self, allf_phi, jump_threshold, mask_outliers = None):
        phi = np.asarray(allf_phi)
        if mask_outliers is not None:
            phi[]

    def plot_mv2e_phi_chi(self, allf_mv2e, allf_phi, allfit_output, plot_time = True, mask_outliers = None, save = False, save_name_add=None):
        if mask_outliers is not None:
            if len(mask_outliers) != len(allf_phi):
                raise ValueError('Mask for outliers is incorrect length, cannot plot')
            # mask data array, mv2e, phi, and fit_output for plotting using outlier mask list
            condition_array = np.array(mask_outliers)
            boolean_mask = (condition_array == 0) #only allows values which have 0 (are good scans from the checks)
            plot_mv2e = allf_mv2e[boolean_mask]
            plot_phi = allf_phi[boolean_mask]
            plot_fit_output = allfit_output[boolean_mask]
        else:
            plot_mv2e = allf_mv2e
            plot_phi = allf_phi
            plot_fit_output = allfit_output

        if plot_time:
            x_arr = self.a_group_timestamps
            t_num = mdates.date2num(self.a_group_timestamps)
            dt = np.median(np.diff(t_num))
            x_label = "Time"
            extent = [t_num[0] - dt/2, t_num[-1] + dt/2, self.vsweep[0], self.vsweep[-1]]
        else:
            x_arr = self.a_group_rounds
            x_label = "Scan #"
            extent = [self.a_group_rounds[0], self.a_group_rounds[-1], self.vsweep[0], self.vsweep[-1]]

        fig, axs = plt.subplots(4, 1, sharex=True, constrained_layout=True, figsize = (10, 8)) #, gridspec_kw={"height_ratios": [2, 1, 1]})
        im = axs[0].imshow(self.a_group_data.T, aspect = 'auto', origin='lower',
                           extent = extent)
        axs[0].set_ylabel("Voltage (mV)")
        cbar = fig.colorbar(im, ax=axs[0], pad=0.01)
        cbar.set_label(f"{self.data_type} Amplitude")

        axs[1].plot(x_arr, plot_mv2e)
        axs[1].set_ylabel("mV to e conv.")

        axs[2].plot(x_arr, plot_phi)
        axs[2].set_ylabel("phi (rad)")

        axs[3].plot(x_arr, plot_fit_output)
        axs[3].set_ylabel("Fake chi2")
        axs[3].set_xlabel(x_label)

        if plot_time:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)

            for ax in axs:
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)

            fig.autofmt_xdate()

        if mask_outliers:
            fig.suptitle(f'Fit Results, mv2e and phi fit, outliers masked, R{self.a_group_rounds[0]}-R{self.a_group_rounds[-1]} (R{self.d_group_rounds[0]}-R{self.d_group_rounds[-1]}) \n {self.dataset_timestamp}')
        else:
            fig.suptitle(f'Fit Results, mv2e and phi fit, outliers masked, R{self.a_group_rounds[0]}-R{self.a_group_rounds[-1]} (R{self.d_group_rounds[0]}-R{self.d_group_rounds[-1]}) \n {self.dataset_timestamp}')
        #plt.tight_layout()

        if save:
            if plot_time:
                full_plot_name = (f'Q{qubit[0] + 1}_3pt_2Fit_time')
            else:
                full_plot_name = (f'Q{qubit[0] + 1}_3pt_2Fit_round')
            if mask_outliers:
                full_plot_name += '_masked'
            full_plot_name += f'_{self.dataset_timestamp}_Rplot{self.a_group_rounds[0]}-{self.a_group_rounds[-1]}_Rgroup{self.d_group_rounds[0]}-{self.d_group_rounds[-1]}'
            if save_name_add is not None:
                full_plot_name += f'_{save_name_add}'
            save = os.path.join(self.plotFolder, full_plot_name)
            plt.savefig(save)
        else:
            plt.show()
        return

    def plot_phi_chi(self, allf_phi, allfit_output, plot_time = True, mask_outliers = None, save = False, save_name_add = None):
        allf_phi = np.asarray(allf_phi)
        allfit_output = np.asarray(allfit_output)

        if mask_outliers is not None:
            if len(mask_outliers) != len(allf_phi):
                raise ValueError('Mask for outliers is incorrect length, cannot plot')
            # count how many scans are masked
            # mask data array, mv2e, phi, and fit_output for plotting using outlier mask list
            condition_array = np.asarray(mask_outliers)
            boolean_mask = (condition_array == 0)

            plot_phi = allf_phi.copy()
            plot_fit_output = allfit_output.copy()

            plot_phi[~boolean_mask] = np.nan
            plot_fit_output[~boolean_mask] = np.nan
        else:
            plot_phi = allf_phi
            plot_fit_output = allfit_output

        if plot_time:
            x_arr = self.a_group_timestamps
            t_num = mdates.date2num(self.a_group_timestamps)
            dt = np.median(np.diff(t_num))
            x_label = "Time"
            extent = [t_num[0] - dt/2, t_num[-1] + dt/2, self.vsweep[0], self.vsweep[-1]]
        else:
            x_arr = self.a_group_rounds
            x_label = "Scan #"
            extent = [self.a_group_rounds[0], self.a_group_rounds[-1], self.vsweep[0], self.vsweep[-1]]

        fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize = (10, 8)) #, gridspec_kw={"height_ratios": [2, 1, 1]})
        im = axs[0].imshow(self.a_group_data.T, aspect = 'auto', origin='lower',
                           extent = extent)
        if mask_outliers is not None:
            mask = np.asarray(mask_outliers)
            bad = mask == 1
            if plot_time:
                t_num = mdates.date2num(self.a_group_timestamps)
                dt = np.median(np.diff(t_num))

                for t in t_num[bad]:
                    axs[0].axvspan(t - dt/2, t + dt/2, color = 'red', alpha = 0.25, lw = 0)
            else:
                for r in np.asarray(self.a_group_rounds)[bad]:
                    axs[0].axvspan(r - 0.5, r + 0.5, color = 'red', alpha = 0.25, lw = 0)
            # count how many scans are masked and put it on plot
            n_masked = int(np.sum(mask == 1))
            n_total = len(mask)
            fig.text(0.99, 0.98, f"Masked scans: {n_masked} / {n_total}",
                     ha='right', va='top',
                     fontsize = 10, color = 'red')
        axs[0].set_ylabel("Voltage (mV)")
        cbar = fig.colorbar(im, ax=axs[0], pad=0.01)
        cbar.set_label(f"{self.data_type} Amplitude")

        axs[1].plot(x_arr, plot_phi)
        axs[1].set_ylabel("phi (rad)")

        axs[2].plot(x_arr, plot_fit_output)
        axs[2].set_ylabel("Fake chi2")
        axs[2].set_xlabel(x_label)

        if plot_time:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)

            for ax in axs:
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)

            fig.autofmt_xdate()

        if mask_outliers is not None:
            fig.suptitle(f'Fit Results, phi fit, outliers masked, R{self.a_group_rounds[0]}-R{self.a_group_rounds[-1]} (R{self.d_group_rounds[0]}-R{self.d_group_rounds[-1]}) \n {self.dataset_timestamp}')
        else:
            fig.suptitle(f'Fit Results, phi fit, outliers masked, R{self.a_group_rounds[0]}-R{self.a_group_rounds[-1]} (R{self.d_group_rounds[0]}-R{self.d_group_rounds[-1]}) \n {self.dataset_timestamp}')
        #plt.tight_layout()

        if save:
            if plot_time:
                full_plot_name = (f'Q{qubit[0] + 1}_3pt_PhiFit_time')
            else:
                full_plot_name = (f'Q{qubit[0] + 1}_3pt_PhiFit_round')
            if mask_outliers is not None:
                full_plot_name += '_masked'
            full_plot_name += f'_{self.dataset_timestamp}_Rplot{self.a_group_rounds[0]}-{self.a_group_rounds[-1]}_Rgroup{self.d_group_rounds[0]}-{self.d_group_rounds[-1]}'
            if save_name_add is not None:
                full_plot_name += f'_{save_name_add}'
            save = os.path.join(self.plotFolder, full_plot_name)
            plt.savefig(save)
        else:
            plt.show()
        return

#### Get the data ####
run = 'run33e'
study = 'Cs_TimeStudy_Tomography' #SC_Tomography' #DDoff_SC_HoleOpen' #'DDon_SC_HoleClosed' #'DDon_SC_HoleOpen' #'Longtime_Study
substudy = 'SingleQ4_Tomography' #'SingleQ4_Tomography' #'AllQ_Tomography' #AllQ_Tomography'

qubit = [3]
num_of_qubits = 4

date = '2025-12-05' #'2025-12-14' #'2025-12-08' #'2025-12-04'
timestamp = ('2025-12-05_15-34-40')  #('2025-12-14_20-56-26') #('2025-12-08_14-48-43') #('2025-12-04_13-36-05')
studyFolder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/study_data'

## if h5 file
file = f'Tomography_Q{qubit[0]+1}_{timestamp}.h5'

data_acq = GetData(studyFolder, num_of_qubits)

start = 0
stop = 9000 #rounds to get from dataset

## Get metadata - pick one
metadata = data_acq.getNPZmetadata()
#metadata = data_acq.getH5metadata(file)
#print(metadata['tot_rounds'])

## Get data - pick one
#data = data_acq.getNPZdata_fromallQ(qubit, rounds = (10, 20))
data = data_acq.getNPZdata_fromsingleQ(qubit, rounds = (start, stop)) #2000))
#data = data_acq.getH5data(file, qubit, rounds = (start, stop))


# #### Plotting #####
# if not os.path.exists(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots'):
#     os.makedirs(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots')
# plotFolder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots'
#
# vsweep = metadata['vsweep_mV']
#
# v1 = datetime(2025, 12, 5, 16, 6, 59)
# v2 = datetime(2025, 12, 5, 16, 7, 5)
#
# plot_2dtomo(vsweep, data, qubit[0], start, stop, saveFolder = plotFolder)

#Note: Takes 15 sec to get and plot 5000 npz saved scans


#### Fitting - 3pt data ####

dataset_timestamp = metadata['file_timestamp']
data_type = 'I'
data_test = data
data_group = [367, 3000] #group of data to put together to get d and nu
analysis_group = [367, 3000] #what data to fit and possibly plot
nvals = 10
mv2e = 0.0065
time_x = True
#
if not os.path.exists(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots/{data_type}_fit_plots/R{data["_rounds"][data_group[0]]}-{data["_rounds"][data_group[1]]}_fits'):
    os.makedirs(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots/{data_type}_fit_plots/R{data["_rounds"][data_group[0]]}-{data["_rounds"][data_group[1]]}_fits')
plotFolder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots/{data_type}_fit_plots/R{data["_rounds"][data_group[0]]}-{data["_rounds"][data_group[1]]}_fits'

fit_3pt = Fit_3pt(metadata, data_test, qubit[0], data_group, analysis_group, plotFolder, data_type)

#### mv2e and phi fit ####
# all_mv2e, all_phi, all_func, all_success = fit_3pt.run_3pt_2fit(data_group, analysis_group, nvals, plot = False, save_name_add = None)
# fit_3pt.plot_mv2e_phi_chi(d_group, a_group, all_mv2e, all_phi, all_func, plot_time = time_x, save = False, save_name_add = None)

#### phi fit ####
all_phi, all_func, all_success = fit_3pt.run_3pt_1fit(nvals, mv2e, plot = False, save_name_add = None)

### Outliers ###
same_threshold = 0.5
single_out = fit_3pt.get_single_outliers(all_phi, same_threshold)
double_out = fit_3pt.get_double_outliers(all_phi, same_threshold)
outliers = fit_3pt.mask_outliers(single_outlier = single_out, double_outlier = double_out)
#plt.plot(single_out)
#plt.show()


fit_3pt.plot_phi_chi(all_phi, all_func, plot_time = time_x, mask_outliers = outliers, save = True, save_name_add = 'both_out_red')


# add flag to throw out single/2 scan jumps that go back to the same value (bad fits) -- done, but bad scans still plotted in color plot. To do: mask them there too and add note to name and savename if masked
# add data type to save name so things don't get overwritten - sort of done, made it so each data type gets a different fit folder
# move the plotFolder making to the class
# add round timestamp to indiv round plots title and save name -- think more about which timestamp to use