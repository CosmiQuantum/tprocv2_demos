import numpy as np
from section_008_save_data_to_h5 import Data_H5
#import sys
import os
#sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import glob
import datetime

class PlotRRData:
    def __init__(self, outerFolder, outerFolder_saveplots):
        #self.date = date
        #self.save_figs = save_figs
        #self.fit_saved = fit_saved
        #self.signal = signal
        # self.run_name = run_name
        # self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        self.outerFolder_saveplots = outerFolder_saveplots

    def process_h5_data(self, data):
        # Check if the data is a byte string; decode if necessary.
        if isinstance(data, bytes):
            data_str = data.decode()
        elif isinstance(data, str):
            data_str = data
        else:
            raise ValueError("Unsupported data type. Data should be bytes or string.")

        # Remove extra whitespace and non-numeric characters.
        cleaned_data = ''.join(c for c in data_str if c.isdigit() or c in ['-', '.', ' ', 'e'])

        # Split into individual numbers, removing empty strings.
        numbers = [float(x) for x in cleaned_data.split() if x]
        return numbers

    def load_qspec(self):
        outerFolder_expt = self.outerFolder + "/Data_h5/qspec_ge"
        all_scans = []
        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
        for h5_file in h5_files:
            save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
            H5_class_instance = Data_H5(h5_file)
            load_data = H5_class_instance.load_from_h5(data_type='QSpec_ge', save_r = int(save_round))
            populated_keys = []
            for q_key in load_data['QSpec_ge']:
                # print(f'found q_keys, {q_key}')
                # Get dates for current q_key
                dates_list = load_data['QSpec_ge'][q_key].get('Dates', [[]])

                # Check it's not NaN
                if any(
                        not np.isnan(date)
                        for date in dates_list[0]  # Iterate over the first batch of dates
                ):
                    populated_keys.append(q_key)

            for q_key in populated_keys:
                for dataset in range(len(load_data['QSpec_ge'][q_key].get('Dates', [])[0])):
                    dates = datetime.datetime.fromtimestamp(load_data['QSpec_ge'][q_key].get('Dates', [])[0][dataset])
                    #print(date)
                    I = self.process_h5_data(load_data['QSpec_ge'][q_key].get('I', [])[0][dataset].decode())
                    Q = self.process_h5_data(load_data['QSpec_ge'][q_key].get('Q', [])[0][dataset].decode())
                    # I_fit = load_data['QSpec'][q_key].get('I Fit', [])[0][dataset]
                    # Q_fit = load_data['QSpec'][q_key].get('Q Fit', [])[0][dataset]
                    freqs = self.process_h5_data(load_data['QSpec_ge'][q_key].get('Frequencies', [])[0][dataset].decode())
                    round_num = load_data['QSpec_ge'][q_key].get('Round Num', [])[0][dataset]
                    batch_num = load_data['QSpec_ge'][q_key].get('Batch Num', [])[0][dataset]

                    exp_config = load_data['QSpec_ge'][q_key].get('Exp Config', [])[0][dataset].decode()
                    safe_globals = {"np": np, "array": np.array, "__builtins__": {}}

                    exp_config = eval(exp_config, safe_globals)

                    if len(I)>0:

                        all_scans.append({
                            "q_key": int(q_key),
                            "date": dates,
                            "round": round_num,
                            "batch": batch_num,
                            "freqs": np.array(freqs, dtype=float),
                            "I": np.array(I, dtype=float),
                            "Q": np.array(Q, dtype=float),
                            "exp_config": exp_config,
                            "filename": os.path.basename(h5_file),
                            "dataset_index": dataset,
                        })
            # all_scans_sorted = sorted(all_scans, key=lambda x: x['timestamp'])
            #
            # freq = all_scans_sorted[0]['freqs']
            #
            # I_all = np.array([scan["I"] for scan in all_scans_sorted])
            # Q_all = np.array([scan["Q"] for scan in all_scans_sorted])
            del H5_class_instance

        return all_scans

    def plot_IQ_long(self, all_scans, qubit, f_min = None, f_max = None, backsub = False):
        scans = [s for s in all_scans if s["q_key"] == qubit]
        found_freqs = [4914.05, 4764.5, 4577, 4782]

        if len(scans) == 0:
            print(f"No scans found for qubit {qubit}")
            return

        freqs = scans[0]['freqs'] #1d freq axis

        if f_min is not None:
            freq_mask = freqs >= f_min
        else:
            freq_mask = np.ones_like(freqs, dtype=bool)
        if f_max is not None:
            freq_mask &= freqs <= f_max
        freqs = freqs[freq_mask]

        I_all = np.array([s['I'][freq_mask] for s in scans]) #(n_scans, n_freq)
        Q_all = np.array([s['Q'][freq_mask] for s in scans])
        dates = np.array([s['date'] for s in scans])

        # if normalize:
        #     I_min, I_max = I_all.min(), I_all.max()
        #     Q_min, Q_max = Q_all.min(), Q_all.max()
        #     I_all = (I_all - I_min) / (I_max - I_min)
        #     Q_all = (Q_all - Q_min) / (Q_max - Q_min)
        if backsub:
            Is = np.array(I_all)
            I_val = Is.astype(float)
            I_bkgd = np.zeros_like(I_val)
            for i in range(0, len(I_val)):
                I_bkgd[i] = np.mean(I_val[i])
                I_val[i] = I_val[i] - I_bkgd[i]
            Qs = np.array(Q_all)
            Q_val = Qs.astype(float)
            Q_bkgd = np.zeros_like(Q_val)
            for i in range(0, len(Q_val)):
                Q_bkgd[i] = np.mean(Q_val[i])
                Q_val[i] = Q_val[i] - Q_bkgd[i]
        if backsub:
            I_plot = I_val
            Q_plot = Q_val
        else:
            I_plot = I_all
            Q_plot = Q_all

        times_numeric = mdates.date2num(dates)

        fig, (axI, axQ) = plt.subplots(2, 1, figsize = (9,7), sharex=True)

        imI = axI.imshow(I_plot.T, aspect = 'auto', origin = 'lower', extent = [times_numeric[0], times_numeric[-1], freqs[0], freqs[-1]])
        axI.axhline(found_freqs[qubit], color = 'red', linestyle = '--', alpha = 0.6)
        axI.set_ylabel('Freq (MHz)')
        plt.colorbar(imI, ax=axI)

        imQ = axQ.imshow(Q_plot.T, aspect = 'auto', origin = 'lower', extent = [times_numeric[0], times_numeric[-1], freqs[0], freqs[-1]])
        axQ.axhline(found_freqs[qubit], color='red', linestyle='--', alpha=0.6)
        axQ.set_ylabel('Freq (MHz)')
        axQ.set_xlabel('Time')
        plt.colorbar(imQ, ax=axQ)

        if backsub:
            title = f"Qubit {qubit + 1} I/Q Spectroscopy Over Time, Bkgd Sub"
        else:
            title = f"Qubit {qubit + 1} I/Q Spectroscopy Over Time"
        fig.suptitle(title)

        axI.xaxis_date()
        axQ.xaxis_date()
        fmt = mdates.DateFormatter("%m-%d\n%H:%M")
        axI.xaxis.set_major_formatter(fmt)
        axQ.xaxis.set_major_formatter(fmt)

        plt.tight_layout()
        if backsub:
            save_name = f'Q{qubit + 1}_Spec_bkgdsub_{dates[0]}'
        else:
            save_name = f'Q{qubit + 1}_Spec_{dates[0]}'
        if f_min is not None:
            save_name += '_zoomed'
        fig_path = os.path.join(self.outerFolder_saveplots, f'{save_name}.png')
        plt.savefig(fig_path)
        #plt.show()
        return

run = 'run33e'
study = 'DDoff_SC_HoleClosed' #'Initial Checkout' #DDoff_SC_HoleOpen' #'DDon_SC_HoleClosed' #'DDon_SC_HoleOpen' #'Longtime_Study
substudy = 'RR_Long' #RR_Q1' #RR_Long'
date = '2025-11-25'

root_folder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}'
folders = sorted(glob.glob(os.path.join(root_folder, f"{date}*")))
print(folders)
if not os.path.exists(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}_analysis_plots'):
    os.makedirs(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}_analysis_plots')
plot_folder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}_analysis_plots'
all_scans = []
for f in folders:
    folder = os.path.join(f, 'study_data')
    data_handler = PlotRRData(folder, plot_folder)
    all_scans.extend(data_handler.load_qspec())

# outerFolder = f"/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/study_data"
# outerFolder_saveplots = f"/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/analysis_plots"
# if not os.path.exists( f"/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/analysis_plots"):
#     os.makedirs(f"/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/analysis_plots")
#
# data_handler = PlotRRData(outerFolder, outerFolder_saveplots)
#
# all_scans = data_handler.load_qspec()
data_handler= PlotRRData(folders[0], plot_folder)
found_freqs = [4914.05, 4764.5, 4577, 4782]
f_min = [4914.05-5, 4764.5-5, 4577-5, 4782-5]
f_max = [4914.05+5, 4764.5+5, 4577+5, 4782+5]
for q in [0, 1, 2, 3]:
    data_handler.plot_IQ_long(all_scans, q, f_min = f_min[q], f_max = f_max[q], backsub = True)
    data_handler.plot_IQ_long(all_scans, q, f_min=None, f_max=None, backsub=True)