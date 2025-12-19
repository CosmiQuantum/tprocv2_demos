import numpy as np
import os
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import glob
from scipy.signal import butter, filtfilt, savgol_filter

from plotly.express.trendline_functions import rolling
from sympy.codegen.ast import continue_


# class Analyze_AllQ_Tomo()
# class Analyze_SingleQ_Tomo()

def load_allQtomo_data(studyFolder):
    contents = os.listdir(studyFolder)
    #print(contents)
    #meta_path = os.path.join(studyFolder, "Tomography_Metadata*")
    metafile = glob.glob(os.path.join(studyFolder, "Tomography_Metadata*.npz"))
    if len(metafile) == 0:
        raise FileNotFoundError("No metadata file found in given folder")
    if len(metafile) > 1:
        raise ValueError("Multiple metadata files found. Specify one.")
    meta_path = metafile[0]

    datafiles = glob.glob(os.path.join(studyFolder, "Tomography_AllQs*")) #AllQs*"))
    if len(datafiles) == 0:
        raise ValueError(f"No tomography files found in given folder, {studyFolder}")

    meta = np.load(meta_path, allow_pickle = True)
    metadata = {k: meta[k].item() if meta[k].shape == () else meta[k] for k in meta}
    print(metadata["q4_cfg"])

    #vsweep = metadata["vsweep"]
    q_list = [3] #[0, 1, 2, 3]

    data = {
        q: {
            #"vsweep": vsweep,
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

        data_arrs = np.load(file, allow_pickle=True)["all_xi_xq"]

        for qi, q in enumerate(q_list):
            I = data_arrs[2*qi]
            Q = data_arrs[2*qi + 1]
            data[q]["xi"].append(I)
            data[q]["xq"].append(Q)

        data["_rounds"] = np.array(cycle_rounds)
        data["_timestamps"] = np.array(cycle_timestamps, dtype = object)

    return metadata, data

def load_singleQtomo_data(studyFolder, qubit, start, stop, all = True):
    contents = os.listdir(studyFolder)
    #print(contents)
    #meta_path = os.path.join(studyFolder, "Tomography_Metadata*")
    metafile = glob.glob(os.path.join(studyFolder, "Tomography_Metadata*.npz"))
    if len(metafile) == 0:
        raise FileNotFoundError("No metadata file found in given folder")
    if len(metafile) > 1:
        raise ValueError("Multiple metadata files found. Specify one.")
    meta_path = metafile[0]

    datafiles_all = glob.glob(os.path.join(studyFolder, f"Tomography_Q{qubit+1}*")) #AllQs*"))
    if all:
        datafiles = datafiles_all
    else:
        datafiles = []

        pattern = re.compile(r"_R(\d+)_")
        for f in datafiles_all:
            base = os.path.basename(f)
            m = pattern.search(f)
            if m:
                num = int(m.group(1))
                if start <= num <= stop:
                    datafiles.append(f)
    if len(datafiles) == 0:
        raise ValueError(f"No tomography files found in given folder, {studyFolder}")

    meta = np.load(meta_path, allow_pickle = True)
    metadata = {k: meta[k].item() if meta[k].shape == () else meta[k] for k in meta}
    #vsweep = metadata["vsweep"]
    q_index = qubit

    data = {
        q_index: {
            #"vsweep": vsweep,
            "xi": [],
            "xq": []
        }
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

        data_arrs = np.load(file, allow_pickle=True)["xi_xq"]

        I = data_arrs[0]
        Q = data_arrs[1]
        data[qubit]["xi"].append(I)
        data[qubit]["xq"].append(Q)

        data["_rounds"] = np.array(cycle_rounds)
        data["_timestamps"] = np.array(cycle_timestamps, dtype = object)

    return metadata, data

# def get_short_scan_avg(data, qubit, data_type):
#     num_of_sets = len(data[qubit]['xi'])
#     scan_avgs = np.full((num_of_sets), np.nan)
#     for set in range(0, num_of_sets):
#         if data_type == 'I':
#             scan_data = data[qubit]['xi'][set]
#         elif data_type == 'Q':
#             scan_data = data[qubit]['xq'][set]
#         elif data_type == 'Amp':
#             I = data[qubit]['xi'][set]
#             Q = data[qubit]['xq'][set]
#             scan_data = np.sqrt(I**2 + Q**2)
#         else:
#             print(f"Data type entered: {data_type} is not acceptable. \n Enter 'I', 'Q', or 'Amp'")
#             return
#         scan_len = len(scan_data)
#         average = sum(scan_data)/scan_len
#         scan_avgs[set] = average
#
#     return scan_avgs
#
# def scan_avg_threshold_checking(scan_avgs, threshold):
#     scan_and_jumps = []
#
#     for scan in range(1, len(scan_avgs)):
#         dif = abs(scan_avgs[scan] - scan_avgs[scan - 1])
#         if dif > threshold:
#             scan_and_jumps.append((scan, dif))
#
#     return scan_and_jumps

#def

def rolling_avg_threshold(data, qubit, index, data_type, window, threshold):
    if data_type == 'I':
        scans = np.array(data[qubit]['xi'])
    elif data_type == 'Q':
        scans = np.array(data[qubit]['xq'])
    elif data_type == 'Amp':
        I = np.array(data[qubit]['xi'])
        Q = np.array(data[qubit]['xq'])
        scans = np.sqrt(I**2 + Q**2)
    else:
        print(f"Data type entered: {data_type} is not acceptable. \n Enter 'I', 'Q', or 'Amp'")
        return

    data_pts = scans[:, index]
    kernel = np.ones(window) / window
    rolling_avg = np.convolve(data_pts, kernel, mode='valid')
    rolling_avg_filt = savgol_filter(rolling_avg, window_length=20, polyorder=2)
    difs = np.abs(np.diff(rolling_avg))
    difs_filtered = np.abs(np.diff(rolling_avg_filt))

    jump_indices_window = np.where(difs_filtered > threshold)[0] + 1
    jump_indices_scan = jump_indices_window + (window - 1)
    jump_timestamp = [data['_timestamps'][i] for i in jump_indices_scan]

    return rolling_avg, rolling_avg_filt, difs, difs_filtered, jump_indices_window, jump_indices_scan, jump_timestamp

def butterworth_filter(data, cutoff, fs, order =4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)

def plot_rollavg_wjumps(qubit, rolling_avg, rolling_avg_filt, jump_indices_window, difs, difs_filtered):
    plt.plot(rolling_avg, label = 'raw')
    #butterfilt = butterworth_filter(rolling_avg, 0.1, 1)
    #plt.plot(butterfilt, label = 'BW')
    #savgol = savgol_filter(rolling_avg, window_length = 9, polyorder = 2)
    plt.plot(rolling_avg_filt, label = 'SG')

    for jump in jump_indices_window:
        plt.axvline(jump, color = 'red', linestyle = '--', alpha = 0.3, linewidth = 0.5)

    plt.legend()
    plt.title(f'Rolling Avg Plot, Q{qubit+1}')
    plt.ylabel('Amplitude')
    plt.xlabel('Scan Num')
    plt.show()

    plt.plot(difs, label = 'raw')
    plt.plot(difs_filtered, label = 'SV filt')
    plt.legend()
    plt.title(f'Rolling Avg Dif, Q{qubit+1}')
    plt.ylabel('Amplitude Dif')
    plt.xlabel('Scan Num')
    plt.show()
    return

def plot_2dtomo_data(metadata, data, qubit, start, stop, backsub = False, saveFolder = None, stitch = False):
    vsweep = np.array(metadata["vsweep"]) * 1000
    xi = np.array(data[qubit]["xi"][:]) #[:1000]
    xq = np.array(data[qubit]["xq"][:]) #[:1000]
    times = data["_timestamps"][:] #[:1000]

    #Get time set up
    t0 = times[0]
    elapsed_min = np.array([(dt - t0).total_seconds() / 60.0 for dt in times])

    if backsub:
        I_val = xi.astype(float)
        Q_val = xq.astype(float)
        for i in range(0, len(I_val)):
            I_val[i] -= np.mean(I_val[i])
            Q_val[i] -= np.mean(Q_val[i])
        xi = I_val
        xq = Q_val

    # xi = np.array(xi)
    # xq = np.array(xq)
    xi_plot = xi.T
    xq_plot = xq.T

    vsweep_diff = np.diff(vsweep)
    if len(vsweep_diff) == 0:
        vsweep_edges = np.array([vsweep[0], vsweep[0]+1])
    else:
        vsweep_edges = np.concatenate([vsweep, [vsweep[-1] + vsweep_diff[-1]]])

    elapsed_diff = np.diff(elapsed_min)
    last_diff = elapsed_diff[-1] if len(elapsed_diff) > 0 else 1
    elapsed_edges = np.concatenate([elapsed_min, [elapsed_min[-1] + last_diff]])

    # print(len(vsweep_edges))
    # print(len(elapsed_edges))

    fig, (axI, axQ) = plt.subplots(2, 1, figsize = (9,7), sharex=True)

    imI = axI.imshow(xi_plot, aspect = 'auto', origin = 'lower', extent = [times[0], times[-1], vsweep[0], vsweep[-1]])
           #pcolormesh(elapsed_edges, vsweep_edges, xi_plot, shading='auto', cmap='viridis'))
        #xi.T, aspect = 'auto', origin = 'lower', extent = [elapsed_min[0], elapsed_min[-1], vsweep[0], vsweep[-1]])
    axI.set_ylabel('Voltage Bias (mV)')
    plt.colorbar(imI, ax=axI, label='I amp')

    imQ = axQ.imshow(xq_plot, aspect = 'auto', origin = 'lower', extent = [times[0], times[-1], vsweep[0], vsweep[-1]])
        #pcolormesh(elapsed_edges, vsweep_edges, xq_plot, shading='auto', cmap='viridis')
        #xq.T, aspect = 'auto', origin = 'lower', extent = [elapsed_min[0], elapsed_min[-1], vsweep[0], vsweep[-1]])
    axQ.set_ylabel('Voltage Bias (mV)')
    axQ.set_xlabel('Timestamp') #'Time (min)')
    plt.colorbar(imQ, ax=axQ, label ='Q amp')

    tot_min = elapsed_min[-1] - elapsed_min[0]
    title = f"Qubit {qubit + 1} I/Q Tomography start-stop" #First 1000"
    if backsub:
        title += ", Bgkd Sub"
    if stitch:
        title += ", Stitched"
    title += f", {tot_min} min"
    fig.suptitle(title)

    axI.xaxis_date()
    axQ.xaxis_date()
    fmt = mdates.DateFormatter("%m-%d\n%H:%M")
    axI.xaxis.set_major_formatter(fmt)
    axQ.xaxis.set_major_formatter(fmt)

    plt.tight_layout()
    if saveFolder is not None:
        save_name = f'Q{qubit + 1}_Tomography_start-stop' #_first1000'
        if backsub:
            save_name += '_bkgdsub'
        if stitch:
            save_name += '_stitched'
        save_name += f'_{times[0].strftime("%Y-%m-%d_%H-%M-%S")}.png'
        fig_path = os.path.join(saveFolder, save_name)
        plt.savefig(fig_path)
    plt.close()
    return

def stitch_data(folders, loader_func):
    merged_data = None
    merged_metadata = None
    q_list = [0, 1, 2, 3]

    for i, folder in enumerate(folders):
        metadata, data = loader_func(folder)

        if merged_data is None:
            merged_metadata = metadata
            merged_data = {
                q: {#"vsweep": data[q]["vsweep"],
                    "xi": data[q]["xi"],
                    "xq": data[q]["xq"]}
                for q in q_list
            }
            merged_data["_rounds"] = np.array(data["_rounds"])
            merged_data["_timestamps"] = np.array(data["_timestamps"], dtype=object)
            continue

        if not np.array_equal(metadata["vsweep"], merged_metadata["vsweep"]):
            raise ValueError("Vsweep mismatch between folders")

        for q in q_list:
            merged_data[q]["xi"] = np.vstack([merged_data[q]["xi"], data[q]["xi"]])
            merged_data[q]["xq"] = np.vstack([merged_data[q]["xq"], data[q]["xq"]])

        merged_data["_rounds"] = np.concatenate([merged_data["_rounds"], data["_rounds"]])
        merged_data["_timestamps"] = np.concatenate([merged_data["_timestamps"], data["_timestamps"]])

    return merged_metadata, merged_data

def find_folders(run, study, substudy):
    root = f"/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}"

    study_folders = glob.glob(os.path.join(root, "*", "*", "study_data"))

    study_folders = sorted(study_folders)

    if not os.path.exists(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/full_analysis_plots'):
        os.makedirs(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/full_analysis_plots')
    plot_folder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/full_analysis_plots'

    return study_folders, plot_folder

run = 'run33e'
study = 'Cs_TimeStudy_Tomography' #SC_Tomography' #DDoff_SC_HoleOpen' #'DDon_SC_HoleClosed' #'DDon_SC_HoleOpen' #'Longtime_Study
substudy = 'AllQ_Tomography' #'SingleQ4_Tomography' #AllQ_Tomography'

qubits = [3] #[0, 1, 2, 3]
qubit = 3


# ### To run single timestamp folder ###
date = '2025-12-04' #'2025-12-11'
timestamp = '2025-12-04_13-36-05' #'2025-12-11_08-06-54'
studyFolder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/study_data'
if not os.path.exists(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots'):
    os.makedirs(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots')
plotFolder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots'

start = 0
stop = 39

metadata, data = load_allQtomo_data(studyFolder)
#plot_2dtomo_data(metadata, data, qubit, start, stop, backsub = False, saveFolder = plotFolder)

#metadata, data = load_singleQtomo_data(studyFolder, qubit, start, stop, all = False) #load_allQtomo_data(studyFolder)
#print(len(data[qubit]['xi'][0]))
#for q in qubits:
#plot_2dtomo_data(metadata, data, qubit, start, stop, backsub = False, saveFolder = plotFolder)
# rolling_avg, rolling_avg_filt, difs, difs_filt, jump_indices_window, jump_indices_scan, jump_timestamps = rolling_avg_threshold(data, qubit, 0, 'I',5, 0.05)
# plot_rollavg_wjumps(qubit, rolling_avg, rolling_avg_filt, jump_indices_window, difs, difs_filt)
# print(jump_timestamps)

### To run all timestamp folders individually in a substudy ###
# study_folders, plot_folder = find_folders(run, study, substudy)
# for f in study_folders:
#     metadata, data = load_allQtomo_data(f)
#     for q in qubits:
#         plot_2dtomo_data(metadata, data, q, backsub= False, saveFolder = plot_folder, stitch = False)

### To run full substudy ###
# study_folders, plot_folder = find_folders(run, study, substudy)
# metadata, data = stitch_data(study_folders, load_allQtomo_data)
# for q in qubits:
#     plot_2dtomo_data(metadata, data, q, backsub = False, saveFolder = plot_folder, stitch = True)

