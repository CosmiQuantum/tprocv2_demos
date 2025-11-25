import numpy as np
import os
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import glob


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

    datafiles = glob.glob(os.path.join(studyFolder, "Tomography_AllQs*"))
    if len(datafiles) == 0:
        raise ValueError(f"No tomography files found in given folder, {studyFolder}")

    meta = np.load(meta_path, allow_pickle = True)
    metadata = {k: meta[k].item() if meta[k].shape == () else meta[k] for k in meta}
    #vsweep = metadata["vsweep"]
    q_list = [0, 1, 2, 3]

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

def plot_2dtomo_data(metadata, data, qubit, backsub = False, saveFolder = None, stitch = False):
    vsweep = np.array(metadata["vsweep"]) * 1000
    xi = np.array(data[qubit]["xi"])
    xq = np.array(data[qubit]["xq"])
    times = data["_timestamps"]

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
    axQ.set_xlabel('Time (min')
    plt.colorbar(imQ, ax=axQ, label ='Q amp')

    tot_min = elapsed_min[-1] - elapsed_min[0]
    title = f"Qubit {qubit + 1} I/Q Tomography"
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
        save_name = f'Q{qubit + 1}_Tomography'
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
study = 'DDoff_SC_HoleOpen' #'DDon_SC_HoleClosed' #'DDon_SC_HoleOpen' #'Longtime_Study
substudy = 'AllQ_Tomography'

qubits = [0, 1, 2, 3]


# ### To run single timestamp folder ###
# date = '2025-11-24'
# timestamp = '2025-11-24_11-46-41'
# studyFolder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/study_data'
# if not os.path.exists(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots'):
#     os.makedirs(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots')
# plotFolder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots'
#
# metadata, data = load_allQtomo_data(studyFolder)
# plot_2dtomo_data(metadata, data, qubit, backsub = True, saveFolder = plotFolder)

# ### To run all timestamp folders individually in a substudy ###
# study_folders, plot_folder = find_folders(run, study, substudy)
# for f in study_folders:
#     metadata, data = load_allQtomo_data(f)
#     for q in qubits:
#         plot_2dtomo_data(metadata, data, q, backsub= True, saveFolder = plot_folder, stitch = False)

### To run full substudy ###
study_folders, plot_folder = find_folders(run, study, substudy)
metadata, data = stitch_data(study_folders, load_allQtomo_data)
for q in qubits:
    plot_2dtomo_data(metadata, data, q, backsub = True, saveFolder = plot_folder, stitch = True)

