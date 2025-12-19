import numpy as np
import h5py
import json
import os
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# H5 file structure
# vsweep (npts)
# q_config (1,)
# total_rounds (1,)
# q_data (rounds, 2, npts)
#   axis 0: rounds
#   axis 1: 0 is I, 1 is Q
#   axis 2: voltage points in vsweep
# timestamps (rounds)
# voltage_ok (rounds, npts)
#   1 = good
#   0 or -1 = voltage setting failed

# Attributes: rows_written (int)

def load_tomo_h5(file, start, stop, print_config = False, print_duration = False):
    # Get file timestamp
    basename = os.path.basename(file)
    match = re.search(r'Tomography_Q\d+_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', basename)
    if match:
        file_timestamp = match.group(1)
    else:
        file_timestamp = "Unknonwn"
        print("Couldn't get file timestamp from name")

    with h5py.File(file, "r") as f:
        rows = f.attrs["rows_written"]
        vsweep_V = f['vsweep'][:]
        q_config_str = f['config'][()]

        if start is None:
            start = 0

        if stop is None:
            stop = rows

        if stop > rows:
            print(f"Stop value ({stop}) is greater than rows in file ({rows})")

        if start >= stop:
            print(f"Invalid start and stop values")


        qdata = f['qdata'][start:stop]
        timestamps = f['timestamps'][start:stop]
        voltage_check = f['voltage_check'][start:stop]

    # timestamps to datetime
    timestamps_dt = np.array([datetime.strptime(t.decode('ascii'), "%Y-%m-%d_%H-%M-%S") for t in timestamps])
    if print_duration:
        round_duration = np.array([
            (timestamps_dt[i+1] - timestamps_dt[i]).total_seconds() for i in range(len(timestamps_dt) -1)
        ])
        print(f"Round durations: {round_duration}")

    vsweep_mV = vsweep_V * 1000

    #print(voltage_check)

    if print_config:
        print(q_config_str)
        if isinstance(q_config_str, bytes):
            q_config_str = q_config_str.decode('ascii')
        q_config = json.loads(q_config_str)
        print(json.dumps(q_config, indent=4))

    return vsweep_mV, qdata, timestamps_dt, voltage_check, file_timestamp, rows

def tomo_colorplot(qubit, vsweep, qdata, timestamps, voltage_check, file_timestamp, start, stop, rows, masked = False, backsub=False, saveFolder = None):
    """
    vsweep          : 1d array, voltage points in mV
    qdata           : ndarray, (rounds, 2, npts)
    timestamps      : array of datettime objects, # of rounds
    voltage_check   : ndarray, (rounds, npts)
    """

    if start is None:
        start = 0
    if stop is None:
        stop = rows


    # get total time and timestamps to matplotlib format
    total_time = (timestamps[-1] - timestamps[0]).total_seconds() / 60.0
    times_plt = mdates.date2num(timestamps)

    xi = qdata[:, 0, :]
    xq = qdata[:, 1, :]

    if backsub:
        I_val = xi.astype(float)
        Q_val = xq.astype(float)
        for i in range(0, len(I_val)):
            I_val[i] -= np.mean(I_val[i])
            Q_val[i] -= np.mean(Q_val[i])
        xi_plot = I_val
        xq_plot = Q_val
    elif masked:
        bad_rounds = np.any(voltage_check <= 0, axis = 1)
        xi_plot = np.ma.masked_array(xi, mask = np.repeat(bad_rounds[:, np.newaxis], xi.shape[1], axis=1))
        xq_plot = np.ma.masked_array(xq, mask = np.repeat(bad_rounds[:, np.newaxis], xq.shape[1], axis=1))
    else:
        xi_plot = xi
        xq_plot = xq

    fig, (axI, axQ) = plt.subplots(2, 1, figsize=(12,7), sharex=True)

    imI = axI.imshow(xi_plot.T, aspect = 'auto', origin = 'lower',
                   extent = [times_plt[0], times_plt[-1], vsweep[0], vsweep[-1]])
    axI.set_ylabel('Voltage Bias (mV)')
    plt.colorbar(imI, ax=axI, label = 'I amp')

    imQ = axQ.imshow(xq_plot.T, aspect = 'auto', origin = 'lower',
                   extent = [times_plt[0], times_plt[-1], vsweep[0], vsweep[-1]])
    axQ.set_ylabel('Voltage Bias (mV)')
    axQ.set_xlabel('Timestamp')
    plt.colorbar(imQ, ax=axQ, label = 'Q amp')

    title = f"Q{qubit+1} I/Q Tomography {start}-{stop}"
    if not masked:
        title += ", Unmasked"
    if backsub:
        title += ", Bgkd Sub"
    title += f", {file_timestamp}, {total_time} min"
    plt.suptitle(title)

    axI.xaxis_date()
    axQ.xaxis_date()
    fmt = mdates.DateFormatter("%m-%d\n%H:%M:%S")
    axI.xaxis.set_major_formatter(fmt)
    axQ.xaxis.set_major_formatter(fmt)

    plt.tight_layout()
    if saveFolder is not None:
        save_name = f'Q{qubit + 1}_Tomography_{start}-{stop}'
        if not masked:
            save_name += '_unmasked'
        if backsub:
            save_name += '_bkgdsub'
        save_name += f'_{file_timestamp}.png'
        fig_path = os.path.join(saveFolder, save_name)
        plt.savefig(fig_path)
    plt.close()
    return


######################## Run plotting ###################################

run = 'run33e'
study = 'Cs_TimeStudy_Tomography_2' #SC_Tomography' #DDoff_SC_HoleOpen' #'DDon_SC_HoleClosed' #'DDon_SC_HoleOpen' #'Longtime_Study
substudy = 'SingleQ4_Tomography' #AllQ_Tomography'

qubits = [3] #[0, 1, 2, 3]
qubit = 3


# ### To run single file ###
date = '2025-12-17'
timestamp = '2025-12-17_11-35-46'
studyFolder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/study_data'
if not os.path.exists(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots'):
    os.makedirs(f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots')
plotFolder = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{date}/{timestamp}/analysis_plots'
filepath = os.path.join(studyFolder, f'Tomography_Q{qubit+1}_{timestamp}.h5')

start = 11115
stop = None

vsweep_mV, qdata, timestamps, voltage_check, file_timestamp, rounds_written = load_tomo_h5(filepath, start, stop, print_config=False, print_duration = False)
tomo_colorplot(qubit, vsweep_mV, qdata, timestamps, voltage_check, file_timestamp, start, stop, rounds_written, masked = False, backsub = False, saveFolder = plotFolder)


