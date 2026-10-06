import numpy as np
import os
import datetime
import re
from matplotlib import pyplot as plt
import h5py
from sklearn.cluster import KMeans

def load_from_h5(filename, data_type, save_r=1):  # Added save_r as parameter.

    data = {data_type: {}}  # Initialize the main dictionary with the data_type.

    with h5py.File(filename, 'r') as f:
        for qubit_group in f.keys():
            qubit_index = int(qubit_group[1:]) - 1
            qubit_data = {}
            group = f[qubit_group]

            for dataset_name in group.keys():
                # Attempt to map HDF5 keys to the target dictionaries' keys.
                if data_type == 'Res':
                    target_keys = {'Dates': 'Dates', 'freq_pts': 'freq_pts', 'freq_center': 'freq_center',
                                       'Amps': 'Amps', 'Found Freqs': 'Found Freqs', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num'}
                elif data_type == 'QSpec':
                    target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Frequencies': 'Frequencies',
                                       'I Fit': 'I Fit', 'Q Fit': 'Q Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num'}
                elif data_type == 'Rabi':
                    target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Gains': 'Gains', 'Fit': 'Fit',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num'}
                elif data_type == 'SS':
                    target_keys = {'Fidelity': 'Fidelity', 'Angle': 'Angle', 'Dates': 'Dates', 'I_g': 'I_g',
                                       'Q_g': 'Q_g', 'I_e': 'I_e', 'Q_e': 'Q_e',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num'}
                elif data_type == 'T1':
                    target_keys = {'T1': 'T1', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num'}
                elif data_type == 'T2':
                    target_keys = {'T2': 'T2', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num'}
                elif data_type == 'T2E':
                    target_keys = {'T2E': 'T2E', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num'}
                elif data_type == 'stark2D':
                    target_keys = {'Dates': 'Dates', 'I':'I', 'Q': 'Q', 'Qu Frequency Sweep':'Qu Frequency Sweep',
                                   'Res Gain Sweep':'Res Gain Sweep','Round Num':'Round Num', 'Batch Num': 'Batch Num',
                                   'Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type =='starkSpec':
                    target_keys = {'Dates': 'Dates', 'I':'I', 'Q': 'Q','P': 'P', 'shots':'shots','Gain Sweep':'Res Gain Sweep','Round Num':'Round Num', 'Batch Num': 'Batch Num',
                                   'Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}

                else:
                        raise ValueError(f"Unsupported data_type: {data_type}")

                try:
                    mapped_key = target_keys[dataset_name]  # Map HDF5 key to target key.
                    qubit_data[mapped_key] = [group[dataset_name][()]] * save_r  # Expand to match the desired length.

                except KeyError:
                    print(f"Warning: Key '{dataset_name}' not found in target dictionary for data_type '{data_type}'. Skipping.")
                    pass

            data[data_type][qubit_index] = qubit_data

    return data

def process_string_of_nested_lists(data):
    # Remove extra whitespace and non-numeric characters.
    data = re.sub(r'\s*\[(\s*.*?\s*)\]\s*', r'[\1]', data)
    data = data.replace('[ ', '[')
    data = data.replace('[ ', '[')
    data = data.replace('[ ', '[')
    cleaned_data = ''.join(c for c in data if c.isdigit() or c in ['-', '.', ' ', 'e', '[', ']'])
    pattern = r'\[(.*?)\]'  # Regular expression to match data within brackets
    matches = re.findall(pattern, cleaned_data)
    result = []
    for match in matches:
        numbers = [float(x.strip('[').strip(']').replace("'", "").replace(" ", "").replace("  ", "")) for x in
                    match.split()]  # Convert strings to integers
    result.append(numbers)

    return result

def process_h5_data(data):
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

##  Load data
path = '/data/QICK_data/run6/6transmon/StarkShift/Junkyard/2025-04-11/Q4/starkSpec/Data_h5/StarkSpec'
# filenames = ["./2025-03-25_22-04-24_StarkSpec_results_batch_0_Num_per_batch1.h5", #0.01 us, 200 us
#             "./2025-03-25_22-32-51_StarkSpec_results_batch_0_Num_per_batch1.h5", #10 us, 200 us
#             "./2025-03-25_22-34-55_StarkSpec_results_batch_0_Num_per_batch1.h5", #15 us, 200 us
#             "./2025-03-25_22-37-08_StarkSpec_results_batch_0_Num_per_batch1.h5", #20 us, 200 us
#             "./2025-03-25_22-07-53_StarkSpec_results_batch_0_Num_per_batch1.h5", #25 us, 200 us
#             "./2025-03-25_22-42-42_StarkSpec_results_batch_0_Num_per_batch1.h5", #30 us, 200 us
#             "./2025-03-25_22-44-23_StarkSpec_results_batch_0_Num_per_batch1.h5", #35 us, 200 us
#             "./2025-03-25_22-46-21_StarkSpec_results_batch_0_Num_per_batch1.h5", #40 us, 200 us
#             "./2025-03-25_22-48-19_StarkSpec_results_batch_0_Num_per_batch1.h5", #45 us, 200 us
#             ]


# filenames = ["./2025-03-25_23-05-19_StarkSpec_results_batch_0_Num_per_batch1.h5"] #15 us, 200 us, 7 steps
# stark_length = [15]

# filenames = ["./2025-03-25_23-11-19_StarkSpec_results_batch_0_Num_per_batch1.h5", #0.01 us, 5 us
#             "./2025-03-25_23-13-01_StarkSpec_results_batch_0_Num_per_batch1.h5", #10 us, 5 us
#             "./2025-03-25_23-14-18_StarkSpec_results_batch_0_Num_per_batch1.h5", #15 us, 5 us
#              "./2025-03-25_23-14-59_StarkSpec_results_batch_0_Num_per_batch1.h5", #20 us, 5 us
#              "./2025-03-25_23-15-45_StarkSpec_results_batch_0_Num_per_batch1.h5", #25 us, 5 us
#              "./2025-03-25_22-42-42_StarkSpec_results_batch_0_Num_per_batch1.h5", #30 us, 5 us
#              "./2025-03-25_23-18-31_StarkSpec_results_batch_0_Num_per_batch1.h5", #35 us, 5 us
#              "./2025-03-25_23-21-03_StarkSpec_results_batch_0_Num_per_batch1.h5", #40 us, 5 us
#              "./2025-03-25_23-21-57_StarkSpec_results_batch_0_Num_per_batch1.h5", #45 us, 5 us
#              ]
# stark_length = [0.01, 10, 15, 20, 25, 30, 35, 40, 45]

# filenames = ["./2025-03-25_23-29-34_StarkSpec_results_batch_0_Num_per_batch1.h5"] #15 us, 1 us, 10 steps, 1000 reps
# stark_length = [15]
# filenames = ["./2025-03-25_23-36-01_StarkSpec_results_batch_0_Num_per_batch1.h5"] #15 us, 1 us, 10 steps, 200 reps
# stark_length = [15]

# filenames = ["./2025-03-27_20-24-42_StarkSpec_results_batch_0_Num_per_batch1.h5"] #5 us, 1 us, 10 steps, 1000 reps
# stark_length = [30]

# filenames = ["./2025-03-28_10-20-44_StarkSpec_results_batch_0_Num_per_batch1.h5",
#             "./2025-03-28_10-38-58_StarkSpec_results_batch_0_Num_per_batch1.h5",
#              "./2025-03-28_10-40-38_StarkSpec_results_batch_0_Num_per_batch1.h5",
#              "./2025-03-28_10-43-08_StarkSpec_results_batch_0_Num_per_batch1.h5",
#              "./2025-03-28_10-45-20_StarkSpec_results_batch_0_Num_per_batch1.h5",
#              "./2025-03-28_10-47-46_StarkSpec_results_batch_0_Num_per_batch1.h5",
#              "./2025-03-28_10-49-16_StarkSpec_results_batch_0_Num_per_batch1.h5",
#              "./2025-03-28_10-51-14_StarkSpec_results_batch_0_Num_per_batch1.h5",
#              "./2025-03-28_10-52-56_StarkSpec_results_batch_0_Num_per_batch1.h5",
#              "./2025-03-28_10-54-30_StarkSpec_results_batch_0_Num_per_batch1.h5",
#              ] #0.01 us, 200 us,  1 steps, 500 reps]
# stark_length = [0.01, 10, 15, 20, 25, 30, 35, 40, 45, 50]

#filenames = ["./2025-03-28_11-30-12_StarkSpec_results_batch_0_Num_per_batch1.h5"]
#filenames = ["./2025-03-28_12-24-00_StarkSpec_results_batch_0_Num_per_batch1.h5"]
#filenames = ["2025-03-31_01-58-27_StarkSpec_results_batch_0_Num_per_batch1.h5"]
filenames = ["2025-04-11_09-09-40_StarkSpec_results_batch_0_Num_per_batch1.h5"]
stark_length = [25]

QubitIndex = 4
#n = 20
#reps = 500
#steps = 250
theta = 0/180 * np.pi
power2shift = -25 #-17.051522699805382 #MHz per resonator probe power]
#freq = np.linspace(0, 20, num=125)
#freq_sweep = np.array([-1*freq[::-1], freq]).reshape([steps,1])
thresh_i = 0
thresh_q = 0

#fig, axes = plt.subplots(3,3,figsize=[8,8])
a = 0
b = 0
count = 0
P = []
for filename in filenames:
    filepath = os.path.join(path, filename)
    load_data = load_from_h5(filepath, 'starkSpec', save_r=1)
    print((load_data['starkSpec'][QubitIndex].get('Gain Sweep', [])))
    #gain_sweep = np.array(process_h5_data(load_data['starkSpec'][QubitIndex].get('Gain Sweep', [])[0].decode()))
    #gain_pts = len(gain_sweep)
    gain_pts = 250
    gain_sweep = np.sqrt(np.linspace(0, 1, num= gain_pts))
    #freq_sweep = np.linspace(-1, 1, num=gain_pts)
    freq_sweep = gain_sweep**2 * power2shift
    n = np.shape(load_data['starkSpec'][QubitIndex].get('I', []))[1]
    print(n)
    reps = int(np.shape(process_h5_data(load_data['starkSpec'][QubitIndex].get('I', [])[0][0].decode()))[0] / gain_pts)
    print(reps)

    start_time = datetime.datetime.fromtimestamp(load_data['starkSpec'][QubitIndex].get('Dates', [])[0][0])
    times = []
    for i in np.arange(0,n):
        date = datetime.datetime.fromtimestamp(load_data['starkSpec'][QubitIndex].get('Dates', [])[0][i])
        times.append((date - start_time).total_seconds())

    I = np.zeros([n, gain_pts, reps])
    Q = np.zeros([n, gain_pts, reps])
    shots = np.zeros([n, gain_pts, reps])

    for i in np.arange(0, n):
        I[i][:][:] = np.array(process_h5_data(load_data['starkSpec'][QubitIndex].get('I', [])[0][i].decode())).reshape([gain_pts, reps])
        Q[i][:][:] = np.array(process_h5_data(load_data['starkSpec'][QubitIndex].get('Q', [])[0][i].decode())).reshape([gain_pts, reps])
        shots[i][:][:] = np.array(process_h5_data(load_data['starkSpec'][QubitIndex].get('shots', [])[0][i].decode())).reshape([gain_pts, reps])

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    m = 0
    j=0

    i_new = I[0][0][:] * np.cos(theta) - Q[0][0][:] * np.sin(theta)
    q_new = I[0][0][:] * np.sin(theta) + Q[0][0][:] * np.cos(theta)
    kmeans = KMeans(n_clusters=2).fit(np.transpose([i_new, q_new]))

    time_idx = 0
    for i in (np.arange(0,6) * round((gain_pts-5)/5)):
        plot = axes[j][m]
        plot.set_box_aspect(1)
        i_new = I[time_idx][i][:] * np.cos(theta) - Q[time_idx][i][:] * np.sin(theta)
        q_new = I[time_idx][i][:] * np.sin(theta) + Q[time_idx][i][:] * np.cos(theta)
    #kmeans = KMeans(n_clusters=3).fit(np.transpose([i_new, q_new]))
        plot.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], c='k')
        idx = kmeans.predict(np.transpose([i_new, q_new]))
        #idx = i_new > thresh_i
        plot.scatter(i_new, q_new, c=idx)
        plot.set_xlabel("I")
        plot.set_ylabel("Q")
        #plot.set_title(f"gain = {gain_sweep[i]}")
        plot.set_title(f"{np.round(freq_sweep[i])} MHz")
        m+=1
        if m == 3:
            m=0
            j=1

    plt.tight_layout()
    plt.show()


    gstate = np.argmin(kmeans.cluster_centers_[:,0])
    estate = np.argmax(kmeans.cluster_centers_[:,0])

    p_new = np.zeros([n, gain_pts])
    for j in np.arange(0, n):
        for i in np.arange(0, gain_pts):
            i_new = I[j][i][:] * np.cos(theta) - Q[0][i][:] * np.sin(theta)
            q_new = I[j][i][:] * np.sin(theta) + Q[0][i][:] * np.cos(theta)
            idx = kmeans.predict(np.transpose([i_new, q_new]))

            #idx = i_new > thresh_i
            idx_post_process = idx
            # idx_post_process = [idx[0]]
            # for k in np.arange(1, reps):
            #     if idx[k-1] == gstate:
            #          idx_post_process.append(idx[k])
            fcount = np.sum(q_new > -1500)
            gcount = np.sum(np.array(idx_post_process) == gstate)
            ecount = np.sum(np.array(idx_post_process) == estate)
            p_new[j][i] = (ecount)/len(idx_post_process)
            #p_new[j][i] = np.mean(i_new)

    #P.append(p_new[0,:])
    #fig, axes = plt.subplots(1,2,figsize=[10,4])

    #plot = axes[0]
    #plot = axes[b][a]
    #for j in np.arange(0,n):
        #plot.plot(gain_sweep**2 * power2shift, p_new[j,:],label=f"round {j}")
        #plot.plot(freq_sweep, p_new[j,:])

    #plot.set_box_aspect(1)
#     plot.set_title(f"stark length = {stark_length[count]} us")
#     a = a+1
#     if a == 3:
#         a = 0
#         b = b+1
#     count = count+1
#
# #plot.legend()
#     #plot.set_xlabel('AC Stark shift [MHz]')
#     plot.set_ylabel('P(MS)=1')
#     plot.set_ylim([0.0, 1.0])

    fig, axes = plt.subplots(1,1, figsize=[5,5], dpi=150)
    plot = axes
    cbar = plt.colorbar(plot.pcolormesh(times, freq_sweep, np.transpose(p_new), shading="nearest", cmap="viridis"), ax=plot)
    cbar.set_label("P(MS=1)")
    plot.set_xlabel('time (s)')
    plot.set_ylabel('AC Stark shift [MHz]')
    plot.set_title('Qubit 4, 25 us stark tone length')

    plt.tight_layout()
    plt.savefig(os.path.join(path,"im4.png"))

    del I
    del Q
    del load_data

# fig, axes = plt.subplots(1,1, figsize=[3,3])
# plot = axes
# plt.colorbar(plot.pcolormesh(stark_length, freq_sweep, np.transpose(P), shading="nearest", cmap="viridis"), ax=plot)
# plot.set_xlabel('stark tone length [us]')
# plot.set_ylabel('AC Stark shift [MHz]')
    # plot.set_title('Qubit 0')
plt.show()


