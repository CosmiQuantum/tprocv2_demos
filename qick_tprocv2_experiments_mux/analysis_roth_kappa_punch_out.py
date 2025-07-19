################################################ imports ####################################
import numpy as np
import h5py
import glob
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
################################################ definitions ####################################
def scalar(thing):
    ## numpy scalars and python scalars are different,
    return thing.item() if isinstance(thing, np.generic) else thing

def read_dataset(dset):
    ## look at the dataset and unpack it
    ## if its a string opject or unicode string convert it to a regular string
    if h5py.check_dtype(vlen=dset.dtype) == str or dset.dtype.kind in ("S", "O", "U"):
        data = dset.asstr()[()]
    else:
        ## if not it could be a boolean or number or array or something
        data = dset[()]
    ## if its a scalar make it into float or int for playing nice in python
    if np.isscalar(data):
        data = scalar(data)
    ## if array just make it a list
    if isinstance(data, np.ndarray):
        data = data.tolist()
    return data

def read_group_recursive(grp):
    ## take the groups and convert to a dictionary
    out = {}
    ## go through all of the items and see if its a dataset and call the right definition
    for name, obj in grp.items():
        if isinstance(obj, h5py.Dataset):
            out[name] = read_dataset(obj)
        else:
            out[name] = read_group_recursive(obj)
    ## we saved some things as attributes so load those too
    for key in grp.attrs:
        out[key] = scalar(grp.attrs[key])
    return out

def load_from_h5(filename, data_type, save_r= None):
    data = {data_type: {}}
    with h5py.File(filename, "r") as f:
        ## each qubit lives in its own group: 'Q1', 'Q2',
        for qname in (k for k in f.keys() if k.startswith("Q")):
            data[data_type][qname] = read_group_recursive(f[qname])
    return data

def lorentzian(f, f0, kappa, depth, offset):
    return offset - depth / (1.0 + 4.0 * ((f - f0) / kappa) ** 2)

def plot_res_sweeps(outerFolder, fpts, fcenter, amps, power_sweep, fit_power_index_low_gain_, fit_power_index_high_gain_, number_of_qubits=6,):
    plt.figure(figsize=(12, 8))

    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 14,  # Base font size
        'axes.titlesize': 18,  # Title font size
        'axes.labelsize': 16,  # Axis label font size
        'xtick.labelsize': 14,  # X-axis tick label size
        'ytick.labelsize': 14,  # Y-axis tick label size
        'legend.fontsize': 14,  # Legend font size
    })


    for i in range(number_of_qubits):
        title = f"Resonator {i + 1} fit"
        kappa_low = None
        kappa_high = None
        freq_low = None
        freq_high = None
        for power_index_ in range(0, len(power_sweep)):
            plt.subplot(2, 3, i + 1)
            plt.subplots_adjust(right=0.88)
            x=[fcenter[power_index_][i][i] + f for f in fpts[power_index_][i]]
            plt.plot( x , amps[power_index_][i][i], '-', linewidth=1.5,
                     label=round(power_sweep[power_index_], 3))

            plt.xlabel("Frequency (MHz)", fontweight='normal')
            plt.ylabel("Amplitude (a.u)", fontweight='normal')

            if round(power_sweep[power_index_],3) == round(fit_power_index_low_gain_[i], 3) or round(power_sweep[power_index_],3) == round(fit_power_index_high_gain_[i], 3):
                if round(power_sweep[power_index_],3) == round(fit_power_index_low_gain_[i], 3):
                    gain_is = 'High gain'
                else:
                    gain_is = 'Low gain'

                freqs = np.asarray([fcenter[power_index_][i][i] + f for f in fpts[power_index_][i]])
                amps_data = np.asarray(amps[power_index_][i][i])

                ## first guesses
                f0_guess = freqs[np.argmin(amps_data)]
                kappa_guess = 0.02 * (freqs.max() - freqs.min())
                depth_guess = amps_data.max() - amps_data.min()
                offset_guess = np.median(amps_data)

                popt, _ = curve_fit(lorentzian, freqs, amps_data, p0=[f0_guess, kappa_guess, depth_guess, offset_guess])
                f0_fit, kappa_fit, depth_fit, offset_fit = popt

                ## store it
                if gain_is == 'Low gain':
                    kappa_low = kappa_fit
                    freq_low = x[amps[power_index_][i][i].index(min(amps[power_index_][i][i]))]

                else:
                    kappa_high = kappa_fit
                    freq_high = x[amps[power_index_][i][i].index(min(amps[power_index_][i][i]))]

                ## overlay fitted curve
                f_fit = np.linspace(freqs.min(), freqs.max(), 1200)
                plt.plot( f_fit, lorentzian(f_fit, *popt),  '--', linewidth=1.8)

        if kappa_low is not None:
            title += f"\nLow gain:  $\\kappa = {kappa_low:.3f}\\,\\mathrm{{MHz}}$"
            title += f"\nLow gain:  $w_r = {freq_low:.3f}\\,\\mathrm{{MHz}}$"
        if kappa_high is not None:
            title += f"\nHigh gain: $\\kappa = {kappa_high:.3f}\\,\\mathrm{{MHz}}$"
            title += f"\nLow gain:  $w_r = {freq_high:.3f}\\,\\mathrm{{MHz}}$"

        plt.title(title, pad=10, fontsize=10)
        plt.legend(loc='upper left',
                   bbox_to_anchor=(1.02, 1.00),
                   fontsize=5, title='Gain')

    plt.suptitle("Resonance At Various Probe Gains", fontsize=24, y=0.95)

    plt.tight_layout(pad=2)
    outerFolder = os.path.join(outerFolder.replace('study_data/Data_h5/Res/',''), "documentation")
    if not os.path.exists(outerFolder):
        os.makedirs(outerFolder)
    file_name = os.path.join(outerFolder, f"fit_kappas.pdf")
    plt.savefig(file_name, dpi=300)
    plt.show()
    plt.close()
    return

def grab(lst, value, power_index, qubit_index):
    lst[power_index][qubit_index]=value[0]

################################################ run ####################################
## replace with unzipped folder name
base_path = '/data/QICK_data/run7/6transmon/'

data_file_path = base_path + 'thomas_punch_out_kappa_data_for_simulation/kappa_punch_out/2025-07-18_16-05-32/study_data/Data_h5/Res/'
print(data_file_path )
## load everything and save it
h5_files = glob.glob(os.path.join(data_file_path, "*.h5"))
print(h5_files)

power_sweep    = [] # is the same for all of the qubits
for h5_file in h5_files:
    ## get the gain info
    power_value = float(h5_file.split('.h5')[0].split('_')[-1].replace('p', '.'))
    if power_value not in power_sweep:
        power_sweep.append(power_value)

## sort according to power
power_sweep = sorted(power_sweep, key=float)
Dates          = [[[] for _ in range(6)] for _ in range(len(power_sweep))]
freq_pts       = [[[] for _ in range(6)] for _ in range(len(power_sweep))]
freq_center    = [[[] for _ in range(6)] for _ in range(len(power_sweep))]
Amps           = [[[] for _ in range(6)] for _ in range(len(power_sweep))]
Found_Freqs    = [[[] for _ in range(6)] for _ in range(len(power_sweep))]
Round_Num      = [[[] for _ in range(6)] for _ in range(len(power_sweep))]
Batch_Num      = [[[] for _ in range(6)] for _ in range(len(power_sweep))]
Exp_Config     = [[[] for _ in range(6)] for _ in range(len(power_sweep))]
Syst_Config    = [[[] for _ in range(6)] for _ in range(len(power_sweep))]

fit_power_index_low_gain =[0.016, 0.016, 0.016, 0.016, 0.016, 0.016]
fit_power_index_high_gain =[0.205, 0.284, 0.189, 0.126, 0.237, 0.079]

for h5_file in h5_files:
    ## get the gain info
    power_value = float(h5_file.split('.h5')[0].split('_')[-1].replace('p', '.'))

    ## find its order in the list
    power_index = power_sweep.index(power_value)

    data = load_from_h5(h5_file, data_type='Res_ge')

    ## get the res data for the qubits, loop through each qubit
    for q_name, q_group in data.get("Res_ge", {}).items():
        qubit_index = int(q_name.split('Q')[-1]) -1

        ## if theres no data for this qubit from this file then continue
        if np.isnan(q_group['Amps'][0]).any():
            continue

        grab(Dates, q_group.get("Dates"), power_index, qubit_index)
        grab(freq_pts, q_group.get("freq_pts"), power_index, qubit_index)
        grab(freq_center, q_group.get("freq_center"), power_index, qubit_index)
        grab(Amps, q_group.get("Amps"), power_index, qubit_index)
        grab(Found_Freqs, q_group.get("Found Freqs"), power_index, qubit_index)
        grab(Round_Num, q_group.get("Round Num"), power_index, qubit_index)
        grab(Batch_Num, q_group.get("Batch Num"), power_index, qubit_index)
        grab(Exp_Config, q_group.get("Exp Config"), power_index, qubit_index)
        grab(Syst_Config, q_group.get("Syst Config"), power_index, qubit_index)

## now plot
plot_res_sweeps(data_file_path, freq_pts, freq_center, Amps, power_sweep, fit_power_index_low_gain, fit_power_index_high_gain)