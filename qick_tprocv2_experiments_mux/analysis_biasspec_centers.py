import os
import numpy as np
import csv
import glob
import matplotlib.pyplot as plt

run_name = 'run35'
study = 'Initial Checkout'
qubit = 4 #1, 2, 3, or 4
substudy = f'Q{qubit}_BiasSpec'

date = '2026-02-24'
timestamp = f'{date}_10-03-03'

plot_sweep = True
plot_2d = True
bkgd_sub = False
save_plots = True

x1 = [4942.6, 4772.2, 4581, 4804.15]
x2 = [4944.3, 4774.7, 4576.9, 4806.55]
xc = [4943.4, 4773.45, 4578.95, 4805.35]

dataFolder = f"/home/nexusadmin/Documents/Data/{run_name}/4charge/{study}/{substudy}/{date}/{timestamp}"
freq_file = glob.glob(os.path.join(dataFolder, "*freqarr.csv"))[0]
I_file = glob.glob(os.path.join(dataFolder, "*Iarr.csv"))[0]
Q_file = glob.glob(os.path.join(dataFolder, "*Qarr.csv"))[0]
amp_file = glob.glob(os.path.join(dataFolder, "*amparr.csv"))[0]
vsweep_file = glob.glob(os.path.join(dataFolder, "*vsweep.csv"))[0]

freq = np.loadtxt(freq_file, delimiter=",")
Iarr = np.loadtxt(I_file, delimiter=",")
Qarr = np.loadtxt(Q_file, delimiter=",")
amp = np.loadtxt(amp_file, delimiter=",")
vsweep = np.loadtxt(vsweep_file, delimiter=",")
I_bs = np.zeros_like(Iarr)
Q_bs = np.zeros_like(Qarr)
amp_bs = np.zeros_like(amp)
for i in range(0, len(Iarr)):
    I_bs[i] = Iarr[i] - np.mean(Iarr[i])
    Q_bs[i] = Qarr[i] - np.mean(Qarr[i])
    amp_bs[i] = amp[i] - np.mean(amp[i])

plt.rcParams.update({
            'font.size': 14,  # Base font size
            'axes.titlesize': 18,  # Title font size
            'axes.labelsize': 16,  # Axis label font size
            'xtick.labelsize': 14,  # X-axis tick label size
            'ytick.labelsize': 14,  # Y-axis tick label size
            'legend.fontsize': 14,  # Legend font size
        })

if plot_sweep:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,10), sharex='all')
    ax1.set_ylabel('I Amplitude (a.u.)', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax2.set_ylabel('Q Amplitude (a.u.)', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax3.set_ylabel('Amplitude (a.u.)', fontsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=16)

    for volt_index in range(len(vsweep)):
        if bkgd_sub:
            ax1.plot(freq, I_bs[volt_index])
            ax2.plot(freq, Q_bs[volt_index])
            ax3.plot(freq, amp_bs[volt_index])
        else:
            ax1.plot(freq, Iarr[volt_index])
            ax2.plot(freq, Qarr[volt_index])
            ax3.plot(freq, amp[volt_index])

    ax1.axvline(x1[qubit-1], color='r', linestyle = '--')
    ax1.axvline(x2[qubit-1], color = 'r', linestyle = '--')
    ax1.axvline(xc[qubit-1], color = 'k', linestyle = '--')

    ax2.axvline(x1[qubit-1], color='r', linestyle='--')
    ax2.axvline(x2[qubit-1], color='r', linestyle='--')
    ax2.axvline(xc[qubit-1], color='k', linestyle='--')

    ax3.axvline(x1[qubit-1], color='r', linestyle='--')
    ax3.axvline(x2[qubit-1], color='r', linestyle='--')
    ax3.axvline(xc[qubit-1], color='k', linestyle='--')

    if bkgd_sub:
        fig.suptitle(f'Q{qubit} Bias Spec Bkgd Sub \n f1 = {x1[qubit - 1]}, f2 = {x2[qubit - 1]}, fc = {xc[qubit - 1]} MHz')
    else:
        fig.suptitle(f'Q{qubit} Bias Spec \n f1 = {x1[qubit-1]}, f2 = {x2[qubit-1]}, fc = {xc[qubit-1]} MHz')
    plt.tight_layout()

    plt.subplots_adjust(top = 0.93)
    if save_plots:
        plots_folder = dataFolder + "/analysis_plots/"
        os.makedirs(plots_folder, exist_ok=True)
        if bkgd_sub:
            file_path = os.path.join(plots_folder, f"{timestamp}_Q{qubit}_annotatedsweeps_bkgdsub.png")
        else:
            file_path = os.path.join(plots_folder, f"{timestamp}_Q{qubit}_annotatedsweeps.png")
        plt.savefig(file_path, dpi=300)
        plt.close()
    else:
        plt.show()

if plot_2d:
    if bkgd_sub:
        I_plot = I_bs
        Q_plot = Q_bs
        amp_plot = amp_bs
    else:
        I_plot = Iarr
        Q_plot = Qarr
        amp_plot = amp_bs
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,10), sharex='all')
    ax1.set_ylabel("Voltage Bias (V)", fontsize = 16)
    ax1.tick_params(axis='both', which = 'major', labelsize=14)
    im1 = ax1.imshow(I_plot, aspect='auto', origin='lower',
                     extent = [float(freq[0]), float(freq[-1]), vsweep[0], vsweep[-1]])
    fig.colorbar(im1, label="I Amplitude (a.u.)")
    ax2.set_ylabel("Voltage Bias (V)", fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    im2 = ax2.imshow(Q_plot, aspect='auto', origin='lower',
                     extent=[float(freq[0]), float(freq[-1]), vsweep[0], vsweep[-1]])
    fig.colorbar(im2, label="Q Amplitude (a.u.)")
    ax3.set_ylabel("Voltage Bias (V)", fontsize = 16)
    ax3.tick_params(axis='both', which = 'major', labelsize=14)
    im3 = ax3.imshow(amp_plot, aspect='auto', origin='lower',
                     extent=[float(freq[0]), float(freq[-1]), vsweep[0], vsweep[-1]])
    fig.colorbar(im3, label="I Amplitude (a.u.)")

    ax1.axvline(x1[qubit - 1], color='r', linestyle='--')
    ax1.axvline(x2[qubit - 1], color='r', linestyle='--')
    ax1.axvline(xc[qubit - 1], color='k', linestyle='--')

    ax2.axvline(x1[qubit - 1], color='r', linestyle='--')
    ax2.axvline(x2[qubit - 1], color='r', linestyle='--')
    ax2.axvline(xc[qubit - 1], color='k', linestyle='--')

    ax3.axvline(x1[qubit - 1], color='r', linestyle='--')
    ax3.axvline(x2[qubit - 1], color='r', linestyle='--')
    ax3.axvline(xc[qubit - 1], color='k', linestyle='--')

    if bkgd_sub:
        fig.suptitle(f"Bias Spec for Q{qubit} Bkgd Sub \n f1 = {x1[qubit - 1]}, f2 = {x2[qubit - 1]}, fc = {xc[qubit - 1]} MHz")
    else:
        fig.suptitle(f"Bias Spec for Q{qubit} \n f1 = {x1[qubit-1]}, f2 = {x2[qubit-1]}, fc = {xc[qubit-1]} MHz")
    plt.tight_layout()
    plt.subplots_adjust(top=0.89)

    if save_plots:
        plots_folder = dataFolder + "/analysis_plots/"
        os.makedirs(plots_folder, exist_ok=True)
        if bkgd_sub:
            file_path = os.path.join(plots_folder, f"{timestamp}_Q{qubit}_annotated2dsweeps_bkgdsub.png")
        else:
            file_path = os.path.join(plots_folder, f"{timestamp}_Q{qubit}_annotated2dsweeps.png")
        plt.savefig(file_path, dpi=300)
        plt.close()
    else:
        plt.show()