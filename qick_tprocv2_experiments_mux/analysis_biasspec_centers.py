import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def load_data(dataFolder):
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

    return freq, Iarr, Qarr, amp, vsweep

def background_subtract(Iarr, Qarr, amp):
    I_bs = np.zeros_like(Iarr)
    Q_bs = np.zeros_like(Qarr)
    amp_bs = np.zeros_like(amp)
    for i in range(0, len(Iarr)):
        I_bs[i] = Iarr[i] - np.mean(Iarr[i])
        Q_bs[i] = Qarr[i] - np.mean(Qarr[i])
        amp_bs[i] = amp[i] - np.mean(amp[i])
    return I_bs, Q_bs, amp_bs

def plot_1d_sweeps(freq, vsweep, Iarr, Qarr, amp, qubit, x1, x2, xc, timestamp, bkgd_sub = False, save = False, save_path = None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex='all')
    ax1.set_ylabel('I Amplitude (a.u.)', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax2.set_ylabel('Q Amplitude (a.u.)', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax3.set_ylabel('Magnitude (a.u.)', fontsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=16)

    if bkgd_sub:
        I_plot, Q_plot, amp_plot = background_subtract(Iarr, Qarr, amp)
    else:
        I_plot, Q_plot, amp_plot = Iarr, Qarr, amp

    for volt_index in range(len(vsweep)):
        ax1.plot(freq, I_plot[volt_index])
        ax2.plot(freq, Q_plot[volt_index])
        ax3.plot(freq, amp_plot[volt_index])

    for ax in (ax1, ax2, ax3):
        ax.axvline(x1[qubit - 1], color='r', linestyle='--')
        ax.axvline(x2[qubit - 1], color='r', linestyle='--')
        ax.axvline(xc[qubit - 1], color='k', linestyle='--')

    if bkgd_sub:
        fig.suptitle(
            f'Q{qubit} Bias Spec Bkgd Sub \n f1 = {x1[qubit - 1]}, f2 = {x2[qubit - 1]}, fc = {xc[qubit - 1]} MHz')
    else:
        fig.suptitle(f'Q{qubit} Bias Spec \n f1 = {x1[qubit - 1]}, f2 = {x2[qubit - 1]}, fc = {xc[qubit - 1]} MHz')
    plt.tight_layout()

    plt.subplots_adjust(top=0.90)
    if save and save_path is not None:
        if bkgd_sub:
            file_path = os.path.join(save_path, f"{timestamp}_Q{qubit}_annotatedsweeps_bkgdsub.png")
        else:
            file_path = os.path.join(save_path, f"{timestamp}_Q{qubit}_annotatedsweeps.png")
        plt.savefig(file_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_2d_bias(freq, vsweep, Iarr, Qarr, amp, qubit, x1, x2, xc, timstamp, bkgd_sub = False, save = False, save_path = None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,10), sharex='all')
    if bkgd_sub:
        I_plot, Q_plot, amp_plot = background_subtract(Iarr, Qarr, amp)
    else:
        I_plot, Q_plot, amp_plot = Iarr, Qarr, amp

    extent = [float(freq[0]), float(freq[-1]), vsweep[0], vsweep[-1]]

    ax1.set_ylabel("Voltage Bias (V)", fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    im1 = ax1.imshow(I_plot, aspect = 'auto', origin = 'lower', extent=extent)
    fig.colorbar(im1, label="I Amplitude (a.u.)")

    ax2.set_ylabel("Voltage Bias (V)", fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    im2 = ax2.imshow(Q_plot, aspect = 'auto', origin = 'lower', extent=extent)
    fig.colorbar(im2, label="Q Amplitude (a.u.)")

    ax3.set_ylabel("Voltage Bias (V)", fontsize=16)
    ax3.set_xlabel("Frequency (MHz)", fontsize=16)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    im3 = ax3.imshow(amp_plot, aspect='auto', origin='lower', extent=extent)
    fig.colorbar(im3, label="Magnitude (a.u.)")

    for ax in (ax1, ax2, ax3):
        ax.axvline(x1[qubit - 1], color='r', linestyle='--')
        ax.axvline(x2[qubit - 1], color='r', linestyle='--')
        ax.axvline(xc[qubit - 1], color='k', linestyle='--')

    if bkgd_sub:
        fig.suptitle(f"Bias Spec for Q{qubit} Bkgd Sub \n f1 = {x1[qubit - 1]}, f2 = {x2[qubit - 1]}, fc = {xc[qubit - 1]} MHz")
    else:
        fig.suptitle(f"Bias Spec for Q{qubit} \n f1 = {x1[qubit-1]}, f2 = {x2[qubit-1]}, fc = {xc[qubit-1]} MHz")
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    if save and save_path is not None:
        if bkgd_sub:
            file_path = os.path.join(plots_folder, f"{timestamp}_Q{qubit}_annotated2dsweeps_bkgdsub.png")
        else:
            file_path = os.path.join(plots_folder, f"{timestamp}_Q{qubit}_annotated2dsweeps.png")
        plt.savefig(file_path, dpi=300)
        plt.close()
    else:
        plt.show()

def fit_single_lorenzian(I, Q, amp, freqs, freq_q, volt, volt_index, sigma_guess = 1, plot_fit = False, save = False, save_path = None, name_mod = None):
    def lorentzian(f, f0, gamma, A, B):
        return A * gamma ** 2 / ((f - f0) ** 2 + gamma ** 2) + B

    def max_offset_difference_with_x(x_values, y_values, offset):
        max_average_difference = -1
        corresponding_x = None
        # average all 3 to avoid noise spikes
        for i in range(len(y_values) - 2):
            # group 3 vals
            y_triplet = y_values[i:i + 3]

            # avg differences for these 3 vals
            average_difference = sum(abs(y - offset) for y in y_triplet) / 3

            # see if this is the highest difference yet
            if average_difference > max_average_difference:
                max_average_difference = average_difference
                # x value for the middle y value in the 3 vals
                corresponding_x = x_values[i + 1]

        return corresponding_x, max_average_difference

    try:
        # Initial guesses for I and Q
        initial_guess_I = [freq_q, sigma_guess, np.max(I), np.min(I)]
        initial_guess_Q = [freq_q, sigma_guess, np.max(Q), np.min(Q)]

        # First round of fits (to get rough estimates)
        params_I, _ = curve_fit(lorentzian, freqs, I, p0=initial_guess_I)
        params_Q, _ = curve_fit(lorentzian, freqs, Q, p0=initial_guess_Q)

        # Use these fits to refine guesses
        x_max_diff_I, max_diff_I = max_offset_difference_with_x(freqs, I, params_I[3])
        x_max_diff_Q, max_diff_Q = max_offset_difference_with_x(freqs, Q, params_Q[3])
        initial_guess_I = [x_max_diff_I, sigma_guess, np.max(I), np.min(I)]
        initial_guess_Q = [x_max_diff_Q, sigma_guess, np.max(Q), np.min(Q)]

        # Second (refined) round of fits, this time capturing the covariance matrices
        params_I, cov_I = curve_fit(lorentzian, freqs, I, p0=initial_guess_I)
        params_Q, cov_Q = curve_fit(lorentzian, freqs, Q, p0=initial_guess_Q)

        # Create the fitted curves
        I_fit = lorentzian(freqs, *params_I)
        Q_fit = lorentzian(freqs, *params_Q)

        # Calculate errors from the covariance matrices
        fit_err_I = np.sqrt(np.diag(cov_I))
        fit_err_Q = np.sqrt(np.diag(cov_Q))

        # Extract fitted means and FWHM (assuming params[0] is the mean and params[1] relates to the width)
        mean_I = params_I[0]
        mean_Q = params_Q[0]
        fwhm_I = 2 * params_I[1]
        fwhm_Q = 2 * params_Q[1]

        # Calculate the amplitude differences from the fitted curves
        amp_I_fit = abs(np.max(I_fit) - np.min(I_fit))
        amp_Q_fit = abs(np.max(Q_fit) - np.min(Q_fit))

        # Choose which curve to use based on the input signal indicator
        if amp_I_fit > amp_Q_fit:
            largest_amp_curve_mean = mean_I
            largest_amp_curve_fwhm = fwhm_I
            # error on the Q fit's center frequency (first parameter):
            qspec_fit_err = fit_err_I[0]
        else:
            largest_amp_curve_mean = mean_Q
            largest_amp_curve_fwhm = fwhm_Q
            qspec_fit_err = fit_err_Q[0]

    except Exception as e:
        print("Error during Lorentzian fit:", e)
        return

    if plot_fit:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex='all')
        ax1.set_ylabel('I Amplitude (a.u.)', fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax2.set_ylabel('Q Amplitude (a.u.)', fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax3.set_ylabel('Magnitude (a.u.)', fontsize=20)
        ax3.tick_params(axis='both', which='major', labelsize=16)

        ax1.plot(freqs, I, label = 'Data')
        ax1.plot(freqs, I_fit, 'r--', label='Lorentzian Fit')
        ax1.axvline(largest_amp_curve_mean, color = 'orange', linestyle='--')
        ax2.plot(freqs, Q)
        ax2.plot(freqs, Q_fit, 'r--')
        ax2.axvline(largest_amp_curve_mean, color='orange', linestyle='--')
        ax3.plot(freqs, amp)
        ax3.axvline(largest_amp_curve_mean, color='orange', linestyle='--')

        for ax in (ax1, ax2, ax3):
            ax.axvline(freq_q, color='k', linestyle='--')

        fig.suptitle(f'Q{qubit} Bias Spec ({name_mod}), {np.round(volt, 4)} V \n Guess: {freq_q}, Fit: {round(largest_amp_curve_mean, 5)} MHz, FWHM: {round(largest_amp_curve_fwhm, 1)}')
        plt.tight_layout()

        plt.subplots_adjust(top=0.90)
        if save and save_path is not None:
            file_path = os.path.join(save_path, f"{timestamp}_Q{qubit}_{volt_index}fitsweep_{name_mod}.png")
            plt.savefig(file_path, dpi=300)
            plt.close()
        else:
            plt.show()

    # Return all desired results including the error on the Q fit
    return mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, qspec_fit_err


def split_scan(freq, Iarr, Qarr, amp, split_freq = None):
    # if split_freq is None:
    #     split_freq = 0.5 * (freq[0] + freq[-1])
    split_index = len(freq) //2

    pt1_freq = freq[:split_index]
    pt1_Iarr = Iarr[:split_index]
    pt1_Qarr = Qarr[:split_index]
    pt1_amp = amp[:split_index]

    pt2_freq = freq[split_index:]
    pt2_Iarr = Iarr[split_index:]
    pt2_Qarr = Qarr[split_index:]
    pt2_amp = amp[split_index:]

    return pt1_freq, pt1_Iarr, pt1_Qarr, pt1_amp, pt2_freq, pt2_Iarr, pt2_Qarr, pt2_amp

run_name = 'run37'
study = 'Initial Checkout'
qubit = 3 #1, 2, 3, or 4
substudy = "Q2Q3_BiasSpec" #f'Q{qubit}_BiasSpec'

date = '2026-06-09'
timestamp = f'{date}_10-59-13'

dataFolder = f"/home/nexusadmin/Documents/Data/{run_name}/4charge/{study}/{substudy}/{date}/{timestamp}"
print(dataFolder)

plot_sweep = True
plot_2d = True
bkgd_sub = True
save_plots = True
fit = False

x1 = [4930, 4787.55, 4582.55, 4799.8] #
x2 = [4931.8, 4790.1, 4586.45, 4802.3]
xc = [4930.9, 4788.8, 4584.5, 4801.1]

freq, Iarr, Qarr, amp, vsweep = load_data(dataFolder)

if save_plots:
    plots_folder = dataFolder + "/analysis_plots/"
    os.makedirs(plots_folder, exist_ok=True)
else:
    plots_folder = None

plt.rcParams.update({
            'font.size': 14,  # Base font size
            'axes.titlesize': 18,  # Title font size
            'axes.labelsize': 16,  # Axis label font size
            'xtick.labelsize': 14,  # X-axis tick label size
            'ytick.labelsize': 14,  # Y-axis tick label size
            'legend.fontsize': 14,  # Legend font size
        })

if plot_sweep:
    plot_1d_sweeps(freq, vsweep, Iarr, Qarr, amp, qubit, x1, x2, xc, timestamp, bkgd_sub = bkgd_sub, save=save_plots, save_path=plots_folder)
if plot_2d:
    plot_2d_bias(freq, vsweep, Iarr, Qarr, amp, qubit, x1, x2, xc, timestamp, bkgd_sub = bkgd_sub, save=save_plots, save_path=plots_folder)

if fit:
    plot_fit = True
    split = False
    volt_index = 10
    I = Iarr[volt_index]
    Q = Qarr[volt_index]
    amp = amp[volt_index]
    volt = vsweep[volt_index]

    if split:
        pt1_freq, pt1_Iarr, pt1_Qarr, pt1_amp, pt2_freq, pt2_Iarr, pt2_Qarr, pt2_amp = split_scan(freq, I, Q, amp)
        print(len(pt1_freq), len(pt1_Iarr), len(pt1_Qarr), len(pt1_amp))
        print(len(pt2_freq), len(pt2_Iarr), len(pt2_Qarr), len(pt2_amp))


        pt1_freq_q = x1[qubit-1]
        mod_pt1 = 'Left'
        pt2_freq_q = x2[qubit-1]
        mod_pt2 = 'Right'
        (mean_I1, mean_Q1, I_fit1, Q_fit1,
         largest_amp_curve_mean1, largest_amp_curve_fwhm1,
         qspec_fit_err1) = fit_single_lorenzian(pt1_Iarr, pt1_Qarr, pt1_amp, pt1_freq, pt1_freq_q, volt, volt_index, plot_fit=plot_fit, save=save_plots,
                                               save_path=plots_folder, name_mod=mod_pt1)
        print("Left")
        print(mean_I1)
        print(mean_Q1)
        print(largest_amp_curve_mean1)
        print(largest_amp_curve_fwhm1)

        (mean_I2, mean_Q2, I_fit2, Q_fit2,
         largest_amp_curve_mean2, largest_amp_curve_fwhm2,
         qspec_fit_err2) = fit_single_lorenzian(pt2_Iarr, pt2_Qarr, pt2_amp, pt2_freq, pt2_freq_q, volt, volt_index,
                                                plot_fit=plot_fit, save=save_plots,
                                                save_path=plots_folder, name_mod=mod_pt2)
        print("Right")
        print(mean_I2)
        print(mean_Q2)
        print(largest_amp_curve_mean2)
        print(largest_amp_curve_fwhm2)

    else:
        freq_q = xc[qubit-1]

        mod = 'Full'
        mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, qspec_fit_err = fit_single_lorenzian(I, Q, amp, freq, freq_q, volt, volt_index, plot_fit = plot_fit, save = save_plots,
                                               save_path = plots_folder, name_mod = mod)
        print(mean_I)
        print(mean_Q)
        print(largest_amp_curve_mean)
        print(largest_amp_curve_fwhm)