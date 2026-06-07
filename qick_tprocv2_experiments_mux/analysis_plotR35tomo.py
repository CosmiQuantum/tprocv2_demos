import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import curve_fit
import os
from datetime import datetime
import json

def decode_time(timestamps, rows_written):
    # Decode bytes, strings, datetime objects
    dt = np.array([
        datetime.strptime(ts.decode("utf-8"), "%Y-%m-%d_%H-%M-%S")
        for ts in timestamps[:rows_written]
    ])
    # Convert to minutes since start
    time_minutes = np.array([
        (t - dt[0]).total_seconds() / 60.0
        for t in dt
    ])
    return time_minutes

def load_singleh5(filepath):
    with h5py.File(filepath, "r") as f:
        vsweep = f["vsweep"][:]         #npts
        qdata = f["qdata"][:]           #(rounds, n_qubits, 2, npts)
        qubits = f["qubits"][:]         #qubits measured ([1, 2, 3, 4], etc)
        timestamps = f["timestamps"][:] #rounds
        rows_written = f.attrs["rows_written"]

    return vsweep, qubits, qdata[:rows_written], timestamps[:rows_written], rows_written

def tomo_colorplot(vsweep, qindex, qid, qdata, rows_written, timestamps = None):
    '''
    qubit : [q_index, q_id (name)]
    '''

    I_data = qdata[:, qindex, 0, :].T
    Q_data = qdata[:, qindex, 1, :].T

    vsweep_mV = vsweep * 1000

    # X-axis handling
    if timestamps is not None:
        time_min = decode_time(timestamps, rows_written)

        x_vals = time_min
        x_label = "Time (min)"
    else:
        x_vals = np.arange(rows_written)
        x_label = "Round"

    fig, (axI, axQ) = plt.subplots(2, 1, figsize=(10,8), sharex=True)

    imI = axI.imshow(I_data, aspect='auto', origin='lower',
                     extent = [x_vals[0], x_vals[-1], vsweep_mV[0], vsweep_mV[-1]])
    axI.set_ylabel('Voltage Bias (mV)')
    cbarI = fig.colorbar(imI, ax=axI)
    cbarI.set_label("I Amplitude (a.u.)")

    imQ = axQ.imshow(Q_data, aspect='auto', origin='lower',
                     extent=[x_vals[0], x_vals[-1], vsweep_mV[0], vsweep_mV[-1]])
    axQ.set_ylabel('Applied Voltage Bias (mV)')
    axQ.set_xlabel(x_label)
    cbarQ = fig.colorbar(imQ, ax=axQ)
    cbarQ.set_label("Q Amplitude (a.u.)")

    fig.suptitle(f'Charge Tomography Q{qid}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    plt.show()


def tomo_colorplot_amp(vsweep, qindex, qid, qdata, rows_written, timestamps = None):
    '''
    qubit : [q_index, q_id (name)]
    '''

    I_data = qdata[:, qindex, 0, :].T
    Q_data = qdata[:, qindex, 1, :].T

    amps = np.sqrt(I_data**2 + Q_data**2)

    vsweep_mV = vsweep * 1000

    # X-axis handling
    if timestamps is not None:
        time_min = decode_time(timestamps, rows_written)

        x_vals = time_min
        x_label = "Time (min)"
    else:
        x_vals = np.arange(rows_written)
        x_label = "Round"

    # Single plot
    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(
        amps,
        aspect='auto',
        origin='lower',
        extent=[x_vals[0], x_vals[-1], vsweep_mV[0], vsweep_mV[-1]]
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel('Applied Voltage Bias (mV)')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Amplitude (a.u.)')

    ax.set_title(f'Charge Tomography Q{qid}', fontsize=14)

    plt.tight_layout()
    plt.show()
    #fig_name = os.path.join('/home/nexusadmin/Documents/Data/run35/4charge/PostCsTomography/Dataset2/2026-03-23_16-15-38/analysis_plots/', 'Tomography_Qs1234_2026-03-24_17-06-21_plot.png') #Hardcode, fix
    #plt.savefig(fig_name)


def tomo_rndplot(vsweep, qindex, qid, qdata, round):
    fig, (axI, axQ) = plt.subplots(2, 1, figsize=(10,8), sharex='all')
    axI.set_ylabel("I Amplitude (a.u.)")
    axI.tick_params(axis='both', which='major', labelsize=14)
    axQ.set_ylabel("Q Amplitude (a.u.)")
    axQ.set_xlabel("Applied Voltage Bias (mV)")
    if type(qindex) is int:
        if type(round) is int:
            axI.plot(vsweep * 1000, qdata[round, qindex, 0, :])
            axQ.plot(vsweep * 1000, qdata[round, qindex, 1, :])
            fig.suptitle(f"Charge Tomography Q{qid}, Round {round}")
        else:
            for r in range(round[0], round[1]):
                axI.plot(vsweep * 1000, qdata[r, qindex, 0, :], label = f"rnd {r}")
                axQ.plot(vsweep * 1000, qdata[r, qindex, 1, :], label = f"rnd {r}")
            fig.suptitle(f"Charge Tomography Q{qid}, Rounds {round[0]}-{round[1]}")
            axI.legend()
            axQ.legend()
    else:
        for index, q in enumerate(qindex):
            axI.plot(vsweep * 1000, qdata[round, q, 0, :], label=f"Q{qid[index]}")
            axQ.plot(vsweep * 1000, qdata[round, q, 1, :], label=f"Q{qid[index]}")
        fig.suptitle(f"Charge Tomography, Round {round}")
        axI.legend()
        axQ.legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    plt.show()

def fit_func(n_g, d, nu, Vconv, phi):
        P = d + nu*np.cos(np.pi*np.cos(2*np.pi*Vconv*n_g - phi))
        return P

def guessfit(data):
    nvals = 5
    fit_params = {}

    sort_data = np.sort(data)
    avg_min = np.mean(sort_data[:nvals])
    avg_max = np.mean(sort_data[-nvals:])
    fit_params["g_d"] = (avg_max + avg_min)/2
    fit_params["g_nu"] = (avg_max - avg_min)/2
    g_d = (avg_max + avg_min) / 2
    g_nu = (avg_max - avg_min) / 2
    fit_params["g_Vconv"] = (1/80) #1 e / period in mV
    #
    fit_params["g_phi"] = np.pi
    return fit_params

def fitdata(fit_params, vsweep, qindex, qid, qdata, rd, signal = "amp", plot = False, saveFolder = None):
    if signal == "I":
        data = qdata[rd, qindex, 0, :]
    elif signal == "Q":
        data = qdata[rd, qindex, 1, :]
    elif signal == "amp":
        I = qdata[rd, qindex, 0, :]
        Q = qdata[rd, qindex, 1, :]
        data = np.sqrt(I**2 + Q**2)
    else:
        print('Incorrect signal input. Must be "I", "Q", or "amp".')
        return

    g_d = fit_params["g_d"],
    g_nu = fit_params["g_nu"]
    g_Vconv = fit_params["g_Vconv"]
    g_phi = fit_params["g_phi"]

    bounds = ((g_d-0.5, 1.5*g_nu, 0.5*g_Vconv, 0), (g_d+0.5, 0.2*g_nu, 1.2*g_Vconv, 2*np.pi))
    popt, pcov = curve_fit(fit_func, vsweep, data, p0 = (g_d, g_nu, g_Vconv, g_phi), bounds = bounds)

    y_fit = fit_func(vsweep, *popt)
    residuals = data - y_fit
    chi_squared = np.sum((residuals / 1)**2)

    fit_params["f_d"] = popt[0]
    fit_params["f_nu"] = popt[1]
    fit_params["f_Vconv"] = popt[2]
    fit_params["f_phi"] = popt[3]
    fit_params["f_chi2"] = chi_squared
    fit_params["f_err"] = np.sqrt(np.diag(pcov))

    if plot:
        plt.plot(vsweep, data, '.', label = 'Data')
        plt.plot(vsweep, fit_func(vsweep, fit_params["f_d"], fit_params["f_nu"], fit_params["f_Vconv"], fit_params["f_phi"]), label='Fit')
        plt.legend()
        plt.xlabel("Applied Voltage Bias (mV)")
        plt.ylabel("Amplitude (a.u.)")
        plt.title(f'Fit Charge Tomography {qid} \n d = {fit_params["f_d"]:.2f}, nu = {fit_params["f_nu"]:.2f}, Vconv = {fit_params["f_Vconv"]:.4f}, phi = {fit_params["f_phi"]:.2f}')
        plt.show()
    return fit_params

#def fit_single_scan()

run = 'run36a'
study = 'BackgroundTomography' #'Tomography_Check' #'PostCsTomography' #'EndOfRunData' #'PostCsTomography' #'BackgroundTomography'
substudy = 'Dataset1' #'HighgainOpt' #'Dataset2' #'Dataset2_neg'
timestamp = '2026-04-25_13-45-53' #'2026-03-23_16-15-38' #'2026-04-03_11-55-28'
file = 'Tomography_Qs1234_2026-04-27_02-50-42.h5' #'Tomography_Qs1234_2026-04-23_19-36-26.h5' #'Tomography_Qs1234_2026-03-24_17-06-21.h5'
path = f'/home/nexusadmin/Documents/Data/{run}/4charge/{study}/{substudy}/{timestamp}/study_data/{file}'

vsweep, qubits, qdata, timestamps, rounds = load_singleh5(path)
print(rounds)
#print(timestamps)
qindex = 0
qid = 1
rd = 0
#guess_params = guessfit(qdata) #Need to fix which data to look (which rd, which qubit, etc) before trying
#full_fitparams = fitdata(guess_params, vsweep, qindex, qid, qdata, rd, signal = "amp", plot = True)
for qindex in range(0, 4):
    tomo_colorplot(vsweep, qindex, qindex+1, qdata, rounds, timestamps = timestamps)
    #tomo_colorplot_amp(vsweep, qindex, qindex+1, qdata, rounds, timestamps = timestamps)
#tomo_rndplot(vsweep, qindex, qid, qdata, [10, 30])