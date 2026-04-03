import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import h5py
import re
import ast

def parse_array(arr_str):
    return np.fromstring(
        arr_str.replace('[','').replace(']','').replace(',',''),
        sep=' '
    )

def load_ssf_dataset(h5_file, q_key):
    '''
    Load single dataset from SSF h5 file and calculate amplitudes of g, e (centers and edges)
    '''

    with h5py.File(h5_file, "r") as f:
        fidelity = f[f'Q{q_key}']['Fidelity'][0]
        I_g = parse_array(f[f'Q{q_key}']['I_g'][0].decode())
        Q_g = parse_array(f[f'Q{q_key}']['Q_g'][0].decode())
        I_e = parse_array(f[f'Q{q_key}']['I_e'][0].decode())
        Q_e = parse_array(f[f'Q{q_key}']['Q_e'][0].decode())

    g_amps = np.sqrt(I_g**2 + Q_g**2)
    e_amps = np.sqrt(I_e**2 + Q_e**2)

    g_amps_mean = np.mean(g_amps)
    e_amps_mean = np.mean(e_amps)

    g_amps_max = np.max(g_amps)
    g_amps_min = np.min(g_amps)

    e_amps_max = np.max(e_amps)
    e_amps_min = np.min(e_amps)

    fidelity = 100*fidelity
    shots = len(g_amps)

    return g_amps_mean, g_amps_max, g_amps_min, e_amps_mean, e_amps_min, e_amps_max, fidelity, shots

def load_res_dataset(h5_file, q_key, data_type = "amps"):
    '''
    Load single dataset from resonator spec h5 file (GE or EF).
    data_type can be 'amps' or 'IQ'
    '''

    with h5py.File(h5_file, "r") as f:
        freq_pts = parse_array(f[f'Q{q_key}']['freq_pts'][0].decode())
        n_freq = len(freq_pts)

        freq_center = parse_array(f[f'Q{q_key}']['freq_center'][0].decode())
        freq_center = freq_center[q_key -1]

        freq_found = parse_array(f[f'Q{q_key}']['Found Freqs'][0].decode())
        freq_found = freq_found[(q_key-1)]

        syst_config = f[f"Q{q_key}/Syst Config"][:]
        text = syst_config[0].decode('utf-8')
        pattern = r"array\(\s*(\[[^\]]*\])\s*(?:,\s*dtype=[^)]+)?\)"
        text_normalized = re.sub(pattern, r"\1", text)
        cfg = ast.literal_eval(text_normalized)
        reps = cfg['reps']
        res_gain = cfg['res_gain_ge'][q_key-1]

        if data_type == 'amps':
            amps_flat = parse_array(f[f'Q{q_key}']['Amps'][0].decode())
            amps = amps_flat.reshape(-1, n_freq) #auto-detect resonator count
            amps_real = amps[q_key -1]
            return freq_pts, freq_center, freq_found, reps, res_gain, amps_real

        elif data_type == 'IQ':
            Iarr_flat = parse_array(f[f'Q{q_key}']['I'][0].decode())
            Iarr = Iarr_flat.reshape(-1, n_freq)
            Iarr_real = Iarr[q_key -1]

            Qarr_flat = parse_array(f[f'Q{q_key}']['Q'][0].decode())
            Qarr = Qarr_flat.reshape(-1, n_freq)
            Qarr_real = Qarr[q_key -1]

            return freq_pts, freq_center, freq_found, reps, res_gain, Iarr_real, Qarr_real

def plot_ge_ef_res(data_folder, save_folder, qubit, round = 0, data_type = 'amps', plot_ssf = False,save = False):
    ge_file = sorted(glob.glob(os.path.join(data_folder, "res_ge", "*.h5")))[round]
    ef_file = sorted(glob.glob(os.path.join(data_folder, "res_ef", "*.h5")))[round]

    if plot_ssf:
        ssf_file = sorted(glob.glob(os.path.join(data_folder, "ss_ge", "*.h5")))[round]
        g_mean, g_max, g_min, e_mean, e_max, e_min, fidelity, shots = load_ssf_dataset(ssf_file, qubit)

    if data_type == 'amps':
        fpts_ge, f_center_ge, f_found_ge, reps, res_gain, amps_ge = load_res_dataset(ge_file, qubit, data_type)
        fpts_ef, f_center_ef, f_found_ef, reps, res_gain, amps_ef = load_res_dataset(ef_file, qubit, data_type)

        plt.figure(figsize=(7,5))

        plt.plot([f + f_center_ge for f in fpts_ge], amps_ge, '-', linewidth = 1.5, color = 'tab:blue', label = "ge res spec")
        plt.plot([f + f_center_ef for f in fpts_ef], amps_ef, '-', linewidth = 1.5, color = 'tab:orange', label = "ef res spec")

        plt.axvline(f_found_ge, linestyle = '--', color = 'tab:blue', alpha = 0.6, label = f'g freq, {f_found_ge} MHz')
        plt.axvline(f_found_ef, linestyle = '--', color = 'tab:orange', alpha = 0.6, label = f'e freq, {f_found_ef} MHz')

        if plot_ssf:
            plt.axhline(g_mean, linestyle = ':', color = 'tab:blue', alpha = 0.6, label = f'ssf g state mean {g_mean}')
            plt.axhline(e_mean, linestyle = ':', color = 'tab:orange', alpha = 0.6, label = f'ssf e state mean {e_mean}')
            # plt.axhline(g_min, linestyle = ':', color = 'lightblue', alpha = 0.6, label = f'ssf g state min {g_min}')
            # plt.axhline(g_max, linestyle = ':', color = 'darkblue', alpha = 0.6, label = f'ssf g state max {g_max}')
            # plt.axhline(e_min, linestyle=':', color='gold', alpha=0.6, label=f'ssf e state min, {e_min}')
            # plt.axhline(e_max, linestyle=':', color='chocolate', alpha=0.6, label=f'ssf e state max {e_max}')


        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Amplitude (a.u.)')

        title = f"Q{qubit} Res GE ({f_found_ge} MHz) and EF ({f_found_ef} MHz), {reps} reps, {res_gain} gain"
        if plot_ssf:
            title += f"\n fidelity = {fidelity:.2f}%, {shots} SSF shots"
        plt.title(title)

        plt.legend()
        plt.tight_layout()
        if save:
            os.makedirs(save_folder, exist_ok = True)
            filename = os.path.join(save_folder, f"R{round}_Q{qubit}_res_ge_ef_amps.png")
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    elif data_type == 'IQ':
        fpts_ge, f_center_ge, f_found_ge, reps, res_gain, Iarr_ge, Qarr_ge = load_res_dataset(ge_file, qubit, data_type)
        fpts_ef, f_center_ef, f_found_ef, reps, res_gain, Iarr_ef, Qarr_ef = load_res_dataset(ef_file, qubit, data_type)

        plt.figure(figsize=(7,7))

        plt.plot(Iarr_ge, Qarr_ge, label='ge')
        plt.plot(Iarr_ef, Qarr_ef, label='ef')

        plt.xlabel('I Amplitude (a.u.)')
        plt.ylabel('Q Amplitude (a.u.)')

        plt.title(f"Q{qubit} Res GE ({f_found_ge} MHz) and EF ({f_found_ef} MHz), {reps} reps, {res_gain} gain")

        plt.legend()
        plt.tight_layout()
        if save:
            os.makedirs(save_folder, exist_ok = True)
            filename = os.path.join(save_folder, f"R{round}_Q{qubit}_res_ge_ef_IQcir.png")
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

run_name = 'run35'
device_name = '4charge'
study_name = 'Initial Checkout'
substudy_name = 'RR_ResSpecGE'

date = '2026-03-13_15-57-16'
outerFolder = f"/home/nexusadmin/Documents/Data/"
data_folder = os.path.join(outerFolder, run_name, device_name, study_name, substudy_name, date, 'study_data', 'Data_h5')
save_folder = os.path.join(outerFolder, run_name, device_name, study_name, substudy_name, date, 'documentation', 'analysis_plots')


#res = 2
round = 0
#plot_ge_ef_res(data_folder, save_folder, 1, round, 'amps', save=False)
for res in range(1, 2):
    plot_ge_ef_res(data_folder, save_folder, res, round, 'amps', plot_ssf= True, save= False)