import numpy as np
import os
from section_008_save_data_to_h5 import Data_H5
import glob
import matplotlib.pyplot as plt
import h5py
def load_res_dataset(h5_file, q_key, data_type = "amps"):
    '''
    Load single dataset from resonator spec h5 file (GE or EF).
    data_type can be 'amps' or 'IQ'
    '''

    with h5py.File(h5_file, "r") as f:

        freq_str = f[f'Q{q_key}']['freq_pts'][0].decode()
        freq_pts = np.fromstring(freq_str.strip('[]'), sep=' ')

        freq_center_str = f[f'Q{q_key}']['freq_center'][0].decode()
        freq_center = np.fromstring(freq_center_str.strip('[]'), sep=' ')

        freq_found_str = f[f'Q{q_key}']['Found Freqs'][0].decode()
        freq_found = np.fromstring(freq_found_str.strip('[]'), sep=' ')

    # H5 = Data_H5(h5_file)
    # data = H5.load_from_h5(data_type='Res', save_r = 0)
    # print("Dates:", data['Res'][q_key]['Dates'])
    # print("freq_pts:", data['Res'][q_key]['freq_pts'])
    #
    # print(data['Res'].keys())
    # print(data['Res'][q_key].keys())
    # print(data['Res'][q_key]['freq_pts'])
    #
    # freq_pts = data['Res'][q_key]['freq_pts'][0].decode()
    # freq_pts = np.array(eval(freq_pts))
    # freq_center = data['Res'][q_key]['freq_center'][0].decode()
    # freq_center = np.array(eval(freq_center))
    # freq_found = data['Res'][q_key]['Found Freqs'][0].decode()

        if data_type == 'amps':
            amps_str = f[f'Q{q_key}']['Amps'][0].decode()
            amps = np.fromstring(amps_str.strip('[]'), sep=' ')
            # amps = data['Res'][q_key]['Amps'][0][dataset].decode()
            # amps = np.array(eval(amps))[0]
            return freq_pts, freq_center, freq_found, amps
        elif data_type == 'IQ':
            Iarr_str = f[f'Q{q_key}']['I'][0].decode()
            Iarr = np.fromstring(Iarr_str.strip('[]'), sep=' ')
            # Iarr = data['Res'][q_key]['I'][0][dataset].decode()
            # Iarr = np.array(eval(Iarr))[0]

            Qarr_str = f[f'Q{q_key}']['Q'][0].decode()
            Qarr = np.fromstring(Qarr_str.strip('[]'), sep=' ')
            # Qarr = data['Res'][q_key]['Q'][0][dataset].decode()
            # Qarr = np.array(eval(Qarr))[0]
            return freq_pts, freq_center, freq_found, Iarr, Qarr

def plot_ge_ef_res(data_folder, qubit, round = 0, data_type = 'amps'):
    ge_file = sorted(glob.glob(os.path.join(data_folder, "res_ge", "*.h5")))[round]
    ef_file = sorted(glob.glob(os.path.join(data_folder, "res_ef", "*.h5")))[round]

    if data_type == 'amps':
        fpts_ge, f_center_ge, f_found_ge, amps_ge = load_res_dataset(ge_file, qubit, data_type)
        fpts_ef, f_center_ef, f_found_ef, amps_ef = load_res_dataset(ef_file, qubit, data_type)

        plt.figure(figsize=(7,5))

        plt.plot([f + f_center_ge for f in fpts_ge], amps_ge, '-', linewidth = 1.5, color = 'blue', label = "ge")
        plt.plot([f + f_center_ge for f in fpts_ef], amps_ef, '-', linewidth = 1.5, color = 'orange', label = "ef")

        plt.axvline(f_found_ge, linestyle = '--', color = 'blue', alpha = 0.4, label = 'ge_f')
        plt.axvline(f_found_ef, linestyle = '--', color = 'orange', alpha = 0.4, label = 'ef_f')

        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Amplitude (a.u.)')

        plt.title(f"Q{qubit} Res GE and EF")

        plt.legend()
        plt.tight_layout()
        plt.show()

    elif data_type == 'IQ':
        fpts_ge, f_center_ge, f_found_ge, Iarr_ge, Qarr_ge = load_res_dataset(ge_file, qubit, data_type)
        fpts_ef, f_center_ef, f_found_ef, Iarr_ef, Qarr_ef = load_res_dataset(ef_file, qubit, data_type)

        plt.figure(figsize=(7,7))

        plt.plot(Iarr_ge, Qarr_ge, label='ge')
        plt.plot(Iarr_ef, Qarr_ef, label='ef')

        plt.xlabel('I Amplitude (a.u.)')
        plt.ylabel('Q Amplitude (a.u.)')

        plt.title(f"Q{qubit} Res GE and EF")

        plt.legend()
        plt.tight_layout()
        plt.show()

run_name = 'run35'
device_name = '4charge'
study_name = 'Initial Checkout'
substudy_name = 'RR_ResSpecGE'

date = '2026-03-12_13-49-29'
outerFolder = f"/home/nexusadmin/Documents/Data/"
data_folder = os.path.join(outerFolder, run_name, device_name, study_name, substudy_name, date, 'study_data', 'Data_h5')

res = 1

plot_ge_ef_res(data_folder, 1, 0, 'amps')