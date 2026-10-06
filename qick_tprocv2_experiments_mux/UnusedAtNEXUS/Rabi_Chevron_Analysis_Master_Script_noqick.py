import h5py
import numpy as np
import matplotlib.pyplot as plt
######################################################### Functions ########################################################################
class combined_Qtemp_studies:
    def __init__(self, figure_quality = 200):
        self.figure_quality = figure_quality

    def qubit_of(self, h5_path):
        """ Tells you which qubit is stored inside the h5 file. Assumes only one qubit is stored inside. """
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                # match "Q1", "Q2", ...
                if key.startswith("Q") and key[1:].isdigit():
                    return int(key[1:]) - 1
        raise ValueError(f"No Q<digit> group in {h5_path}")

    def process_h5_data(self,data):
        # Check if the data is a byte string; decode if necessary.
        if isinstance(data, bytes):
            data_str = data.decode()
        elif isinstance(data, str):
            data_str = data
        else:
            raise ValueError("Unsupported data type. Data should be bytes or string.")

        # Remove anything that isnt part of a float literal
        cleaned = ''.join(c for c in data_str if c.isdigit() or c in ['-', '.', ' ', 'e'])
        # Split on whitespace and convert
        return [float(x) for x in cleaned.split() if x]

    def plot_rabi_chevron(self,h5_path):
        with h5py.File(h5_path, 'r') as f:
            # find the only Q* group
            grp_name = [k for k in f if k.startswith('Q')][0]
            qub = f[grp_name]

            # pull out the raw byte-blobs
            raw_I = qub['I'][0]  # something like b'[[0.1 0.2 ...]\n [ ... ]]'
            raw_Q = qub['Q'][0]
            raw_gains = qub['Gains'][0]
            raw_freqs = qub['Freqs_MHz'][0]

        # process them into flat Python lists
        flat_I = self.process_h5_data(raw_I)
        flat_Q = self.process_h5_data(raw_Q)
        gains_list = self.process_h5_data(raw_gains)
        freqs_list = self.process_h5_data(raw_freqs)

        # reshape into 2D arrays
        n_freqs = len(freqs_list)
        n_gains = len(gains_list)
        I = np.array(flat_I).reshape(n_freqs, n_gains)
        Q = np.array(flat_Q).reshape(n_freqs, n_gains)
        gains = np.array(gains_list)
        freqs = np.array(freqs_list)

        # compute magnitude & plot
        mag = np.sqrt(I ** 2 + Q ** 2)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            mag,
            origin='lower',
            aspect='auto',
            extent=[gains[0], gains[-1], freqs[0], freqs[-1]]
        )
        ax.set_xlabel('Gain (arb. units)')
        ax.set_ylabel('Drive frequency (MHz)')
        ax.set_title(f'Rabi Chevron: {grp_name}')
        plt.colorbar(im, ax=ax, label='v(I²+Q²)')
        plt.tight_layout()
        plt.show()
####################################################################################################################################
twoD_Rabisweeps = combined_Qtemp_studies(figure_quality = 200)

h5_path_2drabi = '/data/QICK_data/run6/6transmon/gain_rabi_ge_qfreq_study/rabi_Qfreq_2Dsweeps_wshots_default_sigmas/2025-05-21_16-06-23/study_data/Data_h5/rabi_ge_chevron_wshots/2025-05-21_16-20-58_rabi_ge_chevron_wshots_results_batch_0_Num_per_batch1.h5'
twoD_Rabisweeps.plot_rabi_chevron(h5_path_2drabi)