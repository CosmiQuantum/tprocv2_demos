import h5py
import numpy as np
import matplotlib.pyplot as plt

######################################################### Functions ########################################################################
class combined_Qtemp_studies:
    def __init__(self, figure_quality):
        self.figure_quality = figure_quality

    def qubit_of(self, h5_path):
        """ Tells you which qubit is stored inside the h5 file. Assumes only one qubit is stored inside. """
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                # match "Q1", "Q2", ...
                if key.startswith("Q") and key[1:].isdigit():
                    return int(key[1:]) - 1
        raise ValueError(f"No Q<digit> group in {h5_path}")

    def plot_rabi_chevron(self, h5_path):
        """
        Re-plot the Rabi chevron from an HDF5 file whose top level is 'Q1','Q2',...
        and where each group has datasets 'I','Q','Gains','Freqs_MHz' saved at round 0.
        """
        # auto-detect qubit index
        qidx = self.qubit_of(h5_path)
        with h5py.File(h5_path, 'r') as f:
            # Navigate into the Q<idx+1> group
            qub = f[f'Q{qidx+1}']

            # Load round-0 data (we saved everything under round 0, do not change this zero).
            I      = qub['I'][0, ...]         # (n_freqs, n_gains)
            Q      = qub['Q'][0, ...]
            gains  = qub['Gains'][0, ...]     # (n_gains,)
            freqs  = qub['Freqs_MHz'][0, ...] # (n_freqs,)

        # Compute magnitude
        mag = np.sqrt(I**2 + Q**2)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            mag,
            origin='lower',
            aspect='auto',
            extent=[gains[0], gains[-1], freqs[0], freqs[-1]])
        ax.set_xlabel('Gain (arb. units)')
        ax.set_ylabel('Drive frequency (MHz)')
        ax.set_title(f'Rabi Chevron: Q{qidx+1}')
        plt.colorbar(im, ax=ax, label='I,Q Magnitude')
        plt.tight_layout()
        plt.show()