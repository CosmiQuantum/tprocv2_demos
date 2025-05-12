import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import os
import datetime
from sklearn.mixture import GaussianMixture

class ssf:
    def ssf_fit_two_gaussians_midpoint(self, ig_new: np.ndarray, ie_new: np.ndarray):
        """
        Fits a 2‑component GMM (double gaussian) to all shots (ig_new + ie_new) from an ssf file
        and chooses the threshold as the midpoint between the two component means.

        ig_new is the rotated I data for the ground state in a g-e SSF measurement.
        ie_new is the rotated I data for the first excited state in a g-e SSF measurement.

        Returns
        -------
        thresh           : (μ_g + μ_e) / 2
        means, sigmas    : np.ndarray shape (2,)
        weights          : np.ndarray shape (2,)
        ground_idx       : component index for ground cluster
        excited_idx      : component index for excited cluster
        """

        # Fit a 2‑component Gaussian mixture
        all_i = np.concatenate([ig_new, ie_new]).reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, covariance_type="full")
        gmm.fit(all_i)

        means = gmm.means_.flatten()
        sigmas = np.sqrt(gmm.covariances_).flatten()
        weights = gmm.weights_

        ground_idx, excited_idx = np.argsort(means)  # smaller mean = ground
        mu_g, mu_e = means[ground_idx], means[excited_idx]

        # Mid‑point threshold
        threshold = 0.5 * (mu_g + mu_e)

        return threshold, means, sigmas, weights, ground_idx, excited_idx

    def plot_ssf_ge_thresh(self, QubitIndex: int, ig_new: np.ndarray, ie_new: np.ndarray, plotting_path: str, numbins: int = 64, figure_quality: int = 200):
        """
        This function plots the SSF data + the two gaussian fits + the means of the gaussians + the g-e threshold for visualization.
        ig_new is the rotated I data for the ground state in a g-e SSF measurement.
        ie_new is the rotated I data for the first excited state in a g-e SSF measurement.
        QubitIndex is expected to start at 0 for qubit 1 and so forth.
        """
        os.makedirs(plotting_path, exist_ok=True)

        # fit & extract numbers
        thresh, means, sigmas, weights, ground_idx, excited_idx = self.ssf_fit_two_gaussians_midpoint(ig_new, ie_new)

        # plot to check things fitted correctly
        fig, ax = plt.subplots(figsize=(7, 4))
        all_i = np.concatenate([ig_new, ie_new])

        # histogram of *all* shots (does not show population overlaps)
        # n, edges, _ = ax.hist(all_i, bins=numbins, alpha=0.35, color="grey", label="all shots")
        # counts, edges = np.histogram(all_i, bins=numbins) # just extracting edges

        # Plot g and e histograms separately (shows populations that overlap)
        edges = np.linspace(all_i.min(), all_i.max(), numbins + 1)
        ax.hist(ig_new, bins=edges, alpha=0.55, color="royalblue", label="g-state")
        ax.hist(ie_new, bins=edges, alpha=0.55, color="crimson", label="e-state")

        x_grid = np.linspace(all_i.min(), all_i.max(), 400)
        g_pdf = (weights[ground_idx] /
                 (np.sqrt(2 * np.pi) * sigmas[ground_idx]) *
                 np.exp(-0.5 * ((x_grid - means[ground_idx]) /
                                sigmas[ground_idx]) ** 2))
        e_pdf = (weights[excited_idx] /
                 (np.sqrt(2 * np.pi) * sigmas[excited_idx]) *
                 np.exp(-0.5 * ((x_grid - means[excited_idx]) /
                                sigmas[excited_idx]) ** 2))

        # Component‑specific scaling. We scale since we want to plot the y-axis in counts instead of probability densities
        counts_g, _ = np.histogram(ig_new, bins=edges)
        counts_e, _ = np.histogram(ie_new, bins=edges)

        peak_g = counts_g.max()
        peak_e = counts_e.max()

        # factor that makes the PDF peak equal the tallest bar
        scale_g = peak_g / g_pdf.max()
        scale_e = peak_e / e_pdf.max()

        ax.plot(x_grid, g_pdf * scale_g, color="blue", lw=2,
                label="ground Gaussian")
        ax.plot(x_grid, e_pdf * scale_e, color="red", lw=2,
                label="excited Gaussian")

        # vertical markers
        ax.axvline(means[ground_idx], color="blue", ls="--")
        ax.axvline(means[excited_idx], color="red", ls="--")
        ax.axvline(thresh, color="black", ls=":",
                   label=f"threshold = {thresh:.2f}")

        ax.set_title(f"Q{QubitIndex + 1}")
        ax.set_xlabel("I'  (rotated)")
        ax.set_ylabel("Counts")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(frameon=False)
        fig.tight_layout()

        fname = os.path.join(plotting_path, f"Q{QubitIndex + 1}_midpoint_fit_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        fig.savefig(fname, dpi=figure_quality)
        plt.close(fig)

        print('Plots saved to:', plotting_path)