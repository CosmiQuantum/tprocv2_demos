import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class QfreqHistPlots:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, base_data_path, plots_path, run_name, fridge):
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.signal = signal
        self.figure_quality = figure_quality
        self.base_data_path = base_data_path
        self.run_name = run_name
        self.plots_path = plots_path
        self.number_of_qubits = number_of_qubits
        self.final_figure_quality = final_figure_quality
        self.top_folder_dates = top_folder_dates
        self.fridge = fridge

    def run(self, qfreq_vals, qfreq_errs, bins=45, rel_err_cutoff=None):
        """
        Plot per-qubit histograms of qubit frequency (MHz), with Gaussian overlay whose
        center/width come from the SAME weighted-mean + median-MAD clipping recipe used
        in the qubit temperature histogram code.

        Parameters
        ----------
        qfreq_vals : dict or list
            dict {qid: [freq_MHz, ...]} OR list-of-lists indexed by qid.
        qfreq_errs : dict or list
            dict {qid: [freq_err_MHz, ...]} OR list-of-lists indexed by qid.
        bins : int
            Histogram bins per subplot.
        rel_err_cutoff : float or None
            Optional filter: keep only points with (err/freq) <= rel_err_cutoff.
        """
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        os.makedirs(self.plots_path, exist_ok=True)

        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("Qubit Frequency Distributions", fontsize=18)

        # iterate stable order 0..N-1 (like your other plots)
        for q in range(self.number_of_qubits):
            # --- fetch lists robustly for dict or list input ---
            if isinstance(qfreq_vals, dict):
                freqs_list = qfreq_vals.get(q, [])
            else:
                freqs_list = qfreq_vals[q] if q < len(qfreq_vals) else []

            if isinstance(qfreq_errs, dict):
                errs_list = qfreq_errs.get(q, [])
            else:
                errs_list = qfreq_errs[q] if q < len(qfreq_errs) else []

            if (not freqs_list) or (not errs_list):
                continue

            # --- continue-style filtering (same style as your temp hist) ---
            freq_vals = []
            freq_errs = []
            for f, e in zip(freqs_list, errs_list):
                try:
                    f = float(f)
                    e = float(e)
                except (TypeError, ValueError):
                    continue

                # basic validity
                if not np.isfinite(f) or not np.isfinite(e):
                    continue
                if f <= 0:
                    continue
                if e <= 0:
                    continue

                # optional relative error cutoff
                if rel_err_cutoff is not None and (e / f) > rel_err_cutoff:
                    continue

                freq_vals.append(f)
                freq_errs.append(e)

            if len(freq_vals) == 0:
                continue

            freqs = np.asarray(freq_vals, dtype=float)
            errs = np.asarray(freq_errs, dtype=float)

            # ---------------------------Weighted mean with robust median-MAD clipping---------------------------
            n_counts = len(freqs)

            # keep only finite pairs
            finite = np.isfinite(freqs) & np.isfinite(errs)
            freqs, errs = freqs[finite], errs[finite]
            if freqs.size == 0:
                mu_1, std_1 = np.nan, np.nan
            else:
                # robust outlier clip around the median
                k = 2.0  # 2-4 is typical; lower = stricter
                med = np.median(freqs)
                mad = np.median(np.abs(freqs - med))
                if mad == 0:
                    mad = max(np.std(freqs), 1e-12)

                keep = np.abs(freqs - med) < k * mad
                freqs, errs = freqs[keep], errs[keep]

                if freqs.size == 0:
                    mu_1, std_1 = np.nan, np.nan
                else:
                    # compute weights and weighted mean/std (USING 1/err to match your temp hist code)
                    err_floor = 1e-12
                    safe_errs = np.clip(errs, err_floor, np.inf)
                    weights = 1.0 / safe_errs

                    w_sum = np.nansum(weights)
                    mu_1 = float(np.nansum(weights * freqs) / w_sum)

                    var = float(np.nansum(weights * (freqs - mu_1) ** 2) / w_sum)
                    std_1 = float(np.sqrt(max(var, 0.0)))

            # --- Histogram (raw counts) ---
            ax = plt.subplot(2, 3, q + 1)
            hist_data, edges = np.histogram(freqs, bins=bins)
            bin_width = np.diff(edges)[0] if len(edges) > 1 else 1.0

            ax.hist(freqs,
                    bins=bins,
                    alpha=0.7,
                    color=colors[q % len(colors)],
                    edgecolor='black',
                    label="Counts")

            # --- Weighted Gaussian overlay, area-matched to histogram ---
            if np.isfinite(mu_1) and np.isfinite(std_1) and std_1 > 0:
                x_vals = np.linspace(freqs.min(), freqs.max(), 400)
                pdf_vals = norm.pdf(x_vals, mu_1, std_1)
                scale_factor = len(freqs) * bin_width  # area-match
                ax.plot(x_vals, pdf_vals * scale_factor,
                        linestyle='--', linewidth=2, color='black',
                        label='Weighted Gaussian fit')

            ax.set_title(f"Q{q + 1}  µ={mu_1:.4f} MHz,  s={std_1:.4f} MHz, c:{n_counts}", fontsize=9, pad=3)
            ax.set_xlabel("Qubit Frequency (MHz)", labelpad=1)
            ax.set_ylabel("Count")
            ax.grid(alpha=0.3)

        plt.tight_layout()

        # --- save ---
        if self.save_figs:
            timestp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            fname = os.path.join(self.plots_path, f"{self.run_name}_{self.fridge}_Qfreq_Hists_{timestp}.png")
            plt.savefig(fname, dpi=self.final_figure_quality)
            plt.close(fig)
            print("Saved qubit frequency histograms to:", fname)
        else:
            plt.show()