import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

show_text = False
colors = ['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred']

T1_vs_qtemps = True
T1_AND_qtemps = False

runs = ([5, 6, 7, 8])

qubit_temps = [
        [201.13,  99.22,  81.46,  71.21],  # Qubit 1
        [302.20,  84.26,  78.32,  75.36],  # Qubit 2
        [170.90, 106.43,  80.11,  78.82],  # Qubit 3
        [339.13, 135.15,  91.27, 100.85],  # Qubit 4
        [174.67,  76.82,  87.88,  74.10],  # Qubit 5
        [225.68,  91.56,  78.55,  64.90],  # Qubit 6
    ]

qtemp_errs = [
    [10.98, 9.99, 2.52, 1.26],  # Qubit 1
    [16.42, 2.74, 1.18, 1.64],  # Qubit 2
    [7.33, 8.37, 1.37, 2.55],  # Qubit 3,
    [19.11, 10.16, 2.38, 7.78],  # Qubit 4,
    [6.17, 5.77, 1.58, 2.07],  # Qubit 5
    [8.18, 1.65, 1.98, 3.96]   # Qubit 6
]

# NOTE: these have 5 entries per qubit, but temps only have 4 (runs).
# We drop the first T1 entry so the LAST 4 align with runs 5-8.
t1_vals = [
        [6.09, 14.94, 64.32, 63.71, 61.14],  # Qubit 1
        [19.61, 27.73, 59.71, 62.47, 63.60],  # Qubit 2
        [7.72, 16.43, 73.72, 52.64, 54.17],  # Qubit 3
        [12.74, 21.53, 59.55, 59.51, 64.76],  # Qubit 4
        [9.28, 12.31, 32.83, 55.77, 48.78],  # Qubit 5
        [9.44, 14.82, 41.52, 32.09, 30.45],  # Qubit 6
    ]

t1_errs = [
        [0.24, 1.41, 4.20, 5.63, 3.06],  # Qubit 1
        [2.38, 3.34, 6.54, 3.98, 3.85],  # Qubit 2
        [0.26, 1.75, 4.18, 1.81, 4.82],  # Qubit 3
        [0.44, 2.49, 3.72, 6.65, 7.72],  # Qubit 4
        [0.46, 1.66, 4.19, 2.45, 2.99],  # Qubit 5
        [0.35, 1.39, 2.98, 2.02, 1.36]   # Qubit 6
    ]

# ---- Physics assumptions ----
T1_baseline_us = 100.0          # Expected low-T plateau (>100 µs)
Gamma0 = 1.0 / (T1_baseline_us * 1e-6)   # baseline rate in 1/s
Delta_over_kB = 2.1             # K (Al junctions)

def T1_model_us(T_mK, Gamma0, B, Delta_over_kB):
    """Return T1(T) in microseconds."""
    T_K = np.asarray(T_mK) * 1e-3
    gamma = Gamma0 + B * np.exp(-Delta_over_kB / T_K)
    return (1.0 / gamma) * 1e6


if T1_vs_qtemps:

    run_markers = {5: "o", 6: "s", 7: "^", 8: "D"}
    num_qubits = len(qubit_temps)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for q in range(num_qubits):

        ax = axes[q]
        color = colors[q % len(colors)]

        x = np.asarray(qubit_temps[q], dtype=float)
        xerr = np.asarray(qtemp_errs[q], dtype=float)

        y_all = np.asarray(t1_vals[q], dtype=float)
        yerr_all = np.asarray(t1_errs[q], dtype=float)

        y = y_all[1:1 + len(runs)]
        yerr = yerr_all[1:1 + len(runs)]

        # ---------- THEORETICAL OVERLAY (readable, anchored) ----------
        # Use Al junction gap scale
        Delta_over_kB = 2.1  # K

        # Baseline expected at low T (<50 mK)
        T1_baseline_us = 100.0
        Gamma0 = 1.0 / (T1_baseline_us * 1e-6)

        # Temperature grid for smooth curves
        T_grid = np.linspace(max(1e-6, x.min() * 0.9), x.max() * 1.1, 400)

        # Anchor curves at a representative temperature (no fitting)
        T_star_mK = 100.0
        T_star_K = T_star_mK * 1e-3

        # Choose a few "target T1" values at T_star to generate 3 curves
        # (These are just visual guides, not fits.)
        targets_us = [80.0, 50.0, 30.0]

        for T1_target_us in targets_us:
            Gamma_target = 1.0 / (T1_target_us * 1e-6)

            # Only meaningful if Gamma_target > Gamma0; otherwise B would be negative
            if Gamma_target <= Gamma0:
                continue

            B = (Gamma_target - Gamma0) * np.exp(Delta_over_kB / T_star_K)

            ax.plot(
                T_grid,
                T1_model_us(T_grid, Gamma0, B, Delta_over_kB),
                color="black",
                alpha=0.18,
                lw=1.4,
                zorder=0
            )

        # Optional: label the overlay meaning on the first subplot only
        if q == 0:
            ax.text(
                0.03, 0.10,
                "Overlay: G1(T)=G0 + B exp[-(?/kB)/T]\n"
                "?/kB=2.1 K (Al junctions), G0 from T10=100 µs\n"
                "B chosen so model hits T1={80,50,30} µs at T=100 mK",
                transform=ax.transAxes, fontsize=7.5, color="k", alpha=0.75,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
            )

        # ---------- DATA ----------
        ax.plot(x, y, "-", color=color, alpha=0.5, zorder=1)

        for r_i, r in enumerate(runs):
            ax.errorbar(
                x[r_i], y[r_i],
                xerr=xerr[r_i], yerr=yerr[r_i],
                fmt=run_markers[int(r)],
                color=color,
                capsize=3,
                elinewidth=1,
                markersize=7,
                zorder=3
            )

        # 50 mK thermalization target
        ax.axvline(50.0, linestyle="--", color="k", linewidth=1, alpha=0.5, zorder=0)

        ax.set_title(f"Qubit {q+1}")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.35)

    fig.supxlabel("Effective Qubit Temperature (mK)", y=0.06)
    fig.supylabel("T1 (µs) [log scale]")

    fig.suptitle(
        "Mean T1 vs Effective Qubit Temperature\n"
        "(Activated qp model overlay, ?/kB = 2.1 K, T10 = 100 µs)",
        y=0.98
    )

    run_handles = [
        Line2D([0], [0], marker=run_markers[int(r)], color="black", lw=0,
               markersize=7, label=f"Run {int(r)}")
        for r in runs
    ]

    fig.legend(
        handles=run_handles,
        frameon=True,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02)
    )

    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    plt.show()