import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ----------------- USER TOGGLES -----------------
show_text = False
colors = ['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred']

T1_vs_qtemps = True
T1_AND_qtemps = False  # (left untouched below; your focus is T1_vs_qtemps)

runs = [5, 6, 7, 8]

# ----------------- DATA ARRAYS -----------------
qubit_temps = [
    [197.35,  99.22,  81.46,  71.21],  # Qubit 1
    [361.39,  84.26,  78.32,  75.36],  # Qubit 2
    [164.24, 106.43,  80.11,  78.82],  # Qubit 3
    [354.34, 135.15,  91.27, 100.85],  # Qubit 4
    [169.67,  76.82,  87.88,  74.10],  # Qubit 5
    [216.55,  91.56,  78.55,  64.90],  # Qubit 6
]

qtemp_errs = [
    [12.76, 9.99, 2.52, 1.26],   # Qubit 1
    [33.57, 2.74, 1.18, 1.64],   # Qubit 2
    [8.37,  8.37, 1.37, 2.55],   # Qubit 3
    [23.71, 10.16, 2.38, 7.78],  # Qubit 4
    [5.94,  5.77, 1.58, 2.07],   # Qubit 5
    [7.80,  1.65, 1.98, 3.96],   # Qubit 6
]

# NOTE: 5 entries per qubit, but temps only have 4 (runs).
# We drop the first T1 entry so the LAST 4 align with runs 5-8.
t1_vals = [
    [6.09, 14.94, 64.32, 63.71, 61.14],   # Qubit 1
    [19.61, 27.73, 59.71, 62.47, 63.60],  # Qubit 2
    [7.72, 16.43, 73.72, 52.64, 54.17],   # Qubit 3
    [12.74, 21.53, 59.55, 59.51, 64.76],  # Qubit 4
    [9.28, 12.31, 32.83, 55.77, 48.78],   # Qubit 5
    [9.44, 14.82, 41.52, 32.09, 30.45],   # Qubit 6
]

t1_errs = [
    [0.24, 1.41, 4.20, 5.63, 3.06],  # Qubit 1
    [2.38, 3.34, 6.54, 3.98, 3.85],  # Qubit 2
    [0.26, 1.75, 4.18, 1.81, 4.82],  # Qubit 3
    [0.44, 2.49, 3.72, 6.65, 7.72],  # Qubit 4
    [0.46, 1.66, 4.19, 2.45, 2.99],  # Qubit 5
    [0.35, 1.39, 2.98, 2.02, 1.36],  # Qubit 6
]

# ----------------- PHYSICS ASSUMPTIONS (SET ONCE) -----------------
DELTA_OVER_KB_K = 2.1       # K (Al junctions)

# This is your "theoretical expectation" for the true T->0 plateau.
# NOTE: Option A will NOT force this on the overlay curves; it stays in the title only.
T1_BASELINE_US = 100.0

# -------- Baseline option toggle (KEEPING YOUR TWO OPTIONS) --------
# "coldest"  -> gamma0 from the coldest observed temperature point
# "bestT1"   -> gamma0 from the best observed T1 (smallest observed rate)
BASELINE_MODE = "bestT1"  # change to "coldest" if desired

def T1_model_us(T_mK, gamma0, B, delta_over_kB_K):
    """
    Return T1(T) in microseconds for:
        G1(T) = G0 + B * exp(-(\u0394/kB)/T)
    where:
        T is in Kelvin,
        \u0394/kB is in Kelvin,
        G0 and B are in 1/s.
    """
    T_K = np.asarray(T_mK, dtype=float) * 1e-3
    T_K = np.maximum(T_K, 1e-6)  # numerical safety
    gamma = gamma0 + B * np.exp(-delta_over_kB_K / T_K)
    return (1.0 / gamma) * 1e6  # µs

def infer_gamma0_from_points(T_mK, T1_us, mode="bestT1"):
    """
    Baseline inference with TWO OPTIONS you requested:

    mode="coldest":
        gamma0 = 1/T1 at coldest observed T
    mode="bestT1":
        gamma0 = smallest observed rate (i.e., max observed T1)
    """
    T_mK = np.asarray(T_mK, float)
    T1_us = np.asarray(T1_us, float)

    if mode == "coldest":
        # pick empirical baseline from coldest observed T
        i0 = np.argmin(T_mK)
        gamma0 = 1.0 / (T1_us[i0] * 1e-6)  # 1/s
    elif mode == "bestT1":
        # baseline = smallest observed rate (highest T1)
        gamma0 = float(np.min(1.0 / (T1_us * 1e-6)))  # 1/s
    else:
        raise ValueError("mode must be 'coldest' or 'bestT1'")

    return float(gamma0)

def implied_B_values(T_mK, T1_us, gamma0, delta_over_kB_K):
    """
    Compute implied B_i for each point:
        B_i = (G1_i - G0) * exp((\u0394/kB)/T_i)
    Keep only points where G1_i > G0 (so B_i > 0).
    """
    T_mK = np.asarray(T_mK, float)
    T1_us = np.asarray(T1_us, float)

    T_K = np.maximum(T_mK * 1e-3, 1e-6)
    gamma1 = 1.0 / (T1_us * 1e-6)

    gamma_th = gamma1 - gamma0
    valid = gamma_th > 0
    if not np.any(valid):
        return np.array([], dtype=float)

    B_i = gamma_th[valid] * np.exp(delta_over_kB_K / T_K[valid])
    return B_i

# ----------------- PLOT: T1 vs Teff (Option A) -----------------
if T1_vs_qtemps:
    run_markers = {5: "o", 6: "s", 7: "^", 8: "D"}

    num_qubits = len(qubit_temps)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    # ----- Step 1: infer per-qubit gamma0_q (baseline) + pool B_i across ALL qubits -----
    gamma0_list = []
    B_pool = []

    for q in range(num_qubits):
        xq = np.asarray(qubit_temps[q], float)

        yq_all = np.asarray(t1_vals[q], float)
        yq = yq_all[1:1 + len(runs)]  # align to runs 58

        gamma0_q = infer_gamma0_from_points(xq, yq, mode=BASELINE_MODE)
        gamma0_list.append(gamma0_q)

        B_i = implied_B_values(xq, yq, gamma0_q, DELTA_OVER_KB_K)
        if B_i.size:
            B_pool.append(B_i)

    if len(B_pool) == 0:
        B_global = 0.0
    else:
        B_global = float(np.median(np.concatenate(B_pool)))

    # ----- Step 2: plot per-qubit curves using (gamma0_q, B_global) -----
    for q in range(num_qubits):
        ax = axes[q]
        color = colors[q % len(colors)]

        # Data arrays
        x = np.asarray(qubit_temps[q], dtype=float)      # mK
        xerr = np.asarray(qtemp_errs[q], dtype=float)    # mK

        y_all = np.asarray(t1_vals[q], dtype=float)      # µs (len 5)
        yerr_all = np.asarray(t1_errs[q], dtype=float)   # µs (len 5)

        # Align T1 arrays to runs 58 (drop the first entry)
        y = y_all[1:1 + len(runs)]
        yerr = yerr_all[1:1 + len(runs)]

        # Optional watermark
        if show_text:
            ax.text(
                0.5, 0.5, "Preliminary",
                fontsize=35, color="lightgray",
                ha="center", va="center",
                alpha=0.3, rotation=45,
                transform=ax.transAxes, zorder=0
            )

        # ---------- THEORETICAL OVERLAY: Option A ----------
        T_grid = np.linspace(max(1e-3, x.min() * 0.9), x.max() * 1.1, 400)

        gamma0_q = gamma0_list[q]

        ax.plot(
            T_grid,
            T1_model_us(T_grid, gamma0_q, B_global, DELTA_OVER_KB_K),
            color="black",
            alpha=0.40,
            lw=2.0,
            zorder=0
        )

        # Annotation per qubit
        ax.text(
            0.03, 0.08,
            rf"$T_{{1,0}}^{{emp}}$={1e6/gamma0_q:.0f} µs,  $B_{{global}}$={B_global:.1e} s$^{{-1}}$",
            transform=ax.transAxes, fontsize=7.5, color="k", alpha=0.75,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.65)
        )

        # ---------- DATA ----------
        ax.plot(x, y, "-", color=color, alpha=0.5, zorder=1)

        for r_i, r in enumerate(runs):
            ax.errorbar(
                x[r_i], y[r_i],
                xerr=xerr[r_i], yerr=yerr[r_i],
                fmt=run_markers[int(r)],
                color=color, capsize=3,
                elinewidth=1, markersize=7,
                zorder=3
            )

        # 50 mK thermalization target
        ax.axvline(50.0, linestyle="--", color="k", linewidth=1, alpha=0.5, zorder=0)

        ax.set_title(f"Qubit {q+1}")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.35)

        # Keep log axis focused on the relevant range
        ax.set_ylim(10, 150)

        # Optional run labels
        if show_text:
            for r_i, r in enumerate(runs):
                ax.text(x[r_i] + 2.0, y[r_i] + 1.0, f"R{int(r)}",
                        fontsize=9, color=color)

    # Shared labels/titles
    fig.supxlabel("Effective Qubit Temperature (mK)", y=0.06)
    fig.supylabel("T1 (µs) [log scale]")

    fig.suptitle(
        "Mean T1 vs Effective Qubit Temperature\n"
        f"(Option A: shared $B$, per-qubit baseline; "
        f"baseline mode = {BASELINE_MODE}; "
        f"\u0394/kB = {DELTA_OVER_KB_K} K, Theoretical T10 = {T1_BASELINE_US:.0f} µs)",
        y=0.98
    )

    # Legend: runs only (since each panel is one qubit)
    run_handles = [
        Line2D([0], [0], marker=run_markers[int(r)], color="black", lw=0,
               markersize=7, label=f"Run {int(r)}")
        for r in runs
    ]
    fig.legend(handles=run_handles, frameon=True, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    plt.show()