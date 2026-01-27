import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

show_text = False
colors = ['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred']

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

# NOTE: these have 5 entries per qubit, but temps only have 4 (runs 5–8).
# I’m assuming the FIRST T1 entry corresponds to an older run (e.g., run 4),
# and we want the LAST 4 entries to match runs 5–8.
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

runs = np.array([5, 6, 7, 8])

# marker per run (so the legend can encode run number)
run_markers = {5: "o", 6: "s", 7: "^", 8: "D"}

num_qubits = len(qubit_temps)

fig, ax = plt.subplots(figsize=(9, 6))

# "Preliminary" watermark
if show_text:
    ax.text(
        0.5, 0.5, 'Preliminary',
        fontsize=70,
        color='lightgray',
        ha='center',
        va='center',
        alpha=0.3,
        rotation=45,
        transform=ax.transAxes,
        zorder=0
    )

# Plot: for each qubit, one point per run: (mean_temp, mean_T1)
for q in range(num_qubits):
    x = np.asarray(qubit_temps[q], dtype=float)
    xerr = np.asarray(qtemp_errs[q], dtype=float)

    y_all = np.asarray(t1_vals[q], dtype=float)
    yerr_all = np.asarray(t1_errs[q], dtype=float)

    # Align T1 arrays to runs 5–8 (drop the first entry)
    y = y_all[1:1 + len(runs)]
    yerr = yerr_all[1:1 + len(runs)]

    color = colors[q % len(colors)]

    # light connecting line across runs for this qubit
    ax.plot(x, y, "-", color=color, alpha=0.5, zorder=1)

    # points + error bars; marker encodes run, color encodes qubit
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

# Axis labels/titles
ax.set_xlabel("Effective Qubit Temperature (mK)")
ax.set_ylabel("T1 (µs)")
ax.set_title("Mean T1 vs Effective Qubit Temperature")

# Optional: annotate values (kept off by default)
if show_text:
    for q in range(num_qubits):
        x = np.asarray(qubit_temps[q], float)
        y = np.asarray(t1_vals[q], float)[1:1 + len(runs)]
        color = colors[q % len(colors)]
        for r_i, r in enumerate(runs):
            ax.text(x[r_i] + 2.0, y[r_i] + 1.0, f"R{int(r)}", fontsize=10, color=color)

ax.grid(True)
fig.tight_layout()

# --- Combined legend: qubits (color) + runs (marker) ---
qubit_handles = [
    Line2D([0], [0], color=colors[q % len(colors)], lw=2, label=f"Qubit {q+1}")
    for q in range(num_qubits)
]

run_handles = [
    Line2D([0], [0], marker=run_markers[int(r)], color="black", lw=0,
           markersize=7, label=f"Run {int(r)}")
    for r in runs
]

all_handles = qubit_handles + run_handles
ax.legend(handles=all_handles, frameon=True, ncol=1)

plt.show()
