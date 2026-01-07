import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

show_text = False
colors = ['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred']

qubit_temps = [
    [200.97, 130.30, 108.42, 86.50],   # Qubit 1 (runs 5-8)
    [284.57, 98.02, 95.54, 85.70],     # Qubit 2
    [172.77, 128.61, 106.26, 101.02],  # Qubit 3
    [337.99, 138.76, 118.59, 112.61],  # Qubit 4
    [176.2, 105.60, 108.56, 76.14],    # Qubit 5
    [227.27, 111.21, 97.49, 82.94]     # Qubit 6
]
qtemp_errs = [
    [10.27, 13.28, 4.81, 2.40],        # Qubit 1
    [15.14, 6.72, 2.71, 3.05],         # Qubit 2
    [7.46, 11.57, 2.20, 8.18],         # Qubit 3
    [19.08, 13.33, 5.51, 23.70],       # Qubit 4
    [6.14, 7.04, 10.38, 4.05],         # Qubit 5
    [8.23, 11.67, 8.45, 14.34]         # Qubit 6
]

# NOTE: these have 5 entries per qubit, but temps only have 4 (runs 5–8).
# I’m assuming the FIRST T1 entry corresponds to an older run (e.g., run 4),
# and we want the LAST 4 entries to match runs 5–8.
t1_vals = [
    [6.09, 14.89, 64.21, 64.20, 60.98],   # Qubit 1
    [19.50, 27.46, 59.03, 62.39, 64.41],  # Qubit 2
    [7.72, 16.33, 73.61, 52.61, 52.84],   # Qubit 3
    [12.73, 21.38, 59.41, 60.09, 64.61],  # Qubit 4
    [9.27, 12.19, 32.44, 55.72, 48.44],   # Qubit 5
    [9.43, 14.72, 41.57, 32.11, 30.45]    # Qubit 6
]
t1_errs = [
    [0.24, 1.42, 4.32, 5.46, 3.03],   # Qubit 1
    [2.35, 3.30, 6.46, 4.06, 3.55],   # Qubit 2
    [0.26, 1.74, 4.16, 1.80, 5.30],   # Qubit 3
    [0.44, 2.55, 3.74, 6.26, 7.65],   # Qubit 4
    [0.47, 1.65, 4.63, 2.49, 2.74],   # Qubit 5
    [0.35, 1.39, 2.89, 1.99, 1.37]    # Qubit 6
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
ax.set_ylabel("Mean T1 (µs)")
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
