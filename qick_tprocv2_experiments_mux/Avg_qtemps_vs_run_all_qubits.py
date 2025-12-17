import matplotlib.pyplot as plt
import numpy as np

show_text = False
colors = ['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred']

qubit_temps = [
    [200.97, 130.30, 108.42, 86.50],  # Qubit 1
    [284.57, 98.02, 95.54, 85.70],  # Qubit 2
    [172.77, 128.61, 106.26, 101.02],  # Qubit 3,
    [337.99, 138.76, 118.59, 112.61],  # Qubit 4,
    [176.2, 105.60, 108.56, 76.14],  # Qubit 5
    [227.27, 111.21, 97.49, 82.94]   # Qubit 6
]
qtemp_errs = [
    [10.27, 13.28, 4.81, 2.40],  # Qubit 1
    [15.14, 6.72, 2.71, 3.05],  # Qubit 2
    [7.46, 11.57, 2.20, 8.18],  # Qubit 3,
    [19.08, 13.33, 5.51, 23.70],  # Qubit 4,
    [6.14, 7.04, 10.38, 4.05],  # Qubit 5
    [8.23, 11.67, 8.45, 14.34]   # Qubit 6
]

runs = np.array([5, 6, 7, 8])

num_qubits = len(qubit_temps)
num_runs =  len(qubit_temps[0])

plt.figure(figsize=(8, 6))

# Add "Preliminary" text in the background
if show_text:
    plt.text(
        0.5, 0.5, 'Preliminary',
        fontsize=70,
        color='lightgray',
        ha='center',
        va='center',
        alpha=0.3,
        rotation=45,
        transform=plt.gca().transAxes,
        zorder=0
    )

for qubit_index, temps in enumerate(qubit_temps):
    temps_array = np.array(temps, dtype=np.float64)
    errs_array = np.array(qtemp_errs[qubit_index], dtype=np.float64)
    color = colors[qubit_index % len(colors)]

    plt.errorbar(
        runs, temps_array,
        yerr=errs_array,
        fmt='-o',
        color=color,
        capsize=3,
        elinewidth=1,
        label=f"Qubit {qubit_index + 1}",
        zorder=2
    )

    if show_text:
        for x, y in zip(runs, temps_array):
            if not np.isnan(y):
                plt.text(x+0.06, y + 2.0, f"{y:.0f}mK", ha='center', va='bottom', fontsize=11, color=color)

plt.xlabel("Run Number")
plt.ylabel("Average Effective Qubit Temperature (mK)")
plt.title("Average Effective Qubit Temperature vs Run Number")
plt.xticks(runs, ['Run 5\n(SSF Meas.)', 'Run 6\n(Rabi Pop. Meas.)', 'Run 7\n(Rabi Pop. Meas.)', 'Run 8\n(Rabi Pop. Meas.)'])
plt.yticks(np.arange(100, 351, 25))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
