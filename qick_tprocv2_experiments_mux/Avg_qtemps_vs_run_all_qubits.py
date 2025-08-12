import matplotlib.pyplot as plt
import numpy as np

# Each sublist corresponds to a qubit, and contains temperatures from Run 2 with qubits (QUIET run 5), Run 3 with qubits (QUIET run 6),
# Run 3 with qubits (QUIET run 6), and Run 4 with qubits (QUIET run 7).

qubit_temps = [
    [192.58, 186.2, 111.01],  # Qubit 1
    [289.7, 98.36, 117.45],  # Qubit 2
    [177.1, 131.18, 111.01],  # Qubit 3, 110
    [346.95, 144.01, 123.55],  # Qubit 4, 195
    [181.27, 114.04, 121.76],  # Qubit 5
    [227.82, 120.45, 103.69]   # Qubit 6
]
runs = np.array([5, 6, 7])

num_qubits = len(qubit_temps)
num_runs =  len(qubit_temps[0])

plt.figure(figsize=(8, 6))

# Add "Preliminary" text in the background
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

# Use a colormap to assign a unique color to each qubit
colors = plt.get_cmap('tab10')  # or 'Set1', 'tab20', etc.
show_text = False
for qubit_index, temps in enumerate(qubit_temps):
    temps_array = np.array(temps, dtype=np.float64)
    color = colors(qubit_index % 10)  # wrap around if >10 qubits

    plt.plot(runs, temps_array, marker='o', label=f"Qubit {qubit_index + 1}", color=color)

    if show_text:
        for x, y in zip(runs, temps_array):
            if not np.isnan(y):
                plt.text(x+0.06, y + 2.0, f"{y:.0f}mK", ha='center', va='bottom', fontsize=11, color=color)

plt.xlabel("Run Number")
plt.ylabel("Average Effective Qubit Temperature (mK)")
plt.title("Average Effective Qubit Temperature vs Run Number")
plt.xticks(runs, ['Run 5\n(SSF Meas.)', 'Run 6\n(Rabi Pop. Meas.)', 'Run 7\n(Rabi Pop. Meas.)'])
plt.yticks(np.arange(100, 351, 25))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
