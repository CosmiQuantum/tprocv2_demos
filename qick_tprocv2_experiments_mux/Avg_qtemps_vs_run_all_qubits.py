import matplotlib.pyplot as plt
import numpy as np

# Each sublist corresponds to a qubit, and contains temperatures from Run 1 with qubits (run 4), Run 2 with qubits (run 5),
# and Run 3 with qubits (run 6).

from_run = 2 # we have a total of 3 runs with qubits thus far

if from_run == 2:
    qubit_temps = [
        [200, 183],  # Qubit 1
        [275, 95],  # Qubit 2
        [160, 131],  # Qubit 3, 110
        [350, 144],  # Qubit 4, 195
        [170, 114],  # Qubit 5
        [220, 120]   # Qubit 6
    ]
    runs = np.array([2, 3])

if from_run == 1:
    qubit_temps = [
        [None, 200, 183],  # Qubit 1
        [None, 275, 95],  # Qubit 2
        [110, 160, 131],  # Qubit 3, 110
        [195, 350, 144],  # Qubit 4, 195
        [None, 170, 114],  # Qubit 5
        [None, 220, 120]   # Qubit 6
    ]
    runs = np.array([1, 2, 3])

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
plt.xticks(runs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
