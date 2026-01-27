import matplotlib.pyplot as plt
import numpy as np

runs6_through_8_rpm = True

show_text = False
colors = ['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred']

if runs6_through_8_rpm is False:
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
else: # Only SSF qubit temps
    qubit_temps = [
        [201.13, 100.98, 89.26, 77.21],  # Qubit 1
        # [302.20, 84.26, 78.32, 75.36],  # Qubit 2
        # [170.90, 106.43, 80.11, 78.82],  # Qubit 3
        # [339.13, 135.15, 91.27, 100.85],  # Qubit 4
        [174.67, 88.59, 93.73, 81.36],  # Qubit 5
        # [225.68, 91.56, 78.55, 64.90],  # Qubit 6
    ]

    qtemp_errs = [
        [10.98, 3.84, 2.26, 1.88],  # Qubit 1
        # [16.42, 2.74, 1.18, 1.64],  # Qubit 2
        # [7.33, 8.37, 1.37, 2.55],  # Qubit 3,
        # [19.11, 10.16, 2.38, 7.78],  # Qubit 4,
        [6.17, 2.98, 2.29, 1.14],  # Qubit 5
        # [8.18, 1.65, 1.98, 3.96]  # Qubit 6
    ]

    runs = np.array([5, 6, 7, 8])

    num_qubits = len(qubit_temps)
    num_runs = len(qubit_temps[0])

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
                    plt.text(x + 0.06, y + 2.0, f"{y:.0f}mK", ha='center', va='bottom', fontsize=11, color=color)

    plt.xlabel("Run Number")
    plt.ylabel("Average Effective Qubit Temperature (mK)")
    plt.title("Average Effective Qubit Temperature vs Run Number")
    plt.xticks(runs, ['Run 5\n(SSF Meas.)', 'Run 6\n(SSF Meas.)', 'Run 7\n(SSF Meas.)',
                      'Run 8\n(SSF Meas.)'])
    plt.yticks(np.arange(100, 351, 25))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
