import matplotlib.pyplot as plt
import numpy as np

do_T1 = False
do_T2R = False
do_T2E = True

if do_T1:
    t1_vals = [
        [6.09, 14.89, 64.21, 64.20, 61.67],  # Qubit 1
        [19.50, 27.46, 59.03, 62.39, 58.99],  # Qubit 2
        [7.72, 16.33, 73.61, 52.61, 54.12],  # Qubit 3, 110
        [12.73, 21.38, 59.41, 60.09, 59.27],  # Qubit 4, 195
        [9.27, 12.19, 32.44, 55.72, 47.83],  # Qubit 5
        [9.43, 14.72, 41.57, 32.11, 31.51]   # Qubit 6
    ]
    runs = np.array([4, 5, 6, 7, 8])

    num_qubits = len(t1_vals)
    num_runs =  len(t1_vals[0])

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
    for qubit_index, temps in enumerate(t1_vals):
        temps_array = np.array(temps, dtype=np.float64)
        color = colors(qubit_index % 10)  # wrap around if >10 qubits

        plt.plot(runs, temps_array, marker='o', label=f"Qubit {qubit_index + 1}", color=color)

        if show_text:
            for x, y in zip(runs, temps_array):
                if not np.isnan(y):
                    plt.text(x+0.06, y + 2.0, f"{y:.0f}mK", ha='center', va='bottom', fontsize=11, color=color)

    plt.xlabel("Run Number")
    plt.ylabel("Average T1 (us)")
    plt.title("Average T1 vs Run Number")
    plt.xticks(runs, ['Run 4','Run 5', 'Run 6', 'Run 7', 'Run 8'])
    plt.yticks(np.arange(0, 100, 10))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#-------------------------------------------- T2 R -------------------------------------------------------------
if do_T2R:
    t2r_vals = [
        [6.99, 12.17, 29.56, 30.72, 60.46],  # Qubit 1
        [14.82, 11.78, 10.52, 60.87, 77.77],  # Qubit 2
        [4.25, 8.76, 26.33, 51.11, 25.65],  # Qubit 3, 110
        [11.86, 13.57, 20.83, 55.01, 8.11],  # Qubit 4, 195
        [8.98, 11.68, 4.95, 57.71, 37.16],  # Qubit 5
        [9.97, 13.69, 22.94, 25.51, 12.73]  # Qubit 6
    ]
    runs = np.array([4, 5, 6, 7, 8])

    num_qubits = len(t2r_vals)
    num_runs = len(t2r_vals[0])

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
    for qubit_index, temps in enumerate(t2r_vals):
        temps_array = np.array(temps, dtype=np.float64)
        color = colors(qubit_index % 10)  # wrap around if >10 qubits

        plt.plot(runs, temps_array, marker='o', label=f"Qubit {qubit_index + 1}", color=color)

        if show_text:
            for x, y in zip(runs, temps_array):
                if not np.isnan(y):
                    plt.text(x + 0.06, y + 2.0, f"{y:.0f}mK", ha='center', va='bottom', fontsize=11, color=color)

    plt.xlabel("Run Number")
    plt.ylabel("Average T2R (us)")
    plt.title("Average T2R vs Run Number")
    plt.xticks(runs, ['Run 4', 'Run 5', 'Run 6', 'Run 7', 'Run 8'])
    plt.yticks(np.arange(0, 100, 10))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#-------------------------------------------- T2 E -------------------------------------------------------------
if do_T2E:
    t2e_vals = [
        [6.32, 12.21, 41.10, 85.05, 95.34],  # Qubit 1
        [15.16, 20.94, 39.65, 93.65, 89.32],  # Qubit 2
        [6.95, 10.11, 35.55, 60.65, 59.04],  # Qubit 3, 110
        [12.47, 17.71, 44.47, 90.95, 75.22],  # Qubit 4, 195
        [8.26, 11.78, 9.65, 75.42, 66.65],  # Qubit 5
        [10.08, 14.78, 37.82, 52.78, 51.39]  # Qubit 6
    ]
    runs = np.array([4, 5, 6, 7, 8])

    num_qubits = len(t2e_vals)
    num_runs = len(t2e_vals[0])

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
    for qubit_index, temps in enumerate(t2e_vals):
        temps_array = np.array(temps, dtype=np.float64)
        color = colors(qubit_index % 10)  # wrap around if >10 qubits

        plt.plot(runs, temps_array, marker='o', label=f"Qubit {qubit_index + 1}", color=color)

        if show_text:
            for x, y in zip(runs, temps_array):
                if not np.isnan(y):
                    plt.text(x + 0.06, y + 2.0, f"{y:.0f}mK", ha='center', va='bottom', fontsize=11, color=color)

    plt.xlabel("Run Number")
    plt.ylabel("Average T2E (us)")
    plt.title("Average T2E vs Run Number")
    plt.xticks(runs, ['Run 4', 'Run 5', 'Run 6', 'Run 7', 'Run 8'])
    plt.yticks(np.arange(0, 100, 10))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()