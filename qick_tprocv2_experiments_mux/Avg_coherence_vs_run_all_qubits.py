import matplotlib.pyplot as plt
import numpy as np

do_T1 = False
do_T2R = True
do_T2E = False

show_text = False
colors = ['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred']

if do_T1:
    t1_vals = [
        [6.09, 14.89, 64.21, 64.20, 61.67],  # Qubit 1
        [19.50, 27.46, 59.03, 62.39, 58.99],  # Qubit 2
        [7.72, 16.33, 73.61, 52.61, 54.12],  # Qubit 3,
        [12.73, 21.38, 59.41, 60.09, 59.27],  # Qubit 4,
        [9.27, 12.19, 32.44, 55.72, 47.83],  # Qubit 5
        [9.43, 14.72, 41.57, 32.11, 31.51]   # Qubit 6
    ]

    t1_errs = [
        [0.24, 1.42, 4.32, 5.46, 2.37],  # Qubit 1
        [2.35, 3.30, 6.46, 4.06, 4.23],  # Qubit 2
        [0.26, 1.74, 4.16, 1.80, 2.66],  # Qubit 3,
        [0.44, 2.55, 3.74, 6.26, 9.80],  # Qubit 4,
        [0.47, 1.65, 4.63, 2.49, 3.36],  # Qubit 5
        [0.35, 1.39, 2.89, 1.99, 1.33]   # Qubit 6
    ]

    runs = np.array([4, 5, 6, 7, 8])

    num_qubits = len(t1_vals)
    num_runs =  len(t1_vals[0])

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

    # Use a colormap to assign a unique color to each qubit
    for qubit_index, vals in enumerate(t1_vals):
        vals_array = np.array(vals, dtype=np.float64)
        errs_array = np.array(t1_errs[qubit_index], dtype=np.float64)
        color = colors[qubit_index % len(colors)]

        plt.errorbar(
            runs, vals_array,
            yerr=errs_array,
            fmt='-o',
            color=color,
            capsize=3,
            elinewidth=1,
            label=f"Qubit {qubit_index + 1}",
            zorder=2
        )

        if show_text:
            for x, y in zip(runs, vals_array):
                if not np.isnan(y):
                    plt.text(x+0.06, y + 2.0, f"{y:.0f}mK", ha='center', va='bottom', fontsize=11, color=color)

    plt.xlabel("Run Number")
    plt.ylabel("(weighted) Mean T1 (us)")
    plt.title("(weighted) Mean T1 vs Run Number")
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
        [4.25, 8.76, 26.33, 51.11, 25.65],  # Qubit 3,
        [11.86, 13.57, 20.83, 55.01, 8.11],  # Qubit 4
        [8.98, 11.68, 4.95, 57.71, 37.16],  # Qubit 5
        [9.97, 13.69, 22.94, 25.51, 12.73]  # Qubit 6
    ]

    t2r_errs = [
        [0.27, 1.0, 3.58, 1.20, 2.34],  # Qubit 1
        [0.73, 1.44, 1.43, 6.81, 13.11],  # Qubit 2
        [0.71, 1.0, 3.38, 1.85, 1.51],  # Qubit 3,
        [0.42, 1.32, 1.12, 4.58, 2.65],  # Qubit 4,
        [0.34, 0.90, 1.69, 2.62, 5.53],  # Qubit 5
        [0.54, 1.0, 2.09, 1.11, 0.64]  # Qubit 6
    ]

    runs = np.array([4, 5, 6, 7, 8])

    num_qubits = len(t2r_vals)
    num_runs = len(t2r_vals[0])

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


    for qubit_index, vals in enumerate(t2r_vals):
        vals_array = np.array(vals, dtype=np.float64)
        errs_array = np.array(t2r_errs[qubit_index], dtype=np.float64)
        color = colors[qubit_index % len(colors)]

        plt.errorbar(
            runs, vals_array,
            yerr=errs_array,
            fmt='-o',
            color=color,
            capsize=3,
            elinewidth=1,
            label=f"Qubit {qubit_index + 1}",
            zorder=2
        )

        if show_text:
            for x, y in zip(runs, vals_array):
                if not np.isnan(y):
                    plt.text(x + 0.06, y + 2.0, f"{y:.0f}mK", ha='center', va='bottom', fontsize=11, color=color)

    plt.xlabel("Run Number")
    plt.ylabel("(weighted) Mean T2R (us)")
    plt.title("(weighted) Mean T2R vs Run Number")
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

    t2e_errs = [
        [0.46, 1.20, 5.36, 6.38, 4.08],  # Qubit 1
        [1.36, 2.27, 2.61, 5.61, 10.53],  # Qubit 2
        [0.32, 1.02, 5.65, 1.64, 4.0],  # Qubit 3,
        [0.47, 1.65, 4.57, 4.79, 7.0],  # Qubit 4,
        [0.44, 1.20, 1.09, 3.93, 5.0],  # Qubit 5
        [0.86, 1.13, 7.07, 1.54, 2.47]  # Qubit 6
    ]

    runs = np.array([4, 5, 6, 7, 8])

    num_qubits = len(t2e_vals)
    num_runs = len(t2e_vals[0])

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

    for qubit_index, vals in enumerate(t2e_vals):
        vals_array = np.array(vals, dtype=np.float64)
        errs_array = np.array(t2e_errs[qubit_index], dtype=np.float64)
        color = colors[qubit_index % len(colors)]

        plt.errorbar(
            runs, vals_array,
            yerr=errs_array,
            fmt='-o',
            color=color,
            capsize=3,
            elinewidth=1,
            label=f"Qubit {qubit_index + 1}",
            zorder=2
        )

        if show_text:
            for x, y in zip(runs, vals_array):
                if not np.isnan(y):
                    plt.text(x + 0.06, y + 2.0, f"{y:.0f}mK", ha='center', va='bottom', fontsize=11, color=color)

    plt.xlabel("Run Number")
    plt.ylabel("(weighted) Mean T2E (us)")
    plt.title("(weighted) Mean T2E vs Run Number")
    plt.xticks(runs, ['Run 4', 'Run 5', 'Run 6', 'Run 7', 'Run 8'])
    plt.yticks(np.arange(0, 100, 10))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()