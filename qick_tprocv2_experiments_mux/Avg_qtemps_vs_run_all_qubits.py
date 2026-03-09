import matplotlib.pyplot as plt
import numpy as np

runs6_through_8_rpm = False
plot_both_SSF_RPM_tog = True
show_text = False
colors = ['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred']

if runs6_through_8_rpm is True and not plot_both_SSF_RPM_tog:
    print("runs6_through_8_rpm")
    qubit_temps = [
        [197.35, 99.22, 81.46, 71.21],  # Qubit 1
        [361.39, 84.26, 78.32, 75.36],  # Qubit 2
        [164.24, 106.43, 80.11, 78.82],  # Qubit 3
        [354.34, 135.15, 91.27, 100.85],  # Qubit 4
        [169.67, 76.82, 87.88, 74.10],  # Qubit 5
        [216.55, 91.56, 78.55, 64.90],  # Qubit 6
    ]

    qtemp_errs = [
        [12.76, 9.99, 2.52, 1.26],  # Qubit 1
        [33.57, 2.74, 1.18, 1.64],  # Qubit 2
        [8.37, 8.37, 1.37, 2.55],  # Qubit 3
        [23.71, 10.16, 2.38, 7.78],  # Qubit 4
        [5.94, 5.77, 1.58, 2.07],  # Qubit 5
        [7.80, 1.65, 1.98, 3.96],  # Qubit 6
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

    # --- Add run-change "bubbles" (one per run) ---
    run_notes = {
        5: "Added eccosorbs\nand LPFs",
        6: "Added 0dB + copper tape to\n mag can pass-through holes\n and qubit package ",
        7: "Mag can lid sealing\n with copper tape\n + changed attenuation",
        8: "HERD1 filter\n+ 0dB",
    }

    # Convert to array for easy column operations
    temps_mat = np.array(qubit_temps, dtype=float)  # shape (n_qubits, n_runs)

    # choose an anchor y-value for each run (median across qubits)
    y_anchor = np.nanmedian(temps_mat, axis=0)

    for i, r in enumerate(runs):
        note = run_notes.get(int(r), None)
        if note is None or np.isnan(y_anchor[i]):
            continue

        # Place bubble slightly above the median point cluster
        xy = (r, y_anchor[i])
        xytext = (r + 0.40, y_anchor[i] + 70) # tweak offsets to taste

        plt.annotate(
            note,
            xy=xy,
            xytext=xytext,
            textcoords="data",
            ha="left",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
            arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"),
            zorder=5
        )

    plt.xlabel("Run Number")
    plt.ylabel("Average Effective Qubit Temperature (mK)")
    plt.title("Average Effective Qubit Temperature vs Run Number")
    plt.xticks(runs, ['Run 5\n(SSF Meas.)', 'Run 6\n(Rabi Pop. Meas.)', 'Run 7\n(Rabi Pop. Meas.)', 'Run 8\n(Rabi Pop. Meas.)'])
    plt.yticks(np.arange(100, 351, 25))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
elif not runs6_through_8_rpm and not plot_both_SSF_RPM_tog: # Only SSF qubit temps, OUTDATED
    qubit_temps = [
        [197.35, 100.98, 89.26, 77.21],  # Qubit 1
        [361.39, None, 85.85, 85.96],  # Qubit 2
        [164.24, None, 84.47, 85.38],  # Qubit 3
        [354.34, None, None, None],  # Qubit 4
        [169.67, 88.59, 93.73, 81.36],  # Qubit 5
        [216.55, None, None, None],  # Qubit 6
    ]

    qtemp_errs = [
        [12.76, 3.84, 2.26, 1.88],  # Qubit 1
        [33.57, None, 2.58, 3.09],  # Qubit 2
        [8.37, None, 2.02, 2.83],  # Qubit 3
        [23.71, None, None, None],  # Qubit 4
        [5.94, 2.98, 2.29, 1.14],  # Qubit 5
        [7.80, None, None, None],  # Qubit 6
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
        temps_array = np.array(temps, dtype=float)
        errs_array = np.array(qtemp_errs[qubit_index], dtype=float)
        color = colors[qubit_index % len(colors)]

        mask = np.isfinite(temps_array) & np.isfinite(errs_array)  # keep only real points w/ real errors

        plt.errorbar(
            runs[mask], temps_array[mask],
            yerr=errs_array[mask],
            fmt='-o',
            color=color,
            capsize=3,
            elinewidth=1,
            label=f"Qubit {qubit_index + 1}",
            zorder=2
        )

        if show_text:
            for x, y in zip(runs[mask], temps_array[mask]):
                plt.text(x + 0.06, y + 2.0, f"{y:.0f}mK",
                         ha='center', va='bottom', fontsize=11, color=color)

        # --- Add run-change "bubbles" (one per run) ---
        run_notes = {
            5: "Added eccosorbs\nand LPFs",
            6: "Added 0dB + copper tape to\n mag can pass-through holes\n and qubit package ",
            7: "Mag can lid sealing\n with copper tape\n + changed attenuation",
            8: "HERD1 filter\n+ 0dB",
        }

        # Convert to array for easy column operations
        temps_mat = np.array(qubit_temps, dtype=float)  # shape (n_qubits, n_runs)

        # choose an anchor y-value for each run (median across qubits)
        y_anchor = np.nanmedian(temps_mat, axis=0)

        for i, r in enumerate(runs):
            note = run_notes.get(int(r), None)
            if note is None or np.isnan(y_anchor[i]):
                continue

            # Place bubble slightly above the median point cluster
            xy = (r, y_anchor[i])
            xytext = (r + 0.40, y_anchor[i] + 60)  # tweak offsets to taste

            plt.annotate(
                note,
                xy=xy,
                xytext=xytext,
                textcoords="data",
                ha="left",
                va="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"),
                zorder=5
            )

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

if plot_both_SSF_RPM_tog:
    # ---------------- USER OPTIONS ----------------
    qubits_to_plot = [1]  # qubit numbers (1-based)
    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred'] # if u want to do per qubit
    ssf_color = "purple" # if u wanna do per type of measurement
    rpm_color = "palevioletred" # if u wanna do per type of measurement
    runs = np.array([5, 6, 7, 8])

    # ---------------- RPM DATA ----------------
    rpm_qubit_temps = [
        [None, 99.22, 81.46, 71.21],  # Qubit 1
        [None, 84.26, 78.32, 75.36],  # Qubit 2
        [None, 106.43, 80.11, 78.82],  # Qubit 3
        [None, 135.15, 91.27, 100.85],  # Qubit 4
        [None, 76.82, 87.88, 74.10],  # Qubit 5
        [None, 91.56, 78.55, 64.90],  # Qubit 6
    ]

    rpm_qtemp_errs = [
        [None, 9.99, 2.52, 1.26],  # Qubit 1
        [None, 2.74, 1.18, 1.64],  # Qubit 2
        [None, 8.37, 1.37, 2.55],  # Qubit 3
        [None, 10.16, 2.38, 7.78],  # Qubit 4
        [None, 5.77, 1.58, 2.07],  # Qubit 5
        [None, 1.65, 1.98, 3.96],  # Qubit 6
    ]

    # ---------------- SSF DATA ----------------
    ssf_qubit_temps = [
        [197.35, 100.98, 89.26, 77.21],  # Qubit 1
        [361.39, None, 85.85, 85.96],  # Qubit 2
        [164.24, None, 84.47, 85.38],  # Qubit 3
        [354.34, None, None, None],  # Qubit 4
        [169.67, 88.59, 93.73, 81.36],  # Qubit 5
        [216.55, None, None, None],  # Qubit 6
    ]

    ssf_qtemp_errs = [
        [12.76, 3.84, 2.26, 1.88],  # Qubit 1
        [33.57, None, 2.58, 3.09],  # Qubit 2
        [8.37, None, 2.02, 2.83],  # Qubit 3
        [23.71, None, None, None],  # Qubit 4
        [5.94, 2.98, 2.29, 1.14],  # Qubit 5
        [7.80, None, None, None],  # Qubit 6
    ]

    # ---------------- PLOT ----------------
    plt.figure(figsize=(9, 6))

    # Background watermark
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

    # Keep track of plotted temps for annotation anchor points
    all_plotted_temps = []

    for qubit_num in qubits_to_plot:
        q_idx = qubit_num - 1
        color = colors[q_idx % len(colors)]

        # ---------- SSF ----------
        ssf_temps = np.array(ssf_qubit_temps[q_idx], dtype=float)
        ssf_errs = np.array(ssf_qtemp_errs[q_idx], dtype=float)
        ssf_mask = np.isfinite(ssf_temps) & np.isfinite(ssf_errs)

        if np.any(ssf_mask):
            plt.errorbar(
                runs[ssf_mask],
                ssf_temps[ssf_mask],
                yerr=ssf_errs[ssf_mask],
                fmt='--s',
                color=ssf_color,
                capsize=3,
                elinewidth=1,
                linewidth=1.8,
                markersize=6,
                label=f"Q{qubit_num}, SSF method ",
                zorder=2
            )
            all_plotted_temps.append(ssf_temps)

            if show_text:
                for x, y in zip(runs[ssf_mask], ssf_temps[ssf_mask]):
                    plt.text(
                        x - 0.06, y + 2.0, f"{y:.0f}mK",
                        ha='center', va='bottom',
                        fontsize=10, color=color
                    )

        # ---------- RPM ----------
        rpm_temps = np.array(rpm_qubit_temps[q_idx], dtype=float)
        rpm_errs = np.array(rpm_qtemp_errs[q_idx], dtype=float)
        rpm_mask = np.isfinite(rpm_temps) & np.isfinite(rpm_errs)

        if np.any(rpm_mask):
            plt.errorbar(
                runs[rpm_mask],
                rpm_temps[rpm_mask],
                yerr=rpm_errs[rpm_mask],
                fmt='-o',
                color=rpm_color,
                capsize=3,
                elinewidth=1,
                linewidth=1.8,
                markersize=6,
                label=f"Q{qubit_num}, RPM method",
                zorder=3
            )
            all_plotted_temps.append(rpm_temps)

            if show_text:
                for x, y in zip(runs[rpm_mask], rpm_temps[rpm_mask]):
                    plt.text(
                        x + 0.06, y + 2.0, f"{y:.0f}mK",
                        ha='center', va='bottom',
                        fontsize=10, color=color
                    )

    # ---------------- Run-change notes ----------------
    run_notes = {
        5: "Added eccosorbs\nand LPFs",
        6: "Added 0dB + copper tape to\nmag can pass-through holes\nand qubit package",
        7: "Mag can lid sealing\nwith copper tape\n+ changed attenuation",
        8: "HERD1 filter\n+ 0dB",
    }

    if len(all_plotted_temps) > 0:
        temps_mat = np.array(all_plotted_temps, dtype=float)
        y_anchor = np.nanmedian(temps_mat, axis=0)

        # for i, r in enumerate(runs):
        #     note = run_notes.get(int(r), None)
        #     if note is None or np.isnan(y_anchor[i]):
        #         continue
        #
        #     xy = (r, y_anchor[i])
        #     xytext = (r , y_anchor[i] + 20)
        #
        #     plt.annotate(
        #         note,
        #         xy=xy,
        #         xytext=xytext,
        #         textcoords="data",
        #         ha="left",
        #         va="center",
        #         fontsize=8,
        #         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
        #         arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"),
        #         zorder=5
        #     )

    # ---------------- Formatting ----------------
    plt.xlabel("Run Number", fontsize=16)
    plt.ylabel("Effective Qubit Temperature (mK)", fontsize=16)
    plt.title("Median Effective Qubit Temperature vs Run Number", fontsize=18)

    plt.xticks(
        runs,
        [
            'Run 5\n(SSF)',
            'Run 6\n(RPM)',
            'Run 7\n(RPM)',
            'Run 8\n(RPM)',
        ],
        fontsize=16
    )

    plt.yticks(np.arange(50, 201, 20), fontsize=16)
    plt.grid(True, alpha=0.7)
    plt.legend(ncol=2, fontsize=16)
    plt.tight_layout()
    plt.show()

    # ---------------- PRINT SSF vs RPM DIFFERENCE ----------------
    print("\nDifference between RPM and SSF temperatures (RPM - SSF)\n")

    for qubit_num in qubits_to_plot:
        q_idx = qubit_num - 1

        rpm = np.array(rpm_qubit_temps[q_idx], dtype=float)
        ssf = np.array(ssf_qubit_temps[q_idx], dtype=float)

        diff = rpm - ssf

        for i, run in enumerate(runs):
            if np.isfinite(diff[i]):
                print(f"Qubit {qubit_num}, Run {run}: \u0394T = {diff[i]:.2f} mK")
