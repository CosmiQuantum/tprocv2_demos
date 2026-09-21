import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import math
import os
import datetime
import matplotlib.ticker as mticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter

qtemp_noisetemp_plot = True
extra_noisepwr_calcs = False

# ------------------------------------------------------------
# Pre-computed effective qubit temperatures from box plots
#
# Central value = median
# Lower spread  = median - Q1
# Upper spread  = Q3 - median
#
# rows = qubits 1-6
# columns = Runs 5-9
# ------------------------------------------------------------

T_qubit_mK = np.array([
    [193.9678, 101.6589, 81.9942,  71.2653, 47.0581],  # Q1
    [374.3313,  84.7427, 78.1897,  75.2878, 50.5346],  # Q2
    [165.4990, 108.3733, 79.9452,  78.8874, 57.2820],  # Q3
    [359.0683, 134.7626, 91.4804, 102.8802, 59.7384],  # Q4
    [170.0469,  78.1949, 87.8694,  73.9253, np.nan],   # Q5
    [216.8889,  91.5844, 79.2936,  68.7476, 52.8615],  # Q6
], dtype=float)


T_qubit_err_lower_mK = np.array([
    [49.4010, 11.7431, 2.7645, 1.1609, 3.8532],  # Q1
    [33.0225,  0.8769, 1.1889, 1.6727, 2.9745],  # Q2
    [ 8.0601,  8.0518, 1.3146, 2.3091, 4.5796],  # Q3
    [22.9851,  7.5443, 1.7632, 6.9082, 3.8344],  # Q4
    [ 5.2266,  3.6156, 1.6989, 2.1162, np.nan],   # Q5
    [ 6.7077,  1.6694, 2.2182, 2.3415, 4.4683],  # Q6
], dtype=float)


T_qubit_err_upper_mK = np.array([
    [13.7742,  7.6677, 2.3406, 1.3248, 5.3324],  # Q1
    [38.3340,  3.3773, 1.0722, 1.4478, 4.8221],  # Q2
    [ 8.7771,  9.2237, 1.2893, 2.5915, 7.0070],  # Q3
    [21.1357, 13.2360, 2.9150, 8.3509, 6.2524],  # Q4
    [ 6.8646, 16.2053, 1.4002, 1.9969, np.nan],   # Q5
    [ 8.4017,  2.5366, 1.8165, 3.1249, 2.8429],  # Q6
], dtype=float)


# Matplotlib asymmetric yerr format:
# shape = (2, nQ, nRuns)
# [0] = lower errors
# [1] = upper errors
T_qubit_err_mK = np.array([
    T_qubit_err_lower_mK,
    T_qubit_err_upper_mK ])

# Kelvin version, needed later for the extra-noise calculations
T_qubit_K = T_qubit_mK / 1e3

# ------------------------------------------------------------
# Qubit frequencies (MHz) (Run 4 column will be dropped to match T_eff data)
# ------------------------------------------------------------
# Updated to use median frequencies from box plots. Used final data from run 9a
# rows = qubits 1-6
# columns = runs 4-9a

f_ge_MHz = [
    [4184.144430, 4181.217477, 4189.849485, 4184.045941, 4194.713716, 4226.109179],
    [3821.154079, 3821.165846, 3818.671952, 3823.357494, 3828.621892, 3853.606453],
    [4155.971724, 4154.372501, 4161.426833, 4162.874276, 4173.689466, 4197.003458],
    [4459.200054, 4458.462616, 4462.441271, 4467.354459, 4474.098290, 4506.531336],
    [4471.116747, 4471.308926, 4471.175122, 4475.031972, 4485.260921, None],
    [4997.857624, 5000.619729, 4999.514502, 5006.154276, 5018.137806, 5050.780159],
]

# Lower frequency error = median - Q1
f_ge_err_lower_MHz = [
    [0.005504, 0.014681, 0.042089, 0.006389, 0.008045, 0.003551],  # Q1
    [0.010188, 0.015217, 0.052229, 0.004570, 0.025381, 0.021308],  # Q2
    [0.288242, 0.005879, 0.022725, 0.004876, 0.017929, 0.005338],  # Q3
    [0.009701, 0.015805, 0.038501, 0.007324, 0.176231, 0.003777],  # Q4
    [0.010733, 0.008222, 1.499276, 0.009855, 0.015326, None],      # Q5
    [0.007531, 0.006948, 0.013807, 0.004278, 0.090382, 0.010142],  # Q6
]

# Upper frequency error = Q3 - median
f_ge_err_upper_MHz = [
    [0.005583, 0.018186, 0.045701, 0.006835, 0.007856, 0.005027],  # Q1
    [0.011733, 0.009682, 0.042681, 0.004667, 0.013229, 0.014331],  # Q2
    [0.278790, 0.009764, 0.023331, 0.003466, 0.025307, 0.003219],  # Q3
    [0.005705, 0.016684, 0.038672, 0.008156, 0.046889, 0.004086],  # Q4
    [0.004826, 0.009995, 6.120989, 0.008055, 0.015631, None],      # Q5
    [0.006427, 0.005675, 0.024143, 0.003753, 0.022988, 0.014160],  # Q6
]

# ------------------------------------------------------------
# Convert + drop Run 4 so arrays align with Runs 5-9
# ------------------------------------------------------------
runs = np.array([5, 6, 7, 8, 9], dtype=int)

f_ge_MHz = np.array(f_ge_MHz, dtype=object).astype(float)[:, 1:]
f_ge_Hz = f_ge_MHz * 1e6

if T_qubit_mK.shape != f_ge_Hz.shape:
    raise ValueError(
        f"Shape mismatch: T_qubit_mK {T_qubit_mK.shape} "
        f"vs f_ge_Hz {f_ge_Hz.shape}"
    )
# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
h = 6.62607015e-34
kB = 1.380649e-23

# ------------------------------------------------------------
# Noise temperature model (MIT supplement style)
# ------------------------------------------------------------
if qtemp_noisetemp_plot:
    def nbar_thermal(f_hz: float, T_K: float):
        if T_K <= 0:
            return 0.0
        x = (h * f_hz) / (kB * T_K)
        if x > 700:
            return 0.0
        return 1.0 / (np.exp(x) - 1.0)

    def Te_from_nbar(f_hz: float, nbar: float):
        if nbar <= 0:
            return 0.0
        x = np.log(1.0 + 1.0 / nbar)
        return (h * f_hz) / (kB * x)

    def Pe_from_nbar(nbar: float):
        # thermal two-level population written directly in terms of effective photon occupation
        if nbar <= 0:
            return 0.0
        return nbar / (2.0 * nbar + 1.0)

    def Te_Pe_from_stages(f_hz: float, stage_temps_K: dict, A_after_stage_dB: dict):
        n_eff = 0.0
        for stage, T in stage_temps_K.items():
            if stage not in A_after_stage_dB:
                raise ValueError(f"Missing attenuation-after-stage entry for '{stage}'")
            A_lin = 10 ** (-A_after_stage_dB[stage] / 10.0)
            n_eff += A_lin * nbar_thermal(f_hz, T)
        return Te_from_nbar(f_hz, n_eff), Pe_from_nbar(n_eff)

    def cumulative_after_stage(config_dB: dict, order):
        cfg = {k: float(config_dB.get(k, 0.0)) for k in order}
        A_after = {}
        for i, stage in enumerate(order):
            A_after[stage] = sum(cfg[order[j]] for j in range(i + 1, len(order)))
        return A_after

    order = ["300K", "50K", "4K", "still", "CP", "MXC"]

    stage_temps_K = {
        "300K": 300.0,
        "50K": 50.0,
        "4K": 4.0,
        "still": 1.0,
        "CP": 0.100,
        "MXC": 0.010,
    }

    IL_marki = 0.9
    IL_eccosorb = 1.0

    # ------------------------------------------------------------
    # A11 input-line loss model at 6 GHz
    # ------------------------------------------------------------
    # The thermal-noise model begins at the 300 K fridge feedthrough,
    # rather than at the external patch panel.
    #
    # Values below are taken from the same batch of VNA measurements:
    #   full measured A11 loss, patch panel -> MCP1       = 80.0 +/- 0.3 dB
    #   patch-panel cable                                 = 3.40 dB
    #   patch panel -> 50 K                               = 4.95 dB
    #   50 K -> 4 K, labeled A11 4K                      = 2.44 dB
    #   4 K -> Still/1 K, labeled A11 Still               = 2.08 dB
    #
    # Since the approximately 3.4 dB patch-panel cable lies upstream of
    # the new 300 K model boundary, it is excluded from the attenuation
    # used in the thermal-noise calculation.
    #
    # Therefore:
    #   full A11 input, 300 K fridge feedthrough -> MCP1 = 80.0 - 3.4 = 76.6 dB
    #   300 K feedthrough -> 50 K                       = 4.95 - 3.4 = 1.55 dB
    #
    # Index each inter-stage line loss by its colder endpoint.
    # This allows cumulative_after_stage() to include the loss
    # for thermal sources originating at all warmer stages:
    #   300 K fridge feedthrough -> 50 K   goes at 50K
    #   50 K -> 4 K                        goes at 4K
    #   4 K -> Still/1 K                   goes at still
    #
    # The remaining line loss is assigned to the lower cold sections
    # using their relative cable lengths:
    #   Still/1 K -> CP/100 mK   = 23.5 cm
    #   CP/100 mK -> MXC/10 mK   = 30.5 cm

    A11_total_line_loss_dB = 10.8  # (patch panel to MCP atten) minus (patch panel cable atten) # 14.2-3.4 dB

    lineloss_by_stage_until_MCP1_dB = {  # from VNA measurements, 6 GHz
        "50K": 1.55,  # 4.95 - 3.4 dB # (atten of patch panel to 50K) minus (patch panel cable atten)
        "4K": 2.44,
        "still": 2.08,
    }

    A11_assigned_loss_dB = sum(lineloss_by_stage_until_MCP1_dB.values())
    A11_remaining_loss_dB = A11_total_line_loss_dB - A11_assigned_loss_dB

    # Split the remaining loss by lower-stage line length
    L_still_to_CP_cm = 23.5
    L_CP_to_MXC_cm = 30.5
    L_lower_total_cm = L_still_to_CP_cm + L_CP_to_MXC_cm

    lineloss_by_stage_until_MCP1_dB["CP"] = A11_remaining_loss_dB * L_still_to_CP_cm / L_lower_total_cm
    lineloss_by_stage_until_MCP1_dB["MXC"] = A11_remaining_loss_dB * L_CP_to_MXC_cm / L_lower_total_cm

    # ------------------------------------------------------------
    # MCP1 -> device line loss by run
    # ------------------------------------------------------------
    # These values are the "Total loss from lines only" estimates for the coaxial lines below MCP1. Measurements by Arianna.
    # Component losses such as the Marki filter and Eccosorbs are already included separately below.
    # Attenuation is stored as a positive dB magnitude. Runs 5-8 use reported upper-limit estimates.

    mcp1_to_device_line_loss_by_run_dB = {
        5: 1.50,
        6: 1.50,
        7: 1.32,
        8: 1.32,
        9: 1.68 }

    # ------------------------------------------------------------
    # In-plate attenuators and filters by run
    # ------------------------------------------------------------
    atten_config_by_run = {
        5: {
            "4K": 20,
            "still": 20,
            "MXC": 20 + 3 * IL_eccosorb + IL_marki
        },
        6: {
            "4K": 20,
            "still": 20,
            "MXC": 20 + 3 * IL_eccosorb + IL_marki
        },
        7: {
            "4K": 20,
            "still": 6,
            "CP": 10,
            "MXC": 30 + 3 * IL_eccosorb + IL_marki
        },
        8: {
            "4K": 20,
            "still": 6,
            "CP": 10,
            "MXC": 30 + 3 * IL_eccosorb + IL_marki
        },
        9: {
            "4K": 20,
            "still": 6,
            "CP": 10,
            "MXC": 30 + 3 * IL_eccosorb + IL_marki
        },
    }

    nQ, nRuns = T_qubit_mK.shape
    Te_mK = np.zeros((nQ, nRuns), dtype=float)
    Pe_pred = np.zeros((nQ, nRuns), dtype=float)

    for qi in range(nQ):
        for ri, r in enumerate(runs):
            r = int(r)
            cfg = dict(atten_config_by_run[r])  # copy so we don't mutate the base dict

            # Add the distributed A11 input-line loss from the 300 K fridge feedthrough down to MCP1.
            # Do NOT put this at 300K; assign each line section to its colder endpoint.
            for stage, loss_dB in lineloss_by_stage_until_MCP1_dB.items():
                cfg[stage] = cfg.get(stage, 0.0) + loss_dB

            # Add the run-dependent line loss between MCP1 and the device.
            # This section lies at the cold end of the chain, so its attenuation is assigned to MXC.
            cfg["MXC"] = cfg.get("MXC", 0.0) + mcp1_to_device_line_loss_by_run_dB[r]

            A_after = cumulative_after_stage(cfg, order)

            f_hz = f_ge_Hz[qi, ri]
            Te_K, Pe = Te_Pe_from_stages(f_hz, stage_temps_K, A_after)

            Te_mK[qi, ri] = 1e3 * Te_K
            Pe_pred[qi, ri] = Pe

    print("\nPredicted Te (mK) and Pe (%) by qubit & run (using per-run f_ge):")
    for qi in range(nQ):
        vals = ", ".join([
            f"R{int(runs[i])}: Te={Te_mK[qi, i]:.4f} mK, Pe={100 * Pe_pred[qi, i]:.4f}%"
            for i in range(nRuns)
        ])
        print(f"  Q{qi + 1}: {vals}")


    if extra_noisepwr_calcs:
        # EXTRA NOISE POWER needed to go from model Te -> measured Tqubit
        # Prints PSD and total power in BOTH linear units and dBm units.
        # Also prints "signed" (direction) and "needed" (clipped at 0) versions.

        def nbar_from_T(f_hz, T_K):
            f = np.asarray(f_hz, dtype=float)
            T = np.asarray(T_K, dtype=float)

            if np.any(T <= 0):
                raise ValueError("nbar_from_T received non-positive temperature(s).")

            x = (h * f) / (kB * T)

            n = np.where(
                x > 700,
                0.0,
                1.0 / (np.exp(x) - 1.0)
            )
            return n


        def to_dBm(P_W):
            """Convert power in Watts to dBm (uses magnitude so signed values don't break log)."""
            return 10.0 * np.log10(np.abs(P_W) / 1e-3 + 1e-300)


        Te_K = Te_mK / 1e3  # (nQ, nRuns)
        Tq_K = T_qubit_K  # (nQ, nRuns)

        n_meas = nbar_from_T(f_ge_Hz, Tq_K)
        n_model = nbar_from_T(f_ge_Hz, Te_K)

        # Signed: tells direction (model above/below)
        delta_n_signed = n_meas - n_model

        # Needed: answers "how much extra is needed" (never negative)
        delta_n_needed = np.maximum(delta_n_signed, 0.0)

        # ---- PSD (W/Hz): PSD = \u0394n * h f
        PSD_W_per_Hz_signed = delta_n_signed * h * f_ge_Hz
        PSD_W_per_Hz_needed = delta_n_needed * h * f_ge_Hz

        PSD_dBm_per_Hz_signed = to_dBm(PSD_W_per_Hz_signed)
        PSD_dBm_per_Hz_needed = to_dBm(PSD_W_per_Hz_needed)

        # ---- Total power in bandwidth B_Hz
        B_Hz = 1.0  # change if you want (e.g., IF BW, resonator linewidth, etc.)

        P_W_signed = PSD_W_per_Hz_signed * B_Hz
        P_W_needed = PSD_W_per_Hz_needed * B_Hz

        P_dBm_signed = to_dBm(P_W_signed)
        P_dBm_needed = to_dBm(P_W_needed)

        print("\nExtra noise to go from model -> measured (reported at device input):")
        print("Signed = (measured - model). Needed = max(Signed, 0).")
        print(f"Using B_Hz = {B_Hz:g} Hz for total power.\n")

        for qi in range(nQ):
            print(f"Q{qi + 1}:")
            for ri, r in enumerate(runs):
                print(
                    f"  R{int(r)} | "
                    f"\u0394n_signed={delta_n_signed[qi, ri]: .3e}, \u0394n_needed={delta_n_needed[qi, ri]: .3e} | "
                    f"PSD_signed={PSD_W_per_Hz_signed[qi, ri]: .3e} W/Hz ({PSD_dBm_per_Hz_signed[qi, ri]: .1f} dBm/Hz), "
                    f"PSD_needed={PSD_W_per_Hz_needed[qi, ri]: .3e} W/Hz ({PSD_dBm_per_Hz_needed[qi, ri]: .1f} dBm/Hz) | "
                    f"P_signed={P_W_signed[qi, ri]: .3e} W ({P_dBm_signed[qi, ri]: .1f} dBm), "
                    f"P_needed={P_W_needed[qi, ri]: .3e} W ({P_dBm_needed[qi, ri]: .1f} dBm)"
                )
            print()

        # ------------------------------------------------------------
        # Print photon occupations and check inequality:
        # 0 < n_model < n_meas << 1   (where "<<1" is heuristic)
        # ------------------------------------------------------------

        print("\nPhoton occupation at f_ge:")
        for qi in range(nQ):
            print(f"Q{qi + 1}:")
            for ri, r in enumerate(runs):
                nm = n_model[qi, ri]
                nM = n_meas[qi, ri]
                print(
                    f"  R{int(r)} | n_model={nm:.3e}  n_meas={nM:.3e}  ratio(meas/model)={(nM / nm if nm > 0 else np.inf):.4f}")

        # Checks
        all_pos_model = np.all(n_model > 0)
        all_pos_meas = np.all(n_meas > 0)
        model_lt_meas = np.all(n_model < n_meas)

        # "<<" is not a strict math symbol, so pick a threshold to report
        # Common choices: 0.1 (very small) or 0.01 (tiny). I'll show both.
        meas_lt_0p1 = np.all(n_meas < 0.1)
        meas_lt_0p01 = np.all(n_meas < 0.01)

        print("\nInequality checks:")
        print(f"  0 < n_model: {all_pos_model}")
        print(f"  0 < n_meas:  {all_pos_meas}")
        print(f"  n_model < n_meas: {model_lt_meas}")

        print("\nSummary stats:")
        print(f"  n_model: min={np.min(n_model):.3e}, max={np.max(n_model):.3e}")
        print(f"  n_meas : min={np.min(n_meas):.3e}, max={np.max(n_meas):.3e}")

        # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Plot: T_qubit(from Pe) vs Predicted Noise Temperature
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    # AB paper run labels
    display_labels = ['2', '3', '4', '5', '6']
    runs_arr = np.asarray(runs)

    for qi in range(nQ):
        ax = axes[qi]

        # --- Run 5 -> Run 6 segment  ---
        ax.errorbar(
            runs[:2], T_qubit_mK[qi][:2],
            yerr=T_qubit_err_mK[:, qi, :2], #Note: T_qubit_err_mK[0, qi, ri] = lower error & T_qubit_err_mK[1, qi, ri] = upper error
            fmt="o-",
            color="#E69F00", #color-blind friendly orange
            capsize=3,
            elinewidth=1,
            label=r"$T_{\mathrm{eff}}$ (SSF)"  # (Run 5 SSF)
        )

        # --- Run 6 -> Run 8 segment ---
        ax.errorbar(
            runs[1:], T_qubit_mK[qi][1:],
            yerr=T_qubit_err_mK[:, qi, 1:],
            fmt="o-",
            color="brown",
            capsize=3,
            elinewidth=1,
            label=r"$T_{\mathrm{eff}}$ (RPM)"  # (Runs 6-9 RPM)
        )

        # --- Predicted noise temperature ---
        ax.plot(
            runs,
            Te_mK[qi],
            "s--",
            color="black",
            linewidth=2,
            label=r"$T_e$ (pred. noise)"
        )

        # ------------------------------------------------------------
        # Inset: automatically use last two finite runs for each qubit
        # ------------------------------------------------------------
        valid = (
                np.isfinite(T_qubit_mK[qi]) &
                np.isfinite(T_qubit_err_lower_mK[qi]) &
                np.isfinite(T_qubit_err_upper_mK[qi]) &
                np.isfinite(Te_mK[qi]) )

        valid_idx = np.where(valid)[0]

        if len(valid_idx) >= 2:
            idx2 = valid_idx[-2:]
            x_zoom = runs_arr[idx2]

            axins = ax.inset_axes([0.54, 0.48, 0.42, 0.45])

            # --- Measured temperature ---
            axins.errorbar(
                x_zoom,
                T_qubit_mK[qi][idx2],
                yerr=T_qubit_err_mK[:, qi, idx2],
                fmt="o-",
                color="brown",
                capsize=2,
                elinewidth=1,
                markersize=4
            )

            # --- Predicted noise temperature ---
            axins.plot(
                x_zoom,
                Te_mK[qi][idx2],
                "s--",
                color="black",
                linewidth=1.5,
                markersize=4)

            # --- Inset x-axis ---
            axins.set_xlim(x_zoom[0] - 0.15, x_zoom[-1] + 0.15)
            axins.set_xticks(x_zoom)
            axins.set_xticklabels([display_labels[i] for i in idx2],fontsize=18)

            # --- Inset y-axis ---
            yvals = np.concatenate([
                T_qubit_mK[qi][idx2] - T_qubit_err_lower_mK[qi][idx2],
                T_qubit_mK[qi][idx2] + T_qubit_err_upper_mK[qi][idx2],
                Te_mK[qi][idx2]
            ])

            yvals = yvals[np.isfinite(yvals)]

            ymin = np.min(yvals)
            ymax = np.max(yvals)

            pad = max(5, 0.12 * (ymax - ymin))

            axins.set_ylim(ymin - pad, ymax + pad)

            # Exactly 3 y-axis ticks
            axins.yaxis.set_major_locator(LinearLocator(4))
            axins.yaxis.set_major_formatter(FormatStrFormatter('%d'))

            axins.tick_params(axis="both", labelsize=16)
            axins.grid(True, alpha=0.35)

        # ------------------------------------------------------------
        # Main axis formatting
        # ------------------------------------------------------------
        ax.set_title(f"Qubit {qi + 1}", fontsize=22)

        ax.set_ylim(0, 450)
        ax.set_yticks(np.arange(0, 401, 100))

        ax.set_xticks(runs)

        # ax.set_xticklabels(['5', '6', '7', '8', '9'], fontsize=22) # original QUIET run labels
        ax.set_xticklabels(['2', '3', '4', '5', '6'], fontsize=22)  # for AB paper

        ax.tick_params(axis='y', labelsize=22)
        ax.tick_params(axis='x', labelsize=22)
        ax.grid(True)

    # ---------------- Big title ----------------
    # fig.suptitle(
    #     "Measured Effective Qubit Temperature ($T_{eff}$) vs Predicted Noise Temperature ($T_e$)",
    #     y=0.96, fontsize=24)

    # ---------------- Axis labels ----------------
    fig.supylabel("Effective Temperature (mK)", fontsize=22, x=0.05)

    # ---------------- Legend ----------------
    handles, labels = axes[0].get_legend_handles_labels()

    # Leave space at bottom for legend
    fig.tight_layout(rect=[0.05, 0.12, 1, 0.95])

    # Center legend and x-label with second column
    second_col_pos = axes[1].get_position()
    second_col_center = second_col_pos.x0 + second_col_pos.width / 2

    fig.supxlabel("Run Number", fontsize=22, x=second_col_center, y=0.12)
    fig.legend(handles, labels, ncol=3, frameon=True, loc = "upper center", bbox_to_anchor=(second_col_center, 1.04), fontsize=22)

    # Leave space at bottom for legend
    fig.tight_layout(rect=[0.05, 0.12, 1, 0.95])

    save_path = "/home/acolonce/Documents/analysis/multirun/qubit_temps/combined_ssf_rpm/Tpred_vs_Tmeas_inputlinemodel.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()