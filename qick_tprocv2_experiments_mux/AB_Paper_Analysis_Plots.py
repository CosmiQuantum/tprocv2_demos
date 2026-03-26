import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import math
import os
import datetime

qtemp_noisetemp_plot = False
lnPe_vs_qfreq_plots_per_run = False
lnPe_vs_qfreq_plots_per_qubit = False
ssf_fid_vs_Pe = False
Pe_variance_std_vs_run = False
# ------------------------------------------------------------
# Measured Pe values (Run 4 column will be dropped)
# ------------------------------------------------------------
# Updated for APS
Pe_meas = [
    [None, 0.262206, 0.122347, 0.080209, 0.056575],  # Qubit 1
    [None, 0.379916, 0.103899, 0.088050, 0.080829],  # Qubit 2
    [None, 0.230638, 0.137551, 0.076606, 0.073828],  # Qubit 3
    [None, 0.355278, 0.170338, 0.088298, 0.111364],  # Qubit 4
    [None, 0.220639, 0.061004, 0.080564, 0.052125],  # Qubit 5
    [None, 0.248520, 0.068520, 0.046608, 0.029532],  # Qubit 6
]

Pe_err = [
    [None, 0.038575, 0.020341, 0.005599, 0.002620],  # Qubit 1
    [None, 0.010940, 0.005009, 0.002714, 0.003743],  # Qubit 2
    [None, 0.010852, 0.017270, 0.002867, 0.005371],  # Qubit 3
    [None, 0.008452, 0.016899, 0.004802, 0.015093],  # Qubit 4
    [None, 0.007670, 0.020126, 0.003180, 0.003983],  # Qubit 5
    [None, 0.007161, 0.003828, 0.003404, 0.003975],  # Qubit 6
]

# ------------------------------------------------------------
# Qubit frequencies (MHz) (Run 4 column will be dropped to match Pe data)
# ------------------------------------------------------------
# updated to use medians from box plots and IQR/2
f_ge_MHz = [
    [4184.144999, 4181.217423, 4189.847610, 4184.045941, 4194.713445],
    [3821.165184, 3821.165380, 3818.674021, 3823.357495, 3828.615932],
    [4155.703143, 4154.373711, 4161.427498, 4162.873654, 4173.694786],
    [4459.199056, 4458.460059, 4462.441896, 4467.354628, 4474.099712],
    [4471.119418, 4471.311651, 4471.163478, 4475.032108, 4485.260921],
    [4997.851927, 5000.618976, 4999.514524, 5006.153588, 5018.134399],
]

f_ge_err_MHz = [
    [0.005309, 0.020415, 0.044104, 0.006472, 0.007646],
    [0.012198, 0.014326, 0.046804, 0.004549, 0.019867],
    [0.152578, 0.007039, 0.023183, 0.004947, 0.021120],
    [0.007611, 0.015656, 0.040708, 0.008552, 0.088323],
    [0.010902, 0.009128, 3.811058, 0.009040, 0.015478],
    [0.002007, 0.006477, 0.018931, 0.003672, 0.060262],
]

# -------------------------------------------------------------------------
# Single-shot fidelity values (Run 4 column will be dropped to match Pe data)
# -------------------------------------------------------------------------
ssf_fid_vals = [
    [v5, v6, v7, v8],  # Qubit 1
    [v5, v6, v7, v8],  # Qubit 2
    [v5, v6, v7, v8],  # Qubit 3
    [v5, v6, v7, v8],  # Qubit 4
    [v5, v6, v7, v8],  # Qubit 5
    [v5, v6, v7, v8],  # Qubit 6
]
# -------------------------------------------------------------------------
# Thermal population variance (spread) values
# -------------------------------------------------------------------------
Pe_variance_vals = [
    [v5, v6, v7, v8],  # Qubit 1
    [v5, v6, v7, v8],  # Qubit 2
    [v5, v6, v7, v8],  # Qubit 3
    [v5, v6, v7, v8],  # Qubit 4
    [v5, v6, v7, v8],  # Qubit 5
    [v5, v6, v7, v8],  # Qubit 6
]

# ------------------------------------------------------------
# Convert + drop Run 4 so arrays align with Runs 5-8
# ------------------------------------------------------------
runs = np.array([5, 6, 7, 8], dtype=int)

Pe_meas = np.array(Pe_meas, dtype=object).astype(float)[:, 1:]   # (6,4)
Pe_err  = np.array(Pe_err,  dtype=object).astype(float)[:, 1:]   # (6,4)

f_ge_MHz     = np.array(f_ge_MHz, dtype=float)[:, 1:]            # (6,4)
f_ge_err_MHz = np.array(f_ge_err_MHz, dtype=float)[:, 1:]        # (6,4)

f_ge_Hz     = f_ge_MHz * 1e6
f_ge_err_Hz = f_ge_err_MHz * 1e6

ssf_fid_vals   = np.array(ssf_fid_vals, dtype=float)[:, 1:]            # (6,4)

if Pe_meas.shape != f_ge_Hz.shape:
    raise ValueError(f"Shape mismatch: Pe_meas {Pe_meas.shape} vs f_ge_Hz {f_ge_Hz.shape}")

# ------------------------------------------------------------
# Infer T_qubit from Pe
# T = hf / (kB * ln((1-Pe)/Pe))
# ------------------------------------------------------------
h  = 6.62607015e-34
kB = 1.380649e-23

def T_from_Pe(Pe, f_Hz):
    return (h * f_Hz) / (kB * np.log((1 - Pe) / Pe))  # Kelvin

T_qubit_K  = T_from_Pe(Pe_meas, f_ge_Hz)
T_qubit_mK = 1e3 * T_qubit_K

# ------------------------------------------------------------
# Propagate errors: Pe_err (and optionally f_err) -> T_err
# ------------------------------------------------------------
L = np.log((1 - Pe_meas) / Pe_meas)

dT_dPe_K = (h * f_ge_Hz / kB) * (1.0 / (L**2)) * (1.0 / (1 - Pe_meas) + 1.0 / Pe_meas)
dT_df_K_per_Hz = h / (kB * L)

T_qubit_err_mK = 1e3 * np.sqrt((dT_dPe_K * Pe_err)**2 + (dT_df_K_per_Hz * f_ge_err_Hz)**2)

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

    def Te_from_stages(f_hz: float, stage_temps_K: dict, A_after_stage_dB: dict):
        n_eff = 0.0
        for stage, T in stage_temps_K.items():
            if stage not in A_after_stage_dB:
                raise ValueError(f"Missing attenuation-after-stage entry for '{stage}'")
            A_lin = 10 ** (-A_after_stage_dB[stage] / 10.0)
            n_eff += A_lin * nbar_thermal(f_hz, T)
        return Te_from_nbar(f_hz, n_eff)

    def cumulative_after_stage(config_dB: dict, order):
        cfg = {k: float(config_dB.get(k, 0.0)) for k in order}
        A_after = {}
        for i, stage in enumerate(order):
            A_after[stage] = sum(cfg[order[j]] for j in range(i + 1, len(order)))
        return A_after

    order = ["300K", "4K", "1K", "100mK", "10mK"]
    stage_temps_K = {"300K": 300.0, "4K": 4.0, "1K": 1.0, "100mK": 0.100, "10mK": 0.010}

    IL_marki = 0.9
    IL_eccosorb = 1.0
    SS_line_loss_total_dB = 14.0

    atten_config_by_run = {
        5: {
            "4K": 20,
            "1K": 20,
            "10mK": 20 + 3 * IL_eccosorb + IL_marki,
        },
        6: {
            "4K": 20,
            "1K": 20,
            "10mK": 20 + 3 * IL_eccosorb + IL_marki,
        },
        7: {
            "4K": 20,
            "1K": 6,
            "100mK": 10,
            "10mK": 30 + 3 * IL_eccosorb + IL_marki,
        },
        8: {
            "4K": 20,
            "1K": 6,
            "100mK": 10,
            "10mK": 30 + 3 * IL_eccosorb + IL_marki,
        },
    }

    nQ, nRuns = Pe_meas.shape
    Te_mK = np.zeros((nQ, nRuns), dtype=float)

    for qi in range(nQ):
        for ri, r in enumerate(runs):
            cfg = dict(atten_config_by_run[int(r)])  # copy so we don't mutate the base dict
            cfg["300K"] = cfg.get("300K", 0.0) + SS_line_loss_total_dB

            A_after = cumulative_after_stage(cfg, order)
            f_hz = f_ge_Hz[qi, ri]
            Te_K = Te_from_stages(f_hz, stage_temps_K, A_after)
            Te_mK[qi, ri] = 1e3 * Te_K

    print("\nPredicted Te (mK) by qubit & run (using per-run f_ge):")
    for qi in range(nQ):
        vals = ", ".join([f"R{int(runs[i])}:{Te_mK[qi, i]:.2f}" for i in range(nRuns)])
        print(f"  Q{qi+1}: {vals}")


    # ------------------------------------------------------------
    # EXTRA NOISE POWER needed to go from model Te -> measured Tqubit
    # Prints PSD and total power in BOTH linear units and dBm units.
    # Also prints "signed" (direction) and "needed" (clipped at 0) versions.
    # ------------------------------------------------------------
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
                f"  R{int(r)} | n_model={nm:.3e}  n_meas={nM:.3e}  ratio(meas/model)={(nM / nm if nm > 0 else np.inf):.2f}")

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

    for qi in range(nQ):
        ax = axes[qi]

        # --- Run 5 -> Run 6 segment (purple) ---
        ax.errorbar(
            runs[:2], T_qubit_mK[qi][:2],
            yerr=T_qubit_err_mK[qi][:2],
            fmt="o-",
            color="purple",
            capsize=3,
            elinewidth=1,
            label=r"$T_{\mathrm{qubit}}$ (Run 5 SSF)"
        )

        # --- Run 6 -> Run 8 segment (palevioletred) ---
        ax.errorbar(
            runs[1:], T_qubit_mK[qi][1:],
            yerr=T_qubit_err_mK[qi][1:],
            fmt="o-",
            color="palevioletred",
            capsize=3,
            elinewidth=1,
            label=r"$T_{\mathrm{qubit}}$ (Runs 6-8 RPM)"
        )

        # --- Predicted noise temperature ---
        ax.plot(
            runs,
            Te_mK[qi],
            "s--",
            color="slateblue",
            linewidth=2,
            label=r"$T_e$ (pred. noise)"
        )

        ax.set_title(f"Q{qi + 1}", fontsize=16)
        ax.set_yticks(np.arange(0, 401, 75))
        ax.set_xticks(runs)
        ax.set_xticklabels(['5', '6', '7', '8'], fontsize=16)

        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)

        ax.grid(True)

    # ---------------- Axis labels ----------------
    fig.supylabel("Effective Temperature (mK)", fontsize=16, x=0.04)  # pushes label left
    fig.supxlabel("Run Number", fontsize=16, y=0.09)  # pulls label closer to plot

    # ---------------- Big title ----------------
    fig.suptitle(
        "Effective Qubit Temperature (from $P_e$) vs Predicted Noise Temperature $T_e$",
        y=0.98,
        fontsize=18
    )

    # ---------------- Legend ----------------
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        ncol=3,
        frameon=True,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),  # moved slightly down
        fontsize=16
    )

    # Leave space at bottom for legend
    fig.tight_layout(rect=[0.05, 0.12, 1, 0.95])

    plt.show()

if lnPe_vs_qfreq_plots_per_run:
    run_labels = ["Run 5", "Run 6", "Run 7", "Run 8"]

    n_qubits = len(Pe_meas)
    n_runs = Pe_meas.shape[1]   # should be 4

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred']
    markers = ['o', 's', '^', 'D']  # Run 5, 6, 7, 8
    for r in range(n_runs):
        ax = axes[r]

        f_run = []
        Pe_run = []

        ax.set_ylim(-4.0, 0)

        for q in range(n_qubits):
            pei = Pe_meas[q, r]
            pei_err = Pe_err[q, r]
            fi = f_ge_MHz[q, r]
            fi_err = f_ge_err_MHz[q, r]

            if np.isfinite(pei) and np.isfinite(pei_err) and pei > 0:
                color = colors[q % len(colors)]
                marker = markers[r % len(markers)]

                lnPe = np.log(pei)
                lnPe_err = pei_err / pei

                f_run.append(fi)
                Pe_run.append(pei)

                ax.errorbar(
                    fi,
                    lnPe,
                    xerr=fi_err,
                    yerr=lnPe_err,
                    fmt=marker,
                    capsize=3,
                    color=color,
                    ecolor=color
                )

                if q == 0 or q == 4:
                    ax.text(fi + 12, lnPe + 0.2, f"Q{q+1}", fontsize=12, color=color)
                else:
                    ax.text(fi - 12, lnPe + 0.5, f"Q{q+1}", fontsize=12, color=color)

        f_run = np.array(f_run, dtype=float)
        Pe_run = np.array(Pe_run, dtype=float)

        # Optional fit
        # if len(f_run) > 1:
        #     lnPe_run = np.log(Pe_run)
        #     coeffs = np.polyfit(f_run, lnPe_run, 1)
        #     f_fit = np.linspace(np.min(f_run), np.max(f_run), 200)
        #     ax.plot(f_fit, np.polyval(coeffs, f_fit), '--', color='black')
        #
        #     ax.text(
        #         0.05, 0.90,
        #         f"slope = {coeffs[0]:.3e} per MHz",
        #         transform=ax.transAxes,
        #         fontsize=10
        #     )

        ax.set_title(run_labels[r])
        if q >= 3:  # bottom row only
            ax.set_xlabel(r"$f_{ge}$ (MHz)")
        ax.set_ylabel(r"$\ln(P_e)$")

    plt.tight_layout()
    plt.show()

if lnPe_vs_qfreq_plots_per_qubit:
    run_labels = ["Run 5", "Run 6", "Run 7", "Run 8"]

    n_qubits = len(Pe_meas)
    n_runs = Pe_meas.shape[1]

    fig, axes = plt.subplots(2, 3, figsize=(14, 10), sharey=True)
    axes = axes.ravel()

    colors = ['orange', 'blue', 'purple', 'green']  # one color per run
    markers = ['o', 's', '^', 'D']  # Run 5, 6, 7, 8

    for q in range(n_qubits):
        ax = axes[q]

        ax.set_ylim(-4.0, 0)

        f_q = []
        Pe_q = []

        for r in range(n_runs):
            pei = Pe_meas[q, r]
            pei_err = Pe_err[q, r]
            fi = f_ge_MHz[q, r]
            fi_err = f_ge_err_MHz[q, r]

            if np.isfinite(pei) and np.isfinite(pei_err) and pei > 0:
                color = colors[r % len(colors)]
                marker = markers[r % len(markers)]

                lnPe = np.log(pei)
                lnPe_err = pei_err / pei

                f_q.append(fi)
                Pe_q.append(pei)

                ax.errorbar(
                    fi,
                    lnPe,
                    xerr=fi_err,
                    yerr=lnPe_err,
                    fmt=marker,
                    capsize=3,
                    color=color,
                    ecolor=color,
                    label=run_labels[r]
                )

        # Optional fit per qubit
        # if len(f_q) > 1:
        #     lnPe_q = np.log(Pe_q)
        #     coeffs = np.polyfit(f_q, lnPe_q, 1)
        #     f_fit = np.linspace(np.min(f_q), np.max(f_q), 200)
        #     ax.plot(f_fit, np.polyval(coeffs, f_fit), '--', color='black')
        #
        #     ax.text(
        #         0.05, 0.90,
        #         f"slope = {coeffs[0]:.3e} per MHz",
        #         transform=ax.transAxes,
        #         fontsize=10
        #     )

        ax.set_title(f"Qubit {q+1}")
        if q >= 3:  # bottom row only
            ax.set_xlabel(r"$f_{ge}$ (MHz)")
        ax.set_ylabel(r"$\ln(P_e)$")

        center = np.median(f_q)
        width = 35
        ax.set_xlim(center - width / 2, center + width / 2)

    # Optional legend (only once to avoid clutter)
    axes[0].legend(fontsize=10)

    plt.tight_layout()
    plt.show()

if ssf_fid_vs_Pe:
    run_labels = ["Run 5", "Run 6", "Run 7", "Run 8"]

    n_qubits = len(Pe_meas)
    n_runs = Pe_meas.shape[1]

    fig, axes = plt.subplots(2, 3, figsize=(14, 10), sharey=True)
    axes = axes.ravel()

    colors = ['orange', 'blue', 'purple', 'green']
    markers = ['o', 's', '^', 'D']

    for q in range(n_qubits):
        ax = axes[q]

        ssf_q = []
        Pe_q = []

        for r in range(n_runs):
            pei = Pe_meas[q, r]
            pei_err = Pe_err[q, r]
            ssfi = ssf_fid_vals[q, r]

            if np.isfinite(pei) and np.isfinite(pei_err) and np.isfinite(ssfi):
                color = colors[r % len(colors)]
                marker = markers[r % len(markers)]

                ssf_q.append(ssfi)
                Pe_q.append(pei)

                ax.errorbar(
                    ssfi,
                    pei,
                    yerr=pei_err,
                    fmt=marker,
                    capsize=3,
                    color=color,
                    ecolor=color,
                    label=run_labels[r]
                )

        ax.set_title(f"Qubit {q + 1}")

        if q >= 3:
            ax.set_xlabel("SSF Fidelity")
        ax.set_ylabel(r"$P_e$")

        if len(ssf_q) > 0:
            center = np.median(ssf_q)
            width = 0.08  # adjust if needed
            ax.set_xlim(center - width / 2, center + width / 2)

    axes[0].legend(fontsize=10)
    plt.tight_layout()
    plt.show()

if Pe_variance_std_vs_run:

    use_std = True   # ?? set False to plot variance instead

    n_qubits = len(Pe_meas)

    fig, axes = plt.subplots(2, 3, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for q in range(n_qubits):
        ax = axes[q]

        var_q = np.array(Pe_variance_vals[q], dtype=float)

        # ?? switch between variance and std
        if use_std:
            yvals = np.sqrt(var_q)
            ylabel = r"$\sigma(P_e)$"
        else:
            yvals = var_q
            ylabel = r"$\mathrm{Var}(P_e)$"

        ax.plot(
            runs,
            yvals,
            'o-',
            color='blue',
            linewidth=2,
            markersize=6
        )

        ax.set_title(f"Qubit {q+1}")

        if q >= 3:
            ax.set_xlabel("Run Number")

        ax.set_ylabel(ylabel)

        ax.set_xticks(runs)
        ax.grid(alpha=0.3)

    # hide unused axes if needed
    for k in range(n_qubits, len(axes)):
        axes[k].set_visible(False)

    plt.tight_layout()
    plt.show()
################################ Definitions, additional plotting funcs ####################################
def print_median_spread_table(run_num_list, box_data, q, units="", mode="q1q3"):
    """
    Prints spread stats for a single qubit q across runs.

    mode="q1q3"  -> prints median (Q1, Q3)  [recommended for papers]
    mode="iqr2"  -> prints median ± IQR/2
    """
    for r, arr in zip(run_num_list, box_data):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            print(f"Run {r}, Q{q+1}: no data")
            continue

        med = np.median(arr)
        q1  = np.percentile(arr, 25)
        q3  = np.percentile(arr, 75)
        iqr = q3 - q1

        if "Hz" in units: # more decimals for qubit freq vals to identify subtle shifts
            if mode.lower() == "iqr2":
                spread = 0.5 * iqr
                print(f"Run {r}, Q{q+1}: {med:.6f} ± {spread:.6f} {units}  (IQR={iqr:.6f}, n={arr.size})")
            else:
                # default: median (Q1, Q3)
                print(f"Run {r}, Q{q+1}: {med:.6f} ({q1:.6f}, {q3:.6f}) {units}  [n={arr.size}]")
        else:
            if mode.lower() == "iqr2":
                spread = 0.5 * iqr
                print(f"Run {r}, Q{q+1}: {med:.2f} ± {spread:.2f} {units}  (IQR={iqr:.2f}, n={arr.size})")
            else:
                # default: median (Q1, Q3)
                print(f"Run {r}, Q{q+1}: {med:.2f} ({q1:.2f}, {q3:.2f}) {units}  [n={arr.size}]")

def boxwhisker_t1t2_per_qubit_vs_run(
    run_num_list,
    t1_vals_by_run=None,
    t2r_vals_by_run=None,
    t2e_vals_by_run=None,
    do_T1=True,
    do_T2R=True,
    do_T2E=True,
    n_qubits=6,
    ylims=(0, 140),
    yticks=np.arange(0, 141, 20),
    showfliers=True, # show outliers?
    whis=1.5,
    mode="together",          # "together" or "separate"
    fig_title_prefix=" vs Run Number (per qubit)",
):
    """
    mode="together": each qubit subplot contains multiple metrics (offset boxplots)
    mode="separate": make one figure per metric, each with 6 subplots (one per qubit)

    Makes 6 subplots (one per qubit). X-axis is run number.
    At each run, draws box-and-whisker distributions for the enabled metrics.

    Expected dict structure:
      t1_vals_by_run[run][q]  -> array-like of samples OR scalar
      t2r_vals_by_run[run][q] -> array-like of samples OR scalar
      t2e_vals_by_run[run][q] -> array-like of samples OR scalar

    If your entries are scalars (one value per run), the box collapses (still plots).


    # ---------------- Example call ----------------
    # run_num_list = [4,5,6,7,8]
    # boxwhisker_per_qubit_vs_run(
    #     run_num_list,
    #     t1_vals_by_run=t1_vals_by_run,
    #     t2r_vals_by_run=t2r_vals_by_run,
    #     t2e_vals_by_run=t2e_vals_by_run,
    #     do_T1=True, do_T2R=True, do_T2E=True
    # )
    """

    # ---------------- colors (set once, consistent everywhere) ----------------
    color_map = {"T1": "tab:blue", "T2R": "tab:orange", "T2E": "tab:green"}

    metric_specs = []
    if do_T1:
        if t1_vals_by_run is None:
            raise ValueError("do_T1=True but t1_vals_by_run is None")
        metric_specs.append(("T1", t1_vals_by_run, color_map["T1"]))
    if do_T2R:
        if t2r_vals_by_run is None:
            raise ValueError("do_T2R=True but t2r_vals_by_run is None")
        metric_specs.append(("T2R", t2r_vals_by_run, color_map["T2R"]))
    if do_T2E:
        if t2e_vals_by_run is None:
            raise ValueError("do_T2E=True but t2e_vals_by_run is None")
        metric_specs.append(("T2E", t2e_vals_by_run, color_map["T2E"]))

    if len(metric_specs) == 0:
        raise ValueError("Enable at least one of do_T1/do_T2R/do_T2E.")

    # helper: each (run, qubit) cell -> 1D array of finite samples
    def cell_to_1d(cell):
        if cell is None:
            return np.array([], dtype=float)
        if np.isscalar(cell):
            arr = np.array([cell], dtype=float)
        else:
            arr = np.asarray(cell, dtype=float).ravel()
        return arr[np.isfinite(arr)]

    # positions: one "cluster" per run
    n_runs = len(run_num_list)
    base_pos = np.arange(1, n_runs + 1)  # 1..n_runs

    def style_boxplot(bp, q_color):
        for b in bp["boxes"]:
            b.set_facecolor(q_color)
            b.set_edgecolor(q_color)
            b.set_alpha(0.3)
            b.set_linewidth(1.3)

        for m in bp["medians"]:
            m.set_color(q_color)
            m.set_linewidth(2.0)

        for w in bp["whiskers"]:
            w.set_color(q_color)
            w.set_linewidth(1.2)

        for c in bp["caps"]:
            c.set_color(q_color)
            c.set_linewidth(1.2)

        for f in bp["fliers"]:
            f.set_marker("o")
            f.set_markersize(3.5)
            f.set_markerfacecolor(q_color)
            f.set_markeredgecolor(q_color)
            f.set_alpha(0.6)

    def add_common_axis_styling(ax):
        ax.set_ylim(*ylims)
        ax.set_yticks(yticks)
        ax.grid(True, alpha=0.35)
        ax.set_xticks(base_pos)
        ax.set_xticklabels([f"{r}" for r in run_num_list])
        ax.tick_params(axis="both", labelsize=16)

    # ------------------------- mode: together -------------------------
    if mode.lower() == "together":
        n_metrics = len(metric_specs)
        offsets = np.linspace(-0.25, 0.25, n_metrics) if n_metrics > 1 else np.array([0.0])
        box_width = 0.22 if n_metrics > 1 else 0.45

        fig, axes = plt.subplots(
            2, 3,
            figsize=(18, 9),
            sharex=True,
            sharey=True,
            constrained_layout=False
        )
        axes = axes.ravel()

        for q in range(n_qubits):
            ax = axes[q]

            for (label, vals_by_run, color), off in zip(metric_specs, offsets):
                box_data = [cell_to_1d(vals_by_run[r][q]) for r in run_num_list]
                print_median_spread_table(run_num_list, box_data, q, units="µs", mode="iqr2")
                positions = base_pos + off

                bp = ax.boxplot(
                    box_data,
                    positions=positions,
                    widths=box_width,
                    patch_artist=True,
                    showfliers=showfliers,
                    whis=whis,
                    manage_ticks=False
                )
                style_boxplot(bp, color)

            ax.set_title(f"Qubit {q + 1}")
            add_common_axis_styling(ax)

            # legend: Patch matches box fill
            handles = [Patch(facecolor=ms[2], edgecolor=ms[2], alpha=0.30, label=ms[0]) for ms in metric_specs]
            ax.legend(handles=handles, loc="upper left", fontsize=16)

        fig.suptitle(fig_title_prefix, fontsize=18)
        fig.supxlabel("Run Number", fontsize=16)
        fig.supylabel("Coherence time (µs)", fontsize=16)
        plt.show()
        return  # done

    # ------------------------- mode: separate (one figure per metric) -------------------------
    elif mode.lower() == "separate":
        for (label, vals_by_run, color) in metric_specs:
            fig, axes = plt.subplots(
                2, 3,
                figsize=(18, 9),
                sharex=True,
                sharey=True,
                constrained_layout=False
            )
            axes = axes.ravel()

            for q in range(n_qubits):
                ax = axes[q]
                box_data = [cell_to_1d(vals_by_run[r][q]) for r in run_num_list]
                print_median_spread_table(run_num_list, box_data, q, units="µs", mode="iqr2")

                bp = ax.boxplot(
                    box_data,
                    positions=base_pos,
                    widths=0.55,
                    patch_artist=True,
                    showfliers=showfliers,
                    whis=whis,
                    manage_ticks=False
                )
                style_boxplot(bp, color)

                ax.set_title(f"Qubit {q + 1}")
                add_common_axis_styling(ax)

                # legend with single entry
                handle = Patch(facecolor=color, edgecolor=color, alpha=0.30, label=label)
                #ax.legend(handles=[handle], loc="upper left", fontsize=16)

            fig.suptitle(f"{label}{fig_title_prefix}", fontsize=18)
            fig.supxlabel("Run Number", fontsize=16)
            fig.supylabel(f"{label} (µs)", fontsize=16)
            plt.show()
        return

    else:
        raise ValueError("mode must be 'together' or 'separate'.")

def boxwhisker_qtemps_per_qubit_vs_run_choice(
    run_num_list,
    rpm_temps_by_run,
    ssf_g_temps_by_run,
    ssf_ge_temps_by_run=None,
    n_qubits=6,
    qubits_to_plot=None,   # NEW: e.g. [0,1,2] for Q1-Q3, or None for first n_qubits
    plot_mode="hybrid",    # "hybrid", "all_ssf", or "compare_methods"
    ssf_kind="g",          # "g" or "ge"
    layout="separate",     # "separate" or "together"
    colors=('orange', 'blue', 'purple', 'green', 'brown', 'palevioletred'),
    ssf_color="purple",
    ylims=(0, 600),
    yticks=np.arange(0, 601, 100),
    showfliers=True, # show outliers?
    whis=1.5,
    fig_title=None,
    ylabel="Effective temperature (mK)",
    suptitle_fs=18,
    title_fs=18,
    label_fs=18,
    tick_fs=18,
    save_plt_path = None
):
    """
    Per-qubit box/whisker vs run.

    plot_mode options
    -----------------
    "hybrid":
        - Run 5 uses SSF
        - Other runs use RPM

    "all_ssf":
        - All runs use SSF

    "compare_methods":
        - Same as hybrid, BUT also overlays available SSF data (in ssf_color)
          for runs other than 5, so RPM and SSF can be visually compared.

    Dict structure expected:
    rpm_temps_by_run[run][q] -> array-like or scalar or []
    ssf_g_temps_by_run[run][q] -> array-like or scalar or []
    ssf_ge_temps_by_run[run][q] -> array-like or scalar or [] (optional unless ssf_kind="ge")
    """

    # ---------------- pick SSF dictionary ----------------
    ssf_kind = ssf_kind.lower()
    if ssf_kind not in ("g", "ge"):
        raise ValueError("ssf_kind must be 'g' or 'ge'.")

    if ssf_kind == "g":
        ssf_dict = ssf_g_temps_by_run
    else:
        if ssf_ge_temps_by_run is None:
            raise ValueError("ssf_kind='ge' requires ssf_ge_temps_by_run.")
        ssf_dict = ssf_ge_temps_by_run

    plot_mode = plot_mode.lower()
    if plot_mode not in ("hybrid", "all_ssf", "compare_methods"):
        raise ValueError("plot_mode must be 'hybrid', 'all_ssf', or 'compare_methods'.")

    layout = layout.lower()
    if layout not in ("separate", "together"):
        raise ValueError("layout must be 'separate' or 'together'")

    # ---------------- choose qubits to plot ----------------
    if qubits_to_plot is None:
        qubits_to_plot = list(range(n_qubits))
    else:
        qubits_to_plot = list(qubits_to_plot)

    n_plot = len(qubits_to_plot)
    if n_plot == 0:
        raise ValueError("qubits_to_plot is empty.")

    # ---------------- helpers ----------------
    def cell_to_1d(cell):
        if cell is None:
            return np.array([], dtype=float)

        try:
            if not np.isscalar(cell) and len(cell) == 0:
                return np.array([], dtype=float)
        except TypeError:
            pass

        if np.isscalar(cell):
            arr = np.array([cell], dtype=float)
        else:
            arr = np.asarray(cell, dtype=float).ravel()

        return arr[np.isfinite(arr)]

    def get_cell(vals_by_run, run, q):
        if vals_by_run is None or run not in vals_by_run:
            return None
        row = vals_by_run[run]
        if row is None or q >= len(row):
            return None
        return row[q]

    def select_cell(run, q):
        if plot_mode == "all_ssf":
            return get_cell(ssf_dict, run, q)

        if run == 5:
            return get_cell(ssf_dict, run, q)

        return get_cell(rpm_temps_by_run, run, q)

    def style_boxplot(bp, color):
        for b in bp["boxes"]:
            b.set_facecolor(color)
            b.set_edgecolor(color)
            b.set_alpha(0.31)
            b.set_linewidth(1.3)

        for m in bp["medians"]:
            m.set_color(color)
            m.set_linewidth(2.0)

        for w in bp["whiskers"]:
            w.set_color(color)
            w.set_linewidth(1.2)

        for c in bp["caps"]:
            c.set_color(color)
            c.set_linewidth(1.2)

        for f in bp["fliers"]:
            f.set_marker("o")
            f.set_markersize(3.5)
            f.set_markerfacecolor(color)
            f.set_markeredgecolor(color)
            f.set_alpha(0.6)

    # ---------------- positions ----------------
    n_runs = len(run_num_list)
    base_pos = np.arange(1, n_runs + 1)
    xtick_labels = [f"{r}" for r in run_num_list]
    multi_qubit_colors = len(set(colors[:max(qubits_to_plot) + 1])) > 1

    # ---------------- title ----------------
    if fig_title is None:
        if plot_mode == "compare_methods":
            fig_title = "Effective Qubit Temperatures vs Run Number"
        else:
            fig_title = "Effective Qubit Temperatures vs Run Number"

    # =====================================================
    # =================== SEPARATE MODE ===================
    # =====================================================
    if layout == "separate":

        # dynamic subplot grid
        ncols = min(3, n_plot)
        nrows = math.ceil(n_plot / ncols)
        fig_w = 6 * ncols
        fig_h = 4.5 * nrows

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fig_w, fig_h),
            sharex=True,
            sharey=True,
            constrained_layout=False
        )

        axes = np.atleast_1d(axes).ravel()

        for ax_idx, q in enumerate(qubits_to_plot):
            ax = axes[ax_idx]
            q_color = colors[q % len(colors)]

            # ---------------- main/hybrid data ----------------
            box_data = [cell_to_1d(select_cell(r, q)) for r in run_num_list]
            if plot_mode == "compare_methods":
                print(f"\nHybrid summary for Qubit {q + 1}:")
                print_median_spread_table(run_num_list, box_data, q, units="mK", mode="iqr2")

                ssf_box_data = [cell_to_1d(get_cell(ssf_dict, r, q)) for r in run_num_list]
                print(f"\nSSF-only summary for Qubit {q + 1}:")
                print_median_spread_table(run_num_list, ssf_box_data, q, units="mK", mode="iqr2")
                print("\n")

                main_positions = np.array([
                    p if r == 5 else p - 0.16
                    for p, r in zip(base_pos, run_num_list)
                ])
                main_width = 0.28
            else:
                main_positions = base_pos
                main_width = 0.55
                print_median_spread_table(run_num_list, box_data, q, units="mK", mode="iqr2")

            bp = ax.boxplot(
                box_data,
                positions=main_positions,
                widths=main_width,
                patch_artist=True,
                showfliers=showfliers,
                whis=whis,
                manage_ticks=False
            )
            style_boxplot(bp, q_color)

            # recolor run 5 as SSF in hybrid / compare_methods
            if plot_mode in ("hybrid", "compare_methods") and 5 in run_num_list:
                i0 = run_num_list.index(5)
                bp["boxes"][i0].set_facecolor(ssf_color)
                bp["boxes"][i0].set_edgecolor(ssf_color)
                bp["medians"][i0].set_color(ssf_color)

                for j in (2 * i0, 2 * i0 + 1):
                    bp["whiskers"][j].set_color(ssf_color)
                    bp["caps"][j].set_color(ssf_color)

                if i0 < len(bp["fliers"]):
                    bp["fliers"][i0].set_markerfacecolor(ssf_color)
                    bp["fliers"][i0].set_markeredgecolor(ssf_color)
                    bp["fliers"][i0].set_alpha(0.6)

            # ---------------- overlay SSF for method comparison ----------------
            if plot_mode == "compare_methods":
                ssf_overlay_data = []
                ssf_overlay_positions = []

                for p, r, arr in zip(base_pos, run_num_list, ssf_box_data):
                    if r == 5:
                        continue
                    if arr.size > 0:
                        ssf_overlay_data.append(arr)
                        ssf_overlay_positions.append(p + 0.16)

                if len(ssf_overlay_data) > 0:
                    bp_ssf = ax.boxplot(
                        ssf_overlay_data,
                        positions=ssf_overlay_positions,
                        widths=0.28,
                        patch_artist=True,
                        showfliers=showfliers,
                        whis=whis,
                        manage_ticks=False
                    )
                    style_boxplot(bp_ssf, ssf_color)

            # ---------------- axis styling ----------------
            ax.set_title(f"Qubit {q + 1}", fontsize=title_fs)
            ax.set_ylim(*ylims)
            ax.set_yticks(yticks)
            ax.tick_params(axis="both", labelsize=tick_fs)
            ax.grid(True, alpha=0.35)

            ax.set_xticks(base_pos)
            ax.set_xticklabels(xtick_labels)
            ax.set_xlim(0.5, len(run_num_list) + 0.5)

            # per-subplot legend if qubit colors differ
            # if multi_qubit_colors:
            #     if plot_mode == "hybrid":
            #         legend_handles = [
            #             Patch(facecolor=ssf_color, edgecolor=ssf_color, alpha=0.30, label="Run 5 (SSF)"),
            #             Patch(facecolor=q_color, edgecolor=q_color, alpha=0.30, label="Runs >5 (RPM)")
            #         ]
            #     elif plot_mode == "all_ssf":
            #         legend_handles = [
            #             Patch(facecolor=ssf_color, edgecolor=ssf_color, alpha=0.30, label="SSF")
            #         ]
            #     else:  # compare_methods
            #         legend_handles = [
            #             Patch(facecolor=q_color, edgecolor=q_color, alpha=0.30, label="RPM / hybrid"),
            #             Patch(facecolor=ssf_color, edgecolor=ssf_color, alpha=0.30, label="SSF")
            #         ]
            #
            #     ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=tick_fs - 2)

        # hide unused axes
        for k in range(n_plot, len(axes)):
            axes[k].set_visible(False)

        fig.subplots_adjust(
            left=0.12,  # move axes slightly left
            bottom=0.12,  # move axes slightly down
            top=0.90,
        )

        # fig.suptitle(fig_title, fontsize=suptitle_fs, y=0.965)

        # fig.supxlabel("Run Number", fontsize=label_fs, y=0.03)

        # fig.supylabel(ylabel, fontsize=label_fs, x=0.02)

        # single figure legend if all qubits use same color
        # if not multi_qubit_colors:
        #     fig.subplots_adjust(right=0.84)
        #
        #     if plot_mode == "hybrid":
        #         legend_handles = [
        #             Patch(facecolor=ssf_color, edgecolor=ssf_color, alpha=0.30, label="Run 5 (SSF)"),
        #             Patch(facecolor=colors[0], edgecolor=colors[0], alpha=0.30, label="Runs >5 (RPM)")
        #         ]
        #     elif plot_mode == "all_ssf":
        #         legend_handles = [
        #             Patch(facecolor=ssf_color, edgecolor=ssf_color, alpha=0.30, label="SSF")
        #         ]
        #     else:
        #         legend_handles = [
        #             Patch(facecolor=colors[0], edgecolor=colors[0], alpha=0.30, label="RPM / hybrid"),
        #             Patch(facecolor=ssf_color, edgecolor=ssf_color, alpha=0.30, label="SSF")
        #         ]
        #
        #     fig.legend(
        #         handles=legend_handles,
        #         loc="center left",
        #         bbox_to_anchor=(0.86, 0.5),
        #         frameon=True,
        #         fontsize=label_fs
        #     )

        if save_plt_path is None:
            plt.show()
        else:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

            fname = os.path.join(
                save_plt_path,
                f"boxwhisk_qtemps_vs_run_num_{timestamp}.pdf"
            )

            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
    # =====================================================
    # ==================== TOGETHER MODE ==================
    # =====================================================
    elif layout == "together":

        fig, ax = plt.subplots(figsize=(12, 6))

        if plot_mode == "compare_methods":
            # reserve extra width because we may show two methods
            offsets = np.linspace(-0.30, 0.00, n_plot) if n_plot > 1 else np.array([-0.15])
            ssf_extra_shift = 0.16
            box_width = 0.22
        else:
            offsets = np.linspace(-0.30, 0.30, n_plot) if n_plot > 1 else np.array([0.0])
            ssf_extra_shift = 0.0
            box_width = 0.80 / max(n_plot, 1)

        for i, q in enumerate(qubits_to_plot):
            q_color = colors[q % len(colors)]
            positions = base_pos + offsets[i]

            box_data = [cell_to_1d(select_cell(r, q)) for r in run_num_list]

            if plot_mode == "compare_methods":
                print(f"\nHybrid summary for Qubit {q + 1}:")
                print_median_spread_table(run_num_list, box_data, q, units="mK", mode="iqr2")
                ssf_box_data = [cell_to_1d(get_cell(ssf_dict, r, q)) for r in run_num_list]
                print(f"\nSSF-only summary for Qubit {q + 1}:")
                print_median_spread_table(run_num_list, ssf_box_data, q, units="mK", mode="iqr2")
                print("\n")
            else:
                print_median_spread_table(run_num_list, box_data, q, units="mK", mode="iqr2")

            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=showfliers,
                whis=whis,
                manage_ticks=False
            )
            style_boxplot(bp, q_color)

            if plot_mode in ("hybrid", "compare_methods") and 5 in run_num_list:
                i0 = run_num_list.index(5)
                bp["boxes"][i0].set_facecolor(ssf_color)
                bp["boxes"][i0].set_edgecolor(ssf_color)
                bp["medians"][i0].set_color(ssf_color)

                for j in (2 * i0, 2 * i0 + 1):
                    bp["whiskers"][j].set_color(ssf_color)
                    bp["caps"][j].set_color(ssf_color)

                if i0 < len(bp["fliers"]):
                    bp["fliers"][i0].set_markerfacecolor(ssf_color)
                    bp["fliers"][i0].set_markeredgecolor(ssf_color)
                    bp["fliers"][i0].set_alpha(0.6)

            # overlay SSF in compare_methods
            if plot_mode == "compare_methods":
                ssf_overlay_data = []
                ssf_overlay_positions = []

                for p, r, arr in zip(positions, run_num_list, ssf_box_data):
                    if r == 5:
                        continue
                    if arr.size > 0:
                        ssf_overlay_data.append(arr)
                        ssf_overlay_positions.append(p + ssf_extra_shift)

                if len(ssf_overlay_data) > 0:
                    bp_ssf = ax.boxplot(
                        ssf_overlay_data,
                        positions=ssf_overlay_positions,
                        widths=box_width,
                        patch_artist=True,
                        showfliers=showfliers,
                        whis=whis,
                        manage_ticks=False
                    )
                    style_boxplot(bp_ssf, ssf_color)

        ax.set_title(fig_title, fontsize=suptitle_fs)
        ax.set_ylim(*ylims)
        ax.set_yticks(yticks)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, alpha=0.35)

        ax.set_xticks(base_pos)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlim(0.5, len(run_num_list) + 0.5)

        ax.set_xlabel("Run Number", fontsize=label_fs)
        ax.set_ylabel(ylabel, fontsize=label_fs)

        if save_plt_path is None:
            plt.show()
        else:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

            fname = os.path.join(
                save_plt_path,
                f"boxwhisk_qtemps_vs_run_num_{timestamp}.pdf"
            )

            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)

######################################################################################
def boxwhisker_pe_per_qubit_vs_run_hybrid(
    run_num_list,
    rpm_pe_by_run,
    ssf_pe_by_run,
    n_qubits=6,
    qubits_to_plot=None,
    colors=('orange', 'blue', 'purple', 'green', 'brown', 'palevioletred'),
    ssf_color="purple",
    ylims=None,
    yticks=None,
    showfliers=True, # show outliers?
    whis=1.5,
    fig_title="Excited-State Population vs Run Number",
    ylabel=r"$P_e$",
    suptitle_fs=18,
    title_fs=18,
    label_fs=18,
    tick_fs=18,
    save_plt_path=None
):
    """
    Hybrid Pe boxplot, using already-grouped per-run dictionaries.

    Expected input shape
    --------------------
    rpm_pe_by_run[run][q] = [Pe, ...]
    ssf_pe_by_run[run][q] = [Pe, ...]

    Hybrid logic
    ------------
    - Run 5 uses SSF Pe data
    - Runs > 5 use RPM Pe data
    """

    # ---------------- choose qubits to plot ----------------
    if qubits_to_plot is None:
        qubits_to_plot = list(range(n_qubits))
    else:
        qubits_to_plot = list(qubits_to_plot)

    n_plot = len(qubits_to_plot)
    if n_plot == 0:
        raise ValueError("qubits_to_plot is empty.")

    # ---------------- helpers ----------------
    def cell_to_1d(cell):
        if cell is None:
            return np.array([], dtype=float)

        try:
            if not np.isscalar(cell) and len(cell) == 0:
                return np.array([], dtype=float)
        except TypeError:
            pass

        if np.isscalar(cell):
            arr = np.array([cell], dtype=float)
        else:
            arr = np.asarray(cell, dtype=float).ravel()

        return arr[np.isfinite(arr)]

    def get_cell(vals_by_run, run, q):
        if vals_by_run is None or run not in vals_by_run:
            return None

        row = vals_by_run[run]
        if row is None or q >= len(row):
            return None

        return row[q]

    def style_boxplot(bp, color):
        for b in bp["boxes"]:
            b.set_facecolor(color)
            b.set_edgecolor(color)
            b.set_alpha(0.31)
            b.set_linewidth(1.3)

        for m in bp["medians"]:
            m.set_color(color)
            m.set_linewidth(2.0)

        for w in bp["whiskers"]:
            w.set_color(color)
            w.set_linewidth(1.2)

        for c in bp["caps"]:
            c.set_color(color)
            c.set_linewidth(1.2)

        for f in bp["fliers"]:
            f.set_marker("o")
            f.set_markersize(3.5)
            f.set_markerfacecolor(color)
            f.set_markeredgecolor(color)
            f.set_alpha(0.6)

    def select_cell(run, q):
        if run == 5:
            return get_cell(ssf_pe_by_run, run, q)
        return get_cell(rpm_pe_by_run, run, q)

    # ---------------- axis defaults ----------------
    if ylims is None:
        ylims = (0, 0.5)
    if yticks is None:
        yticks = np.arange(0, 0.51, 0.1)

    n_runs = len(run_num_list)
    base_pos = np.arange(1, n_runs + 1)
    xtick_labels = [f"{r}" for r in run_num_list]

    # ---------------- dynamic subplot grid ----------------
    if n_plot == 1:
        nrows, ncols = 1, 1
    elif n_plot == 2:
        nrows, ncols = 1, 2
    elif n_plot <= 4:
        nrows, ncols = 2, 2
    else:
        ncols = 3
        nrows = math.ceil(n_plot / 3)

    fig_w = 5.8 * ncols
    fig_h = 4.6 * nrows

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        constrained_layout=False
    )
    axes = np.atleast_1d(axes).ravel()

    for ax_idx, q in enumerate(qubits_to_plot):
        ax = axes[ax_idx]
        q_color = colors[q % len(colors)]

        box_data = [cell_to_1d(select_cell(r, q)) for r in run_num_list]

        # compute variance and std per run, for this qubit
        variance_per_run = [np.var(arr, ddof=1) if len(arr) > 1 else np.nan for arr in box_data]
        std_per_run = [np.std(arr, ddof=1) if len(arr) > 1 else np.nan for arr in box_data]

        print(f"\nQubit {q + 1} noise summary:")
        for run, arr, var, std in zip(run_num_list, box_data, variance_per_run, std_per_run):
            n = len(arr)
            if np.isfinite(var):
                print(f"Run {run}: std(Pe) = {std:.3e}  (var = {var:.3e}, n = {n})")
            else:
                print(f"Run {run}: insufficient data (n = {n})")

        # summary print
        print(f"\nQubit {q + 1} Pe summary:")
        print_median_spread_table(
            run_num_list,
            box_data,
            q,
            mode="iqr2"  # or "q1q3" if you want paper-style output
        )

        bp = ax.boxplot(
            box_data,
            positions=base_pos,
            widths=0.55,
            patch_artist=True,
            showfliers=showfliers,
            whis=whis,
            manage_ticks=False
        )
        style_boxplot(bp, q_color)

        # recolor run 5 as SSF
        if 5 in run_num_list:
            i0 = run_num_list.index(5)

            bp["boxes"][i0].set_facecolor(ssf_color)
            bp["boxes"][i0].set_edgecolor(ssf_color)
            bp["medians"][i0].set_color(ssf_color)

            for j in (2 * i0, 2 * i0 + 1):
                bp["whiskers"][j].set_color(ssf_color)
                bp["caps"][j].set_color(ssf_color)

            if i0 < len(bp["fliers"]):
                bp["fliers"][i0].set_markerfacecolor(ssf_color)
                bp["fliers"][i0].set_markeredgecolor(ssf_color)
                bp["fliers"][i0].set_alpha(0.6)

        ax.set_title(f"Qubit {q + 1}", fontsize=title_fs)
        ax.set_ylim(*ylims)
        ax.set_yticks(yticks)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, alpha=0.35)

        ax.set_xticks(base_pos)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlim(0.5, len(run_num_list) + 0.5)

    # hide unused axes
    for k in range(n_plot, len(axes)):
        axes[k].set_visible(False)

    left_margin   = 0.12 + 0.015 * max(nrows - 1, 0)
    bottom_margin = 0.14 + 0.025 * max(nrows - 1, 0)
    top_margin    = 0.90 - 0.015 * max(nrows - 1, 0)
    right_margin  = 0.96

    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        bottom=bottom_margin,
        top=top_margin,
        wspace=0.25,
        hspace=0.35
    )

    fig.suptitle(fig_title, fontsize=suptitle_fs, y=0.975)
    fig.supxlabel("Run Number", fontsize=label_fs, y=0.04)
    fig.supylabel(ylabel, fontsize=label_fs, x=0.035)

    if save_plt_path is None:
        plt.show()
    else:
        fname = os.path.join(save_plt_path, "boxwhisk_pe_vs_run_num.pdf")
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)

def boxwhisker_qfreq_per_qubit_vs_run(
    run_num_list,
    qfreq_vals_by_run,
    qfreq_errs_by_run,
    qfreq_centers,              # user provides center for each qubit
    n_qubits=6,
    freq_window=20.0,           # same total width for every qubit
    yticks_per_qubit=None,
    showfliers=True,
    whis=1.5,
    rel_err_cutoff=None,
    fig_title="Qubit Frequency vs Run Number (per qubit)",
    ylabel="Qubit Frequency (MHz)",
    save_plt_path = None
):
    """
    Makes 6 subplots (one per qubit). X-axis is run number.
    At each run, draws a box-and-whisker distribution for qubit frequency.

    Parameters
    ----------
    run_num_list : list
        Example: [5, 6, 7, 8]

    qfreq_vals_by_run : dict
        qfreq_vals_by_run[run][q] -> array-like of frequency samples OR scalar

    qfreq_errs_by_run : dict
        qfreq_errs_by_run[run][q] -> array-like of frequency errors OR scalar

    qfreq_centers : list
        Required. One center frequency per qubit.

    freq_window : float
        Total y-axis width for every qubit subplot.
        Example: 20.0 means center ± 10 MHz.
    """
    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

    def to_1d_array(cell):
        if cell is None:
            return np.array([], dtype=float)
        if np.isscalar(cell):
            return np.array([cell], dtype=float)
        return np.asarray(cell, dtype=float).ravel()

    def filter_freq_cell(freq_cell, err_cell):
        freqs = to_1d_array(freq_cell)
        errs = to_1d_array(err_cell)

        if freqs.size == 0 or errs.size == 0:
            return np.array([], dtype=float)

        n = min(len(freqs), len(errs))
        freqs = freqs[:n]
        errs = errs[:n]

        keep = []
        for f, e in zip(freqs, errs):
            if not np.isfinite(f) or not np.isfinite(e):
                continue
            if f <= 0 or e <= 0:
                continue
            if rel_err_cutoff is not None and (e / f) > rel_err_cutoff:
                continue
            keep.append(f)

        return np.asarray(keep, dtype=float)

    def style_boxplot(bp, q_color):
        for b in bp["boxes"]:
            b.set_facecolor(q_color)
            b.set_edgecolor(q_color)
            b.set_alpha(0.30)
            b.set_linewidth(1.3)

        for m in bp["medians"]:
            m.set_color(q_color)
            m.set_linewidth(2.0)

        for w in bp["whiskers"]:
            w.set_color(q_color)
            w.set_linewidth(1.2)

        for c in bp["caps"]:
            c.set_color(q_color)
            c.set_linewidth(1.2)

        for f in bp["fliers"]:
            f.set_marker("o")
            f.set_markersize(3.5)
            f.set_markerfacecolor(q_color)
            f.set_markeredgecolor(q_color)
            f.set_alpha(0.6)

    base_pos = np.arange(1, len(run_num_list) + 1)
    half_window = freq_window / 2.0

    fig, axes = plt.subplots(
        2, 3,
        figsize=(18, 9),
        sharex=True,
        sharey=False
    )
    axes = axes.ravel()

    for q in range(n_qubits):
        ax = axes[q]
        q_color = colors[q % len(colors)]

        box_data = []
        for run in run_num_list:
            freq_cell = qfreq_vals_by_run[run][q]
            err_cell = qfreq_errs_by_run[run][q]
            freqs_filtered = filter_freq_cell(freq_cell, err_cell)
            box_data.append(freqs_filtered)

        # summary print
        print(f"\nQubit {q + 1} frequency summary:")
        print_median_spread_table(
            run_num_list,
            box_data,
            q,
            units="MHz",
            mode="iqr2"  # or "q1q3" if you want paper-style output
        )

        bp = ax.boxplot(
            box_data,
            positions=base_pos,
            widths=0.55,
            patch_artist=True,
            showfliers=showfliers,
            whis=whis,
            manage_ticks=False
        )
        style_boxplot(bp, q_color)

        # set same-width y-range, centered on user-provided value
        center = qfreq_centers[q]
        ax.set_ylim(center - half_window, center + half_window)

        if yticks_per_qubit is not None:
            ax.set_yticks(yticks_per_qubit[q])

        ax.set_title(f"Qubit {q + 1}", fontsize=16)
        ax.set_xticks(base_pos)
        ax.set_xticklabels([f"{r}" for r in run_num_list])
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, alpha=0.35)

        # handle = Patch(facecolor=q_color, edgecolor=q_color, alpha=0.30, label=f"Q{q+1}")
        # ax.legend(handles=[handle], loc="upper left", fontsize=12)

    fig.suptitle(fig_title, fontsize=18)
    fig.supxlabel("Run Number", fontsize=16)
    fig.supylabel(ylabel, fontsize=16)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    if save_plt_path is None:
        plt.show()
    else:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        fname = os.path.join(
            save_plt_path,
            f"boxwhisk_ge_qfreqs_vs_run_num_{timestamp}.pdf"
        )

        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)

def boxwhisker_ssf_per_qubit_vs_run(
    run_num_list,
    ssf_vals_by_run,
    n_qubits=6,
    qubits_to_plot=None,
    colors="navy", # default: same color for all qubits, can pass multiple too: ('orange', 'blue', 'purple', 'green', 'brown', 'palevioletred')
    ylims=None,
    yticks=None,
    showfliers=True,
    whis=1.5,
    fig_title="Single-Shot Fidelity vs Run Number",
    ylabel="Single-Shot Fidelity",
    suptitle_fs=18,
    title_fs=18,
    label_fs=18,
    tick_fs=18,
    save_plt_path=None,
    save_name="boxwhisk_ssf_vs_run_num.pdf"
):
    """
    SSF boxplot, using already-grouped per-run dictionaries.

    Expected input shape
    --------------------
    ssf_vals_by_run[run][q] = [ssf, ssf, ...]

    Notes
    -----
    - This is the SSF-only version of the hybrid-style plotting function.
    - Default color is navy for all qubits.
    - If you want different colors per qubit, pass something like:
          colors=('orange', 'blue', 'purple', 'green', 'brown', 'palevioletred')
    """

    # ---------------- choose qubits to plot ----------------
    if qubits_to_plot is None:
        qubits_to_plot = list(range(n_qubits))
    else:
        qubits_to_plot = list(qubits_to_plot)

    n_plot = len(qubits_to_plot)
    if n_plot == 0:
        raise ValueError("qubits_to_plot is empty.")

    # ---------------- helpers ----------------
    def cell_to_1d(cell):
        if cell is None:
            return np.array([], dtype=float)

        try:
            if not np.isscalar(cell) and len(cell) == 0:
                return np.array([], dtype=float)
        except TypeError:
            pass

        if np.isscalar(cell):
            arr = np.array([cell], dtype=float)
        else:
            arr = np.asarray(cell, dtype=float).ravel()

        return arr[np.isfinite(arr)]

    def get_cell(vals_by_run, run, q):
        if vals_by_run is None or run not in vals_by_run:
            return None

        row = vals_by_run[run]
        if row is None or q >= len(row):
            return None

        return row[q]

    def style_boxplot(bp, color):
        for b in bp["boxes"]:
            b.set_facecolor(color)
            b.set_edgecolor(color)
            b.set_alpha(0.31)
            b.set_linewidth(1.3)

        for m in bp["medians"]:
            m.set_color(color)
            m.set_linewidth(2.0)

        for w in bp["whiskers"]:
            w.set_color(color)
            w.set_linewidth(1.2)

        for c in bp["caps"]:
            c.set_color(color)
            c.set_linewidth(1.2)

        for f in bp["fliers"]:
            f.set_marker("o")
            f.set_markersize(3.5)
            f.set_markerfacecolor(color)
            f.set_markeredgecolor(color)
            f.set_alpha(0.6)

    def get_qubit_color(q):
        if isinstance(colors, str):
            return colors
        return colors[q % len(colors)]

    # ---------------- axis defaults ----------------
    if ylims is None:
        ylims = (0, 1.0)
    if yticks is None:
        yticks = np.arange(0, 1.01, 0.1)

    n_runs = len(run_num_list)
    base_pos = np.arange(1, n_runs + 1)
    xtick_labels = [f"{r}" for r in run_num_list]

    # ---------------- dynamic subplot grid ----------------
    if n_plot == 1:
        nrows, ncols = 1, 1
    elif n_plot == 2:
        nrows, ncols = 1, 2
    elif n_plot <= 4:
        nrows, ncols = 2, 2
    else:
        ncols = 3
        nrows = math.ceil(n_plot / 3)

    fig_w = 5.8 * ncols
    fig_h = 4.6 * nrows

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        constrained_layout=False
    )
    axes = np.atleast_1d(axes).ravel()

    for ax_idx, q in enumerate(qubits_to_plot):
        ax = axes[ax_idx]
        q_color = get_qubit_color(q)

        box_data = [cell_to_1d(get_cell(ssf_vals_by_run, r, q)) for r in run_num_list]

        # summary print
        print(f"\nQubit {q + 1} SSF summary:")
        print_median_spread_table(
            run_num_list,
            box_data,
            q,
            mode="iqr2"
        )

        bp = ax.boxplot(
            box_data,
            positions=base_pos,
            widths=0.55,
            patch_artist=True,
            showfliers=showfliers,
            whis=whis,
            manage_ticks=False
        )
        style_boxplot(bp, q_color)

        ax.set_title(f"Qubit {q + 1}", fontsize=title_fs)
        ax.set_ylim(*ylims)
        ax.set_yticks(yticks)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, alpha=0.35)

        ax.set_xticks(base_pos)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlim(0.5, len(run_num_list) + 0.5)

    # ---------------- hide unused axes ----------------
    for k in range(n_plot, len(axes)):
        axes[k].set_visible(False)

    left_margin   = 0.12 + 0.015 * max(nrows - 1, 0)
    bottom_margin = 0.14 + 0.025 * max(nrows - 1, 0)
    top_margin    = 0.90 - 0.015 * max(nrows - 1, 0)
    right_margin  = 0.96

    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        bottom=bottom_margin,
        top=top_margin,
        wspace=0.25,
        hspace=0.35
    )

    fig.suptitle(fig_title, fontsize=suptitle_fs, y=0.975)
    fig.supxlabel("Run Number", fontsize=label_fs, y=0.04)
    fig.supylabel(ylabel, fontsize=label_fs, x=0.035)

    if save_plt_path is None:
        plt.show()
    else:
        fname = os.path.join(save_plt_path, save_name)
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)