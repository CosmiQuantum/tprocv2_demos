import numpy as np
import matplotlib.pyplot as plt

qtemp_noisetemp_plot = True

# ------------------------------------------------------------
# Measured Pe values (Run 4 column will be dropped)
# ------------------------------------------------------------
Pe_meas = [
    [None, 0.2698, 0.1179, 0.0788, 0.0564],
    [None, 0.3541, 0.1028, 0.0884, 0.0810],
    [None, 0.2385, 0.1359, 0.0770, 0.0735],
    [None, 0.3483, 0.1716, 0.0879, 0.1079],
    [None, 0.2269, 0.0577, 0.0806, 0.0524],
    [None, 0.2385, 0.0685, 0.0454, 0.0234],
]

Pe_err = [
    [None, 0.0107, 0.0208, 0.0057, 0.0027],
    [None, 0.0078, 0.0065, 0.0028, 0.0039],
    [None, 0.0088, 0.0155, 0.0030, 0.0056],
    [None, 0.0078, 0.0168, 0.0049, 0.0158],
    [None, 0.0075, 0.0110, 0.0032, 0.0040],
    [None, 0.0073, 0.0030, 0.0033, 0.0048],
]

# ------------------------------------------------------------
# Qubit frequencies (MHz) (Run 4 column will be dropped to match Pe data)
# ------------------------------------------------------------
f_ge_MHz = [
    [4184.1449, 4181.2182, 4189.8486, 4184.0449, 4194.7143],
    [3821.1544, 3821.1662, 3818.6675, 3823.3575, 3828.6250],
    [4155.9588, 4154.3727, 4161.4257, 4162.8739, 4173.6868],
    [4459.1987, 4458.4630, 4462.4391, 4467.3555, 4474.0803],
    [4471.1153, 4471.3083, 4474.0269, 4475.0307, 4485.2600],
    [4997.8579, 5000.6202, 4999.5140, 5006.1538, 5018.1472],
]

f_ge_err_MHz = [
    [0.0057, 0.0112, 0.0466, 0.0064, 0.0080],
    [0.0112, 0.0113, 0.0527, 0.0045, 0.0165],
    [0.2402, 0.0082, 0.0240, 0.0042, 0.0210],
    [0.0083, 0.0155, 0.0397, 0.0072, 0.0785],
    [0.0090, 0.0091, 4.0512, 0.0084, 0.0157],
    [0.0068, 0.0066, 0.0182, 0.0042, 0.0175],
]

# ------------------------------------------------------------
# Convert + drop Run 4 so arrays align with Runs 5-8
# ------------------------------------------------------------
runs = np.array([int(5), int(6), int(7), int(8)], dtype=int)

Pe_meas = np.array(Pe_meas, dtype=object).astype(float)[:, 1:]   # (6,4)
Pe_err  = np.array(Pe_err,  dtype=object).astype(float)[:, 1:]   # (6,4)

f_ge_MHz = np.array(f_ge_MHz, dtype=float)[:, 1:]            # (6,4)
f_ge_err_MHz = np.array(f_ge_err_MHz, dtype=float)[:, 1:]        # (6,4)

f_ge_Hz = f_ge_MHz * 1e6
f_ge_err_Hz = f_ge_err_MHz * 1e6

# Sanity check
if Pe_meas.shape != f_ge_Hz.shape:
    raise ValueError(f"Shape mismatch: Pe_meas {Pe_meas.shape} vs f_ge_Hz {f_ge_Hz.shape}")

# ------------------------------------------------------------
# Infer T_qubit from Pe
# T = hf / (kB * ln((1-Pe)/Pe))

h  = 6.62607015e-34
kB = 1.380649e-23

def T_from_Pe(Pe, f_Hz):
    return (h * f_Hz) / (kB * np.log((1 - Pe) / Pe))  # Kelvin

T_qubit_K  = T_from_Pe(Pe_meas, f_ge_Hz)
T_qubit_mK = 1e3 * T_qubit_K

# ------------------------------------------------------------
# Noise temperature model (MIT supplement style)
# Uses f_ge_Hz[qi, ri] per run
# ------------------------------------------------------------
if qtemp_noisetemp_plot:
    # These functions model photons leaking down the input line from multiple temperature stages through attenuation.
    def nbar_thermal(f_hz: float, T_K: float):
        # Gives the photon occupation number of a bosonic mode (microwave field) at frequency f
        if T_K <= 0:
            return 0.0
        x = (h * f_hz) / (kB * T_K)
        if x > 700:
            return 0.0
        return 1.0 / (np.exp(x) - 1.0) # formula

    def Te_from_nbar(f_hz: float, nbar: float):
        if nbar <= 0:
            return 0.0
        x = np.log(1.0 + 1.0 / nbar)
        return (h * f_hz) / (kB * x) # T formula

    def Te_from_stages(f_hz: float, stage_temps_K: dict, A_after_stage_dB: dict):
        # Used to sum attenuated contributions. This assumes power attenuation (not voltage attenuation) in dB.
        # A stage's thermal photons are reduced by attenuation after that stage.
        # This gives the effective photon occupation at the device input, which we use to calculate the predicted effective noise temp.
        n_eff = 0.0
        for stage, T in stage_temps_K.items():
            if stage not in A_after_stage_dB:
                raise ValueError(f"Missing attenuation-after-stage entry for '{stage}'")
            A_lin = 10 ** (-A_after_stage_dB[stage] / 10.0)
            n_eff += A_lin * nbar_thermal(f_hz, T)
        return Te_from_nbar(f_hz, n_eff)

    def cumulative_after_stage(config_dB: dict, order):
        # How we compute attenuation AFTER each stage
        cfg = {k: float(config_dB.get(k, 0.0)) for k in order}
        A_after = {}
        for i, stage in enumerate(order):
            A_after[stage] = sum(cfg[order[j]] for j in range(i + 1, len(order)))
        return A_after

    # stages. 300K stage added to account for linea attenuation that we measured warm
    order = ["300K", "4K", "1K", "100mK", "10mK"]
    stage_temps_K = {"300K": 300.0, "4K": 4.0, "1K": 1.0, "100mK": 0.100, "10mK": 0.010}

    # attenuation configs
    # Approximate insertion losses (dB)
    IL_marki = 0.9
    IL_eccosorb = 1.0  # adjust if you extract better number
    SS_line_loss_total_dB = 14.0   # total distributed SS cable loss on INPUT line (top of fridge to mcp)

    # ------------------------
    # Run-specific attenuation configs (INPUT line only)
    # ------------------------

    atten_config_by_run = {

        5: {  # Run 5: 3 eccosorbs + 1 Marki on input
            "4K": 20,
            "1K": 20,
            "10mK": 20 + 3 * IL_eccosorb + IL_marki,
        },

        6: {  # Run 6: same filtering as Run 5
            "4K": 20,
            "1K": 20,
            "10mK": 20 + 3 * IL_eccosorb + IL_marki,
        },

        7: {  # Run 7: attenuation redistribution
            "4K": 20,
            "1K": 6,
            "100mK": 10,
            "10mK": 30 + 3 * IL_eccosorb + IL_marki,
        },

        8: {  # Run 8: added HERD on input
            "4K": 20,
            "1K": 6,
            "100mK": 10,
            "10mK": 30 + 3 * IL_eccosorb + IL_marki,
        }
    }

    # compute Te per qubit per run using per-run mean frequency
    nQ, nRuns = Pe_meas.shape
    Te_mK = np.zeros((nQ, nRuns), dtype=float)

    for qi in range(nQ):
        for ri, r in enumerate(runs):
            cfg = atten_config_by_run[int(r)]

            # Lump all SS cable loss as at room temp (simple total-budget approach)
            cfg["300K"] = cfg.get("300K", 0.0) + SS_line_loss_total_dB

            A_after = cumulative_after_stage(cfg, order)
            f_hz = f_ge_Hz[qi, ri]  # per-run frequency
            Te_K = Te_from_stages(f_hz, stage_temps_K, A_after)
            Te_mK[qi, ri] = 1e3 * Te_K

    print("\nPredicted Te (mK) by qubit & run (using per-run f_ge):")
    for qi in range(nQ):
        vals = ", ".join([f"R{int(runs[i])}:{Te_mK[qi, i]:.2f}" for i in range(nRuns)])
        print(f"  Q{qi+1}: {vals}")

    # ------------------------------------------------------------
    # Plot: T_qubit(from Pe) vs Predicted Noise Temperature
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for qi in range(nQ):
        ax = axes[qi]
        ax.plot(runs, T_qubit_mK[qi], "o-", label=r"$T_{\mathrm{qubit}}$ (from $P_e$)")
        ax.plot(runs, Te_mK[qi], "s--", label=r"$T_e$ (pred. noise)")

        ax.set_title(f"Q{qi+1}")
        plt.xticks(runs, ['Run 5', 'Run 6', 'Run 7', 'Run 8'])

        if qi == 0:
            ax.text(
                5.5, 300,
                "Run 5: LPFs, 3 eccosorbs on input, 1 on output\n"
                "Run 6: cryo terminators, 0dBs, copper tape\n"
                "Run 7: attenuation changes, 0 eccosorbs on output\n"
                "Run 8: HERD filter + 0dB\n",
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )
        ax.grid(True)

    # fig.supxlabel("Run number", y=0.06)
    fig.supylabel("Temperature (mK)")
    fig.suptitle("Qubit Temperature (from $P_e$) vs Predicted Noise Temperature $T_e$", y=0.98)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=True, loc="lower center", bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    plt.show()