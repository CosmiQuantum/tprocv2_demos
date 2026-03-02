import numpy as np
import matplotlib.pyplot as plt

qtemp_noisetemp_plot = False

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
runs = np.array([5, 6, 7, 8], dtype=int)

Pe_meas = np.array(Pe_meas, dtype=object).astype(float)[:, 1:]   # (6,4)
Pe_err  = np.array(Pe_err,  dtype=object).astype(float)[:, 1:]   # (6,4)

f_ge_MHz     = np.array(f_ge_MHz, dtype=float)[:, 1:]            # (6,4)
f_ge_err_MHz = np.array(f_ge_err_MHz, dtype=float)[:, 1:]        # (6,4)

f_ge_Hz     = f_ge_MHz * 1e6
f_ge_err_Hz = f_ge_err_MHz * 1e6

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
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for qi in range(nQ):
        ax = axes[qi]

        ax.errorbar(
            runs, T_qubit_mK[qi],
            yerr=T_qubit_err_mK[qi],
            fmt="o-", capsize=3, elinewidth=1,
            label=r"$T_{\mathrm{qubit}}$ (from $P_e$)"
        )

        ax.plot(runs, Te_mK[qi], "s--", label=r"$T_e$ (pred. noise)")

        ax.set_title(f"Q{qi+1}")
        ax.set_xticks(runs)
        ax.set_xticklabels(['Run 5', 'Run 6', 'Run 7', 'Run 8'])
        ax.grid(True)

    fig.supylabel("Temperature (mK)")
    fig.suptitle("Qubit Temperature (from $P_e$) vs Predicted Noise Temperature $T_e$", y=0.98)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=True, loc="lower center", bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    plt.show()

#---------------------------------------- Definitions, additional plotting funcs ----------------------

def boxwhisker_per_qubit_vs_run(
    run_num_list,
    t1_vals_by_run=None,
    t2r_vals_by_run=None,
    t2e_vals_by_run=None,
    do_T1=True,
    do_T2R=True,
    do_T2E=True,
    n_qubits=6,
    ylims=(0, 120),
    yticks=np.arange(0, 121, 20),
    showfliers=True,
    whis=1.5,                 # Tukey whiskers: 1.5*IQR (use (5,95) for percentiles)
):
    """
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

    # ---------- fixed colors so they never change ----------
    metric_specs = []
    if do_T1:
        if t1_vals_by_run is None:
            raise ValueError("do_T1=True but t1_vals_by_run is None")
        metric_specs.append(("T1", t1_vals_by_run, "tab:blue"))
    if do_T2R:
        if t2r_vals_by_run is None:
            raise ValueError("do_T2R=True but t2r_vals_by_run is None")
        metric_specs.append(("T2R", t2r_vals_by_run, "tab:orange"))
    if do_T2E:
        if t2e_vals_by_run is None:
            raise ValueError("do_T2E=True but t2e_vals_by_run is None")
        metric_specs.append(("T2E", t2e_vals_by_run, "tab:green"))

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

    # positions: one "cluster" per run, with small offsets per metric
    n_runs = len(run_num_list)
    base_pos = np.arange(1, n_runs + 1)  # 1..n_runs
    n_metrics = len(metric_specs)
    offsets = np.linspace(-0.25, 0.25, n_metrics) if n_metrics > 1 else np.array([0.0])
    box_width = 0.22 if n_metrics > 1 else 0.45

    # ---------- figure ----------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True, constrained_layout=True)
    axes = axes.ravel()

    for q in range(n_qubits):
        ax = axes[q]

        for (label, vals_by_run, color), off in zip(metric_specs, offsets):
            # build list of arrays, one per run, for THIS qubit
            box_data = []
            for r in run_num_list:
                # vals_by_run[r] is list indexed by qubit
                box_data.append(cell_to_1d(vals_by_run[r][q]))

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

            # style
            for b in bp["boxes"]:
                b.set_facecolor(color)
                b.set_alpha(0.30)
                b.set_edgecolor(color)
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

        ax.set_title(f"Qubit {q+1}")
        ax.set_ylim(*ylims)
        ax.set_yticks(yticks)
        ax.grid(True, alpha=0.35)

        ax.set_xticks(base_pos)
        ax.set_xticklabels([f"Run {r}" for r in run_num_list])

        # legend (manual)
        handles = [plt.Line2D([0], [0], color=ms[2], lw=6, alpha=0.6) for ms in metric_specs]
        ax.legend(handles, [ms[0] for ms in metric_specs], loc="upper left", fontsize=9)

    fig.suptitle("Coherence vs Run Number (per qubit): Box & Whisker", fontsize=16)
    fig.supxlabel("Run Number", fontsize=12)
    fig.supylabel("Coherence time (µs)", fontsize=12)
    plt.show()

######################################################################################