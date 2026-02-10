import numpy as np
import matplotlib.pyplot as plt

MB_distribution_plot = False
qtemp_noisetemp_plot = True

if qtemp_noisetemp_plot:
    # ---------------------------
    # Predicted noise temperature model (MIT supplement logic)
    # Sum attenuated thermal sources -> effective noise temperature Te
    # :contentReference[oaicite:1]{index=1}
    # ---------------------------

    h = 6.62607015e-34  # Planck [J*s]
    kB = 1.380649e-23  # Boltzmann [J/K]

    def nbar_thermal(f_hz: float, T_K: float) -> float:
        """Thermal photon number at frequency f for temperature T (Planck)."""
        if T_K <= 0:
            return 0.0
        x = (h * f_hz) / (kB * T_K)
        if x > 700:
            return 0.0
        return 1.0 / (np.exp(x) - 1.0)


    def Te_from_nbar(f_hz: float, nbar: float) -> float:
        """Invert nbar(T) = 1/(exp(hf/kT)-1) -> Te."""
        if nbar <= 0:
            return 0.0
        x = np.log(1.0 + 1.0 / nbar)  # hf/kT
        return (h * f_hz) / (kB * x)


    def Te_from_stages(f_hz: float, stage_temps_K: dict, A_after_stage_dB: dict) -> float:
        """
        stage_temps_K: stage_name -> physical temp (K) of the source
        A_after_stage_dB: stage_name -> total attenuation AFTER that stage to device (dB)
        returns Te (K)
        """
        n_eff = 0.0
        for stage, T in stage_temps_K.items():
            if stage not in A_after_stage_dB:
                raise ValueError(f"Missing attenuation-after-stage entry for '{stage}'")
            A_lin = 10 ** (-A_after_stage_dB[stage] / 10.0)  # power attenuation
            n_eff += A_lin * nbar_thermal(f_hz, T)
        return Te_from_nbar(f_hz, n_eff)


    def cumulative_after_stage(config_dB: dict, order) -> dict:
        """
        config_dB: attenuators placed at stages (dB)
        order: stage order from warm -> cold
        returns: A_after_stage_dB (sum of attenuators at colder stages)
        """
        cfg = {k: float(config_dB.get(k, 0.0)) for k in order}
        A_after = {}
        for i, stage in enumerate(order):
            A_after[stage] = sum(cfg[order[j]] for j in range(i + 1, len(order)))
        return A_after


    # ---------------------------
    # Your run list
    # ---------------------------
    runs = np.array([5, 6, 7, 8])

    # ---------------------------
    # Your qubit frequencies (MHz -> Hz)
    # ---------------------------
    fq_MHz = np.array([4189.8773, 3820.4723, 4161.3726, 4463.15226, 4471.43854, 4997.86])
    fq_Hz = fq_MHz * 1e6

    # ---------------------------
    # Your qubit effective temperatures (mK) for runs 58
    # ---------------------------
    qubit_temps_mK = [
        [201.13, 99.22, 81.46, 71.21],  # Q1
        [302.20, 84.26, 78.32, 75.36],  # Q2
        [170.90, 106.43, 80.11, 78.82],  # Q3
        [339.13, 135.15, 91.27, 100.85],  # Q4
        [174.67, 76.82, 87.88, 74.10],  # Q5
        [225.68, 91.56, 78.55, 64.90],  # Q6
    ]
    qtemp_errs_mK = [
        [10.98, 9.99, 2.52, 1.26],  # Q1
        [16.42, 2.74, 1.18, 1.64],  # Q2
        [7.33, 8.37, 1.37, 2.55],  # Q3
        [19.11, 10.16, 2.38, 7.78],  # Q4
        [6.17, 5.77, 1.58, 2.07],  # Q5
        [8.18, 1.65, 1.98, 3.96],  # Q6
    ]

    # ---------------------------
    # Stage temps + your attenuation change at Run 7
    # Stages you mentioned: 4K, 1K, 100mK, 10mK
    # ---------------------------
    order = ["4K", "1K", "100mK", "10mK"]

    stage_temps_K = {
        "4K": 4.0,
        "1K": 1.0,
        "100mK": 0.100,
        "10mK": 0.010,
    }

    # Old (runs 56): 20 dB at 4K, 20 dB at 1K, 20 dB at 10 mK
    old_atten_config = {"4K": 20, "1K": 20, "10mK": 20}

    # New (runs 78): 20 dB at 4K, 6 dB at 1K, 10 dB at 100 mK, 30 dB at 10 mK
    new_atten_config = {"4K": 20, "1K": 6, "100mK": 10, "10mK": 30}

    A_after_old = cumulative_after_stage(old_atten_config, order)
    A_after_new = cumulative_after_stage(new_atten_config, order)

    print("Attenuation AFTER each stage (dB):")
    print("  Old (runs 56):", A_after_old)
    print("  New (runs 78):", A_after_new)

    # ---------------------------
    # Compute Te per qubit per run
    # Te[q, r_i] in mK
    # ---------------------------
    Te_mK = np.zeros((6, len(runs)), dtype=float)

    for qi in range(6):
        f_hz = fq_Hz[qi]
        for ri, r in enumerate(runs):
            A_after = A_after_old if int(r) < 7 else A_after_new
            Te_K = Te_from_stages(f_hz, stage_temps_K, A_after)
            Te_mK[qi, ri] = 1e3 * Te_K

    print("\nPredicted Te (mK) by qubit & run:")
    for qi in range(6):
        vals = ", ".join([f"R{int(r)}:{Te_mK[qi, i]:.2f}" for i, r in enumerate(runs)])
        print(f"  Q{qi + 1} (f={fq_MHz[qi]:.2f} MHz): {vals}")

    # ---------------------------
    # PLOT 1: T_qubit vs Te (6 panels)
    # ---------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for qi in range(6):
        ax = axes[qi]
        Tq = np.array(qubit_temps_mK[qi], dtype=float)
        Tq_err = np.array(qtemp_errs_mK[qi], dtype=float)

        ax.errorbar(runs, Tq, yerr=Tq_err, fmt="o-", capsize=3, label=r"$T_{\mathrm{qubit}}$")
        ax.plot(runs, Te_mK[qi], "s--", label=r"$T_e$ (pred. noise)")

        ax.set_title(f"Qubit {qi + 1}  ({fq_MHz[qi]:.1f} MHz)")
        ax.grid(True)

    fig.supxlabel("Run number", y=0.06)
    fig.supylabel("Temperature (mK)")
    fig.suptitle("Qubit Temperature vs Predicted Noise Temperature $T_e$ (attenuation change at Run 7)", y=0.98)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=True, loc="lower center", bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    plt.show()

    # ---------------------------
    # PLOT 2: Convergence metric ?T = T_qubit - Te (6 panels)
    # ---------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for qi in range(6):
        ax = axes[qi]
        Tq = np.array(qubit_temps_mK[qi], dtype=float)
        Tq_err = np.array(qtemp_errs_mK[qi], dtype=float)

        delta = Tq - Te_mK[qi]
        ax.errorbar(runs, delta, yerr=Tq_err, fmt="o-", capsize=3)
        ax.axhline(0, ls="--", lw=1)

        ax.set_title(f"Qubit {qi + 1}  ({fq_MHz[qi]:.1f} MHz)")
        ax.grid(True)

    fig.supxlabel("Run number", y=0.06)
    fig.supylabel(r"$T_{\mathrm{qubit}} - T_e$ (mK)")
    fig.suptitle("Convergence Check: $T_{\\mathrm{qubit}} - T_e$ (closer to 0 ? better thermalization)", y=0.98)

    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    plt.show()

if MB_distribution_plot:
    # Physical constants
    h = 6.62607015e-34      # Planck constant [J·s]
    kB = 1.380649e-23      # Boltzmann constant [J/K]

    # Qubit parameters
    f_ge = 4189.8773*(10**6)  # qubit frequency [Hz]
    E_ge = h * f_ge        # qubit energy splitting [J]

    # Temperature range (bath temperature)
    T_mK = np.linspace(5, 200, 300)      # mK
    T = T_mK * 1e-3                      # K

    # Maxwell-Boltzmann excited-state population (2-level system)
    P_e = np.exp(-E_ge / (kB * T))
    P_e = P_e / (1 + P_e)   # proper normalization

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(T_mK, 100 * P_e, lw=2, label=r"$P_e$ (Maxwell-Boltzmann)")

    plt.xlabel("Bath Temperature (mK)")
    plt.ylabel("Excited-State Population (%)")
    plt.title("Thermal Excited-State Population vs Temperature")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()