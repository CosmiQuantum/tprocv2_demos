import numpy as np
from section_011_qubit_temperatures_efRabipt3_noqick_analysis import Temps_EFAmpRabiExperiment
from analysis_021_plot_allRR_noqick import PlotRR_noQick
from tprocv2_demos.qick_tprocv2_experiments_mux.analysis_021_plot_allRR_noqick import PlotRR_noQick


def make_fake_rabi_IQ_1scan(
    *, # all arguments must be passed by keyword, not position, to avoid passing numbers incorrectly
    gains=None,
    npts=81,
    gain_max=1.0,
    # shared cosine argument params
    b=0.65,          # cycles per gain (because you use cos(2p*b*g + c))
    c=0.0,           # phase (rad)
    dI=-8.0, aI=4.0, # I ~ [-12, -4]
    dQ=-2.0, aQ=4.0, # Q ~ [-6,  2]
    noise_sigma_I=0.25,
    noise_sigma_Q=0.25,
    seed=20250101,
    min_radius=2.0 # safety: keep magnitude away from 0 to avoid W behavior
):

    rng = np.random.default_rng(seed) # fixed seed guarantees same IQ every time, same plots, same fit results, etc

    # In case you don't want to pass gains, it calculates them for you based on the number of pts you pass
    if gains is None:
        gains = np.linspace(0.0, gain_max, npts)
    else:
        gains = np.asarray(gains, dtype=float)

    # Shared cosine phase (Keeps I and Q synchronized).
    # b = how many oscillations you get per unit gain
    # c = phase offset (where the oscillation starts)
    theta = 2*np.pi*b*gains + c

    # aI, aQ = Oscillation size
    # dI, dQ = DC offset (moves the curve up/down)
    # what we do for I and Q below is: offset + amplitude x cos(theta)
    I_true = dI + aI * np.cos(theta)
    Q_true = dQ + aQ * np.cos(theta)

    # Adds Gaussian noise. Same noise every run because of fixed seed. Mean = 0
    I = I_true + rng.normal(0.0, noise_sigma_I, size=gains.size)
    Q = Q_true + rng.normal(0.0, noise_sigma_Q, size=gains.size)

    truth = {
        "b": b, "c": c,
        "dI": dI, "aI": aI,
        "dQ": dQ, "aQ": aQ,
        "seed": seed,
        "I_true": I_true,
        "Q_true": Q_true,
    }
    return gains, I, Q, truth


gains, I, Q, truth = make_fake_rabi_IQ_1scan(
    npts=160,
    b=0.65,
    c=0.0,
    dI=-8.0, aI=4.0,
    dQ=-2.0, aQ=4.0,
    noise_sigma_I=0.25,
    noise_sigma_Q=0.25,
    seed=20250101,
)

## ---------------------------------- Run the above function to generate one fake rabi population measurement scan: ----------------------------------------------
# QubitIndex = 0
# list_of_all_qubits = [0,1,2,3,4,5]
# number_of_qubits = len(list_of_all_qubits)
# outerFolder = "/data/QICK_data/run7/6transmon/round_robin_benchmark/AB_paper_data/benchmark_analysis_plots/RPM_analysis/fake_data_study"
# round_num = 0 # not relevant here
# signal = "None" # let it choose between I or Q by itself
# save_figs = True
#
# temps_class = Temps_EFAmpRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits,  outerFolder, round_num, signal, save_figs)
# best_signal_fit, pi_amp, A_amp, A_err, amp_fit, R2 = temps_class.plot_results(I, Q, gains)
# print("Recovered A_amplitude:", A_amp, "+/-", A_err)

# ----------------------------------- Make two scans, to simulate rabi population measurements for effective temperatures ---------------------------------------
def make_fake_rpm_two_scans(
    *,
    # shared geometry / physics
    gains=None,
    npts=160,
    gain_max=1.0,
    b=0.65,
    c=0.0,
    dI=-8.0,
    dQ=-2.0,

    # Pg (ground) parameters
    aI_g=4.0,
    aQ_g=4.0,
    noise_sigma_I_g=0.20,
    noise_sigma_Q_g=0.20,
    seed_g=20250101,

    # Pe (excited) parameters
    aI_e=0.8,
    aQ_e=0.8,
    noise_sigma_I_e=0.35,
    noise_sigma_Q_e=0.35,
    seed_e=20250102,
):
    # Pg scan
    gains, Ig, Qg, truth_g = make_fake_rabi_IQ_1scan(
        gains=gains,
        npts=npts,
        gain_max=gain_max,
        b=b,
        c=c,
        dI=dI,
        dQ=dQ,
        aI=aI_g,
        aQ=aQ_g,
        noise_sigma_I=noise_sigma_I_g,
        noise_sigma_Q=noise_sigma_Q_g,
        seed=seed_g,
    )

    # Pe scan (reuse same gains!)
    _, Ie, Qe, truth_e = make_fake_rabi_IQ_1scan(
        gains=gains,
        b=b,
        c=c,
        dI=dI,
        dQ=dQ,
        aI=aI_e,
        aQ=aQ_e,
        noise_sigma_I=noise_sigma_I_e,
        noise_sigma_Q=noise_sigma_Q_e,
        seed=seed_e,
    )

    truth = {
        "Pg": truth_g,
        "Pe": truth_e,
    }

    return gains, Ig, Qg, Ie, Qe, truth


QubitIndex = 0 # starts at zero
save_figs = True
list_of_all_qubits = [0,1,2,3,4,5]
number_of_qubits = len(list_of_all_qubits)
outerFolder_save_plots = r"C:\Users\Arianna\Documents\Grad\Research\CosmicQ\QUIET\run8\rpm_analysis_fake_data"
    #"/data/QICK_data/run7/6transmon/round_robin_benchmark/AB_paper_data/benchmark_analysis_plots/RPM_analysis/fake_data_study"
round_num = 0 # not relevant here
signal = "None" # let it choose between I or Q by itself
fit_saved = False
run_name = None
outerFolder = ""
unique_folder_path = ""
run_num = 8
filter_out_bad_amp_fits = True
figure_quality = 200
date = None
use_iminuit_instead = False

avail_ge_Qfreqs = [4194.77, 3828.69, 4173.69, 4474.23, 4485.38, 5018.12]
qubit_freq_MHz = avail_ge_Qfreqs[QubitIndex]

sigma_qfreq_MHz = 0.00474 # chosen based on run 8 typical qspec sigma from a Q1 lorentzian fit

temps_class_plts = Temps_EFAmpRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits,  outerFolder_save_plots, round_num, signal, save_figs)

_, Ig, Qg, Ie, Qe, _ = make_fake_rpm_two_scans(
    aI_g=5.4, aQ_g=5.4, # Oscillation size Pg
    aI_e=0.8, aQ_e=0.8,
    noise_sigma_I_g=0.20,
    noise_sigma_I_e=0.35)

# Ground
_, _, A_g, sigma_Ag, _, _ = temps_class_plts.plot_results(Ig, Qg, gains, use_iminuit_instead = use_iminuit_instead)

# Excited
_, _, A_e, sigma_Ae, _, _ = temps_class_plts.plot_results(Ie, Qe, gains, use_iminuit_instead = use_iminuit_instead)

temp_class_calcs = PlotRR_noQick(date, figure_quality, save_figs, fit_saved, signal, run_name, number_of_qubits, outerFolder,
                 outerFolder_save_plots, unique_folder_path, run_num, filter_out_bad_amp_fits)

T_K, T_mK, Pe, _ = temp_class_calcs.Qubit_Temperature_Convert(A_e, A_g, qubit_freq_MHz)

sigma_T_mK = temp_class_calcs.compute_temperature_error_RPM(
    A_e, A_g, Pe, T_mK,
    qubit_freq_MHz,
    sigma_Ae, sigma_Ag,
    sigma_qfreq_MHz,
)

print('Fake data results:')
print(f'Amplitudes: Ae = {A_e}, Ag = {A_g}')
print(f'ge qubit freq: {qubit_freq_MHz} MHz')
print(f'Pe = {Pe}')
print(f'Temperature: {T_mK} +/- {sigma_T_mK} mK')

import numpy as np

# ------------------------------------------------------------------------------------------------------------------------------------
#                                           Calculating Pe different ways for an analysis test
# ------------------------------------------------------------------------------------------------------------------------------------
# Pg scan (ground sequence) amplitudes from individual fits
A_I_g = 5.4129
A_Q_g = 5.4155

# Pe scan (excited sequence) amplitudes from individual fits
A_I_e = 0.8477
A_Q_e = 0.7810

# Magnitude-fit amplitudes (green curve "Fit to Magnitude Data, A=...")
# NOTE: use ABS for amplitudes if your fitter can return negative A due to phase conventions.
A_mag_g = abs(5.6448)
A_mag_e = abs(-1.0049)

# ----------------------------
# 4 amplitude methods (for each pair of scans)
# ----------------------------

# Method 1: magnitude-fit amplitude (green curve amplitude)
Ag_1 = A_mag_g
Ae_1 = A_mag_e
Pe_1 = Ae_1 / (Ae_1 + Ag_1)

# Method 2: use I-only amplitudes
Ag_2 = abs(A_I_g)
Ae_2 = abs(A_I_e)
Pe_2 = Ae_2 / (Ae_2 + Ag_2)

# Method 3: use Q-only amplitudes
Ag_3 = abs(A_Q_g)
Ae_3 = abs(A_Q_e)
Pe_3 = Ae_3 / (Ae_3 + Ag_3)

# Method 4: vector amplitude from I & Q fit amplitudes
Ag_4 = np.sqrt(A_I_g**2 + A_Q_g**2)
Ae_4 = np.sqrt(A_I_e**2 + A_Q_e**2)
Pe_4 = Ae_4 / (Ae_4 + Ag_4)

# ----------------------------
# Print results
# ----------------------------
print("Inputs:")
print(f"  Pg sequence: A_I={A_I_g}, A_Q={A_Q_g}, A_mag_fit={A_mag_g}")
print(f"  Pe sequence: A_I={A_I_e}, A_Q={A_Q_e}, A_mag_fit={A_mag_e}")
print()

print("Method 1: Magnitude-fit A (green curve, this is the Geerlings et al way)")
print(f"  Ag={Ag_1:.6f}, Ae={Ae_1:.6f}, Pe={Pe_1:.6f}")
print()

print("Method 2: I-only A")
print(f"  Ag={Ag_2:.6f}, Ae={Ae_2:.6f}, Pe={Pe_2:.6f}")
print()

print("Method 3: Q-only A")
print(f"  Ag={Ag_3:.6f}, Ae={Ae_3:.6f}, Pe={Pe_3:.6f}")
print()

print("Method 4: sqrt(A_I^2 + A_Q^2)")
print(f"  Ag={Ag_4:.6f}, Ae={Ae_4:.6f}, Pe={Pe_4:.6f}")
