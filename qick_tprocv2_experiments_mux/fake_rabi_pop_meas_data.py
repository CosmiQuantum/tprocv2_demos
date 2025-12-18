import numpy as np
from section_011_qubit_temperatures_efRabipt3_noqick_analysis import Temps_EFAmpRabiExperiment

def make_fake_rabi_IQ_for_plot_results(
    *,
    gains=None,
    npts=81,
    gain_max=1.0,
    # shared cosine argument params
    b=0.65,          # cycles per gain (because you use cos(2p*b*g + c))
    c=0.0,           # phase (rad)
    # choose offsets/amplitudes to match the look
    dI=-8.0, aI=4.0, # I ~ [-12, -4]
    dQ=-2.0, aQ=4.0, # Q ~ [-6,  2]
    noise_sigma_I=0.25,
    noise_sigma_Q=0.25,
    seed=20250101,
    # safety: keep magnitude away from 0 so it stays smooth
    min_radius=2.0,
):
    rng = np.random.default_rng(seed)

    if gains is None:
        gains = np.linspace(0.0, gain_max, npts)
    else:
        gains = np.asarray(gains, dtype=float)

    theta = 2*np.pi*b*gains + c

    I_true = dI + aI * np.cos(theta)
    Q_true = dQ + aQ * np.cos(theta)

    I = I_true + rng.normal(0.0, noise_sigma_I, size=gains.size)
    Q = Q_true + rng.normal(0.0, noise_sigma_Q, size=gains.size)

    # If you *really* want to guarantee no weirdness: shift both up if needed
    rmin = np.min(np.sqrt(I**2 + Q**2))
    if rmin < min_radius:
        shift = (min_radius - rmin) + 1e-6
        I += shift
        Q += shift
        I_true += shift
        Q_true += shift

    truth = {
        "b": b, "c": c,
        "dI": dI, "aI": aI,
        "dQ": dQ, "aQ": aQ,
        "seed": seed,
        "I_true": I_true,
        "Q_true": Q_true,
    }
    return gains, I, Q, truth


gains, I, Q, truth = make_fake_rabi_IQ_for_plot_results(
    npts=160,
    b=0.65,
    c=0.0,
    dI=-8.0, aI=4.0,
    dQ=-2.0, aQ=4.0,
    noise_sigma_I=0.25,
    noise_sigma_Q=0.25,
    seed=20250101,
)

QubitIndex = 0
list_of_all_qubits = [0,1,2,3,4,5]
number_of_qubits = len(list_of_all_qubits)
outerFolder = "/data/QICK_data/run7/6transmon/round_robin_benchmark/AB_paper_data/benchmark_analysis_plots/RPM_analysis/fake_data_study"
round_num = 0 # not relevant here
signal = "None" # let it choose between I or Q by itself
save_figs = True

temps_class = Temps_EFAmpRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits,  outerFolder, round_num, signal, save_figs)
best_signal_fit, pi_amp, A_amp, A_err, amp_fit, R2 = temps_class.plot_results(I, Q, gains)
print("Recovered A_amplitude:", A_amp, "+/-", A_err)