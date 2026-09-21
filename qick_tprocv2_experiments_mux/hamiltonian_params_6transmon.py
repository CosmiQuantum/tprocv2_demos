import numpy as np
import pandas as pd

print("\n" + "=" * 70)
print("PARAMETER DEFINITIONS")
print("=" * 70)
parameter_definitions = """
f_ge       : qubit ground-to-excited transition frequency
f_ef       : qubit excited-to-second-excited transition frequency
f_res,g    : resonator frequency with qubit prepared in |g>
f_res,e    : resonator frequency with qubit prepared in |e>

alpha      : qubit anharmonicity
f_02       : qubit |0> to |2> transition frequency
EC/h       : charging energy expressed as a frequency
EJ/h       : Josephson energy expressed as a frequency
EJ/EC      : Josephson-to-charging energy ratio

2chi       : separation between resonator frequencies for |g> and |e>
chi        : dispersive qubit-resonator frequency shift

Delta      : qubit-resonator detuning
Sigma      : qubit-resonator frequency sum

g          : qubit-resonator coupling strength
|g/Delta|  : normalized coupling relative to detuning
n_crit     : approximate critical photon number
"""

print(parameter_definitions)
print("=" * 70)

# ============================================================
# INPUT FREQUENCIES
# All frequencies should be entered in MHz.
# Each list must contain 6 values: Q1, Q2, Q3, Q4, Q5, Q6.
# ============================================================

qubit_freq_ge = [4228.49, 3852.44, 4201.92, 4506.32, 4518.33, 5054.72]
qubit_freq_ef = [4054.38, 3674.86, 4027.53, 4334.48, 4346.29, 4887.39]

res_freq_ge = [6236.2014, 6296.685, 6358.140, 6427.895, 6494.4, 6562.500]
res_freq_ef = [6236.157, 6296.658, 6358.029, 6427.713, 6494.240, 6562.153]

qubit_freq_ge = np.asarray(qubit_freq_ge, dtype=float)
qubit_freq_ef = np.asarray(qubit_freq_ef, dtype=float)
res_freq_ge = np.asarray(res_freq_ge, dtype=float)
res_freq_ef = np.asarray(res_freq_ef, dtype=float)

# ============================================================
# INPUTS
# ============================================================

frequency_lists = {"qubit_freq_ge": qubit_freq_ge, "qubit_freq_ef": qubit_freq_ef, "res_freq_ge": res_freq_ge, "res_freq_ef": res_freq_ef}

for name, values in frequency_lists.items():
    if len(values) != 6:
        raise ValueError(
            f"{name} must contain exactly 6 values. "
            f"Currently contains {len(values)}." )

# ============================================================
# QUBIT PARAMETERS
# ============================================================

# Qubit anharmonicity:
# alpha = f_ef - f_ge
# This convention gives a negative alpha for a transmon.
alpha_MHz = qubit_freq_ef - qubit_freq_ge

# 0 -> 2 transition frequency:
# f_02 = f_ge + f_ef
f02_MHz = qubit_freq_ge + qubit_freq_ef

# Half of the 0 -> 2 transition frequency.
# Useful for comparison with two-photon spectroscopy.
f02_half_MHz = f02_MHz / 2

# Charging energy:
# alpha ~ -E_C / h
# Therefore E_C/h is expressed here in MHz.
EC_over_h_MHz = -alpha_MHz

# Josephson energy using the transmon approximation:
# f_ge ~ sqrt(8 EJ EC)/h - EC/h
# Rearranging:
# EJ/h = (f_ge + EC/h)^2 / (8 EC/h)
EJ_over_h_MHz = (((qubit_freq_ge + EC_over_h_MHz) ** 2) / (8 * EC_over_h_MHz))

# Dimensionless EJ/EC ratio
EJ_over_EC = EJ_over_h_MHz / EC_over_h_MHz

# ============================================================
# RESONATOR / DISPERSIVE PARAMETERS
# ============================================================

# Difference between resonator frequency measured for the
# excited-state and ground-state preparations.
# By convention here:
# 2chi = f_res,e - f_res,g
two_chi_MHz = res_freq_ef - res_freq_ge

# Dispersive shift
chi_MHz = two_chi_MHz / 2

# Magnitudes can also be convenient for comparison.
abs_two_chi_MHz = np.abs(two_chi_MHz)
abs_chi_MHz = np.abs(chi_MHz)

# ============================================================
# DETUNING
# ============================================================

# Since we do not yet have the bare resonator frequency,
# use the measured ground-state resonator frequency to
# approximate qubit-resonator detuning
# This method is using the measured ground-state dressed resonator frequency.
# For more accuracy, replace res_freq_ge with the bare resonator frequency later.
# Delta = f_ge - f_r
Delta_MHz = qubit_freq_ge - res_freq_ge

# Counter-rotating sum frequency
# Sigma = f_ge + f_r
Sigma_MHz = qubit_freq_ge + res_freq_ge

# ============================================================
# QUBIT-RESONATOR COUPLING g
# ============================================================

# Standard multilevel transmon dispersive approximation:
#
# chi = g^2 * alpha / [Delta * (Delta + alpha)]
#
# where:
#   alpha = f_ef - f_ge
#   Delta = f_ge - f_r
#
# Solving for g:
# g^2 = chi * Delta * (Delta + alpha) / alpha

g_squared_MHz2 = (chi_MHz * Delta_MHz * (Delta_MHz + alpha_MHz) / alpha_MHz)

# Warn if the chosen sign conventions produce an unphysical negative g^2
if np.any(g_squared_MHz2 < 0):
    print("WARNING: Negative g^2 found. Check chi and detuning sign conventions.")

g_MHz = np.sqrt(g_squared_MHz2)

# ============================================================
# DISPERSIVE-RATIO PARAMETERS
# ============================================================

# Dimensionless coupling-to-detuning ratio
g_over_Delta = np.abs(g_MHz / Delta_MHz)

# Critical photon number, using the common two-level
# dispersive approximation:
# n_crit = Delta^2 / (4 g^2)
n_crit = Delta_MHz**2 / (4 * g_MHz**2)

# ============================================================
# UNIT CONVERSIONS FOR OUTPUT
# ============================================================

alpha_GHz = alpha_MHz / 1000
EC_over_h_GHz = EC_over_h_MHz / 1000
EJ_over_h_GHz = EJ_over_h_MHz / 1000
g_GHz = g_MHz / 1000
Delta_GHz = Delta_MHz / 1000

# ============================================================
# BUILD OUTPUT TABLE
# ============================================================

results = pd.DataFrame({
    "Qubit": np.arange(1, 7),

    # Measured frequencies
    "f_ge [MHz]": qubit_freq_ge,
    "f_ef [MHz]": qubit_freq_ef,
    "f_res,g [MHz]": res_freq_ge,
    "f_res,e [MHz]": res_freq_ef,

    # Qubit parameters
    "alpha [MHz]": alpha_MHz,
    "f_02 [MHz]": f02_MHz,
    "f_02/2 [MHz]": f02_half_MHz,
    "EC/h [MHz]": EC_over_h_MHz,
    "EJ/h [GHz]": EJ_over_h_GHz,
    "EJ/EC": EJ_over_EC,

    # Dispersive parameters
    "2chi [MHz]": two_chi_MHz,
    "chi [MHz]": chi_MHz,
    "|2chi| [MHz]": abs_two_chi_MHz,
    "|chi| [MHz]": abs_chi_MHz,

    # Detuning
    "Delta [MHz]": Delta_MHz,
    "Sigma [MHz]": Sigma_MHz,

    # Coupling
    "g [MHz]": g_MHz,
    "|g/Delta|": g_over_Delta,
    "n_crit": n_crit,
})

# Display more cleanly
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print(results.round(6))