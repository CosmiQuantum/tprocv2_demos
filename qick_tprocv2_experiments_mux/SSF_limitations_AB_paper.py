import numpy as np
import math
import pandas as pd


# ============================================================
# Helper functions
# ============================================================

def snr_to_overlap_error(snr):
    """
    Convert median readout SNR to Gaussian-overlap assignment error.

    Assumes:
        SNR = |mu_e - mu_g| / sqrt((sigma_g^2 + sigma_e^2)/2)

    Approximate equal-width Gaussian overlap error:
        epsilon_SNR = 0.5 * erfc(SNR / (2*sqrt(2)))
    """
    if snr is None or not np.isfinite(snr):
        return np.nan

    return 0.5 * math.erfc(snr / (2.0 * math.sqrt(2.0)))


def t1_decay_probability(readout_length_us, t1_us):
    """
    Estimate probability that an excited state decays during readout.

        P_decay = 1 - exp(-t_ro / T1)

    Inputs should both be in microseconds.
    """
    if not np.isfinite(readout_length_us) or not np.isfinite(t1_us):
        return np.nan

    if readout_length_us < 0 or t1_us <= 0:
        return np.nan

    return 1.0 - np.exp(-readout_length_us / t1_us)


def percent(x):
    """
    Convert fraction to percent.
    """
    if x is None or not np.isfinite(x):
        return np.nan

    return 100.0 * x


# ============================================================
# Run 9 median inputs
# qid is zero-indexed:
# Q1 -> 0, Q2 -> 1, Q3 -> 2, Q4 -> 3, Q6 -> 5
# ============================================================

run9_medians = {
    0: {
        "qubit_label": "Q1",
        "snr": 3.0547,
        "Pe": 0.0135,
        "ie_new_Pg": 0.0773,
        "T1_us": 58.2909,
        "readout_length_us": 5.6000,
    },
    1: {
        "qubit_label": "Q2",
        "snr": 2.6534,
        "Pe": 0.0254,
        "ie_new_Pg": 0.1008,
        "T1_us": 51.2307,
        "readout_length_us": 6.0000,
    },
    2: {
        "qubit_label": "Q3",
        "snr": 2.9665,
        "Pe": 0.0292,
        "ie_new_Pg": 0.0831,
        "T1_us": 71.4829,
        "readout_length_us": 5.7000,
    },
    3: {
        "qubit_label": "Q4",
        "snr": 2.8052,
        "Pe": 0.0264,
        "ie_new_Pg": 0.1091,
        "T1_us": 67.0882,
        "readout_length_us": 6.8500,
    },
    5: {
        "qubit_label": "Q6",
        "snr": 2.5195,
        "Pe": 0.0103,
        "ie_new_Pg": 0.1577,
        "T1_us": 49.5618,
        "readout_length_us": 7.7500,
    },
}


# ============================================================
# Main function: one run at a time
# ============================================================

def calculate_ssf_limitations_for_one_run(run_num, run_medians):
    """
    Calculate and print SSF limitation contributions for one run.

    Parameters
    ----------
    run_num : int
        Run number.

    run_medians : dict
        run_medians[qid] = {
            "qubit_label": "Q1",
            "snr": ...,
            "Pe": ...,
            "ie_new_Pg": ...,
            "T1_us": ...,
            "readout_length_us": ...
        }

    Returns
    -------
    df : pandas.DataFrame
        Table of limitation estimates.
    """

    rows = []

    print(f"\n================ SSF LIMITATION ESTIMATES: RUN {run_num} ================")

    for qid, vals in run_medians.items():
        qlabel = vals.get("qubit_label", f"Q{qid + 1}")

        snr = float(vals["snr"])
        Pe = float(vals["Pe"])
        ie_new_Pg = float(vals["ie_new_Pg"])
        T1_us = float(vals["T1_us"])
        readout_length_us = float(vals["readout_length_us"])

        # 1. Finite-SNR Gaussian overlap error
        snr_overlap_error = snr_to_overlap_error(snr)

        # 2. Thermal population contribution
        thermal_error = Pe

        # 3. T1 decay during readout
        t1_decay = t1_decay_probability(readout_length_us, T1_us)

        # 4. Residual excited-prep/readout loss
        # ie_new_Pg contains T1 decay + residual preparation/readout loss.
        # Subtract T1 decay to avoid double-counting.
        if np.isfinite(ie_new_Pg) and np.isfinite(t1_decay):
            residual_e_prep_loss = max(0.0, ie_new_Pg - t1_decay)
        else:
            residual_e_prep_loss = np.nan

        # Total estimated limitation/error budget
        # This is a first-order additive estimate.
        estimated_total_error = np.nansum([
            snr_overlap_error,
            thermal_error,
            t1_decay,
            residual_e_prep_loss,
        ])

        # Same total can also be viewed as:
        # SNR overlap + Pe + ie_new_Pg
        # whenever ie_new_Pg > T1 decay.
        estimated_total_error_combined_iepg = np.nansum([
            snr_overlap_error,
            thermal_error,
            ie_new_Pg,
        ])

        row = {
            "Run": run_num,
            "Qubit": qlabel,

            "SNR": snr,
            "SNR_overlap_error_frac": snr_overlap_error,
            "SNR_overlap_error_percent": percent(snr_overlap_error),

            "Pe_frac": Pe,
            "Pe_percent": percent(Pe),

            "T1_us": T1_us,
            "readout_length_us": readout_length_us,
            "T1_decay_frac": t1_decay,
            "T1_decay_percent": percent(t1_decay),

            "ie_new_Pg_frac": ie_new_Pg,
            "ie_new_Pg_percent": percent(ie_new_Pg),

            "residual_e_prep_loss_frac": residual_e_prep_loss,
            "residual_e_prep_loss_percent": percent(residual_e_prep_loss),

            "estimated_total_error_frac": estimated_total_error,
            "estimated_total_error_percent": percent(estimated_total_error),

            "estimated_total_error_using_iePg_frac": estimated_total_error_combined_iepg,
            "estimated_total_error_using_iePg_percent": percent(estimated_total_error_combined_iepg),
        }

        rows.append(row)

        print(f"\n{qlabel}")
        print(f"  Median SNR: {snr:.4f}")
        print(f"  SNR overlap error: {percent(snr_overlap_error):.2f}%")
        print(f"  Thermal Pe: {percent(Pe):.2f}%")
        print(f"  T1 median: {T1_us:.2f} us")
        print(f"  Readout length: {readout_length_us:.2f} us")
        print(f"  T1 decay estimate: {percent(t1_decay):.2f}%")
        print(f"  ie_new Pg observed: {percent(ie_new_Pg):.2f}%")
        print(
            "  Residual excited-prep/readout loss "
            f"(ie_new Pg - T1 decay): {percent(residual_e_prep_loss):.2f}%"
        )
        print(f"  Estimated total limitation: {percent(estimated_total_error):.2f}%")

    df = pd.DataFrame(rows)

    return df


# ============================================================
# Run calculation for Run 9
# ============================================================

ssf_limit_run9_df = calculate_ssf_limitations_for_one_run(
    run_num=9,
    run_medians=run9_medians
)

print("\n\n================ RUN 9 SUMMARY TABLE ================")

cols_to_print = [
    "Run",
    "Qubit",
    "SNR",
    "SNR_overlap_error_percent",
    "Pe_percent",
    "T1_decay_percent",
    "ie_new_Pg_percent",
    "residual_e_prep_loss_percent",
    "estimated_total_error_percent",
]

print(ssf_limit_run9_df[cols_to_print].to_string(index=False))


# Optional: save to CSV
# save_path = "/home/acolonce/Documents/analysis/multirun/readout_fidelity/limitations/run9_ssf_limit_budget.csv"
# ssf_limit_run9_df.to_csv(save_path, index=False)
# print(f"Saved: {save_path}")