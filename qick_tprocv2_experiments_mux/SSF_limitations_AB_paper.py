import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Helpers
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
    Estimate probability that an excited state decays during a time window.

        P_decay = 1 - exp(-t / T1)

    Inputs should both be in microseconds.
    """
    if not np.isfinite(readout_length_us) or not np.isfinite(t1_us):
        return np.nan

    if readout_length_us < 0 or t1_us <= 0:
        return np.nan

    return 1.0 - np.exp(-readout_length_us / t1_us)


def caught_t1_decay_probability(readout_length_us, t1_us, sensitive_fraction=0.5):
    """
    Estimate the probability that a T1 decay occurs early enough in the readout
    window to be visible in the averaged IQ readout.

    Following the rough estimate from the contribution-analysis note:
        use only the first half of the readout window by default.

        P_caught_decay = 1 - exp(-(sensitive_fraction * t_ro) / T1)

    sensitive_fraction = 0.5 means only decays in the first half of the
    readout window are counted as likely to affect classification.
    """
    if not np.isfinite(readout_length_us) or not np.isfinite(t1_us):
        return np.nan

    if readout_length_us < 0 or t1_us <= 0:
        return np.nan

    effective_readout_length_us = sensitive_fraction * readout_length_us

    return 1.0 - np.exp(-effective_readout_length_us / t1_us)


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
        "snr": 4.3200,
        "Pe": 0.0135,
        "ie_new_Pg": 0.0773,
        "T1_us": 58.2909,
        "readout_length_us": 5.6000,
        "SSF": 0.8920,
    },
    1: {
        "qubit_label": "Q2",
        "snr": 3.7524,
        "Pe": 0.0254,
        "ie_new_Pg": 0.1008,
        "T1_us": 51.2307,
        "readout_length_us": 6.0000,
        "SSF": 0.8336,
    },
    2: {
        "qubit_label": "Q3",
        "snr": 4.1953,
        "Pe": 0.0292,
        "ie_new_Pg": 0.0831,
        "T1_us": 71.4829,
        "readout_length_us": 5.7000,
        "SSF": 0.8633,
    },
    3: {
        "qubit_label": "Q4",
        "snr": 3.9671,
        "Pe": 0.0264,
        "ie_new_Pg": 0.1091,
        "T1_us": 67.0882,
        "readout_length_us": 6.8500,
        "SSF": 0.8394,
    },
    5: {
        "qubit_label": "Q6",
        "snr": 3.5631,
        "Pe": 0.0103,
        "ie_new_Pg": 0.1577,
        "T1_us": 49.5618,
        "readout_length_us": 7.7500,
        "SSF": 0.7983,
    },
}


# ============================================================
# Main function: one run at a time
# ============================================================

def calculate_ssf_limitations_for_one_run(run_num, run_medians, sensitive_fraction=0.5):
    """
    Calculate and print SSF limitation contributions for one run.

    This version follows the contribution-analysis estimate:

        observed excited-state ground remnant =
            thermal Pe
            + caught T1 decay during readout
            + residual unexplained loss

    where:

        caught T1 decay =
            1 - exp(-(sensitive_fraction * readout_length) / T1)

    By default, sensitive_fraction = 0.5, meaning only decays in the first half
    of the readout are counted as likely to be caught by the averaged IQ readout.

    The residual term can include:
        - pi-pulse failure
        - state-preparation error
        - thresholding/fitting imperfections
        - other prep/readout loss

    Finite-SNR Gaussian overlap is kept as a separate estimate.
    """

    rows = []

    print(f"\n================ SSF LIMITATION ESTIMATES: RUN {run_num} ================")
    print(f"Using sensitive_fraction = {sensitive_fraction:.2f}")
    print("Caught T1 decay uses only this fraction of the readout window.\n")

    for qid, vals in run_medians.items():
        qlabel = vals.get("qubit_label", f"Q{qid + 1}")

        snr = float(vals["snr"])
        Pe = float(vals["Pe"])
        ie_new_Pg = float(vals["ie_new_Pg"])
        T1_us = float(vals["T1_us"])
        readout_length_us = float(vals["readout_length_us"])
        SSF = float(vals["SSF"])

        # 1. Finite-SNR Gaussian overlap error
        snr_overlap_error = snr_to_overlap_error(snr)

        # 2. Thermal population contribution
        thermal_error = Pe

        # 3. Full-window T1 decay probability
        # This is shown only as a diagnostic.
        full_window_t1_decay_prob = t1_decay_probability(readout_length_us, T1_us)

        # 4. Caught T1 decay estimate using the first half of the readout window
        caught_t1_decay_prob = caught_t1_decay_probability(
            readout_length_us=readout_length_us,
            t1_us=T1_us,
            sensitive_fraction=sensitive_fraction,
        )

        # 5. Residual unexplained excited-state ground remnant,
        # following the colleague's estimate:
        #
        # ie_new_Pg = Pe + caught_T1_decay + residual
        #
        # residual = ie_new_Pg - Pe - caught_T1_decay
        if (
            np.isfinite(ie_new_Pg)
            and np.isfinite(thermal_error)
            and np.isfinite(caught_t1_decay_prob)
        ):
            residual_unexplained_loss = max(
                0.0,
                ie_new_Pg - thermal_error - caught_t1_decay_prob,
            )

            explained_by_pe_and_t1 = min(
                ie_new_Pg,
                thermal_error + caught_t1_decay_prob,
            )

            if ie_new_Pg > 0:
                fraction_explained_by_pe_and_t1 = explained_by_pe_and_t1 / ie_new_Pg
                fraction_unexplained = residual_unexplained_loss / ie_new_Pg
            else:
                fraction_explained_by_pe_and_t1 = np.nan
                fraction_unexplained = np.nan
        else:
            residual_unexplained_loss = np.nan
            explained_by_pe_and_t1 = np.nan
            fraction_explained_by_pe_and_t1 = np.nan
            fraction_unexplained = np.nan

        # 6. Main rough SSF infidelity budget.
        #
        # This is a rough additive budget:
        #   SNR overlap + Pe + caught T1 decay + residual unexplained loss
        #
        # Since residual_unexplained_loss is defined from ie_new_Pg - Pe - caught_T1,
        # this is usually equivalent to:
        #   SNR overlap + ie_new_Pg
        #
        # when ie_new_Pg > Pe + caught_T1.
        estimated_total_error = np.nansum([
            snr_overlap_error,
            thermal_error,
            caught_t1_decay_prob,
            residual_unexplained_loss,
        ])

        # Alternative direct budget using observed excited-state loss.
        estimated_total_error_using_iePg = np.nansum([
            snr_overlap_error,
            ie_new_Pg,
        ])

        measured_ssf_infidelity = 1.0 - SSF

        row = {
            "Run": run_num,
            "Qubit": qlabel,

            # Inputs
            "SNR": snr,

            "SSF_frac": SSF,
            "SSF_percent": percent(SSF),

            "Measured_SSF_infidelity_frac": measured_ssf_infidelity,
            "Measured_SSF_infidelity_percent": percent(measured_ssf_infidelity),

            "Pe_frac": Pe,
            "Pe_percent": percent(Pe),

            "ie_new_Pg_frac": ie_new_Pg,
            "ie_new_Pg_percent": percent(ie_new_Pg),

            "T1_us": T1_us,
            "readout_length_us": readout_length_us,

            # Finite-SNR contribution
            "SNR_overlap_error_frac": snr_overlap_error,
            "SNR_overlap_error_percent": percent(snr_overlap_error),

            # T1 diagnostics
            "full_window_T1_decay_prob_frac": full_window_t1_decay_prob,
            "full_window_T1_decay_prob_percent": percent(full_window_t1_decay_prob),

            "caught_T1_decay_frac": caught_t1_decay_prob,
            "caught_T1_decay_percent": percent(caught_t1_decay_prob),

            # Colleague-style decomposition of ie_new_Pg
            "Pe_plus_caught_T1_frac": thermal_error + caught_t1_decay_prob,
            "Pe_plus_caught_T1_percent": percent(thermal_error + caught_t1_decay_prob),

            "explained_by_Pe_and_T1_frac": explained_by_pe_and_t1,
            "explained_by_Pe_and_T1_percent": percent(explained_by_pe_and_t1),

            "fraction_explained_by_Pe_and_T1_frac": fraction_explained_by_pe_and_t1,
            "fraction_explained_by_Pe_and_T1_percent": percent(fraction_explained_by_pe_and_t1),

            "residual_unexplained_loss_frac": residual_unexplained_loss,
            "residual_unexplained_loss_percent": percent(residual_unexplained_loss),

            "fraction_unexplained_frac": fraction_unexplained,
            "fraction_unexplained_percent": percent(fraction_unexplained),

            # Main rough totals
            "estimated_total_error_frac": estimated_total_error,
            "estimated_total_error_percent": percent(estimated_total_error),

            "estimated_total_error_using_iePg_frac": estimated_total_error_using_iePg,
            "estimated_total_error_using_iePg_percent": percent(estimated_total_error_using_iePg),
        }

        rows.append(row)

        print(f"\n{qlabel}")
        print(f"  Median SSF: {percent(SSF):.2f}%")
        print(f"  Measured SSF infidelity: {percent(measured_ssf_infidelity):.2f}%")
        print(f"  Median SNR: {snr:.4f}")
        print(f"  Finite-SNR misassignment: {percent(snr_overlap_error):.2f}%")
        print(f"  Thermal Pe: {percent(Pe):.2f}%")
        print(f"  Observed excited-state loss, ie_new Pg: {percent(ie_new_Pg):.2f}%")
        print(f"  T1 median: {T1_us:.2f} us")
        print(f"  Readout length: {readout_length_us:.2f} us")
        print(f"  Full-window T1 decay probability: {percent(full_window_t1_decay_prob):.2f}%")
        print(f"  Caught T1 decay estimate: {percent(caught_t1_decay_prob):.2f}%")
        print(f"  Pe + caught T1 decay: {percent(thermal_error + caught_t1_decay_prob):.2f}%")
        print(f"  Residual unexplained loss: {percent(residual_unexplained_loss):.2f}%")
        print(f"  Fraction of ie_new Pg explained by Pe + caught T1: {percent(fraction_explained_by_pe_and_t1):.1f}%")
        print(f"  Fraction of ie_new Pg unexplained: {percent(fraction_unexplained):.1f}%")
        print(f"  Estimated SSF infidelity budget: {percent(estimated_total_error):.2f}%")

    df = pd.DataFrame(rows)

    return df


# ============================================================
# Run calculation for Run 9
# ============================================================

ssf_limit_run9_df = calculate_ssf_limitations_for_one_run(
    run_num=9,
    run_medians=run9_medians,
    sensitive_fraction=0.5,
)


# ============================================================
# Print Run 9 summary table
# ============================================================

print("\n\n================ RUN 9 SUMMARY TABLE ================")

cols_to_print = [
    "Run",
    "Qubit",
    "SNR",
    "SSF_percent",
    "Measured_SSF_infidelity_percent",
    "SNR_overlap_error_percent",
    "Pe_percent",
    "ie_new_Pg_percent",
    "caught_T1_decay_percent",
    "Pe_plus_caught_T1_percent",
    "residual_unexplained_loss_percent",
    "fraction_unexplained_percent",
    "full_window_T1_decay_prob_percent",
    "estimated_total_error_percent",
]

summary_df = ssf_limit_run9_df[cols_to_print].copy()

summary_df = summary_df.rename(columns={
    "SNR": "Readout SNR",
    "SSF_percent": "Median SSF (%)",
    "Measured_SSF_infidelity_percent": "Measured SSF infidelity (%)",
    "SNR_overlap_error_percent": "Finite-SNR misassignment (%)",
    "Pe_percent": "Thermal population Pe (%)",
    "ie_new_Pg_percent": "Observed excited-state loss (%)",
    "caught_T1_decay_percent": "Caught T1 decay estimate (%)",
    "Pe_plus_caught_T1_percent": "Pe + caught T1 (%)",
    "residual_unexplained_loss_percent": "Residual unexplained loss (%)",
    "fraction_unexplained_percent": "Unexplained fraction of observed loss (%)",
    "full_window_T1_decay_prob_percent": "Full-window T1 decay diagnostic (%)",
    "estimated_total_error_percent": "Estimated SSF infidelity budget (%)",
})

summary_df = summary_df.round({
    "Readout SNR": 4,
    "Median SSF (%)": 2,
    "Measured SSF infidelity (%)": 2,
    "Finite-SNR misassignment (%)": 2,
    "Thermal population Pe (%)": 2,
    "Observed excited-state loss (%)": 2,
    "Caught T1 decay estimate (%)": 2,
    "Pe + caught T1 (%)": 2,
    "Residual unexplained loss (%)": 2,
    "Unexplained fraction of observed loss (%)": 1,
    "Full-window T1 decay diagnostic (%)": 2,
    "Estimated SSF infidelity budget (%)": 2,
})

print(summary_df.to_string(index=False))


# ============================================================
# Optional: save full unrounded table to CSV
# ============================================================

save_path = "/home/acolonce/Documents/analysis/multirun/SSF_limitations/run9_ssf_limit_budget.csv"
ssf_limit_run9_df.to_csv(save_path, index=False)
print(f"\nSaved: {save_path}")


# ============================================================
# Optional: save summary table as an image
# ============================================================

# fig, ax = plt.subplots(figsize=(26, 4.5))
# ax.axis("off")
#
# table = ax.table(
#     cellText=summary_df.values,
#     colLabels=summary_df.columns,
#     cellLoc="center",
#     colLoc="center",
#     loc="center",
# )
#
# table.auto_set_font_size(False)
# table.set_fontsize(7)
# table.scale(1.1, 1.5)
#
# # Bold header row
# for col_idx in range(len(summary_df.columns)):
#     table[(0, col_idx)].set_text_props(weight="bold")
#
# plt.savefig(
#     "/home/acolonce/Documents/analysis/multirun/SSF_limitations/run9_table.png",
#     dpi=300,
#     bbox_inches="tight"
# )
#
# plt.close(fig)
#
# print("Saved table image")