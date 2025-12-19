import numpy as np
import os
from qicklab.analysis.t1 import AnaT1
from qicklab.analysis.ssf import AnaSSF
from qicklab.analysis.auto_threshold import AnaAutoThreshold
import matplotlib.pyplot as plt

def run_qicklab_t1_all_qubits(
    data_dir,
    dataset,
    qubits_to_analyze=6,
    res_phase=None,
    ro_length=None,
    method_ssf="max_contrast",
    ssf_numbins=55,
    do_thresholding=True,
    iminuit_method_t1fit=True,
    verbose=False,
    selected_rounds=(0,),
    per_pt_errs=True,
    plot_threshold=False,
    save_plot_t1_round=False,      # save plots for ACCEPTED rounds (no show)
    save_rejected_plots=False,     # save plots for REJECTED rounds (no show)
    rejected_plots_dir="",
    save_plt_dir="",
    max_t1_keep=200.0,             # reject T1 >= this (and non-finite)
):
    """
    Runs SSF -> AutoThreshold (optional) -> T1 for each qubit index in [0..qubits_to_analyze-1].

    Returns:
      out = {
        "theta": {q: theta},
        "threshold": {q: threshold},
        "t1_vals": {q: [T1_est per selected_round]},
        "t1_errs": {q: [T1_err per selected_round]},
        "dates": {q: [date strings per selected_round]},  # from AnaT1 loader
        "rejected": {q: [{"round": r, "t1": T1_est, "t1_err": T1_err}, ...]}
      }
    """
    if res_phase is None:
        res_phase = [0.0] * qubits_to_analyze
    if ro_length is None:
        ro_length = [None] * qubits_to_analyze  # allow qicklab defaults if it can

    out = {
        "theta": {q: None for q in range(qubits_to_analyze)},
        "threshold": {q: None for q in range(qubits_to_analyze)},
        "t1_vals": {q: [] for q in range(qubits_to_analyze)},
        "t1_errs": {q: [] for q in range(qubits_to_analyze)},
        "dates": {q: [] for q in range(qubits_to_analyze)},
        "rejected": {q: [] for q in range(qubits_to_analyze)},
    }

    for q in range(qubits_to_analyze):
        print(f"Analyzing qubit {q + 1} data using Qicklab funcs")

        # -------------------- threshold (via SSF -> AutoThreshold) --------------------
        theta = 0.0
        threshold = 0.0

        if do_thresholding:
            # SSF (seed for AutoThreshold)
            ssf_ana_params = {"method": method_ssf, "numbins": ssf_numbins}
            opt_ssf_ge = AnaSSF(data_dir, dataset, q, folder="study_data", ana_params=ssf_ana_params)
            _ = opt_ssf_ge.load_all(verbose=verbose)
            ssf_result = opt_ssf_ge.run_analysis(verbose=verbose)

            ssf_theta = ssf_result["thetas"][0]
            ssf_threshold = ssf_result["thresholds"][0]

            # AutoThreshold
            auto_params = {
                "idx": 0,
                "plot": bool(plot_threshold),
                "method": "from_ssf",
                "ssf_theta": ssf_theta,
                "ssf_threshold": ssf_threshold,
                "res_phase": res_phase[q],
                "ro_length": ro_length[q],
            }
            auto = AnaAutoThreshold(data_dir, dataset, q, expt_name="t1_ge", datagroup="T1", ana_params=auto_params)
            _ = auto.load_all()
            auto_result = auto.run_analysis(verbose=verbose)

            theta = auto_result["theta"]
            threshold = auto_result["threshold"]

            auto.cleanup()
            del auto

        out["theta"][q] = theta
        out["threshold"][q] = threshold

        # -------------------- T1 --------------------
        t1_params = {
            "theta": theta,
            "threshold": threshold,
            "thresholding": bool(do_thresholding),
            "iminuit_fitting": bool(iminuit_method_t1fit),
            "per_pt_errs": bool(per_pt_errs),
        }

        t1_ge = AnaT1(data_dir, dataset, q, ana_params=t1_params)
        t1_data = t1_ge.load_all(verbose=verbose)
        _ = t1_ge.run_analysis(verbose=verbose)

        dates = t1_data.get("dates", [])

        # Directories (optionally per-qubit subfolders for sanity)
        good_dir_base = save_plt_dir or os.path.join(data_dir, "t1_plots")
        rej_dir_base = rejected_plots_dir or os.path.join(data_dir, "rejected_t1_plots")
        good_dir_q = os.path.join(good_dir_base, f"Q{q+1}")
        rej_dir_q = os.path.join(rej_dir_base, f"Q{q+1}")

        if save_plot_t1_round:
            os.makedirs(good_dir_q, exist_ok=True)
        if save_rejected_plots:
            os.makedirs(rej_dir_q, exist_ok=True)

        for r in selected_rounds:
            # First: compute T1 without showing/saving
            try:
                _, T1_err, T1_est = t1_ge.get_round(
                    r,
                    plot=False,  # never show
                    iminuit_method=iminuit_method_t1fit,
                    verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"[Q{q + 1}] get_round({r}) failed: {e}")
                continue

            rejected = (not np.isfinite(T1_est)) or (T1_est >= float(max_t1_keep))

            # If rejected, optionally save plot (still no show)
            if rejected:
                out["rejected"][q].append({"round": int(r), "t1": float(T1_est), "t1_err": float(T1_err)})

                if save_rejected_plots:
                    t1_ge.get_round(
                        r,
                        plot=False,
                        save_fig=True,
                        save_plt_dir=rej_dir_q,
                        iminuit_method=iminuit_method_t1fit,
                        verbose=verbose
                    )
                    if verbose:
                        print(f"[Q{q + 1}] REJECTED round {r}: T1={T1_est:.3g} µs (saved plot)")
                continue

            # Accepted: optionally save plot
            if save_plot_t1_round:
                t1_ge.get_round(
                    r,
                    plot=False,
                    save_fig=True,
                    save_plt_dir=good_dir_q,
                    iminuit_method=iminuit_method_t1fit,
                    verbose=verbose
                )

            # Store accepted values
            out["t1_vals"][q].append(T1_est)
            out["t1_errs"][q].append(T1_err)
            out["dates"][q].append(
                dates[r] if (isinstance(dates, (list, tuple)) and len(dates) > r) else None
            )

        # t1_ge.cleanup()  # optional

    return out

def comp_t1_methods_allQs_offline_vs_qicklab(
    qicklab_out,
    offline_tuple,
    out_dir=None,
    title_prefix="T1: QICKLab shots(thresholded) vs Offline shots(unthresholded)",
    dpi=140,
    save_comp_results_plot=True,
    sharey=True,
    max_cols=3,
    max_t1 = 200.0 # filter out data above this value
):
    """
    Compare:
      - QICKLab shot-based T1 extraction (often rotated+thresholded internally)
      - Offline shot processing T1 extraction (your method)

    Inputs
    ------
    qicklab_out : dict
        Must include:
          qicklab_out["t1_vals"][q] -> list of T1 estimates
          qicklab_out["t1_errs"][q] -> list of T1 errors
        (Optionally qicklab_out["dates"][q])

    offline_tuple : tuple
        (date_times, t1_vals, t1_errs, I_per_pt_errs, Q_per_pt_errs)
        where each of date_times/t1_vals/t1_errs is a dict keyed by qubit index.

    Returns
    -------
    comp : dict keyed by qubit with:
        t1_qicklab, err_qicklab, t1_offline, err_offline,
        delta = offline - qicklab (over overlap),
        mean_delta, rms_delta, max_abs_delta, frac_offline_bigger, n_overlap,
        plus date fields if present.
    """
    # --------- unpack offline tuple ----------
    try:
        date_times_off, t1_vals_off, t1_errs_off, I_per_pt_errs, Q_per_pt_errs = offline_tuple
    except Exception as e:
        raise ValueError(
            "offline_tuple must be (date_times, t1_vals, t1_errs, I_per_pt_errs, Q_per_pt_errs)"
        ) from e

    # --------- determine qubits to compare ----------
    qubits_qick = set(qicklab_out.get("t1_vals", {}).keys())
    qubits_off  = set(t1_vals_off.keys())
    qubits = sorted(qubits_qick.union(qubits_off))

    if len(qubits) == 0:
        print("[T1 compare] No qubits found in inputs.")
        return {}

    # --------- build dict ----------
    comp = {}
    for q in qubits:
        t1_q = qicklab_out.get("t1_vals", {}).get(q, [])
        e_q  = qicklab_out.get("t1_errs", {}).get(q, [])

        t1_o = t1_vals_off.get(q, [])
        e_o  = t1_errs_off.get(q, [])

        # arrays (allow missing errs -> NaNs)
        t1_q_arr = np.asarray(t1_q, float)
        e_q_arr  = np.asarray(e_q, float) if len(e_q) else np.full_like(t1_q_arr, np.nan, dtype=float)

        t1_o_arr = np.asarray(t1_o, float)
        e_o_arr  = np.asarray(e_o, float) if len(e_o) else np.full_like(t1_o_arr, np.nan, dtype=float)

        n = int(min(len(t1_q_arr), len(t1_o_arr)))

        # ---------------- filter out unphysical T1 values ----------------
        # restrict to overlapping region first
        t1_q_use = t1_q_arr[:n]
        e_q_use = e_q_arr[:n]
        t1_o_use = t1_o_arr[:n]
        e_o_use = e_o_arr[:n]

        # keep only points where both methods are <= max_t1
        good = (
                np.isfinite(t1_q_use) & np.isfinite(t1_o_use) &
                (t1_q_use <= max_t1) & (t1_o_use <= max_t1)
        )

        t1_q_use = t1_q_use[good]
        e_q_use = e_q_use[good]
        t1_o_use = t1_o_use[good]
        e_o_use = e_o_use[good]

        # update overlap count after filtering
        n = int(t1_q_use.size)
        # -----------------------------------------------------------------

        if n > 0:
            delta = t1_o_use - t1_q_use # Offline - QICKLab  (your convention)
            mean_delta = float(np.nanmean(delta))
            rms_delta  = float(np.sqrt(np.nanmean(delta**2)))
            max_abs_delta = float(np.nanmax(np.abs(delta)))
            frac_offline_bigger = float(np.mean(delta > 0))
        else:
            delta = np.array([], dtype=float)
            mean_delta = np.nan
            rms_delta = np.nan
            max_abs_delta = np.nan
            frac_offline_bigger = np.nan

        # optional dates
        dates_q = qicklab_out.get("dates", {}).get(q, []) if isinstance(qicklab_out.get("dates", {}), dict) else []
        dates_o = date_times_off.get(q, []) if isinstance(date_times_off, dict) else []

        comp[q] = {
            "t1_qicklab": t1_q_use,
            "err_qicklab": e_q_use,
            "t1_offline": t1_o_use,
            "err_offline": e_o_use,
            "delta": delta,
            "mean_delta": mean_delta,
            "rms_delta": rms_delta,
            "max_abs_delta": max_abs_delta,
            "frac_offline_bigger": frac_offline_bigger,
            "n_overlap": n,
            "dates_qicklab": dates_q,
            "dates_offline": dates_o,
        }

        # print summary line (mirrors your old behavior)
        if n > 0:
            print(
                f"[T1 Δ] Q{q + 1}: mean ΔT1 (Offline − QICKLab) = {mean_delta:+.3f} µs, "
                f"max |ΔT1| = {max_abs_delta:.3f} µs, "
                f"Offline > QICKLab in {frac_offline_bigger * 100:.1f}% of overlap "
                f"(N_overlap = {n})"
            )
        else:
            print(f"[T1 Δ] Q{q + 1}: no overlap to compare (len_offline={len(t1_o_arr)}, len_qicklab={len(t1_q_arr)}).")

    # --------- plot (grid, like your plot_t1_methods_comparison_all_qubits) ----------
    if save_comp_results_plot:
        if out_dir is None:
            raise ValueError("out_dir must be provided when save_plot=True")
        os.makedirs(out_dir, exist_ok=True)

        qubit_indices = sorted(comp.keys())
        n_qubits = len(qubit_indices)
        if n_qubits == 0:
            print("[T1 summary] No T1 results to plot.")
            return comp

        ncols = min(int(max_cols), n_qubits)
        nrows = int(np.ceil(n_qubits / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4 * ncols, 3.5 * nrows),
            sharey=bool(sharey)
        )
        axes = np.atleast_1d(axes).ravel()

        for ax, q in zip(axes, qubit_indices):
            t1_q = comp[q]["t1_qicklab"]
            t1_o = comp[q]["t1_offline"]

            n = comp[q]["n_overlap"]
            max_diff = comp[q]["max_abs_delta"]
            mean_delta = comp[q]["mean_delta"]
            frac_offline_bigger = comp[q]["frac_offline_bigger"]

            n_pts = int(max(len(t1_q), len(t1_o)))
            x = np.arange(n_pts)

            ax.plot(x[:len(t1_q)], t1_q, "o-", label="QICKLab-processed shots (thresholding)", linewidth=1)
            ax.plot(x[:len(t1_o)], t1_o, "s--", label="Offline-processed shots (no thresholding)", linewidth=1)
            ax.legend(fontsize=9, frameon=True)

            if n > 0 and np.isfinite(max_diff):
                frac_pct = int(round(frac_offline_bigger * 100)) if np.isfinite(frac_offline_bigger) else 0
                ax.set_title(
                    f"Q{q + 1}, max |Δ T1|={max_diff:.3g} µs\n"
                    f"⟨ΔT1⟩={mean_delta:+.3g} µs (Offline−QICKLab), {frac_pct}% > 0)",
                    fontsize=9
                )
            else:
                ax.set_title(f"Q{q + 1}, ΔT1 n/a", fontsize=9)

            ax.set_xlabel("Dataset index")
            ax.grid(alpha=0.3)
            ax.set_ylabel(r"$T_1$ (µs)")

        # remove unused axes
        for j in range(len(qubit_indices), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(title_prefix, fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])

        fname = os.path.join(out_dir, "T1_Comparison_All_Qubits.png")
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)

        print(f"[T1 summary] Saved comparison plot to: {fname}")

    return comp