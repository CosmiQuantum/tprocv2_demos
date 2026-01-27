import matplotlib.pyplot as plt
import numpy as np

# What coherence metrics do you want to plot?
do_T1 = False
do_T2R = False
do_T2E = False

# do you want to join them in a single plot?
plot_all_together = False

# Plot diagram with changes made per run?
plot_run_diagram = True

# set up axes for single plots vs a plot of three plots------------
def get_ax(i):
    if plot_all_together:
        return axes[i]
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        return ax

if plot_all_together:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, constrained_layout=True)
    
#------------------------------------------------------------------

show_text = False
colors = ['orange', 'blue', 'purple', 'green', 'brown', 'palevioletred']

# -------------------------------------------- T1 -------------------------------------------------------------
if do_T1:
    ax = get_ax(0)

    t1_vals = [
        [6.09, 14.94, 64.32, 63.71, 61.14],  # Qubit 1
        [19.61, 27.73, 59.71, 62.47, 63.60],  # Qubit 2
        [7.72, 16.43, 73.72, 52.64, 54.17],  # Qubit 3
        [12.74, 21.53, 59.55, 59.51, 64.76],  # Qubit 4
        [9.28, 12.31, 32.83, 55.77, 48.78],  # Qubit 5
        [9.44, 14.82, 41.52, 32.09, 30.45],  # Qubit 6
    ]

    t1_errs = [
        [0.24, 1.41, 4.20, 5.63, 3.06],  # Qubit 1
        [2.38, 3.34, 6.54, 3.98, 3.85],  # Qubit 2
        [0.26, 1.75, 4.18, 1.81, 4.82],  # Qubit 3
        [0.44, 2.49, 3.72, 6.65, 7.72],  # Qubit 4
        [0.46, 1.66, 4.19, 2.45, 2.99],  # Qubit 5
        [0.35, 1.39, 2.98, 2.02, 1.36]   # Qubit 6
    ]

    runs = np.array([4, 5, 6, 7, 8])

    # Add "Preliminary" text in the background
    if show_text:
        ax.text(
            0.5, 0.5, 'Preliminary',
            fontsize=70,
            color='lightgray',
            ha='center',
            va='center',
            alpha=0.3,
            rotation=45,
            transform=ax.transAxes,
            zorder=0
        )

    for qubit_index, vals in enumerate(t1_vals):
        vals_array = np.array(vals, dtype=np.float64)
        errs_array = np.array(t1_errs[qubit_index], dtype=np.float64)
        color = colors[qubit_index % len(colors)]

        ax.errorbar(
            runs, vals_array,
            yerr=errs_array,
            fmt='-o',
            color=color,
            capsize=3,
            elinewidth=1,
            label=f"Qubit {qubit_index + 1}",
            zorder=2
        )

        if show_text:
            for x, y in zip(runs, vals_array):
                if not np.isnan(y):
                    ax.text(x + 0.06, y + 2.0, f"{y:.0f}mK",
                            ha='center', va='bottom', fontsize=11, color=color)

    ax.set_xlabel("Run Number")
    ax.set_ylabel("(weighted) Mean T1 (us)")
    ax.set_title("(weighted) Mean T1 vs Run Number")
    ax.set_xticks(runs)
    ax.set_xticklabels(['Run 4', 'Run 5', 'Run 6', 'Run 7', 'Run 8'])
    ax.set_yticks(np.arange(0, 120, 10))
    ax.grid(True)

    if plot_all_together:
        ax.set_ylabel("us")
        ax.set_ylim(0, 120)

    if not plot_all_together:
        ax.legend()
        plt.show()

# -------------------------------------------- T2R -------------------------------------------------------------
if do_T2R:
    ax = get_ax(1)

    t2r_vals = [
        [11.18, 14.07, 30.08, 31.33, 64.49],  # Qubit 1
        [18.43, 14.69, 13.27, 60.64, 83.91],  # Qubit 2
        [8.45, 10.06, 27.58, 52.43, 28.42],   # Qubit 3
        [15.40, 15.54, 22.40, 58.78, 12.38],  # Qubit 4
        [13.51, 13.95, 6.33, 58.29, 51.98],   # Qubit 5
        [14.83, 14.47, 23.41, 25.70, 12.59],  # Qubit 6
    ]

    t2r_errs = [
        [1.02, 1.57, 3.59, 1.33, 4.71],       # Qubit 1
        [1.53, 2.03, 1.98, 6.25, 14.01],      # Qubit 2
        [1.77, 1.21, 3.64, 2.61, 1.90],       # Qubit 3
        [0.48, 1.78, 1.28, 6.46, 3.05],       # Qubit 4
        [0.88, 1.33, 1.77, 2.71, 5.53],       # Qubit 5
        [1.04, 1.09, 2.53, 1.23, 0.91]        # Qubit 6
    ]

    runs = np.array([4, 5, 6, 7, 8])

    if show_text:
        ax.text(
            0.5, 0.5, 'Preliminary',
            fontsize=70,
            color='lightgray',
            ha='center',
            va='center',
            alpha=0.3,
            rotation=45,
            transform=ax.transAxes,
            zorder=0
        )

    for qubit_index, vals in enumerate(t2r_vals):
        vals_array = np.array(vals, dtype=np.float64)
        errs_array = np.array(t2r_errs[qubit_index], dtype=np.float64)
        color = colors[qubit_index % len(colors)]

        ax.errorbar(
            runs, vals_array,
            yerr=errs_array,
            fmt='-o',
            color=color,
            capsize=3,
            elinewidth=1,
            label=f"Qubit {qubit_index + 1}",
            zorder=2
        )

        if show_text:
            for x, y in zip(runs, vals_array):
                if not np.isnan(y):
                    ax.text(x + 0.06, y + 2.0, f"{y:.0f}mK",
                            ha='center', va='bottom', fontsize=11, color=color)

    ax.set_xlabel("Run Number")
    ax.set_ylabel("(weighted) Mean T2R (us)")
    ax.set_title("(weighted) Mean T2R vs Run Number")
    ax.set_xticks(runs)
    ax.set_xticklabels(['Run 4', 'Run 5', 'Run 6', 'Run 7', 'Run 8'])
    ax.set_yticks(np.arange(0, 120, 10))
    ax.grid(True)

    if plot_all_together:
        ax.set_ylabel("")
        ax.set_ylim(0, 120)
        ax.tick_params(axis='y', left=False, labelleft=False)

    if not plot_all_together:
        ax.legend()
        plt.show()

# -------------------------------------------- T2E -------------------------------------------------------------
if do_T2E:
    ax = get_ax(2)

    t2e_vals = [
        [12.18, 14.13, 44.69, 87.75, 104.29],  # Qubit 1
        [19.75, 23.61, 43.52, 94.90, 105.93],  # Qubit 2
        [12.32, 11.10, 36.48, 62.83, 66.75],   # Qubit 3
        [17.48, 18.56, 46.98, 96.67, 82.34],   # Qubit 4
        [13.83, 14.06, 11.32, 76.37, 78.07],   # Qubit 5
        [16.06, 15.64, 39.32, 53.74, 49.81],   # Qubit 6
    ]

    t2e_errs = [
        [0.82, 1.32, 5.56, 6.58, 6.78],        # Qubit 1
        [1.48, 2.63, 3.40, 5.85, 9.44],        # Qubit 2
        [0.72, 1.27, 5.77, 1.37, 7.13],        # Qubit 3
        [0.67, 2.14, 4.58, 8.28, 9.57],        # Qubit 4
        [0.83, 1.69, 1.11, 4.09, 8.56],        # Qubit 5
        [1.04, 1.33, 7.63, 1.63, 4.18]         # Qubit 6
    ]

    runs = np.array([4, 5, 6, 7, 8])

    if show_text:
        ax.text(
            0.5, 0.5, 'Preliminary',
            fontsize=70,
            color='lightgray',
            ha='center',
            va='center',
            alpha=0.3,
            rotation=45,
            transform=ax.transAxes,
            zorder=0
        )

    for qubit_index, vals in enumerate(t2e_vals):
        vals_array = np.array(vals, dtype=np.float64)
        errs_array = np.array(t2e_errs[qubit_index], dtype=np.float64)
        color = colors[qubit_index % len(colors)]

        ax.errorbar(
            runs, vals_array,
            yerr=errs_array,
            fmt='-o',
            color=color,
            capsize=3,
            elinewidth=1,
            label=f"Qubit {qubit_index + 1}",
            zorder=2
        )

        if show_text:
            for x, y in zip(runs, vals_array):
                if not np.isnan(y):
                    ax.text(x + 0.06, y + 2.0, f"{y:.0f}mK",
                            ha='center', va='bottom', fontsize=11, color=color)

    ax.set_xlabel("Run Number")
    ax.set_ylabel("(weighted) Mean T2E (us)")
    ax.set_title("(weighted) Mean T2E vs Run Number")
    ax.set_xticks(runs)
    ax.set_xticklabels(['Run 4', 'Run 5', 'Run 6', 'Run 7', 'Run 8'])
    ax.set_yticks(np.arange(0, 120, 10))
    ax.grid(True)

    if plot_all_together:
        ax.set_ylabel("")
        ax.set_ylim(0, 120)
        ax.tick_params(axis='y', left=False, labelleft=False)
    if not plot_all_together:
        ax.legend()
        plt.show()


# -------------------------------------------- finalize joint plot ---------------------------------------------
if plot_all_together:
    axes[0].legend(title="Qubit", fontsize=9)
    fig.suptitle("Qubit Coherence Metrics vs Run Number", fontsize=16)
    fig.tight_layout()
    plt.show()
# ------------------------------- Plot diagram outlining modifications per run --------------------------------
if plot_run_diagram:
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import numpy as np
    import textwrap

    runs = [
        ("QUIET Run 4", [
            "One Eccosorb filter on input line",
            "No low-pass filters at mixing chamber stage",
            "3 circulators on output line",
        ]),
        ("QUIET Run 5", [
            "7.5 GHz low-pass filter on input line",
            "7.5 GHz low-pass filter on output line",
            "Two more Eccosorb filters on input line",
            "One new Eccosorb filter on output line",
        ]),
        ("QUIET Run 6", [
            "0 dB attenuator on input line",
            "Replaced cryo terminators on the circulators",
        ]),
        ("QUIET Run 7", [
            "Taped around the mag can seam with copper tape",
            "Qick board: adjusted attenuation on qubit input line (Johnson-noise-based)",
            "DAC attenuation change: 20 dB@4K, 6 dB@1K, 10 dB@100mK, 20 dB@10mK + extra 10 dB (with copper strap at 10mK)",
            "Removed Eccosorb on qubit output side (SNR concern)",
            "TWPA removed in this run (was previously not connected; just for tests)",
            "Qick Box change happened 7/21/2025 (AB paper data taken prior)",
        ]),
        ("QUIET Run 8", [
            "Replaced NbTi line (sub-optimal bending before)",
            "Used Qick Box: removed most warm amps/filters except the two warm DC blocks on the qubit input and output lines",
            "Output side: added + thermalized Herd1 filter + 0 dB to MCP1 using copper straps",
        ]),
    ]

    # ------------------- layout knobs -------------------
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    LEFT = 0.06
    W = 0.88
    GAP = 0.03
    ROUNDING = 0.02

    TITLE_FS = 16
    BODY_FS = 11

    # wrapping width (characters). Smaller = more wrapping, taller boxes.
    WRAP = 70

    # vertical spacing per line (axis units). Tune if needed.
    LINE_H = 0.028
    TITLE_PAD = 0.05
    TOP_PAD = 0.035
    BOTTOM_PAD = 0.03

    # ------------- precompute wrapped text + required heights -------------
    wrapped_runs = []
    heights = []

    for title, bullets in runs:
        wrapped_bullets = []
        n_lines = 0
        for b in bullets:
            lines = textwrap.wrap(b, width=WRAP)
            if not lines:
                lines = [""]
            wrapped_bullets.append(lines)
            n_lines += len(lines)
        # title line + bullet lines + padding
        h = TOP_PAD + TITLE_PAD + n_lines * LINE_H + BOTTOM_PAD
        wrapped_runs.append((title, wrapped_bullets))
        heights.append(h)

    heights = np.array(heights, float)
    total_h = heights.sum() + GAP * (len(runs) - 1)

    # scale everything to fit vertically in [0.05, 0.95]
    avail = 0.90
    scale = min(1.0, avail / total_h)
    heights *= scale
    gap = GAP * scale
    line_h = LINE_H * scale
    title_pad = TITLE_PAD * scale
    top_pad = TOP_PAD * scale
    bottom_pad = BOTTOM_PAD * scale

    # ------------------- draw -------------------
    y = 0.95
    boxes = []

    for (title, wrapped_bullets), h in zip(wrapped_runs, heights):
        y0 = y - h

        box = FancyBboxPatch(
            (LEFT, y0), W, h,
            boxstyle=f"round,pad=0.012,rounding_size={ROUNDING}",
            linewidth=1.6,
            facecolor="white",
            edgecolor="black",
        )
        ax.add_patch(box)
        boxes.append((y0, y))

        # title
        ax.text(
            LEFT + 0.02, y - top_pad,
            title, fontsize=TITLE_FS, fontweight="bold",
            va="top", ha="left"
        )

        # bullets (wrapped)
        yy = y - (top_pad + title_pad)
        for lines in wrapped_bullets:
            # first line gets the bullet symbol, continuation lines align under text
            ax.text(LEFT + 0.03, yy, f" {lines[0]}", fontsize=BODY_FS, va="top", ha="left")
            yy -= line_h
            for cont in lines[1:]:
                ax.text(LEFT + 0.055, yy, cont, fontsize=BODY_FS, va="top", ha="left")
                yy -= line_h

        y = y0 - gap

    # arrows
    for i in range(len(boxes) - 1):
        y0, _ = boxes[i]  # bottom of current
        _, ytop_next = boxes[i + 1]  # top of next
        start = (0.5, y0 - 0.004)
        end = (0.5, ytop_next + 0.004)
        ax.add_patch(FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.4,
            color="black",
        ))

    ax.set_title("QUIET Setup Changes by Run", fontsize=20, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.show()