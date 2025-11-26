"""
Map h5 data files to their corresponding png plot files and dump
the associations (with timestamps) into a new HDF5 file.

Motivation: the timestamp on an h5 file name taken during RR (for all runs except the run6 SCIENCE RUN) correspond to
when the h5 file was saved but not when each measurement was actually carried out (all h5 files get saved at the end of
a round, once all measurements in the round have been carried out for each qubit). The pngs that were saved in each round
were created during each measurement though, so the timestamp in each png filename is a better representation of the
time at which each measurement was done.
"""
import argparse
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import h5py

# Mapping between data folders and plot folders (relative to study_data/Data_h5 and documentation respectively) -------

class create_h5_png_map():
    DATA_TO_PLOTS = {
        "qspec_ef": "qubit_spec_ef_plots",
        "qspec_ge": "qubit_spec_ge_plots",
        "q_temperatures": "q_temperatures_plots",
        "rabi_ge": "power_rabi_ge_plots",
        "res_ef": "res_spec_ef_plots",
        "res_ge": "res_spec_ge_plots",
        "ss_ge": "ss_ge_plots",  # <-- has extra qubit-level folders (Q1, Q5, etc)
        "t1_ge": "T1_ge",
        "t2e_ge": "SpinEcho_ge",
        "t2_ge": "Ramsey_ge",
    }
    def __init__(self, DATA_TO_PLOTS = None):
        self.DATA_TO_PLOTS = DATA_TO_PLOTS or self.DATA_TO_PLOTS
    # --- regex helpers ----------------------------------------------------
    TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}") # timestamps
    # PNG: uses R_1, R_2, ...
    ROUND_RE = re.compile(r"(?:^|_)[Rr]_(\d+)(?:_|\.|$)") # Rounds
    # h5: uses batch_1, batch_2, ...
    BATCH_RE = re.compile(r"(?:^|_)[Bb]atch_(\d+)(?:_|\.|$)") # Batches
    # matches '_q1', 'Q_1', 'Q1', 'q2', etc.
    QUBIT_RE = re.compile(r"[qQ][ _]?(\d+)") # Qubits


    def extract_timestamp(self, name: str):
        """Return 'YYYY-MM-DD_HH-MM-SS' substring from filename (or '')."""
        m = self.TIMESTAMP_RE.search(name)
        return m.group(0) if m else ""


    def extract_round_from_png(self, name: str):
        """Return integer round number from 'R_#' in PNG filename, or None."""
        m = self.ROUND_RE.search(name)
        return int(m.group(1)) if m else None


    def extract_batch_from_h5(self, name: str):
        """Return integer batch number from 'batch_#' in h5 filename, or None."""
        m = self.BATCH_RE.search(name)
        return int(m.group(1)) if m else None


    def extract_qubit_from_string(self, s: str):
        """Return integer qubit index from a string like '_q1', 'Q_1', 'Q1'."""
        m = self.QUBIT_RE.search(s)
        return int(m.group(1)) if m else None


    def extract_qubit_from_path(self, p: Path):
        """
        Try to infer qubit index from filename, then from parent directory name
        (useful for ss_ge_plots/Q1/whatever.png).
        """
        q = self.extract_qubit_from_string(p.name)
        if q is not None:
            return q
        # try immediate parent
        q = self.extract_qubit_from_string(p.parent.name)
        return q


    def index_h5_files(self, data_root: Path):
        """
        Build index: exp_type -> batch -> metadata for each h5 file.
        'batch' is treated as 'round' for matching the PNGs since we have been saving 1 round per h5 file lately.
        data_root should be .../study_data/Data_h5
        """
        index: dict[str, dict[int, dict]] = defaultdict(dict)

        for exp_dir in data_root.iterdir():
            if not exp_dir.is_dir():
                continue
            exp_type = exp_dir.name

            for h5_path in exp_dir.glob("*.h5"):
                b = self.extract_batch_from_h5(h5_path.name)
                if b is None:
                    print(f"[WARN] Could not find batch in h5 filename: {h5_path}")
                    continue

                ts = self.extract_timestamp(h5_path.name)
                if b in index[exp_type]:
                    print(
                        f"[WARN] Duplicate h5 for {exp_type} batch/round {b}: "
                        f"{index[exp_type][b]['h5_path']} vs {h5_path}"
                    )

                index[exp_type][b] = {
                    "h5_path": h5_path,
                    "h5_name": h5_path.name,
                    "h5_timestamp": ts,
                }

        return index

    def collect_matches(self, timestamp_dir: Path):
        """
        For a given timestamp_dir that contains 'study_data/Data_h5'
        and 'documentation', build a list of matched records.

        Matching key:
            - normal case: h5 batch number  <-->  png R_#
            - res_ef special case: match by nearest-in-time h5 timestamp
        """
        study_data = timestamp_dir / "study_data" / "Data_h5"
        docs_root = timestamp_dir / "documentation"

        if not study_data.is_dir():
            raise FileNotFoundError(f"Missing directory: {study_data}")
        if not docs_root.is_dir():
            raise FileNotFoundError(f"Missing directory: {docs_root}")

        h5_index = self.index_h5_files(study_data)
        records: list[dict] = []

        for data_exp, plot_exp in self.DATA_TO_PLOTS.items():
            if data_exp not in h5_index:
                continue  # no data of this type here

            plot_dir = docs_root / plot_exp
            if not plot_dir.is_dir():
                print(f"[WARN] Plot directory not found for '{data_exp}': {plot_dir}")
                continue

            # --- get the list of png paths, handling the ss_ge extra level ----
            png_paths: list[Path] = []
            if data_exp == "ss_ge":
                # structure: ss_ge_plots/Q1/*.png, ss_ge_plots/Q5/*.png, ...
                for q_dir in plot_dir.iterdir():
                    if q_dir.is_dir():
                        png_paths.extend(q_dir.glob("*.png"))
            else:
                png_paths = list(plot_dir.glob("*.png"))

            # for res_ef, pre-compute h5 batches sorted by timestamp
            if data_exp == "res_ef":
                # list of (batch, info) sorted by h5_timestamp
                h5_batches = sorted(
                    h5_index[data_exp].items(),
                    key=lambda kv: kv[1]["h5_timestamp"],
                )
            else:
                h5_batches = None  # unused

            for png_path in png_paths:
                png_ts = self.extract_timestamp(png_path.name)

                # --------- choose round / batch depending on experiment ---------
                if data_exp == "res_ef":
                    # Special case: R_0 bug, so use timestamps instead of R_#
                    if not png_ts:
                        print(f"[WARN] No timestamp in res_ef png filename: {png_path}")
                        continue

                    # find first h5 with timestamp >= png_ts, else fall back to last
                    chosen_batch = None
                    chosen_info = None
                    for b, info in h5_batches:
                        if info["h5_timestamp"] >= png_ts:
                            chosen_batch, chosen_info = b, info
                            break
                    if chosen_batch is None:
                        # png is after last h5; assign to last batch
                        chosen_batch, chosen_info = h5_batches[-1]

                    r = chosen_batch
                    h5_info = chosen_info

                else:
                    # Normal case: use R_# from png name
                    r = self.extract_round_from_png(png_path.name)
                    if r is None:
                        print(f"[WARN] No round (R_#) in png filename: {png_path}")
                        continue

                    if r not in h5_index[data_exp]:
                        print(
                            f"[WARN] No matching h5 for exp='{data_exp}' "
                            f"batch/round {r} for png: {png_path}"
                        )
                        continue

                    h5_info = h5_index[data_exp][r]

                # --------- qubit + record construction (shared) -----------------
                qubit = self.extract_qubit_from_path(png_path)
                if qubit is None:
                    print(f"[WARN] No qubit tag in png filename or parent dir: {png_path}")
                    qubit = -1  # keep it but mark as unknown

                records.append(
                    {
                        "experiment": data_exp,
                        "round": r,
                        "qubit": qubit,
                        "h5_relpath": str(h5_info["h5_path"].relative_to(timestamp_dir)),
                        "h5_filename": h5_info["h5_name"],
                        "h5_timestamp": h5_info["h5_timestamp"],
                        "png_relpath": str(png_path.relative_to(timestamp_dir)),
                        "png_filename": png_path.name,
                        "png_timestamp": png_ts,
                    }
                )

        return records

    def save_to_h5(self, output_path: Path, source_dir: Path, records: list[dict]) -> None:
        """Write the matched records to a new HDF5 file."""
        if not records:
            print("[INFO] No matches found; nothing to write.")
            return

        str_dt = h5py.string_dtype("utf-8")
        dtype = np.dtype(
            [
                ("experiment", str_dt),
                ("round", np.int32),  # here: batch number from h5 == R_# from png
                ("qubit", np.int32),
                ("h5_relpath", str_dt),
                ("h5_filename", str_dt),
                ("h5_timestamp", str_dt),
                ("png_relpath", str_dt),
                ("png_filename", str_dt),
                ("png_timestamp", str_dt),
            ]
        )

        arr = np.zeros(len(records), dtype=dtype)
        for i, rec in enumerate(records):
            arr[i] = (
                rec["experiment"],
                rec["round"],
                rec["qubit"],
                rec["h5_relpath"],
                rec["h5_filename"],
                rec["h5_timestamp"],
                rec["png_relpath"],
                rec["png_filename"],
                rec["png_timestamp"],
            )

        with h5py.File(output_path, "w") as f:
            dset = f.create_dataset("file_matches", data=arr)
            dset.attrs["description"] = (
                "Mapping between h5 data files (indexed by batch number) and "
                "per-qubit png plot files (indexed by R_#), including filenames, "
                "relative paths, and timestamps parsed from the filenames."
            )
            f.attrs["source_timestamp_dir"] = str(source_dir)
            print(f"[INFO] Wrote {len(records)} records to {output_path}")


    def run(self):
        parser = argparse.ArgumentParser(
            description="Map h5 data files to png plots and store the mapping in an HDF5 file."
        )
        parser.add_argument(
            "timestamp_dir",
            help="Directory that contains 'study_data/Data_h5' and 'documentation'",
        )
        parser.add_argument(
            "--output",
            "-o",
            default="h5_png_timestamp_map.h5",
            help="Output HDF5 file name (default: h5_png_timestamp_map.h5)",
        )
        args = parser.parse_args()

        ts_dir = Path(args.timestamp_dir).resolve()
        out_path = Path(args.output).resolve()

        records = self.collect_matches(ts_dir)
        self.save_to_h5(out_path, ts_dir, records)

#######################################################################################################################
class load_h5_png_map():
    """
    Below you will find utility functions for querying the HDF5 h5–png mapping file.
    Usage: filter mapping by qubit or experiment, get png paths, get h5 paths.

    Example use:
    data = m.load_map("h5_png_timestamp_map.h5") # load info and mapping
    q1 = m.filter_by_qubit(data, 1) #  extract all info for one qubit (1-indexed)
    ss5 = m.filter_by(data, experiment="ss_ge", qubit=5) # extract all info for a specific experiment and qubit
    pngs_q1 = m.get_png_paths(q1) # This takes the filtered dataset q1 and extracts the relative paths of the PNG files
    h5_q1 = m.get_h5_paths(q1) # This takes the filtered dataset q1 and extracts the unique paths of the h5 files
    m.summary(data) # Prints a quick snapshot of the entire dataset you loaded (list of expts, qubits, rounds, and # entries
    """
    # ---------------------------------------------------------
    # Core loader
    # ---------------------------------------------------------
    def load_map(self, path):
        """
        Load the file_matches dataset from the mapping HDF5.
        Returns a structured numpy array.
        """
        with h5py.File(path, "r") as f:
            data = f["file_matches"][:]
        return data

    # ---------------------------------------------------------
    # Filtering helpers
    # ---------------------------------------------------------
    def filter_by_qubit(self, data, qubit):
        """
        Return only rows corresponding to a specific qubit index (int).
        """
        return data[data["qubit"] == qubit]


    def filter_by_experiment(self, data, experiment):
        """
        Return only rows with a given experiment (string, e.g. 'ss_ge').
        """
        # stored as bytes in HDF5, convert user string → bytes
        exp_b = experiment.encode()
        return data[data["experiment"] == exp_b]


    def filter_by(self, data, experiment=None, qubit=None, round=None):
        """
        Flexible combined filter.
        Any of the fields can be provided:
            experiment='ss_ge', qubit=1, round=3
        """
        mask = np.ones(len(data), dtype=bool)

        if experiment is not None:
            mask &= data["experiment"] == experiment.encode()

        if qubit is not None:
            mask &= data["qubit"] == qubit

        if round is not None:
            mask &= data["round"] == round

        return data[mask]


    # ---------------------------------------------------------
    # Extracting paths / filenames / timestamps
    # ---------------------------------------------------------
    def get_png_paths(self, data):
        """
        Return a list of png relative paths (decoded to str).
        """
        return [row["png_relpath"].decode() for row in data]


    def get_h5_paths(self, data):
        """
        Return a list of unique h5 relative paths (decoded to str).
        """
        return sorted({row["h5_relpath"].decode() for row in data})


    def get_png_filenames(self, data):
        return [row["png_filename"].decode() for row in data]


    def get_h5_filenames(self, data):
        return sorted({row["h5_filename"].decode() for row in data})


    def get_png_timestamps(self, data):
        return [row["png_timestamp"].decode() for row in data]


    def get_h5_timestamps(self, data):
        return sorted({row["h5_timestamp"].decode() for row in data})


    # ---------------------------------------------------------
    # Pretty-print helpers (optional but nice)
    # ---------------------------------------------------------
    def summary(self, data):
        """
        Print a quick snapshot of the mapping content.
        """
        qubits = sorted({int(q) for q in data["qubit"]})
        exps = sorted({row.decode() for row in data["experiment"]})
        rounds = sorted({int(r) for r in data["round"]})

        print("Experiments:", exps)
        print("Qubits:", qubits)
        print("Rounds:", rounds)
        print("Total entries:", len(data))


####################################### Using the Classes here ############################################
# list of timestamp directories you want to process
# timestamp_dirs = [
#     "/data/QICK_data/run8/6transmon/round_robin/temperature_sweep_qubit_data/temperature_sweep_run8_25dBDAC_onechan_day1/2025-11-18_12-40-59",
# ]

# for ts in timestamp_dirs:
#     ts_dir = Path(ts).resolve()
#     # name output per dataset; you can customize this pattern
#     out_path = ts_dir / "documentation/h5_png_timestamp_map.h5"
#
#     m = load_h5_png_map()
#     data = m.load_map(out_path)
#     qtemps = m.filter_by(data, experiment="q_temperatures", qubit=1, round=1)
#     print(len(qtemps))  # should be 2
#     print(m.get_png_filenames(qtemps))
#     print(m.get_png_timestamps(qtemps))

# ts_dir = Path(timestamp_dirs[0]).resolve()
# m = load_h5_png_map()                       # create instance
# data = m.load_map(ts_dir / "documentation/h5_png_timestamp_map.h5")  # use instance method
# m.summary(data)                              # use summary

# def print_entry(data, i):
#     row = data[i]
#     print(f"--- Entry {i} ---")
#     for field in row.dtype.names:
#         val = row[field]
#         if isinstance(val, bytes):
#             val = val.decode()
#         print(f"{field}: {val}")
#
# print_entry(data, 1)