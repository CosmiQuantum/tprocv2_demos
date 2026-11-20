"""
Map H5 data files to their corresponding PNG plot files and save the
associations, including their timestamps, into a new HDF5 mapping file.

PURPOSE
-------
During round-robin data taking, H5 files are generally saved after all
qubit measurements in a round have been completed. Consequently, the
timestamp in an H5 filename reflects when the file was saved rather
than when an individual qubit was measured.

PNG plots are generated during individual measurements, so their
filename timestamps provide a more accurate estimate of when each
measurement was performed.

This mapper associates PNG files with their corresponding H5 files
so that downstream analysis can use PNG timestamps instead.

MATCHING LOGIC
--------------
The mapper examines H5 batch numbers (batch_#) and PNG round numbers
(R_#) independently for each experiment and selects a matching method
based on the numbering patterns found in the filenames.

1. Normal numbering:
   - H5 batches begin at 1 and increase.
   - PNG rounds begin at 1 and increase.
   - Files are matched directly using H5 batch number = PNG round number.

2. H5 batch counter stuck at zero:
   - H5 filenames repeatedly contain batch_0.
   - PNG round numbers begin at 1 and increase.
   - Files are matched using timestamps, associating each PNG with
     the first H5 file saved at or after the PNG timestamp.
   - The PNG round number is retained as the mapping's round identifier.

3. PNG round counter stuck at zero:
   - H5 batch numbers begin at 1 and increase.
   - PNG filenames repeatedly contain R_0.
   - Files are matched using timestamps, associating each PNG with
     the first H5 file saved at or after the PNG timestamp.
   - The H5 batch number is retained as the mapping's round identifier.

4. Both counters stuck at zero:
   - H5 filenames repeatedly contain batch_0.
   - PNG filenames repeatedly contain R_0.
   - The experiment is skipped because neither filename counter
     provides a reliable round identifier.

5. Unexpected or inconsistent numbering:
   - Numbering patterns that do not satisfy the supported matching
     conditions are flagged and skipped.
   - Missing or interrupted rounds may produce gaps in numbering.
     Such gaps are reported but are not automatically disqualifying.

REPEATED PNG FILES
------------------
Multiple PNGs may have been accidentally saved for the same measurement.

- Every PNG that successfully matches an H5 file is retained.
- Repeated PNGs with the same round identifier can therefore map
  to the same H5 file, qubit, and round.
- Repeated PNGs are also retained during timestamp-based matching,
  but possible duplicates are flagged with warnings.
- When PNG round numbers are stuck at R_0, timestamps alone cannot
  establish whether multiple PNGs correspond to repeated saves
  or different measurements.
- Mapping records are sorted by PNG timestamp, allowing downstream
  analysis to use subset[0] to select the earliest matching PNG.

AMBIGUOUS OR MISSING MATCHES
----------------------------
- PNGs without valid timestamps, round numbers, or qubit identifiers
  are skipped with warnings.
- PNGs without a corresponding H5 file are skipped.
- During timestamp matching, conflicting associations between
  different PNG round numbers and H5 files are flagged and skipped.
- Duplicate H5 batch identifiers are checked to avoid ambiguous
  direct matching.
- If an experiment contains H5 files but its PNG directory is
  missing or contains no PNG files, a FileNotFoundError is raised.
- PNGs occurring after the last available H5 save are not assigned
  to that H5 file during timestamp matching.

OUTPUT AND COMPATIBILITY
------------------------
Each mapping record contains the experiment type, logical round
number, qubit number, H5 and PNG relative paths, filenames, and
timestamps.

The logical round number is taken from the functioning PNG round
counter or H5 batch counter, depending on the matching strategy.

The mapping is automatically recreated and the previous mapping
file overwritten every time load_h5_png_map.load_map() is called.
Existing mapping files are never trusted without regeneration.

The loader supports filtering by experiment, qubit, and round,
and handles both bytes and string values when reading HDF5 fields.

DIAGNOSTICS
-----------
The mapper prints numbering checks, detected gaps, the selected
matching strategy, numbers of successful and unsuccessful matches,
possible duplicate PNGs, and warnings for ambiguous associations.

IMPORTANT ASSUMPTIONS
---------------------
- Each H5 file generally contains one completed data-taking round,
  potentially including measurements from multiple qubits.
- PNG timestamps precede or coincide with their corresponding
  H5 save timestamps during normal acquisition.
- Timestamp-based associations can be uncertain when acquisitions
  are interrupted or filenames contain insufficient information.
- No internal H5 'Round Num' or 'Batch Num' values are inspected.
  Matching relies exclusively on filenames, timestamps, and paths.
"""
import argparse
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import h5py


class create_h5_png_map:
    DATA_TO_PLOTS = {
        "qspec_ef": "qubit_spec_ef_plots",
        "qspec_ge": "qubit_spec_ge_plots",
        "q_temperatures": "q_temperatures_plots",
        "rabi_ge": "power_rabi_ge_plots",
        "res_ef": "res_spec_ef_plots",
        "res_ge": "res_spec_ge_plots",
        "ss_ge": "ss_ge_plots",
        "t1_ge": "T1_ge",
        "t2e_ge": "SpinEcho_ge",
        "t2_ge": "Ramsey_ge",
    }

    TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")
    ROUND_RE = re.compile(r"(?:^|_)[Rr]_(\d+)(?:_|\.|$)")
    BATCH_RE = re.compile(r"(?:^|_)[Bb]atch_(\d+)(?:_|\.|$)")
    QUBIT_RE = re.compile(r"[qQ][ _]?(\d+)")

    def __init__(self, DATA_TO_PLOTS=None):
        self.DATA_TO_PLOTS = DATA_TO_PLOTS if DATA_TO_PLOTS is not None else self.DATA_TO_PLOTS
        self.last_report = {}

    # --------------------------- Filename helpers ---------------------------

    def extract_timestamp(self, name):
        m = self.TIMESTAMP_RE.search(str(name))
        return m.group(0) if m else ""

    def extract_round_from_png(self, name):
        m = self.ROUND_RE.search(str(name))
        return int(m.group(1)) if m else None

    def extract_batch_from_h5(self, name):
        m = self.BATCH_RE.search(str(name))
        return int(m.group(1)) if m else None

    def extract_qubit_from_string(self, value):
        m = self.QUBIT_RE.search(str(value))
        return int(m.group(1)) if m else None

    def extract_qubit_from_path(self, path):
        path = Path(path)
        q = self.extract_qubit_from_string(path.name)
        return q if q is not None else self.extract_qubit_from_string(path.parent.name)

    # --------------------------- Numbering checks ---------------------------

    def check_numbering(self, values):
        """Check chronological unique IDs; gaps may indicate interrupted acquisitions."""
        values = list(values)
        ids = list(dict.fromkeys(values))
        constant_zero = bool(ids) and set(ids) == {0}

        return {
            "values": ids,
            "starts_at_1": bool(ids) and ids[0] == 1,
            "constant": len(ids) == 1,
            "constant_zero": constant_zero,
            "increases_by_one": len(ids) > 1 and all(b == a + 1 for a, b in zip(ids, ids[1:])),
            "strictly_increasing": all(b > a for a, b in zip(ids, ids[1:])),
            "gaps": sorted(set(range(min(ids), max(ids) + 1)) - set(ids)) if ids else [],
        }

    def check_h5_png_numbering(self, h5_paths, png_paths):
        h5_paths = sorted(map(Path, h5_paths), key=lambda p: (self.extract_timestamp(p.name), p.name))
        png_paths = sorted(map(Path, png_paths), key=lambda p: (self.extract_timestamp(p.name), p.name))

        batches = [self.extract_batch_from_h5(p.name) for p in h5_paths]
        rounds = [self.extract_round_from_png(p.name) for p in png_paths]

        return {
            "h5": self.check_numbering([n for n in batches if n is not None]),
            "png": self.check_numbering([n for n in rounds if n is not None]),
            "missing_h5_ids": batches.count(None),
            "missing_png_ids": rounds.count(None),
            "h5_file_count": len(h5_paths),
            "png_file_count": len(png_paths),
        }

    # --------------------------- H5 indexing ---------------------------

    def index_h5_files(self, data_root):
        """Retain all H5 files, including files with repeated batch numbers."""
        index = defaultdict(list)

        for exp_dir in sorted(Path(data_root).iterdir()):
            if not exp_dir.is_dir():
                continue

            for path in sorted(exp_dir.glob("*.h5")):
                batch = self.extract_batch_from_h5(path.name)
                ts = self.extract_timestamp(path.name)

                if batch is None or not ts:
                    print(f"[WARN] Skipping H5 with missing batch/timestamp: {path}")
                    continue

                index[exp_dir.name].append({
                    "batch": batch,
                    "h5_path": path,
                    "h5_name": path.name,
                    "h5_timestamp": ts,
                })

        return index

    # --------------------------- Matching strategy ---------------------------

    def _strategy(self, exp, diag):
        h, p = diag["h5"], diag["png"]

        if diag["missing_h5_ids"] or diag["missing_png_ids"]:
            print(f"[WARN] {exp}: filenames missing batch_# or R_#; skipping experiment.")
            return None

        if h["constant_zero"] and p["constant_zero"]:
            print(f"[WARN] {exp}: both H5 batches and PNG rounds are zero; skipping experiment.")
            return None

        # A functioning counter starts at 1 and increases strictly.
        # Gaps are allowed because acquisitions may have been interrupted.
        good_h = h["starts_at_1"] and h["strictly_increasing"]
        good_p = p["starts_at_1"] and p["strictly_increasing"]

        if good_h and good_p:
            return "direct"

        if (h["constant_zero"] and good_p) or (p["constant_zero"] and good_h):
            return "timestamp"

        print(f'[WARN] {exp}: unsupported numbering: H5={h["values"]}, PNG={p["values"]}; skipping.')
        return None

    # --------------------------- Collect matches ---------------------------

    def collect_matches(self, timestamp_dir):
        timestamp_dir = Path(timestamp_dir).resolve()
        docs = timestamp_dir / "documentation"

        if not docs.is_dir():
            raise FileNotFoundError(f"Missing documentation directory: {docs}")

        roots = [
            timestamp_dir / x / "Data_h5"
            for x in ("study_data", "optimization")
            if (timestamp_dir / x / "Data_h5").is_dir()
        ]

        if not roots:
            print(f"[WARN] No Data_h5 directory in {timestamp_dir}. Skipping this date folder.")
            return None

        index = defaultdict(list)
        for root in roots:
            for exp, entries in self.index_h5_files(root).items():
                index[exp].extend(entries)

        records = []
        self.last_report = {}

        for exp, plots in self.DATA_TO_PLOTS.items():
            hs = sorted(index.get(exp, []), key=lambda h: (h["h5_timestamp"], str(h["h5_path"])))

            if not hs:
                continue

            plot_dir = docs / plots

            if not plot_dir.is_dir():
                raise FileNotFoundError(f"{exp}: missing PNG directory: {plot_dir}")

            paths = sorted(plot_dir.rglob("*.png") if exp == "ss_ge" else plot_dir.glob("*.png"))

            if not paths:
                raise FileNotFoundError(f"{exp}: no PNGs in {plot_dir}")

            ps = []

            for path in paths:
                ts = self.extract_timestamp(path.name)
                rnd = self.extract_round_from_png(path.name)
                q = self.extract_qubit_from_path(path)

                if not ts or rnd is None or q is None:
                    print(f"[WARN] {exp}: skipping PNG without timestamp, round or qubit: {path}")
                    continue

                ps.append({
                    "path": path,
                    "ts": ts,
                    "round": rnd,
                    "qubit": q,
                })

            if not ps:
                raise FileNotFoundError(f"{exp}: no usable PNG files in {plot_dir}")

            ps.sort(key=lambda p: (p["ts"], str(p["path"])))

            diag = self.check_h5_png_numbering(
                [h["h5_path"] for h in hs],
                [p["path"] for p in ps]
            )

            strategy = self._strategy(exp, diag)

            if strategy is None:
                self.last_report[exp] = {**diag, "strategy": "skipped"}
                continue

            print(
                f'[CHECK] {exp}: H5 {diag["h5"]["values"]}, PNG {diag["png"]["values"]}, '
                f'H5 starts1={diag["h5"]["starts_at_1"]}, PNG starts1={diag["png"]["starts_at_1"]}, '
                f'H5 +1={diag["h5"]["increases_by_one"]}, PNG +1={diag["png"]["increases_by_one"]}, '
                f'H5 gaps={diag["h5"]["gaps"]}, PNG gaps={diag["png"]["gaps"]}, strategy={strategy}'
            )

            by_batch = defaultdict(list)
            for h in hs:
                by_batch[h["batch"]].append(h)

            if strategy == "direct" and any(len(v) != 1 for v in by_batch.values()):
                print(f"[WARN] {exp}: duplicate H5 batches; skipping ambiguous direct matches.")
                continue

            if strategy == "timestamp" and len({h["h5_timestamp"] for h in hs}) != len(hs):
                print(f"[WARN] {exp}: identical H5 save timestamps; skipping ambiguous timestamp matching.")
                continue

            # Group PNGs by qubit before timestamp matching.
            # Each H5 file contains measurements for multiple qubits.
            by_qubit = defaultdict(list)

            for p in ps:
                by_qubit[p["qubit"]].append(p)

            provisional = defaultdict(list)
            unmatched = []

            for q, qps in by_qubit.items():
                for p in qps:
                    if strategy == "direct":
                        candidates = by_batch.get(p["round"], [])
                        h = candidates[0] if len(candidates) == 1 else None
                    else:
                        # Match to the first H5 save occurring after the PNG measurement.
                        h = next(
                            (candidate for candidate in hs if candidate["h5_timestamp"] >= p["ts"]),
                            None
                        )

                        # Verify numbering when both identifiers are nonzero.
                        if h is not None and h["batch"] and p["round"] and h["batch"] != p["round"]:
                            h = None

                    if h is None or p["ts"] > h["h5_timestamp"]:
                        unmatched.append(p)
                    else:
                        provisional[(q, h["h5_path"])].append((p, h))

            accepted = []
            ambiguous = []
            duplicate_candidates = 0

            for (q, h5_path), group in provisional.items():
                if strategy == "timestamp":
                    nonzero = {p["round"] for p, _ in group if p["round"] != 0}

                    # Different nonzero PNG round numbers for the same H5/qubit
                    # are conflicting measurements, not duplicate plot saves.
                    if len(nonzero) > 1:
                        ambiguous.extend(p for p, _ in group)
                        continue

                    # Several PNGs with the same round ID can be repeated saves.
                    # If R_0 is stuck, we cannot prove they are duplicates, but
                    # retain them as provisional matches and warn instead of
                    # discarding all of them. Analysis can use subset[0].
                    if len(group) > 1:
                        duplicate_candidates += len(group) - 1
                        print(
                            f'[WARN] {exp} Q{q}: {len(group)} PNGs matched H5 '
                            f'{Path(h5_path).name}; retaining all as possible repeated saves'
                            + (' (PNG round counter is 0; association is timestamp-based).'
                               if all(p["round"] == 0 for p, _ in group) else '.')
                        )

                accepted.extend(group)

            # Reject conflicting associations between PNG rounds and H5 files.
            if strategy == "timestamp" and any(p["round"] != 0 for p, _ in accepted):
                round_to_h5 = defaultdict(set)
                h5_to_round = defaultdict(set)

                for p, h in accepted:
                    if p["round"]:
                        round_to_h5[p["round"]].add(h["h5_path"])
                        h5_to_round[h["h5_path"]].add(p["round"])

                bad_rounds = {r for r, files in round_to_h5.items() if len(files) > 1}
                bad_h5s = {path for path, rs in h5_to_round.items() if len(rs) > 1}

                good = []

                for p, h in accepted:
                    if p["round"] in bad_rounds or h["h5_path"] in bad_h5s:
                        ambiguous.append(p)
                    else:
                        good.append((p, h))

                accepted = good

            valid = []

            for p, h in accepted:
                # Preserve the identifier used by experiment filter_by(round=round_num).
                logical_round = p["round"] if p["round"] != 0 else h["batch"]

                valid.append({
                    "experiment": exp,
                    "round": int(logical_round),
                    "qubit": int(p["qubit"]),
                    "h5_relpath": str(h["h5_path"].relative_to(timestamp_dir)),
                    "h5_filename": h["h5_name"],
                    "h5_timestamp": h["h5_timestamp"],
                    "png_relpath": str(p["path"].relative_to(timestamp_dir)),
                    "png_filename": p["path"].name,
                    "png_timestamp": p["ts"],
                })

            # Make subset[0] deterministic: earliest PNG timestamp first.
            valid.sort(key=lambda r: (r["png_timestamp"], r["png_filename"]))
            records.extend(valid)

            self.last_report[exp] = {
                **diag,
                "strategy": strategy,
                "matched": len(valid),
                "unmatched": len(unmatched),
                "ambiguous": len(ambiguous),
                "possible_duplicate_pngs": duplicate_candidates,
            }

            print(
                f"[SUMMARY] {exp}: H5={len(hs)}, PNG={len(paths)}, "
                f"matched={len(valid)}, unmatched={len(unmatched)}, ambiguous={len(ambiguous)}, "
                f"possible_duplicate_pngs={duplicate_candidates}"
            )

            for p in unmatched:
                print(f'[WARN] {exp} Q{p["qubit"]}: unmatched PNG: {p["path"].name}')

            for p in ambiguous:
                print(f'[WARN] {exp} Q{p["qubit"]}: ambiguous PNG: {p["path"].name}')

        if not records:
            print(f"[WARN] No valid PNG/H5 associations in {timestamp_dir}; saving an empty map.")

        return records

    # --------------------------- Save mapping ---------------------------

    def save_to_h5(self, output_path, source_dir, records):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        str_dt = h5py.string_dtype("utf-8")

        dtype = np.dtype([
            ("experiment", str_dt),
            ("round", np.int32),
            ("qubit", np.int32),
            ("h5_relpath", str_dt),
            ("h5_filename", str_dt),
            ("h5_timestamp", str_dt),
            ("png_relpath", str_dt),
            ("png_filename", str_dt),
            ("png_timestamp", str_dt),
        ])

        arr = np.zeros(len(records), dtype=dtype)

        for i, rec in enumerate(records):
            arr[i] = tuple(rec[name] for name in dtype.names)

        with h5py.File(output_path, "w") as f:
            dset = f.create_dataset("file_matches", data=arr)
            dset.attrs["description"] = "H5/PNG mapping; round is the functional filename identifier."
            f.attrs["source_timestamp_dir"] = str(source_dir)

        print(f"[INFO] Wrote {len(records)} records to {output_path}")

    # --------------------------- Command line ---------------------------

    def run(self):
        parser = argparse.ArgumentParser(description="Build an H5-to-PNG timestamp mapping.")
        parser.add_argument("timestamp_dir")
        parser.add_argument("--output", "-o", default=None)
        args = parser.parse_args()

        ts_dir = Path(args.timestamp_dir).resolve()
        out = Path(args.output).resolve() if args.output else ts_dir / "documentation" / "h5_png_timestamp_map.h5"

        records = self.collect_matches(ts_dir)
        self.save_to_h5(out, ts_dir, records)


class load_h5_png_map:
    """
    Utility functions for querying the H5-PNG mapping.

    load_map() always rebuilds the mapping from its source files.
    """

    # --------------------------- Text conversion helper ---------------------------

    def _as_str(self, value):
        """Convert HDF5 bytes or Python strings into a regular string."""
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    # --------------------------- Load mapping ---------------------------

    def load_map(self, path):
        path = Path(path).resolve()

        if path.parent.name != "documentation":
            raise ValueError(f"Cannot infer source date folder from mapping path: {path}")

        source_dir = path.parent.parent

        # Always recreate mapping, even if it already exists.
        mapper = create_h5_png_map()
        records = mapper.collect_matches(source_dir)

        if records is None:
            return None  # Skip this date folder; do not save or load an old mapping

        mapper.save_to_h5(path, source_dir, records)

        with h5py.File(path, "r") as f:
            return f["file_matches"][:]

    # --------------------------- Experiment name helpers ---------------------------

    def _experiment_name(self, experiment):
        aliases = {
            "t1": "t1_ge",
            "t2": "t2_ge",
            "t2e": "t2e_ge",
            "qspec": "qspec_ge",
            "res": "res_ge",
            "rabi": "rabi_ge",
            "ss": "ss_ge",
        }
        return aliases.get(self._as_str(experiment), self._as_str(experiment))

    # --------------------------- Filtering helpers ---------------------------

    def filter_by_qubit(self, data, qubit):
        return data[data["qubit"] == qubit]

    def filter_by_experiment(self, data, experiment):
        exp_name = self._experiment_name(experiment)
        return data[np.array([self._as_str(v) == exp_name for v in data["experiment"]])]

    def filter_by(self, data, experiment=None, qubit=None, round=None):
        mask = np.ones(len(data), dtype=bool)

        if experiment is not None:
            exp_name = self._experiment_name(experiment)
            mask &= np.array([self._as_str(v) == exp_name for v in data["experiment"]])

        if qubit is not None:
            mask &= data["qubit"] == qubit

        if round is not None:
            mask &= data["round"] == round

        return data[mask]

    # --------------------------- Extract paths / filenames / timestamps ---------------------------

    def get_png_paths(self, data):
        return [self._as_str(row["png_relpath"]) for row in data]

    def get_h5_paths(self, data):
        return sorted({self._as_str(row["h5_relpath"]) for row in data})

    def get_png_filenames(self, data):
        return [self._as_str(row["png_filename"]) for row in data]

    def get_h5_filenames(self, data):
        return sorted({self._as_str(row["h5_filename"]) for row in data})

    def get_png_timestamps(self, data):
        return [self._as_str(row["png_timestamp"]) for row in data]

    def get_h5_timestamps(self, data):
        return sorted({self._as_str(row["h5_timestamp"]) for row in data})

    # --------------------------- Summary ---------------------------

    def summary(self, data):
        qubits = sorted({int(q) for q in data["qubit"]})
        exps = sorted({self._as_str(row) for row in data["experiment"]})
        rounds = sorted({int(r) for r in data["round"]})

        print("Experiments:", exps)
        print("Qubits:", qubits)
        print("Rounds:", rounds)
        print("Total entries:", len(data))


if __name__ == "__main__":
    create_h5_png_map().run()
