from tprocv2_demos.qick_tprocv2_experiments_mux.socProxy import makeProxy
import datetime
import os
import io
import sys
import numpy as np

# --- Set output path ---
save_dir = "/data/QICK_data/run8/6transmon/run8_soccfg_params"   # <--- change to where you want it
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = os.path.join(save_dir, f"soccfg_full_dump_{timestamp}_firmware_during_run8_updated.txt")

# # --- Connect to QICK ---
# soc, soccfg = makeProxy()
#
# # --- Capture full soccfg print output ---
# buf = io.StringIO()
# _stdout = sys.stdout
# sys.stdout = buf
# try:
#     print(soccfg)
# finally:
#     sys.stdout = _stdout
# soccfg_text = buf.getvalue()
#
# lines = []
# lines.append(f"QICK soccfg Full Configuration Dump for {timestamp}\n")
# lines.append("=" * 100 + "\n\n")
# lines.append("[FULL PRINT OUTPUT OF SOCCFG]\n\n")
# lines.append(soccfg_text)
# lines.append("\n" + "=" * 100 + "\n\n")
# lines.append("[READOUT CHANNEL DIAGNOSTICS]\n")
#
# # --- Extract readout info ---
# readouts = getattr(soccfg, "readouts", None)
# if readouts is None:
#     try:
#         readouts = soccfg["readouts"]
#     except Exception:
#         readouts = None
#
# if isinstance(readouts, dict):
#     iterable = readouts.items()
# elif isinstance(readouts, (list, tuple)):
#     iterable = enumerate(readouts)
# else:
#     iterable = []
#
# # --- Main per-channel extraction ---
# for ro_ch, rocfg in iterable:
#     try:
#         iq_offset_raw = float(rocfg.get('iq_offset', 0.0))
#         b_dds = rocfg.get('b_dds', None)
#         ro_regs = soccfg.calc_ro_regs(rocfg, phase=0, sel='product')
#         pfb_ch = ro_regs.get('pfb_ch')
#         f_int  = ro_regs.get('f_int')
#
#         offset_doubled = False
#         if b_dds is not None and pfb_ch is not None and f_int is not None:
#             fs_int = 2 ** b_dds
#             if f_int == (pfb_ch % 2) * (fs_int // 2):
#                 offset_doubled = True
#
#         iq_offset_effective = iq_offset_raw * (2.0 if offset_doubled else 1.0)
#
#         # --- Compute exact decimated MHz ---
#         us_per_cycle = soccfg.cycles2us(ro_ch=ro_ch, cycles=1)
#         decimated_MHz_exact = 1.0 / us_per_cycle  # MHz = 1 / (us per cycle)
#
#         # --- Write diagnostics ---
#         lines.append(f"\n--- Readout Channel {ro_ch} ---\n")
#         lines.append(f"decimated_MHz_exact:  {decimated_MHz_exact:.8f}\n")
#         lines.append(f"iq_offset_raw:        {iq_offset_raw}\n")
#         lines.append(f"pfb_ch:               {pfb_ch}\n")
#         lines.append(f"f_int:                {f_int}\n")
#         lines.append(f"b_dds:                {b_dds}\n")
#         lines.append(f"offset_doubled:       {offset_doubled}\n")
#         lines.append(f"iq_offset_effective:  {iq_offset_effective}\n")
#
#     except Exception as e:
#         lines.append(f"\n[!] Failed to extract info for RO channel {ro_ch}: {e}\n")
#
# lines.append("\n" + "=" * 100 + "\n")
#
# # --- Save to text file ---
# with open(save_path, "w") as f:
#     f.writelines(lines)
#
# print(f"\nFull soccfg and diagnostics saved to:\n{save_path}")

#############################################################################################################################
import re, numpy as np, h5py, io, ast

# ---------- helpers ----------
def safe_eval_cfg(cfg_str: str):
    s = re.sub(r"<qick\.asm_v2\.QickParam object at 0x[0-9a-fA-F]+>", "None", cfg_str)
    s = re.sub(r"np\.float64\(\s*([^)]+)\s*\)", r"float(\1)", s)
    s = s.replace("array(", "np.array(")
    return eval(s, {"np": np, "array": np.array, "float": float, "__builtins__": {}})

def parse_dump(dump_text: str):
    info = {}
    for m in re.finditer(r"^\s*(\d+):\s*axis_.*readout.*?decimated\s*=\s*([0-9.]+)\s*MHz",
                         dump_text, flags=re.MULTILINE):
        ch = int(m.group(1)); dec = float(m.group(2))
        info.setdefault(ch, {})["decimated_MHz"] = dec
    for m in re.finditer(r"decimated_MHz_exact\s*=\s*([0-9.]+)\s*MHz\s*\(ro_ch\s*=\s*(\d+)\)",
                         dump_text, flags=re.MULTILINE):
        dec = float(m.group(1)); ch = int(m.group(2))
        info.setdefault(ch, {})["decimated_MHz_exact"] = dec
    m = re.search(r"\[READOUT CHANNEL DIAGNOSTICS\](.*)$", dump_text, flags=re.S)
    if m:
        diag = m.group(1)
        for mm in re.finditer(r"--- Readout Channel\s+(\d+)\s+---\s*([\s\S]*?)(?=(?:--- Readout Channel|\Z))", diag):
            ch = int(mm.group(1)); body = mm.group(2)
            m2 = re.search(r"iq_offset_effective:\s*([\-0-9.]+)", body)
            if m2:
                eff = float(m2.group(1))
                info.setdefault(ch, {})["iq_offset_effective"] = eff
    return info

def ro_cycles_from(syst_cfg: dict, dump_info: dict, q_index: int):
    ro_field = syst_cfg["ro_ch"]
    ro_ch = int(ro_field[q_index] if isinstance(ro_field, (list, tuple)) else ro_field)
    res_len_field = (syst_cfg.get('res_length') or syst_cfg.get('res_len') or syst_cfg.get('read_length'))
    if res_len_field is None:
        raise KeyError("system config missing res_length/read_length")
    res_us = float(res_len_field[q_index] if isinstance(res_len_field, (list, tuple)) else res_len_field)
    d = dump_info.get(ro_ch, {})
    dec = float(d.get("decimated_MHz_exact", d.get("decimated_MHz", 307.2)))
    cycles = int(np.trunc(res_us * dec))
    iq_off = float(d.get("iq_offset_effective", 0.0))
    return ro_ch, res_us, dec, cycles, iq_off

# ---- robust HDF5 loaders (works for numeric arrays OR byte-string dumps) ----
def _decode_to_text(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        return x
    # Sometimes datasets are object arrays with a single bytes element
    try:
        # h5py scalar read
        if hasattr(x, "dtype") and x.dtype.kind == "O":
            return _decode_to_text(x[()])
    except Exception:
        pass
    return None

def _coerce_text(val):
    """Return a Python str if val is a bytes/str or a text-like ndarray; else None."""
    # plain bytes/str
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="ignore")
    if isinstance(val, str):
        return val
    # ndarray that stores text (bytes, unicode, or objects)
    if isinstance(val, np.ndarray) and val.dtype.kind in {"S", "U", "O"}:
        # unwrap a single element if that's how it was saved
        if val.ndim == 0:
            elem = val.item()
        elif val.size == 1:
            elem = val.reshape(-1)[0]
        else:
            # join multiple parts with whitespace (rare in your files)
            elems = [e.decode("utf-8", "ignore") if isinstance(e, (bytes, bytearray, np.bytes_)) else str(e)
                     for e in val.reshape(-1)]
            return " ".join(elems)
        return elem.decode("utf-8", "ignore") if isinstance(elem, (bytes, bytearray, np.bytes_)) else str(elem)
    return None

def load_numeric_vector(ds_or_val):
    """Return a 1D float array from a dataset or pre-read value (bytes/str/array)."""
    # read dataset if needed
    val = ds_or_val[()] if isinstance(ds_or_val, h5py.Dataset) else ds_or_val

    # fast path: already numeric ndarray
    if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
        return np.asarray(val, dtype=float).ravel()

    # try to coerce to text and parse numbers
    txt = _coerce_text(val)
    if txt is not None:
        s = txt.strip()
        # common cases: "[1 2 3]" or "1 2 3"
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        arr = np.fromstring(s, sep=" ")
        if arr.size > 0:
            return arr.astype(float).ravel()
        # fallback to literal_eval for comma-separated lists
        import ast
        return np.asarray(ast.literal_eval(txt), dtype=float).ravel()

    # last resort: try numeric cast
    return np.asarray(val, dtype=float).ravel()

def load_numeric_array(ds_or_val):
    """Return a float array from dataset or pre-read value (any shape)."""
    val = ds_or_val[()] if isinstance(ds_or_val, h5py.Dataset) else ds_or_val

    if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
        return np.asarray(val, dtype=float)

    txt = _coerce_text(val)
    if txt is not None:
        import ast
        try:
            return np.asarray(ast.literal_eval(txt), dtype=float)
        except Exception:
            s = txt.strip().strip("[]")
            return np.fromstring(s, sep=" ").astype(float)

    return np.asarray(val, dtype=float)

def replay_qick_math_from_shots(Ishots, Qshots, ro_length, iq_offset, reps, rounds):
    def ensure_rNr(A):
        A = np.asarray(A)
        if A.ndim == 3:
            return A
        if A.ndim == 2:
            N, R = A.shape
            return A.reshape(1, N, R)
        raise ValueError(f"Unexpected shape {A.shape}; need (rounds,N,reps) or (N,reps)")
    I3 = ensure_rNr(Ishots)
    Q3 = ensure_rNr(Qshots)

    # average over reps
    I_avg = I3.mean(axis=2)
    Q_avg = Q3.mean(axis=2)
    # divide by RO length and subtract effective offset
    I_corr = I_avg / float(ro_length) - iq_offset
    Q_corr = Q_avg / float(ro_length) - iq_offset
    # average over rounds
    I_final = I_corr.mean(axis=0)
    Q_final = Q_corr.mean(axis=0)
    return I_final, Q_final

# ---------- USER PATHS ----------
dump_path = "/data/QICK_data/run8/6transmon/run8_soccfg_params/soccfg_full_dump_2025-11-10_15-14-35_firmware_during_run8_updated.txt"
h5_path   = "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_T1_Analysis/2025-10-24_13-58-37/study_data/Data_h5/t1_ge/2025-10-24_14-31-13_t1_ge_results_batch_1_Num_per_batch1.h5"

# ---------- run ----------
with open(dump_path, "r") as f:
    dump_text = f.read()
dump_info = parse_dump(dump_text)

with h5py.File(h5_path, "r") as f:
    for q_key in f.keys():
        if not q_key.startswith("Q"):
            continue
        q_index = int(q_key[1:]) - 1

        syst_cfg_str = f[f"{q_key}/Syst Config"][0].decode()
        exp_cfg_str  = f[f"{q_key}/Exp Config"][0].decode()
        syst_cfg = safe_eval_cfg(syst_cfg_str)
        exp_cfg  = safe_eval_cfg(exp_cfg_str)
        T1 = exp_cfg["T1_ge"]
        steps = int(T1.get("steps") or T1.get("n_steps") or T1.get("n_expts"))
        reps  = int(T1.get("reps")  or T1.get("nreps"))
        rounds= int(T1.get("rounds") or T1.get("soft_avgs"))

        ro_ch, res_us, dec, cycles, iq_off = ro_cycles_from(syst_cfg, dump_info, q_index)

        # --- load datasets robustly (NO [0] here) ---
        I_online = load_numeric_vector(f[f"{q_key}/I"]).reshape(-1)[:steps]
        Q_online = load_numeric_vector(f[f"{q_key}/Q"]).reshape(-1)[:steps]

        Ishots = load_numeric_array(f[f"{q_key}/Ishots"])
        Qshots = load_numeric_array(f[f"{q_key}/Qshots"])

        # Coerce to (rounds, steps, reps) if needed
        Ishots = np.asarray(Ishots)
        Qshots = np.asarray(Qshots)
        if Ishots.ndim == 2:  # (steps, reps) -> add 1 round
            Ishots = Ishots[None, ...]
            Qshots = Qshots[None, ...]
        # trim/reshape steps if theres any mismatch
        Ishots = Ishots[:, :steps, :reps]
        Qshots = Qshots[:, :steps, :reps]

        # --- offline replay ---
        I_off, Q_off = replay_qick_math_from_shots(Ishots, Qshots,
                                                   ro_length=cycles,
                                                   iq_offset=iq_off,
                                                   reps=reps, rounds=rounds)

        # --- diagnostics ---
        print(f"\n[{q_key}] ro_ch={ro_ch}, res={res_us:.4f} us, dec={dec:.6f} MHz, length={cycles}, iq_off={iq_off:.6g}")
        print("   median |I_online - I_off|:", float(np.median(np.abs(I_online - I_off))))
        print("   median |Q_online - Q_off|:", float(np.median(np.abs(Q_online - Q_off))))
        # Optional quick sanity scale:
        eps = 1e-15
        rawI_mean = Ishots.mean(axis=2).mean(axis=0)
        rawQ_mean = Qshots.mean(axis=2).mean(axis=0)
        scale_I = float(np.median(I_online / (rawI_mean / float(cycles) - iq_off + eps)))
        scale_Q = float(np.median(Q_online / (rawQ_mean / float(cycles) - iq_off + eps)))
        print("   sanity scale (online / (raw-mean/len - off))  I", scale_I, " Q", scale_Q)

print("\nDone.")