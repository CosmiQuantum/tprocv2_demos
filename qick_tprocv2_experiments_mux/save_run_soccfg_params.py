from tprocv2_demos.qick_tprocv2_experiments_mux.socProxy import makeProxy
import datetime
import os
import io
import sys
import re, numpy as np, h5py

"""
This script has two main jobs. The first part is set up to dump QICK soccfg hardware/readout configuration 
info to a text file, and the second part uses that dump plus an HDF5 experiment file to compute readout lengths 
in clock cycles for each qubit.
"""
# ============================================================
# FLAGS
RUN_SOCCFG_DUMP = False
RUN_RO_LENGTH_FROM_H5 = True
# ============================================================

if RUN_SOCCFG_DUMP:
    # --- Set output path ---
    save_dir = "/data/QICK_data/run8/6transmon/run8_soccfg_params"   # <--- change to where you want it
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f"soccfg_full_dump_{timestamp}_firmware_during_run8_updated.txt")

    # --- Connect to QICK ---
    soc, soccfg = makeProxy()

    # --- Capture full soccfg print output ---
    buf = io.StringIO()
    _stdout = sys.stdout
    sys.stdout = buf
    try:
        print(soccfg)
    finally:
        sys.stdout = _stdout
    soccfg_text = buf.getvalue()

    lines = []
    lines.append(f"QICK soccfg Full Configuration Dump for {timestamp}\n")
    lines.append("=" * 100 + "\n\n")
    lines.append("[FULL PRINT OUTPUT OF SOCCFG]\n\n")
    lines.append(soccfg_text)
    lines.append("\n" + "=" * 100 + "\n\n")
    lines.append("[READOUT CHANNEL DIAGNOSTICS]\n")

    # --- Extract readout info ---
    readouts = getattr(soccfg, "readouts", None)
    if readouts is None:
        try:
            readouts = soccfg["readouts"]
        except Exception:
            readouts = None

    if isinstance(readouts, dict):
        iterable = readouts.items()
    elif isinstance(readouts, (list, tuple)):
        iterable = enumerate(readouts)
    else:
        iterable = []

    # --- Main per-channel extraction ---
    for ro_ch, rocfg in iterable:
        try:
            iq_offset_raw = float(rocfg.get('iq_offset', 0.0))
            b_dds = rocfg.get('b_dds', None)
            ro_regs = soccfg.calc_ro_regs(rocfg, phase=0, sel='product')
            pfb_ch = ro_regs.get('pfb_ch')
            f_int  = ro_regs.get('f_int')

            offset_doubled = False
            if b_dds is not None and pfb_ch is not None and f_int is not None:
                fs_int = 2 ** b_dds
                if f_int == (pfb_ch % 2) * (fs_int // 2):
                    offset_doubled = True

            iq_offset_effective = iq_offset_raw * (2.0 if offset_doubled else 1.0)

            # --- Compute exact decimated MHz ---
            us_per_cycle = soccfg.cycles2us(ro_ch=ro_ch, cycles=1)
            decimated_MHz_exact = 1.0 / us_per_cycle  # MHz = 1 / (us per cycle)

            # --- Write diagnostics ---
            lines.append(f"\n--- Readout Channel {ro_ch} ---\n")
            lines.append(f"decimated_MHz_exact:  {decimated_MHz_exact:.8f}\n")
            lines.append(f"iq_offset_raw:        {iq_offset_raw}\n")
            lines.append(f"pfb_ch:               {pfb_ch}\n")
            lines.append(f"f_int:                {f_int}\n")
            lines.append(f"b_dds:                {b_dds}\n")
            lines.append(f"offset_doubled:       {offset_doubled}\n")
            lines.append(f"iq_offset_effective:  {iq_offset_effective}\n")

        except Exception as e:
            lines.append(f"\n[!] Failed to extract info for RO channel {ro_ch}: {e}\n")

    lines.append("\n" + "=" * 100 + "\n")

    # --- Save to text file ---
    with open(save_path, "w") as f:
        f.writelines(lines)

    print(f"\nFull soccfg and diagnostics saved to:\n{save_path}")

#############################################################################################################################
if RUN_RO_LENGTH_FROM_H5:
    # --- user path ---
    # where to dump the text file
    dump_path  = "/data/QICK_data/run8/6transmon/run8_soccfg_params/soccfg_full_dump_2025-11-10_15-14-35_firmware_during_run8_updated.txt"
    h5_path   = "/data/QICK_data/run8/6transmon/round_robin/AB_paper_datadump_T1_Analysis/2025-10-27_22-04-57/study_data/Data_h5/t1_ge/2025-10-27_22-46-49_t1_ge_results_batch_1_Num_per_batch1.h5"
    # --- helpers ---
    def safe_eval_cfg(cfg_str):
        s = re.sub(r"<qick\.asm_v2\.QickParam object at 0x[0-9a-fA-F]+>", "None", cfg_str)
        s = re.sub(r"np\.float64\(\s*([^)]+)\s*\)", r"float(\1)", s)
        s = s.replace("array(", "np.array(")
        import numpy as np
        return eval(s, {"np": np, "array": np.array, "float": float, "__builtins__": {}})

    def parse_dump(txt):
        info = {}
        for m in re.finditer(r"--- Readout Channel\s+(\d+)\s+---([\s\S]*?)(?=--- Readout Channel|\Z)", txt):
            ch = int(m.group(1))
            body = m.group(2)
            dec = re.search(r"decimated_MHz_exact:\s*([0-9.]+)", body)
            off = re.search(r"iq_offset_effective:\s*([\-0-9.]+)", body)
            info[ch] = {
                "decimated_MHz_exact": float(dec.group(1)) if dec else None,
                "iq_offset_effective": float(off.group(1)) if off else None
            }
        return info

    # --- read soccfg dump ---
    with open(dump_path, "r") as f:
        dump_info = parse_dump(f.read())

    # --- read H5 + print lengths ---
    ro_lengths = []
    with h5py.File(h5_path, "r") as f:
        for q_key in f.keys():
            if not q_key.startswith("Q"):
                continue
            qidx = int(q_key[1:]) - 1
            syst_cfg_str = f[f"{q_key}/Syst Config"][0].decode()
            syst_cfg = safe_eval_cfg(syst_cfg_str)

            ro_field = syst_cfg["ro_ch"]
            if isinstance(ro_field, (list, tuple, np.ndarray)):
                ro_ch = int(ro_field[qidx])
            else:
                ro_ch = int(ro_field)

            res_len_field = (syst_cfg.get("res_length") or syst_cfg.get("res_len") or syst_cfg.get("read_length"))
            if isinstance(res_len_field, (list, tuple, np.ndarray)):
                res_us = float(res_len_field[qidx])
            else:
                res_us = float(res_len_field)

            d = dump_info.get(ro_ch, {})
            dec = float(d.get("decimated_MHz_exact", 307.2))
            ro_cycles = int(np.trunc(res_us * dec))
            ro_lengths.append(ro_cycles)

            print(f"Q{qidx+1}: ro_ch={ro_ch} | res_len={res_us:.4f} µs | decimated={dec:.3f} MHz | ro_cycles={ro_cycles}")

    print("\nro_length =", ro_lengths)

if not RUN_SOCCFG_DUMP and not RUN_RO_LENGTH_FROM_H5:
    print("No flags are enabled.")