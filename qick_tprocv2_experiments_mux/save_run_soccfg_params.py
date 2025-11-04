from tprocv2_demos.qick_tprocv2_experiments_mux.socProxy import makeProxy
import datetime
import os
import io
import sys

# --- Set output path ---
save_dir = "/data/QICK_data/run6/6transmon/loud2_soccfg_params"   # <--- change to where you want it
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = os.path.join(save_dir, f"soccfg_full_dump_{timestamp}_firmware_during_run6.txt")

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

        lines.append(f"\n--- Readout Channel {ro_ch} ---\n")
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