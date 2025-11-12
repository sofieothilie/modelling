import os
import numpy as np
from seispy import Seismic
from datetime import datetime

# header format, example:
# trans_id=[216 217 218 ... 244 245 246]
# n_traces=186930
# n_samples=768
# dt=8.334999999999999e-07
# dtype=float32
# endianness=little
# trace_file=aligned_Baseline1_mig_IF1__20D_traces.bin
# notes=Exported from aligned_Baseline1_mig_IF1__20D.pkl on 2025-11-12T09:53:38.338636Z



# ----- user-editable -----
paths = [
    r".\data\new\aligned_Monitor1S1_mig_IF1__20D.pkl",
    r".\data\new\aligned_Baseline1_mig_IF1__20D.pkl",
    r".\data\new\150kHz_Single_source_file.pkl"
]
out_dir = r".\data\extracted"

# threshold above which traces are streamed row-by-row to avoid extra big memory copies
STREAMING_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MB
# -------------------------

os.makedirs(out_dir, exist_ok=True)

def is_regular(samples, rel_tol=1e-6, abs_tol=1e-12):
    samples = np.asarray(samples, dtype=float)
    if samples.size < 2:
        return True, 0.0
    dt = np.diff(samples)
    dt_mean = float(dt.mean())
    max_abs = float(np.max(np.abs(dt - dt_mean)))
    max_rel = max_abs / (abs(dt_mean) + 1e-30)
    regular = (max_abs <= abs_tol) or (max_rel <= rel_tol)
    return regular, dt_mean

def write_header_txt(hdr_path, meta):
    with open(hdr_path, "w", encoding="utf8") as fh:
        fh.write(f"trans_id={meta['trans_id']}\n")
        fh.write(f"n_traces={meta['n_traces']}\n")
        fh.write(f"n_samples={meta['n_samples']}\n")
        fh.write(f"dt={meta['dt']}\n")
        fh.write(f"dtype={meta['dtype']}\n")
        fh.write(f"endianness={meta['endianness']}\n")
        fh.write(f"trace_file={meta['trace_file']}\n")
        fh.write(f"notes=Exported from {meta['source_pkl']} on {meta['export_time']}\n")

# normalise input paths for portability
paths = [os.path.normpath(p) for p in paths]
out_dir = os.path.normpath(out_dir)

for p in paths:
    base = os.path.basename(p)
    name = os.path.splitext(base)[0]
    print("="*60)
    print("Processing:", p)
    seis = Seismic()
    try:
        seis.load_seis_obj(p)
    except Exception as e:
        print("  ERROR loading file:", e)
        continue

    traces = getattr(seis, "traces", None)

    if traces is None:
        print("  No suitable amplitude array found in object. Skipping.")
        continue

    traces = np.asarray(traces)
    if traces.ndim != 2:
        print(f"  Found traces but it's not 2D (shape={traces.shape}). Skipping.")
        continue

    header = getattr(seis, "header", {}) or {}
    samples = np.asarray(header.get("samples", []), dtype=float)

    # detect regularity and prepare sample handling
    samples_dtype = ""
    dt = 0.0

    if samples.size >= 2:
        regular, dt_mean = is_regular(samples, rel_tol=1e-6, abs_tol=1e-12)
        if regular:
            dt = dt_mean
            print(f"  samples are regular: dt={dt:.17g}")
        else:
            # write samples.bin as float64 for safety and record dtype
            samples_file = os.path.join(out_dir, f"{name}_samples.bin")
            samples.astype(np.float64).tofile(samples_file)
            samples_dtype = "float64"
            print(f"  samples are irregular -> wrote samples to {samples_file} (float64)")
    else:
        dt = float(header.get("dt", 0.0))
        print(f"  No samples array found; using dt={dt:.17g}")

    # output paths
    traces_bin = os.path.join(out_dir, f"{name}_traces.bin")
    header_txt = os.path.join(out_dir, f"{name}_header.txt")

    # little-endian float32 for traces 
    # If the array is too large: stream row-by-row
    total_bytes = traces.nbytes # current memory usage
    print(f"  traces currently use ~{total_bytes / (1024*1024):.1f} MB memory")
    n_traces, n_samples = traces.shape
    print(traces.astype('<f4'))

    try:
        if total_bytes <= STREAMING_THRESHOLD_BYTES:
            # small enough: do single tofile with explicit little-endian float32
            traces.astype('<f4').tofile(traces_bin)
        else:
            # stream per-row to avoid extra big temporary allocations
            print("  Large traces detected — streaming rows to file to avoid extra memory copies.")
            # open file and write row-by-row as little-endian float32
            with open(traces_bin, "wb") as fh:
                for i in range(n_traces):
                    row = np.asarray(traces[i, :], dtype='<f4')  # little-endian float32
                    fh.write(row.tobytes())
        print(f"  Saved binary traces to {traces_bin}")
    except Exception as e:
        print("  ERROR writing traces bin:", e)
        continue

    # build header meta
    meta = {
        "trans_id": header.get("trans_id", None),
        "n_traces": int(n_traces),
        "n_samples": int(n_samples),
        "dt": float(dt),
        "dtype": "float32",
        "endianness": "little",
        "samples_dtype": samples_dtype,
        "trace_file": os.path.basename(traces_bin),
        "source_pkl": os.path.basename(p),
        "export_time": datetime.now().isoformat() + "Z"
    }

    # write header
    write_header_txt(header_txt, meta)
    print(f"  Wrote header to {header_txt}")

    # check file size and small sample equality
    try:
        data = np.fromfile(traces_bin, dtype=np.float32)
        expected_count = int(n_traces) * int(n_samples)
        if data.size != expected_count:
            raise RuntimeError(f"traces.bin size mismatch: got {data.size}, expected {expected_count}")
        # reshape into (n_traces, n_samples) in memory for quick checks (only if it fits)
        check_n = min(10, n_samples)
        # compare first trace first check_n samples
        first_samples = data[:check_n]
        orig_first = traces[0, :check_n].astype(np.float32)
        if not np.allclose(first_samples, orig_first, atol=1e-6, rtol=1e-6):
            print("  WARNING: verification mismatch for first samples of trace 0.")
            print("   wrote:", first_samples.tolist())
            print("   orig :", orig_first.tolist())
        else:
            print("  Verification ok: traces.bin size and first-sample checks passed.")
    except Exception as e:
        print("  Verification step failed:", e)

    print("  DONE for", name)
