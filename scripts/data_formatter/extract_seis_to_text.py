import os
import struct
import numpy as np
from seispy import Seismic

# contains:
# n_traces: uint32
# n_samples: uint32
# dt: float32
# for each trace:
#   sx: float32
#   sy: float32
#   samples: float32[n_samples]
# format:
# | n_traces | n_samples | dt | sx_1 | sy_1 | samples_1... | sx_2 | sy_2 | samples_2... | ... | sx_n | sy_n | samples_n... |


"""
ImHex pattern for the output binary files:
#pragma endian little

u32 n_traces out;
u32 n_samples out;

struct File { // Total: 54 bytes
 u32 n_traces; // Magic identifier: 0x4d42
 u32 n_samples;
 float dt;

};


File file @ 0x00;
n_traces = file.n_traces;
n_samples = file.n_samples;

struct JPGHeader {
    u16 sof0;
    u16 segmentlength;
    u8 bitsprpixel;
    u16 height;
    u16 width;
    u8 n_components;
};

struct trace {
    float sx;
    float sy;
    float samples[n_samples];
};

trace traces[file.n_traces] @ addressof(file) + sizeof(File);
"""

# ----- user-editable -----
paths = [
    r"./data/new/aligned_Monitor1S1_mig_IF1__20D.pkl",
    r"./data/new/aligned_Baseline1_mig_IF1__20D.pkl",
    r"./data/new/150kHz_Single_source_file.pkl"
]


out_dir = r"./data/extracted"

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

def write_single_file(path, sx_arr, sy_arr, dt, traces):
    # traces: (n_traces, n_samples) float32
    traces = np.asarray(traces, dtype='<f4')
    n_traces, n_samples = traces.shape

    # prepare header
    header = struct.pack('<II f', n_traces, n_samples, float(dt))  # uint32, uint32, float32

    with open(path, 'wb') as f:
        f.write(header)
        # write records: [sx, sy, sample...]
        for i in range(n_traces):
            meta = np.array([float(sx_arr[i]), float(sy_arr[i])], dtype='<f4')
            f.write(meta.tobytes())
            f.write(traces[i].tobytes())

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
    dt = 0.0

    if samples.size >= 2:
        regular, dt_mean = is_regular(samples, rel_tol=1e-6, abs_tol=1e-12)
        if regular:
            dt = dt_mean
            print(f"  samples are regular: dt={dt:.17g}")
        else:
            print(f"  samples are irregular")
            exit()
    else:
        print(f"  No samples array found")
        exit()

    # output paths
    traces_bin = os.path.join(out_dir, f"{name}_traces.bin")

    # If the array is too large: stream row-by-row
    total_bytes = traces.nbytes # current memory usage
    print(f"  traces currently use ~{total_bytes / (1024*1024):.1f} MB memory")
    n_traces, n_samples = traces.shape

    if (n_traces != len(header.get("sourceX", []))):
        print(f"  WARNING: n_traces ({n_traces}) does not match header sourceX length ({len(header.get('sourceX', []))}). Exiting.")
        exit()

    write_single_file(traces_bin, header.get("sourceX", []), header.get("sourceY", []), dt, traces)
    
    print("  DONE for", name)

