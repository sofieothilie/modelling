import os
import struct
import numpy as np
from seispy import Seismic

"""
ImHex pattern for the output meta binary files:
#pragma endian little

// fixed path field size used by the generator
const u32 META_PATH_FIELD_SIZE = 256u;

struct MetaHeader {
u32 n_samples;
u32 n_traces;
float dt;
char data_path[META_PATH_FIELD_SIZE]; // null-padded UTF-8
u32 n_sources;
};

MetaHeader meta @ 0x00;

struct Source {
float sx;
float sy;
u32 n_receivers;
float gx[n_receivers]; // receiver x positions
float gy[n_receivers]; // receiver y positions
};


// sources array starts right after the header
Source sources[meta.n_sources] @ addressof(meta) + sizeof(MetaHeader);



ImHex pattern for the output trace binary files:

#pragma endian little
#pragma array_limit 186932 // set based on values from metadata


u32 n_traces = 186930; // set based on values from metadata
u32 n_samples = 768; // set based on values from metadata
 
struct Trace {
 float values[n_samples];
};

Trace traces[n_traces] @ 0x000;

"""

paths = [
    r"./data/new/aligned_Monitor1S1_mig_IF1__20D.pkl",
    r"./data/new/aligned_Baseline1_mig_IF1__20D.pkl",
    r"./data/new/150kHz_Single_source_file.pkl"
]

out_dir = r"./data/extracted"

# threshold above which traces are streamed row-by-row to avoid extra big memory copies
STREAMING_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MB

# size in bytes of the path field written in the binary meta file (fixed-size)
META_PATH_FIELD_SIZE = 256

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

def write_data_file(path, traces, dt):
    # traces: (n_traces, n_samples) float32
    traces = np.asarray(traces, dtype='<f4')
    n_traces, n_samples = traces.shape

    # header: n_traces (u32), n_samples (u32), dt (f32)
    header = struct.pack('<II f', n_traces, n_samples, float(dt))
    with open(path, 'wb') as f:
        f.write(header)
        for i in range(n_traces):
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

    # If the array is too large: stream row-by-row
    total_bytes = traces.nbytes # current memory usage
    print(f"  traces currently use ~{total_bytes / (1024*1024):.1f} MB memory")
    n_traces, n_samples = traces.shape

    if (n_traces != len(header.get("sourceX", []))):
        print(f"  WARNING: n_traces ({n_traces}) does not match header sourceX length ({len(header.get('sourceX', []))}). Exiting.")
        exit()

    # group traces by unique source positions (sx, sy)
    sx_arr = np.asarray(header.get("sourceX", []), dtype=float)
    sy_arr = np.asarray(header.get("sourceY", []), dtype=float)
    gx_arr = np.asarray(header.get("groupX", []), dtype=float)
    gy_arr = np.asarray(header.get("groupY", []), dtype=float)

    if len(sx_arr) != n_traces or len(sy_arr) != n_traces or len(gx_arr) != n_traces or len(gy_arr) != n_traces:
        print(f"  WARNING: header arrays lengths do not match number of traces. Exiting.")
        exit()

    # output meta data structure
    output_meta = {
        "n_samples": int(n_samples),
        "n_traces": int(n_traces),
        "dt": float(dt),
        "data": None,  # to be filled
        "n_sources": 0,
        "sources": []
    }

    # Determine unique (sx, sy) pairs in order of first appearance
    coords = np.vstack([sx_arr, sy_arr]).T
    # Use dict to preserve order and group indices
    groups = {}
    for idx, (sx, sy) in enumerate(coords.tolist()):
        key = (float(sx), float(sy))
        groups.setdefault(key, []).append(idx)

    # for each world (unique sx,sy) create a per-source block
    world_idx = 0
    for (sx, sy), indices in groups.items():
        world_idx += 1

        gx_vals = [float(gx_arr[i]) for i in indices]
        gy_vals = [float(gy_arr[i]) for i in indices]

        source_block = {
            "sx": float(sx),
            "sy": float(sy),
            "n_receivers": int(len(indices)),
            "gx": gx_vals,
            "gy": gy_vals
        }

        output_meta["sources"].append(source_block)

    # now write a single data file containing all traces (with header)
    data_file = os.path.join(out_dir, f"{name}_traces.bin")
    # if small enough we can write in a single shot, otherwise stream rows
    if total_bytes <= STREAMING_THRESHOLD_BYTES:
        write_data_file(data_file, traces, dt)
    else:
        header_bin = struct.pack('<II f', n_traces, n_samples, float(dt))
        with open(data_file, 'wb') as df:
            df.write(header_bin)
            for i in range(n_traces):
                df.write(traces[i].astype('<f4').tobytes())

    # write single meta file describing the dataset and the source blocks
    output_meta["data"] = data_file
    output_meta["n_sources"] = int(len(groups))

    meta_bin_file = os.path.join(out_dir, f"{name}.bin")
    with open(meta_bin_file, 'wb') as bf:
        # header
        bf.write(struct.pack('<I', int(n_samples)))
        bf.write(struct.pack('<I', int(n_traces)))
        bf.write(struct.pack('<f', float(dt)))

        # path field: fixed bytes
        path_bytes = data_file.encode('utf-8')
        if len(path_bytes) >= META_PATH_FIELD_SIZE:
            print(f"  WARNING: data path too long ({len(path_bytes)} bytes) — truncating to {META_PATH_FIELD_SIZE} bytes")
            path_bytes = path_bytes[:META_PATH_FIELD_SIZE-1]
        # null-pad
        path_padded = path_bytes + b'\x00' * (META_PATH_FIELD_SIZE - len(path_bytes))
        bf.write(path_padded)

        # number of sources
        bf.write(struct.pack('<I', int(len(groups))))

        # write source blocks
        for (sx, sy), indices in groups.items():
            bf.write(struct.pack('<f', float(sx)))
            bf.write(struct.pack('<f', float(sy)))
            bf.write(struct.pack('<I', int(len(indices))))

            # write gx array
            for i in indices:
                bf.write(struct.pack('<f', float(gx_arr[i])))
            # write gy array
            for i in indices:
                bf.write(struct.pack('<f', float(gy_arr[i])))

    # print(f"  WROTE meta json: {json_meta_file}")
    print(f"  WROTE binary meta: {meta_bin_file} (path field size={META_PATH_FIELD_SIZE} bytes)")
    
    print("  DONE for", name)

