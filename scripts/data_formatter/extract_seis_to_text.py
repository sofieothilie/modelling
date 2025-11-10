# extract_seis_to_text.py
import os
import json
import numpy as np
from seispy import Seismic

paths = [
    r".\data\new\aligned_Monitor1S1_mig_IF1__20D.pkl",
    r".\data\new\aligned_Baseline1_mig_IF1__20D.pkl",
    r".\data\new\150kHz_Single_source_file.pkl"
]
out_dir = r".\data\extracted"
os.makedirs(out_dir, exist_ok=True)

# Header explanation:
# trans_id: unique (?) transmitter identifier
# sourceX and sourceY: gives the source coordinates for each trace
# groupX and groupY: gives the receiver coordinates for each trace
# SrcElev and RcvElev: elevation of source and receiver, z coordinate

# Max size (bytes)
# max_full_text_bytes = 200 * 1024 * 1024

def save_header_json(hdr, outpath):
    # convert numpy arrays to lists
    serial = {}
    for k,v in hdr.items():
        try:
            json.dumps({k: v})
            serial[k] = v
        except TypeError:
            try:
                serial[k] = np.asarray(v).tolist()
            except Exception:
                serial[k] = str(v)
    with open(outpath, 'w') as fh:
        json.dump(serial, fh, indent=2)

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

    # Basic metadata
    traces = getattr(seis, "traces", None)
    header = getattr(seis, "header", {})
    if traces is None:
        print("  No 'traces' attribute found in Seismic object.Skipping.")
        continue

    traces = np.asarray(traces)  # ensure ndarray
    n_traces, n_samples = traces.shape
    print(f"  traces.shape = {traces.shape} (n_traces, n_samples)")

    # List useful geometry fields
    geom_keys = ['trans_id', 'sourceX','sourceY','groupX','groupY','SrcElev','RcvElev']
    present = {k: header.get(k, None) for k in geom_keys if k in header}
    if present:
        print("  Geometry/header fields present:", ", ".join(present.keys()))
    else:
        print("  No standard geometry keys found in header")

    # Save header JSON
    hdr_file = os.path.join(out_dir, f"{name}_header.json")
    print("  Saving header to", hdr_file)
    save_header_json(header, hdr_file)

    # Save full traces as text if not too large
    est_bytes = traces.size * 4  # estimate, float32
    trace_file = os.path.join(out_dir, f"{name}_traces.bin")
    traces.astype(np.float32).tofile(trace_file)
    print(f"Saved binary traces to {trace_file}")


    print("  DONE for", name)

