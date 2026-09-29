"""PHM2010: raw signals to the processed training table.

Reads   data/raw/phm2010/raw/<tool>/<tool>/c_<n>_<cut>.csv   (7 channels at 50 kHz, no header)
Writes  data/processed/cuts.csv                             (one row per tool and cut)

Columns
    identity and clock : tool, cut, n_samples, dur_s, t_cut_s
    labels             : vb1, vb2, vb3, vb_max
    features           : 10 statistics x 7 channels, all from this one cut

Leakage rule: no column aggregates other cuts, because every row reads one signal file. Lags
and windows are the training script's problem, and it may only look backwards. Needs pandas
and numpy to get through 18 GB; the model code imports neither.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "phm2010", "raw")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "cuts.csv")
FS = 50000.0
CHANNELS = ["force_x", "force_y", "force_z", "vib_x", "vib_y", "vib_z", "ae_rms"]
TOOLS = ["c1", "c2", "c3", "c4", "c5", "c6"]


def file_stats(path):
    a = pd.read_csv(path, header=None, dtype="float32").to_numpy()
    if a.ndim != 2 or a.shape[1] != len(CHANNELS):
        raise ValueError(f"{path}: expected {len(CHANNELS)} columns, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{path}: NaN or inf, so the file has a header or is corrupt")
    mean = a.mean(axis=0)
    std = a.std(axis=0)
    rms = np.sqrt((a ** 2).mean(axis=0))
    mx = a.max(axis=0)
    absmax = np.abs(a).max(axis=0)
    z = (a - mean) / np.where(std == 0, 1.0, std)
    st = {"mean": mean, "std": std, "rms": rms, "min": a.min(axis=0), "max": mx, "ptp": mx - a.min(axis=0),
          "absmax": absmax, "skew": (z ** 3).mean(axis=0), "kurt": (z ** 4).mean(axis=0),
          "crest": np.where(rms == 0, 0.0, absmax / rms)}
    return {f"{k}_{c}": float(v) for k, arr in st.items() for c, v in zip(CHANNELS, arr)}, len(a)


def load_wear(tool):
    p = os.path.join(RAW, tool, f"{tool}_wear.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    return df.rename(columns={"flute_1": "vb1", "flute_2": "vb2", "flute_3": "vb3"})


def main(tools=TOOLS):
    rows = []
    for tool in tools:
        wear = load_wear(tool)
        with_wear = set() if wear is None else set(wear["cut"])
        t = 0.0
        files = sorted(glob.glob(os.path.join(RAW, tool, tool, "c_*.csv")))
        for path in files:
            cut = int(re.search(r"_(\d+)\.csv$", path).group(1))
            feats, n = file_stats(path)
            t += n / FS
            row = {"tool": tool, "cut": cut, "n_samples": n, "dur_s": n / FS, "t_cut_s": t}
            if cut in with_wear:
                w = wear[wear["cut"] == cut].iloc[0]
                row.update(vb1=float(w["vb1"]), vb2=float(w["vb2"]), vb3=float(w["vb3"]))
                row["vb_max"] = max(row["vb1"], row["vb2"], row["vb3"])
            row.update(feats)
            rows.append(row)
        print(f"{tool}: {len(files)} cuts, total t_cut = {t:.1f} s", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    feat_cols = [c for c in df.columns if c not in
                 ("tool", "cut", "n_samples", "dur_s", "t_cut_s", "vb1", "vb2", "vb3", "vb_max")]
    for tool, g in df.groupby("tool"):
        assert g["cut"].is_monotonic_increasing, f"{tool}: cuts out of order"
        assert g["t_cut_s"].is_monotonic_increasing and g["t_cut_s"].iloc[-1] > 1000, tool
    assert len(feat_cols) == 10 * len(CHANNELS), len(feat_cols)
    assert df[feat_cols].notna().all().all(), "feature with NaN"
    lab = df[df["vb_max"].notna()]
    assert (lab["vb_max"] == lab[["vb1", "vb2", "vb3"]].max(axis=1)).all()
    assert not any(c.startswith("vb") for c in feat_cols), "a label leaked into the feature list"

    print(f"\n{OUT}\n{df.shape[0]} rows ({df['tool'].nunique()} tools) x {df.shape[1]} columns")
    print(f"labelled cuts per tool: {lab['tool'].value_counts().to_dict()}")
    print(f"{len(feat_cols)} features, first few: {', '.join(feat_cols[:8])}")
    print("\ncut duration and accumulated cutting time per tool:")
    print(df.groupby("tool").agg(cuts=("cut", "count"), dur_min=("dur_s", "min"),
                                 dur_med=("dur_s", "median"), dur_max=("dur_s", "max"),
                                 t_total_s=("t_cut_s", "max")).round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or TOOLS)
