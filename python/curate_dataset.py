"""Clean the recorded windows and write a training-ready copy to data/clean/.

Steps
1. drop malformed rows (non-numeric, NaN, physically impossible values) and incomplete windows
2. drop duplicate windows (same samples saved twice)
3. detect and remove a constant gyro offset per recording session
   (the first session had -24 / +47 / +23 °/s because the board moved during calibration;
   rest windows of a session should average 0 °/s, so their mean is the offset)
4. drop motion-energy outliers per class (robust z-score on gyro energy)
5. balance classes by random down-sampling (skip with --no-balance)

Writes data/clean/data_<class>.csv, artifacts/dataset_report.png and artifacts/curation_report.md.

    python curate_dataset.py --no-balance
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

# --------------------------------------------------------------------------- window CSV files
# Long format, one row per sample, WINDOW rows per window:
#     window_id,label,session,sample,timestamp_us,ax,ay,az,gx,gy,gz
# `session` = recording session (capture start time), used to correct a gyro offset per session.
# capture_server.py, train.py and simulate_esp32.py import these helpers from here.
COLUMNS = ["window_id", "label", "session", "sample", "timestamp_us"] + C.AXES


def csv_path(label, data_dir=None):
    return Path(data_dir or C.DATA_DIR) / f"data_{label}.csv"


def append_window(path, window_id, label, session, timestamps_us, samples):
    """Append one window (WINDOW x 6 array) to a CSV, writing the header if the file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        for i, (ts, row) in enumerate(zip(timestamps_us, samples)):
            w.writerow([window_id, label, session, i, int(ts)] + [f"{v:.5f}" for v in row])


def remove_last_window(path):
    """Delete the most recent window from a CSV (capture_server 'z' = undo). Returns its id."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    last = df["window_id"].iloc[-1]
    df[df["window_id"] != last].to_csv(path, index=False)
    return last


def next_window_id(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    ids = pd.read_csv(path, usecols=["window_id"])["window_id"]
    return int(ids.max()) + 1 if len(ids) else 0


def read_raw(paths):
    """Concatenate CSV files into one DataFrame (no checks)."""
    frames = [pd.read_csv(p) for p in paths if Path(p).exists() and Path(p).stat().st_size > 0]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)


def frame_to_windows(df):
    """DataFrame -> (X [n, WINDOW, 6] float32, labels [n], sessions [n]). Skips incomplete windows."""
    X, y, sessions = [], [], []
    for (label, wid), g in df.groupby(["label", "window_id"], sort=False):
        if len(g) != C.WINDOW:
            continue
        g = g.sort_values("sample")
        X.append(g[C.AXES].to_numpy(np.float32))
        y.append(label)
        sessions.append(g["session"].iloc[0])
    if not X:
        return np.zeros((0, C.WINDOW, C.N_AXES), np.float32), np.array([]), np.array([])
    return np.stack(X), np.array(y), np.array(sessions)


def windows_to_frame(X, labels, sessions=None):
    """Inverse of frame_to_windows; re-numbers window ids per label."""
    rows = []
    counters = {}
    for i, (win, label) in enumerate(zip(X, labels)):
        wid = counters.get(label, 0)
        counters[label] = wid + 1
        session = sessions[i] if sessions is not None else 0
        for s in range(len(win)):
            rows.append([wid, label, session, s, s * 10_000] + list(map(float, win[s])))
    return pd.DataFrame(rows, columns=COLUMNS)


def load_dataset(data_dir, classes):
    """Load data_<class>.csv for every class in `classes` from `data_dir`."""
    df = read_raw([csv_path(c, data_dir) for c in classes])
    return frame_to_windows(df)


# --------------------------------------------------------------------------- curation
OFFSET_MIN_DPS = 3.0     # session offsets smaller than this are left alone
OUTLIER_Z = 3.5          # robust z-score on log motion energy


def drop_malformed(df):
    n0 = len(df)
    for col in C.AXES + ["window_id", "sample", "timestamp_us"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=C.AXES + ["window_id", "sample"])
    acc_ok = (df[["ax", "ay", "az"]].abs() <= C.ACCEL_RANGE_G * 1.05).all(axis=1)
    gyr_ok = (df[["gx", "gy", "gz"]].abs() <= C.GYRO_RANGE_DPS * 1.05).all(axis=1)
    df = df[acc_ok & gyr_ok]
    return df, n0 - len(df)


def drop_duplicates(X, y, s):
    keys = [hash(np.round(w, 3).tobytes()) for w in X]
    _, first = np.unique(keys, return_index=True)
    keep = np.sort(first)
    return X[keep], y[keep], s[keep], len(X) - len(keep)


def fix_gyro_offset(X, y, s):
    """Subtract a per-session gyro offset estimated from that session's rest windows."""
    X = X.copy()
    report = []
    for session in np.unique(s):
        in_s = s == session
        rest = in_s & (y == C.REST_CLASS)
        source = "rest windows"
        if rest.sum() >= 3:
            offset = np.median(X[rest][:, :, 3:].mean(axis=1), axis=0)
        else:
            # no rest recorded in this session: signs average out over many windows
            offset = np.median(X[in_s][:, :, 3:].mean(axis=1), axis=0)
            source = "all windows (no rest in session)"
        if np.abs(offset).max() >= OFFSET_MIN_DPS:
            X[in_s, :, 3:] -= offset
            report.append((session, offset, int(in_s.sum()), source))
    return X, report


def drop_outliers(X, y, s):
    energy = np.log1p((X[:, :, 3:].std(axis=1) ** 2).sum(axis=1))
    keep = np.ones(len(X), bool)
    for c in np.unique(y):
        m = y == c
        med = np.median(energy[m])
        mad = np.median(np.abs(energy[m] - med)) * 1.4826 + 1e-9
        z = np.abs(energy[m] - med) / mad
        idx = np.where(m)[0]
        keep[idx[z > OUTLIER_Z]] = False
    return X[keep], y[keep], s[keep], int((~keep).sum())


def balance(X, y, s, rng):
    counts = {c: int((y == c).sum()) for c in np.unique(y)}
    n = min(counts.values())
    keep = np.concatenate([rng.choice(np.where(y == c)[0], n, replace=False) for c in counts])
    keep.sort()
    return X[keep], y[keep], s[keep]


def plot_report(X, y, path):
    classes = [c for c in C.CLASSES if (y == c).any()]
    fig, axes = plt.subplots(len(classes), 2, figsize=(11, 1.9 * len(classes)), sharex=True, squeeze=False)
    t = np.arange(C.WINDOW) / C.SAMPLE_RATE_HZ
    for r, c in enumerate(classes):
        mean = X[y == c].mean(axis=0)
        for k in range(3):
            axes[r, 0].plot(t, mean[:, k], label=C.AXES[k])
            axes[r, 1].plot(t, mean[:, 3 + k], label=C.AXES[3 + k])
        axes[r, 0].set_ylabel(f"{c}\n(n={int((y == c).sum())})")
    axes[0, 0].set_title("mean accelerometer (g)")
    axes[0, 1].set_title("mean gyroscope (°/s)")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    axes[0, 1].legend(fontsize=7, loc="upper left")
    axes[-1, 0].set_xlabel("time in window (s)")
    axes[-1, 1].set_xlabel("time in window (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    ap.add_argument("--out-dir", default=None, help="default: <data-dir>/clean")
    ap.add_argument("--classes", choices=list(C.CLASS_SETS), default=C.CLASS_SET)
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    C.use_class_set(args.classes)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "clean"
    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = read_raw([csv_path(c, data_dir) for c in C.CLASSES])
    if df.empty:
        raise SystemExit(f"no data_<class>.csv files in {data_dir}")
    df, bad_rows = drop_malformed(df)
    X, y, s = frame_to_windows(df)
    n_raw = len(X)
    X, y, s, dups = drop_duplicates(X, y, s)
    X, offsets = fix_gyro_offset(X, y, s)
    X, y, s, outliers = drop_outliers(X, y, s)
    if not args.no_balance:
        X, y, s = balance(X, y, s, np.random.default_rng(args.seed))

    out_dir.mkdir(parents=True, exist_ok=True)
    for c in C.CLASSES:
        m = y == c
        windows_to_frame(X[m], y[m], s[m]).to_csv(csv_path(c, out_dir), index=False)

    plot_report(X, y, C.ARTIFACTS_DIR / "dataset_report.png")

    lines = ["# Dataset curation report", "",
             f"- input: `{data_dir}` · output: `{out_dir}`",
             f"- malformed rows dropped: {bad_rows}",
             f"- complete windows read: {n_raw}",
             f"- duplicate windows dropped: {dups}",
             f"- motion-energy outliers dropped: {outliers}",
             f"- balanced: {'no' if args.no_balance else 'yes'}", "",
             "## Gyro offset corrections", ""]
    if offsets:
        lines += ["| session | windows | offset gx / gy / gz (°/s) | estimated from |", "|---|---|---|---|"]
        for session, off, n, src in offsets:
            lines.append(f"| {session} | {n} | {off[0]:+.1f} / {off[1]:+.1f} / {off[2]:+.1f} | {src} |")
    else:
        lines.append("none needed")
    lines += ["", "## Clean windows per class", "", "| class | windows |", "|---|---|"]
    for c in C.CLASSES:
        lines.append(f"| {c} | {int((y == c).sum())} |")
    lines.append(f"| **total** | **{len(y)}** |")
    (C.ARTIFACTS_DIR / "curation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
