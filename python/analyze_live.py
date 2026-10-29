"""Analyse guided live tests (logs/inference_test_*.csv from inference_server.py --test).

For a confidence threshold T, the outcome of one trial is the FIRST window during the GO phase
whose top class is a sign with confidence >= T (or "rest" if there is none). Then:

    correct        trial outcome == prompted word (rest trial: nothing triggered)
    missed sign    sign prompted, nothing triggered
    wrong class    sign prompted, another sign triggered
    false trigger  rest prompted, a sign triggered

Outputs a trial-level confusion matrix with P/R/F1/kappa/Wilson CI, the most frequent
confusions, a threshold sweep 0.50-0.99 and the latency distribution, written to
artifacts/<name>_live_validation.md and .png.

Recommended threshold: the accuracy-maximising thresholds usually form a plateau; we take its
upper end minus a 0.05 margin, which keeps false triggers low without sitting on the edge
where weak signs start to be missed.

    python analyze_live.py ../logs/inference_test_*.csv --threshold 0.84
    python analyze_live.py ../logs/inference_test_*.csv --stats ../logs/device_stats_*.csv --name final
"""
import argparse
import glob
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

# --------------------------------------------------------------------------- metrics
# Shared with train.py and export_model.py (rows = true class, columns = predicted):
#   accuracy = sum(TP)/N · precision = TP/(TP+FP) · recall = TP/(TP+FN) · F1 = 2PR/(P+R)
#   kappa = (p_o - p_e)/(1 - p_e), p_e = sum(row_total*col_total)/N^2 · 95 % CI = Wilson score
def confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(cm, class_names):
    """Everything the reports need from one confusion matrix."""
    cm = np.asarray(cm)
    n = int(cm.sum())
    tp = np.diag(cm).astype(float)
    col = cm.sum(axis=0).astype(float)
    row = cm.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(col > 0, tp / col, 0.0)
        recall = np.where(row > 0, tp / row, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    acc = tp.sum() / n if n else 0.0
    p_e = (row * col).sum() / (n * n) if n else 0.0
    kappa = (acc - p_e) / (1 - p_e) if p_e < 1 else 0.0
    present = row > 0          # macro average only over classes that occur
    lo, hi = wilson_ci(int(tp.sum()), n)
    return {
        "n": n,
        "correct": int(tp.sum()),
        "accuracy": float(acc),
        "ci95": [float(lo), float(hi)],
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "kappa": float(kappa),
        "chance": 1.0 / len(class_names),
        "per_class": {
            c: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                "support": int(row[i]), "tp": int(tp[i])}
            for i, c in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
        "classes": list(class_names),
    }


def report_text(s, title=""):
    """Plain-text report in the style of sklearn's classification_report, plus kappa and CI."""
    lines = [title] if title else []
    lines.append(f"{'class':>12} {'precision':>10} {'recall':>8} {'f1':>8} {'support':>8}")
    for c, m in s["per_class"].items():
        lines.append(f"{c:>12} {m['precision']:10.4f} {m['recall']:8.4f} {m['f1']:8.4f} {m['support']:8d}")
    lo, hi = s["ci95"]
    lines += [
        "",
        f"accuracy        {s['accuracy']:.4f}  ({s['correct']}/{s['n']}, 95% CI {lo:.3f}-{hi:.3f})",
        f"macro precision {s['macro_precision']:.4f}",
        f"macro recall    {s['macro_recall']:.4f}",
        f"macro F1        {s['macro_f1']:.4f}",
        f"Cohen's kappa   {s['kappa']:.4f}",
        f"chance level    {s['chance']:.4f}",
        "",
        "confusion matrix (rows = true, columns = predicted)",
        " " * 12 + "".join(f"{c[:9]:>10}" for c in s["classes"]),
    ]
    for c, r in zip(s["classes"], s["confusion_matrix"]):
        lines.append(f"{c:>12}" + "".join(f"{v:10d}" for v in r))
    return "\n".join(lines)


def markdown_confusion(s):
    cls = s["classes"]
    lines = ["| true \\ pred | " + " | ".join(cls) + " |", "|---" * (len(cls) + 1) + "|"]
    for c, r in zip(cls, s["confusion_matrix"]):
        lines.append(f"| **{c}** | " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def top_confusions(s, k=5):
    cm = np.array(s["confusion_matrix"])
    pairs = [(cm[i, j], s["classes"][i], s["classes"][j])
             for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
    return sorted(pairs, reverse=True)[:k]


# --------------------------------------------------------------------------- live analysis
SWEEP = np.round(np.arange(0.50, 0.995, 0.01), 2)
MARGIN = 0.05


def load(paths):
    files = sorted({f for p in paths for f in glob.glob(p)})
    if not files:
        raise SystemExit("no log files matched")
    frames = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        df["file"] = Path(f).name
        df["trial_key"] = df["trial"].astype(str) + f"@{i}"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df[df["trial"] >= 0], files


def trial_outcomes(df, threshold):
    go = df[df["phase"] == "go"]
    out = []
    for key, g in go.groupby("trial_key", sort=False):
        expected = g["expected"].iloc[0]
        hit = g[(g["label"] != C.REST_CLASS) & (g["conf"] >= threshold)]
        out.append((expected, hit["label"].iloc[0] if len(hit) else C.REST_CLASS))
    return out


def evaluate(df, threshold, classes):
    outcomes = trial_outcomes(df, threshold)
    idx = {c: i for i, c in enumerate(classes)}
    s = summarize(confusion([idx[e] for e, _ in outcomes], [idx[p] for _, p in outcomes], len(classes)),
                    classes)
    sign = [(e, p) for e, p in outcomes if e != C.REST_CLASS]
    rest = [(e, p) for e, p in outcomes if e == C.REST_CLASS]
    s["missed"] = sum(p == C.REST_CLASS for _, p in sign)
    s["wrong_class"] = sum(p not in (e, C.REST_CLASS) for e, p in sign)
    s["false_trigger"] = sum(p != C.REST_CLASS for _, p in rest)
    return s


def recommend(sweep):
    best = max(r["accuracy"] for r in sweep)
    plateau = [r["threshold"] for r in sweep if r["accuracy"] >= best - 1e-9]
    upper = max(plateau)
    pick = max(min(plateau), round(upper - MARGIN, 2))
    return pick, best, (min(plateau), upper)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--threshold", type=float, default=C.THRESHOLD)
    ap.add_argument("--stats", nargs="*", default=[], help="device_stats_*.csv for on-device numbers")
    ap.add_argument("--name", default="live", help="report name: artifacts/<name>_live_validation.*")
    args = ap.parse_args()

    df, files = load(args.logs)
    prob_cols = [c for c in df.columns if c.startswith("p_")]
    classes = [c[2:] for c in prob_cols] or list(C.CLASSES)

    s = evaluate(df, args.threshold, classes)
    sweep = []
    for t in SWEEP:
        r = evaluate(df, t, classes)
        sweep.append({"threshold": float(t), "accuracy": r["accuracy"], "missed": r["missed"],
                      "wrong_class": r["wrong_class"], "false_trigger": r["false_trigger"]})
    rec, best, plateau = recommend(sweep)

    inv_ms = df["invoke_us"].astype(float) / 1000
    stats = pd.concat([pd.read_csv(f) for p in args.stats for f in glob.glob(p)], ignore_index=True) \
        if args.stats else pd.DataFrame()

    # ---- markdown
    lo, hi = s["ci95"]
    lines = [f"# Live validation ({args.name})", "",
             f"Logs: {', '.join(Path(f).name for f in files)}", "",
             f"**Accuracy = {s['correct']} / {s['n']} = {s['accuracy']:.2%}** (95 % CI {lo:.1%}–{hi:.1%}) · "
             f"macro precision {s['macro_precision']:.2%} · macro recall {s['macro_recall']:.2%} · "
             f"**macro F1 {s['macro_f1']:.2%}** · κ = {s['kappa']:.3f} · threshold {args.threshold:.2f}", "",
             f"Missed signs {s['missed']} · wrong class {s['wrong_class']} · false triggers {s['false_trigger']}",
             "", "| Class | Correct / trials | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for c, m in s["per_class"].items():
        lines.append(f"| {c} | {m['tp']} / {m['support']} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} |")
    lines += ["", markdown_confusion(s), "", "Most frequent confusions: " +
              (", ".join(f"{t} → {p} ({n}×)" for n, t, p in top_confusions(s)) or "none"), "",
              "## Threshold sweep", "",
              f"Best accuracy {best:.2%} for thresholds {plateau[0]:.2f}–{plateau[1]:.2f}; "
              f"**recommended threshold {rec:.2f}** (upper end of the plateau − {MARGIN}).", "",
              "| threshold | accuracy | missed | wrong class | false trigger |", "|---|---|---|---|---|"]
    for r in sweep:
        if round(r["threshold"] * 100) % 5 == 0 or r["threshold"] in (rec, args.threshold):
            lines.append(f"| {r['threshold']:.2f} | {r['accuracy']:.2%} | {r['missed']} | {r['wrong_class']} | "
                         f"{r['false_trigger']} |")
    lines += ["", "## Latency", "",
              f"`Invoke()` from #PRED lines: mean {inv_ms.mean():.2f} ms · min {inv_ms.min():.2f} · "
              f"max {inv_ms.max():.2f} · CPU load at one inference per {C.STRIDE / C.SAMPLE_RATE_HZ:.1f} s: "
              f"{inv_ms.mean() / (1000 * C.STRIDE / C.SAMPLE_RATE_HZ):.1%} of one core",
              f"End of sign → word shown: ≈ {inv_ms.mean() / 1000:.2f}–"
              f"{inv_ms.mean() / 1000 + C.STRIDE / C.SAMPLE_RATE_HZ:.2f} s (latency + 0–0.5 s stride wait)"]
    if not stats.empty:
        num = stats.apply(pd.to_numeric, errors="coerce")
        lines += ["", "Device stats (`#STAT`, averaged):", ""]
        for k in ["invoke_avg_us", "pre_us", "arena", "heap", "rate_hz", "lost"]:
            if k in num:
                lines.append(f"- {k}: {num[k].mean():.1f}")
    out_md = C.ARTIFACTS_DIR / f"{args.name}_live_validation.md"
    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    cm = np.array(s["confusion_matrix"])
    ax[0].imshow(cm, cmap="Blues")
    ax[0].set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax[0].set_yticks(range(len(classes)), classes)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax[0].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[0].set_title(f"live trials @ {args.threshold:.2f}: {s['accuracy']:.1%}")
    ax[0].set_xlabel("predicted")
    ax[0].set_ylabel("prompted")
    th = [r["threshold"] for r in sweep]
    ax[1].plot(th, [r["accuracy"] for r in sweep], label="accuracy")
    n = max(s["n"], 1)
    for k in ["missed", "wrong_class", "false_trigger"]:
        ax[1].plot(th, [r[k] / n for r in sweep], label=k.replace("_", " "), alpha=0.8)
    ax[1].axvline(rec, color="k", ls="--", lw=1, label=f"recommended {rec:.2f}")
    ax[1].set_xlabel("confidence threshold")
    ax[1].set_title("threshold sweep (fraction of trials)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[2].hist(inv_ms, bins=30)
    ax[2].set_xlabel("Invoke() latency (ms)")
    ax[2].set_title("on-device latency")
    fig.tight_layout()
    out_png = out_md.with_suffix(".png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    print("\n".join(lines))
    print(f"\nwrote {out_md} and {out_png}")


if __name__ == "__main__":
    main()
