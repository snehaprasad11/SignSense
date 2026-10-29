"""Train the Conv1D sign classifier.

Network (input = 1 s window, 100 timesteps x 6 axes):

    Conv1D(32, 5, relu) -> MaxPool(2) -> Dropout(0.3)       100 -> 50 timesteps
    Conv1D(64, 3, relu) -> MaxPool(2) -> Dropout(0.3)        50 -> 25 timesteps
    Flatten (25 x 64 = 1600)   |   GlobalAveragePooling (64)     <- --arch flatten | gap
    Dense(64, relu) -> Dropout(0.3) -> Dense(n_classes, softmax)

    7 classes: Flatten = 110,119 parameters, GAP = 11,815 parameters.

Flatten keeps WHEN in the window each motion feature fires (needed to separate COME from YES);
GAP averages over time and is 9x smaller but loses that timing.

Pipeline
- stratified split: 20 % test, then 15 % of the rest for validation (seed 42)
- per-axis z-score normalisation computed on the training split only (no test leakage)
- optional augmentation (--augment): each extra copy gets a random time shift of up to
  ±--max-shift samples (the device classifies every 0.5 s, so live signs appear anywhere in
  the window), Gaussian noise and ±10 % amplitude scaling
- Adam, early stopping (patience 12, best weights restored), LR halved on plateau
- reports: accuracy, macro P/R/F1, Cohen's kappa, Wilson 95 % CI, confusion matrix,
  and accuracy on test windows shifted by 0.5 s (robustness to live timing)

Outputs in artifacts/: gesture_model.keras, norm_params.json, splits.npz, train_metrics.json,
classification_report.txt, training_report.png (and arch_comparison.md with --compare).

    python train.py --arch flatten --augment --aug-copies 6 --max-shift 50
    python train.py --compare --augment            # Flatten vs GAP, deploys --arch
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import config as C
import analyze_live as M      # metrics
import curate_dataset as io   # window CSV helpers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --------------------------------------------------------------------------- data
def load(data_dir):
    X, y_str, _ = io.load_dataset(data_dir, C.CLASSES)
    if len(X) == 0:
        raise SystemExit(f"no windows found in {data_dir} (run curate_dataset.py first)")
    index = {c: i for i, c in enumerate(C.CLASSES)}
    y = np.array([index[s] for s in y_str], dtype=np.int64)
    return X.astype(np.float32), y


def split(X, y, seed):
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.15, stratify=y_tmp,
                                                      random_state=seed)
    return X_train, y_train, X_val, y_val, X_test, y_test


def norm_params(X_train):
    mean = X_train.reshape(-1, C.N_AXES).mean(axis=0)
    std = X_train.reshape(-1, C.N_AXES).std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def time_shift(window, shift):
    """Shift a window by `shift` samples (+ = later), padding with the edge sample."""
    if shift == 0:
        return window
    if shift > 0:
        return np.concatenate([np.repeat(window[:1], shift, axis=0), window[:-shift]], axis=0)
    return np.concatenate([window[-shift:], np.repeat(window[-1:], -shift, axis=0)], axis=0)


def augment(X, y, copies, max_shift, rng, noise=0.05, scale=0.10):
    """Return originals + `copies` randomly shifted / noisy / scaled copies (X already normalised)."""
    out_x, out_y = [X], [y]
    for _ in range(copies):
        shifts = rng.integers(-max_shift, max_shift + 1, size=len(X)) if max_shift else np.zeros(len(X), int)
        Xa = np.stack([time_shift(w, int(s)) for w, s in zip(X, shifts)])
        Xa = Xa * rng.uniform(1 - scale, 1 + scale, size=(len(X), 1, C.N_AXES))
        Xa = Xa + rng.normal(0, noise, size=Xa.shape)
        out_x.append(Xa.astype(np.float32))
        out_y.append(y)
    return np.concatenate(out_x), np.concatenate(out_y)


# --------------------------------------------------------------------------- model
def build_model(arch, n_classes, dropout=0.3):
    inp = keras.Input(shape=(C.WINDOW, C.N_AXES), name="imu_window")
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    if arch == "flatten":
        x = layers.Flatten()(x)
    elif arch == "gap":
        x = layers.GlobalAveragePooling1D()(x)
    else:
        raise ValueError(arch)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name=f"conv1d_{arch}")


def fit(arch, data, args, verbose=2):
    X_train, y_train, X_val, y_val = data
    tf.keras.utils.set_random_seed(args.seed)
    model = build_model(arch, len(C.CLASSES))
    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs,
                     batch_size=args.batch_size, callbacks=callbacks, verbose=verbose)
    return model, hist.history


def evaluate(model, X, y):
    pred = model.predict(X, verbose=0).argmax(axis=1)
    return M.summarize(M.confusion(y, pred, len(C.CLASSES)), C.CLASSES)


def shifted_accuracy(model, X_norm, y, shift):
    """Accuracy when every test window is shifted by `shift` samples (sign appears earlier)."""
    Xs = np.stack([time_shift(w, -shift) for w in X_norm])
    return float((model.predict(Xs, verbose=0).argmax(axis=1) == y).mean())


# --------------------------------------------------------------------------- reports
def plot_training(history, summary, path, title):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].plot(history["loss"], label="train")
    ax[0].plot(history["val_loss"], label="validation")
    ax[0].set_title("loss")
    ax[1].plot(history["accuracy"], label="train")
    ax[1].plot(history["val_accuracy"], label="validation")
    ax[1].set_title("accuracy")
    for a in ax[:2]:
        a.set_xlabel("epoch")
        a.legend()
        a.grid(alpha=0.3)
    cm = np.array(summary["confusion_matrix"])
    ax[2].imshow(cm, cmap="Blues")
    ax[2].set_xticks(range(len(C.CLASSES)), C.CLASSES, rotation=45, ha="right")
    ax[2].set_yticks(range(len(C.CLASSES)), C.CLASSES)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax[2].text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[2].set_xlabel("predicted")
    ax[2].set_ylabel("true")
    ax[2].set_title(f"test confusion matrix (acc {summary['accuracy']:.2%})")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None, help="default: data/clean (falls back to data/)")
    ap.add_argument("--classes", choices=list(C.CLASS_SETS), default=C.CLASS_SET)
    ap.add_argument("--arch", choices=["flatten", "gap"], default="flatten")
    ap.add_argument("--compare", action="store_true", help="train both archs, write arch_comparison.md")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--aug-copies", type=int, default=6)
    ap.add_argument("--max-shift", type=int, default=50, help="max time shift in samples (50 = 0.5 s)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--out-dir", default=str(C.ARTIFACTS_DIR))
    args = ap.parse_args()
    C.use_class_set(args.classes)

    data_dir = Path(args.data_dir) if args.data_dir else C.CLEAN_DIR
    if args.data_dir is None and not any(data_dir.glob("data_*.csv")):
        data_dir = C.DATA_DIR
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X, y = load(data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = split(X, y, args.seed)
    print(f"data: {data_dir}  windows {len(X)}  train {len(X_train)} · val {len(X_val)} · test {len(X_test)}")

    mean, std = norm_params(X_train)
    n = lambda a: (a - mean) / std
    Xtr, Xva, Xte = n(X_train), n(X_val), n(X_test)
    if args.augment:
        Xtr, ytr = augment(Xtr, y_train, args.aug_copies, args.max_shift, np.random.default_rng(args.seed))
        print(f"augmented training set: {len(Xtr)} windows ({args.aug_copies} extra copies, "
              f"shift ±{args.max_shift} samples)")
    else:
        ytr = y_train

    archs = ["flatten", "gap"] if args.compare else [args.arch]
    results = {}
    for arch in archs:
        print(f"\n=== training {arch} ===")
        model, history = fit(arch, (Xtr, ytr, Xva, y_val), args)
        summary = evaluate(model, Xte, y_test)
        summary["shift_0.5s_accuracy"] = shifted_accuracy(model, Xte, y_test, C.STRIDE)
        results[arch] = (model, history, summary)
        print(M.report_text(summary, f"\n{arch}: test set"))
        print(f"accuracy with test windows shifted by 0.5 s: {summary['shift_0.5s_accuracy']:.4f}")

    model, history, summary = results[args.arch]
    model.save(out / "gesture_model.keras")
    (out / "norm_params.json").write_text(json.dumps(
        {"classes": C.CLASSES, "mean": mean.tolist(), "std": std.tolist(), "axes": C.AXES,
         "window": C.WINDOW, "sample_rate_hz": C.SAMPLE_RATE_HZ, "arch": args.arch}, indent=2))
    np.savez_compressed(out / "splits.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        classes=np.array(C.CLASSES))
    (out / "train_metrics.json").write_text(json.dumps({
        "arch": args.arch, "params": int(model.count_params()), "data_dir": str(data_dir),
        "split": {"train": len(X_train), "val": len(X_val), "test": len(X_test), "train_augmented": len(Xtr)},
        "augment": {"enabled": args.augment, "copies": args.aug_copies, "max_shift": args.max_shift},
        "epochs_run": len(history["loss"]),
        "test": summary,
    }, indent=2))
    (out / "classification_report.txt").write_text(
        M.report_text(summary, f"Conv1D + {args.arch} — test set"), encoding="utf-8")
    plot_training(history, summary, out / "training_report.png",
                  f"Conv1D + {args.arch} · {model.count_params():,} parameters")

    if args.compare:
        rows = ["# Architecture comparison: Flatten vs Global Average Pooling", "",
                f"Data: `{data_dir}` · test windows: {len(X_test)} · augmentation: "
                f"{'on' if args.augment else 'off'}", "",
                "| | Flatten | GAP |", "|---|---|---|"]
        f, g = results["flatten"], results["gap"]
        rows += [
            f"| Parameters | {f[0].count_params():,} | {g[0].count_params():,} |",
            f"| Test accuracy | {f[2]['accuracy']:.2%} | {g[2]['accuracy']:.2%} |",
            f"| Macro F1 | {f[2]['macro_f1']:.2%} | {g[2]['macro_f1']:.2%} |",
            f"| Cohen's κ | {f[2]['kappa']:.3f} | {g[2]['kappa']:.3f} |",
            f"| Accuracy, test shifted 0.5 s | {f[2]['shift_0.5s_accuracy']:.2%} | {g[2]['shift_0.5s_accuracy']:.2%} |",
            f"| Float32 size (approx.) | {f[0].count_params() * 4 / 1024:.1f} KB | {g[0].count_params() * 4 / 1024:.1f} KB |",
            "", f"Deployed: **{args.arch}**.",
        ]
        (out / "arch_comparison.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print("\n".join(rows))

    print(f"\nsaved {out / 'gesture_model.keras'}  ({model.count_params():,} parameters)")


if __name__ == "__main__":
    main()
