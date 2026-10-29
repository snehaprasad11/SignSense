# Results — Sign Recognition on ESP32 + MPU6050 (TFLite Micro)

Six sign words plus rest — **HELLO, THANK YOU, YES, PLEASE, HELP, COME, rest** — classified
on-device from 1-second windows (100 samples × 6 axes at 100 Hz) of a single MPU6500/MPU6050
mounted on the back of the hand.

- **Test set** = 20 % of the recordings held back from training (never seen by the model).
- **Live** = guided tests on the ESP32: the PC prompts a word, the user signs it, the device's
  predictions are logged (`python/inference_server.py --test`, analysed with `analyze_live.py`).

---

## 1. Formulas

Confusion matrix: rows = true class, columns = predicted class.

| Metric | Formula |
|---|---|
| Accuracy | Σ TP / N (correct predictions ÷ all predictions) |
| Precision (class c) | TP / (TP + FP) = diagonal ÷ column sum |
| Recall (class c) | TP / (TP + FN) = diagonal ÷ row sum |
| F1 (class c) | 2 · P · R / (P + R) |
| Macro average | mean of a metric over the 7 classes |
| Cohen's κ | (p_o − p_e) / (1 − p_e), p_o = accuracy, p_e = Σ(row total × column total) / N² |
| 95 % confidence interval | Wilson score interval for k correct out of n |
| Chance level | 1 / 7 classes = **14.3 %** |

**Worked example (COME, test set):** the model predicted COME 27 times, 25 correctly (2 were YES)
→ **P = 25/27 = 92.59 %**. There are 30 real COME windows, 25 found → **R = 25/30 = 83.33 %**.
**F1 = 2(0.9259 × 0.8333)/(0.9259 + 0.8333) = 87.72 %**.

---

## 2. Final model — test set (N = 236 windows)

**Accuracy = 224 / 236 = 94.92 %** (95 % CI 91.3–97.1 %) · macro precision 95.38 % ·
macro recall 94.99 % · **macro F1 95.11 %** · κ = 0.941

| Class | TP / support | Precision | Recall | F1 |
|---|---|---|---|---|
| COME | 25 / 30 | 92.59 % | 83.33 % | 87.72 % |
| HELLO | 30 / 30 | 100 % | 100 % | 100 % |
| HELP | 35 / 38 | 94.59 % | 92.11 % | 93.33 % |
| PLEASE | 30 / 30 | 100 % | 100 % | 100 % |
| rest | 32 / 32 | 100 % | 100 % | 100 % |
| THANK YOU | 36 / 38 | 94.74 % | 94.74 % | 94.74 % |
| YES | 36 / 38 | 85.71 % | 94.74 % | 90.00 % |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | come | hello | help | please | rest | thank_you | yes |
|---|---|---|---|---|---|---|---|
| **come** | 25 | 0 | 0 | 0 | 0 | 0 | 5 |
| **hello** | 0 | 30 | 0 | 0 | 0 | 0 | 0 |
| **help** | 0 | 0 | 35 | 0 | 0 | 2 | 1 |
| **please** | 0 | 0 | 0 | 30 | 0 | 0 | 0 |
| **rest** | 0 | 0 | 0 | 0 | 32 | 0 | 0 |
| **thank_you** | 0 | 0 | 2 | 0 | 0 | 36 | 0 |
| **yes** | 2 | 0 | 0 | 0 | 0 | 0 | 36 |

Main confusion: COME → YES (5×).

---

## 3. Live on the ESP32 — final test (N = 52 trials, threshold 0.84)

**Accuracy = 42 / 52 = 80.77 %** (95 % CI 68.1–89.2 %) · macro precision 83.65 % ·
macro recall 81.63 % · **macro F1 80.02 %** · κ = 0.776

| Class | Correct / trials | Precision | Recall | F1 |
|---|---|---|---|---|
| COME | 7 / 7 | 77.78 % | 100 % | 87.50 % |
| HELLO | 4 / 8 | 100 % | 50.00 % | 66.67 % |
| HELP | 7 / 7 | 77.78 % | 100 % | 87.50 % |
| PLEASE | 8 / 8 | 66.67 % | 100 % | 80.00 % |
| rest | 5 / 7 | 83.33 % | 71.43 % | 76.92 % |
| THANK YOU | 7 / 7 | 100 % | 100 % | 100 % |
| YES | 4 / 8 | 80.00 % | 50.00 % | 61.54 % |

| true \ pred | come | hello | help | please | rest | thank_you | yes |
|---|---|---|---|---|---|---|---|
| **come** | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| **hello** | 0 | 4 | 1 | 3 | 0 | 0 | 0 |
| **help** | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| **please** | 0 | 0 | 0 | 8 | 0 | 0 | 0 |
| **rest** | 0 | 0 | 0 | 1 | 5 | 0 | 1 |
| **thank_you** | 0 | 0 | 0 | 0 | 0 | 7 | 0 |
| **yes** | 2 | 0 | 1 | 0 | 1 | 0 | 4 |

Main confusions: HELLO → PLEASE (3×), YES → COME (2×).

---

## 4. Live accuracy across iterations

| Live test | Change made | Accuracy | 95 % CI |
|---|---|---|---|
| #1 | first 6-word model | 66.2 % (43/65) | 54–76 % |
| #2 | ±0.5 s time-shift augmentation | 83.1 % (54/65) | 72–90 % |
| #3 | same model, 15 repetitions per word | 78.4 % (80/102) | 69–85 % |
| #4 | + extra THANK YOU / YES / HELP recordings | 80.8 % (42/52) | 68–89 % |

The confidence intervals of #2–#4 overlap, so they are statistically about the same (~80 %).
The significant improvement was #1 → #2: recordings always had the sign at the end of the
window, while live windows (0.5 s stride) see it at any position. Augmenting training data
with ±0.5 s time shifts fixed this (test accuracy with a 0.5 s shift: 78.5 % → 90.3 %).

---

## 5. Model comparisons

| Comparison | Test accuracy | Size |
|---|---|---|
| Signs — Flatten CNN (**deployed**) | **94.92 %** | 110,119 params, 437 KB |
| Signs — Global Average Pooling CNN | 93.22 % | 11,815 params, 46 KB |
| Directions (up/down/left/right) — GAP | 99.32 % | 11,685 params |
| Directions — Flatten | 97.95 % | 109,989 params |
| Float32 TFLite | 94.92 % | 437.1 KB |
| INT8 TFLite (full-integer, post-training) | 95.34 % (99.58 % agreement with float32) | 119.6 KB (3.65× smaller) |

GAP wins for the direction gestures but loses for signs: it averages over time, discarding
*when* motion happens in the window — the information that separates COME from YES.

---

## 6. On-device performance (ESP32-D0WD-V3, 240 MHz, no PSRAM)

| Quantity | Value | How |
|---|---|---|
| `Invoke()` latency | **279.1 ms** (min 279.08, max 279.25) | measured |
| Pre-processing (normalise window) | 0.22 ms | measured |
| Max inference rate | 3.58 / s | 1 ÷ 0.2791 s |
| CPU load (1 inference every 0.5 s) | 55.8 % of one core | 279.1 ÷ 500 |
| Sampling rate | 99.9 Hz, 0 samples lost | measured (sampler task on core 0) |
| Tensor arena used | 28 KB (internal RAM) | measured |
| Free heap | 186 KB | measured |
| Flash used by sketch | 856 KB (27 %) | Arduino build output |
| Delay from end of sign to word shown | ≈ 0.28–0.78 s | latency + 0–0.5 s stride wait |

---

## 7. Summary

**94.9 % accuracy on held-out data, ~81 % in live use on the ESP32, 279 ms on-device inference.**
The offline–live gap comes from variation in how signs are performed when prompted live,
mainly YES (small nods resemble the start of other signs) and HELLO (a long wave ends looking
like PLEASE's circle). Recording training data under the same prompted conditions as use is the
most direct way to close it.

Source files: `artifacts/train_metrics.json`, `artifacts/classification_report.txt`,
`artifacts/quantization_report.md`, `artifacts/*_live_validation.md`,
`logs/inference_test_*.csv`, `logs/device_stats_*.csv`.
