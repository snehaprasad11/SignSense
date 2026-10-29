"""Test the software without hardware.

A synthetic signer produces 6-axis IMU signals with a different motion pattern per class
(wave, nod, circle, ...). The data are SYNTHETIC: they only exercise the code paths, they are
not a substitute for the recorded dataset and give no meaningful accuracy numbers.

Modes
    dataset   write synthetic data_<class>.csv files (default data/synthetic/) for
              curate_dataset.py / train.py smoke tests
    capture   act as the capture firmware: stream 100 Hz JSON to capture_server.py over a
              WebSocket, performing a random sign every few seconds
    serial    act as the inference firmware: run the exported TFLite model on a synthetic
              stream every 0.5 s and print #CLASSES / #PRED / #STAT lines
              (inference_server.py --sim uses SimulatedDevice from this file)

    python simulate_esp32.py dataset --per-class 150
    python simulate_esp32.py capture --host 127.0.0.1
    python simulate_esp32.py serial --seconds 20
"""
import argparse
import asyncio
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np

import config as C
import curate_dataset as io  # window CSV helpers

# --------------------------------------------------------------------------- synthetic signer
# (axis index, waveform, cycles, amplitude) per class; axes 0-2 accel (g), 3-5 gyro (°/s)
TEMPLATES = {
    "hello": [(5, "sin", 3.0, 120), (1, "sin", 3.0, 0.25)],                      # repeated wave
    "thank_you": [(3, "half", 1.0, 150), (1, "half", 1.0, 0.45)],                # chin forward/down
    "yes": [(3, "sin", 2.0, 80)],                                                # fist nod
    "please": [(3, "sin", 1.5, 90), (4, "cos", 1.5, 90)],                        # circle on chest
    "help": [(2, "half", 1.0, 0.6), (4, "half", 1.0, 70)],                       # lift up
    "come": [(3, "half", 2.0, -110), (5, "half", 1.0, 40)],                      # beckon
    "up": [(2, "sin", 1.0, 0.7)],
    "down": [(2, "sin", 1.0, -0.7)],
    "left": [(0, "sin", 1.0, 0.7)],
    "right": [(0, "sin", 1.0, -0.7)],
}
GRAVITY = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], np.float32)   # back of hand facing up


class SyntheticSigner:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def rest(self, n):
        x = np.tile(GRAVITY, (n, 1))
        x[:, :3] += self.rng.normal(0, 0.01, (n, 3))
        x[:, 3:] += self.rng.normal(0, 0.6, (n, 3))
        return x.astype(np.float32)

    def sign(self, label):
        """One sign performance (variable speed and size), without the rest baseline."""
        n = int(self.rng.integers(55, 86))
        t = np.linspace(0, 1, n)
        env = np.sin(np.pi * t)                       # smooth onset and stop
        amp_scale = self.rng.uniform(0.8, 1.2)
        x = np.zeros((n, C.N_AXES), np.float32)
        for axis, kind, cycles, amp in TEMPLATES[label]:
            phase = 2 * np.pi * cycles * t
            wave = {"sin": np.sin(phase), "cos": np.cos(phase) - 1, "half": np.abs(np.sin(phase / 2))}[kind]
            x[:, axis] += amp * amp_scale * wave * env
        return x

    def window(self, label, end_jitter=25):
        """A 1 s window like the recordings: the sign ends 0..end_jitter samples before the end."""
        w = self.rest(C.WINDOW)
        if label == C.REST_CLASS:
            return w
        s = self.sign(label)
        end = C.WINDOW - int(self.rng.integers(0, end_jitter + 1))
        start = max(0, end - len(s))
        w[start:end] += s[len(s) - (end - start):]
        return np.clip(w, [-2] * 3 + [-250] * 3, [2] * 3 + [250] * 3)


class Stream:
    """Continuous 100 Hz stream: rest, plus signs queued with perform()."""

    def __init__(self, seed=0):
        self.signer = SyntheticSigner(seed)
        self.pending = np.zeros((0, C.N_AXES), np.float32)
        self.performing = None

    def perform(self, label):
        if label != C.REST_CLASS:
            self.pending = self.signer.sign(label)
            self.performing = label

    def next(self, n):
        x = self.signer.rest(n)
        k = min(n, len(self.pending))
        if k:
            x[:k] += self.pending[:k]
            self.pending = self.pending[k:]
        if not len(self.pending):
            self.performing = None
        return x


# --------------------------------------------------------------------------- dataset mode
def make_dataset(out_dir, per_class, classes, seed):
    signer = SyntheticSigner(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in classes:
        X = np.stack([signer.window(c) for _ in range(per_class)])
        io.windows_to_frame(X, [c] * per_class, [0] * per_class).to_csv(io.csv_path(c, out_dir), index=False)
        print(f"wrote {per_class} synthetic windows -> {io.csv_path(c, out_dir)}")


# --------------------------------------------------------------------------- capture mode
async def stream_capture(host, port, seconds, seed):
    import websockets
    stream = Stream(seed)
    rng = np.random.default_rng(seed)
    labels = [c for c in C.CLASSES if c != C.REST_CLASS]
    seq, t0 = 0, time.time()
    next_sign = t0 + 2.0
    async with websockets.connect(f"ws://{host}:{port}") as ws:
        print(f"streaming synthetic IMU data to ws://{host}:{port}")
        while seconds <= 0 or time.time() - t0 < seconds:
            if time.time() >= next_sign:
                label = labels[rng.integers(len(labels))]
                stream.perform(label)
                print(f"performing {label} (press its key when it ends)")
                next_sign = time.time() + 3.0
            x = stream.next(1)[0]
            msg = {"seq": seq, "timestamp_us": int(seq * 10_000), "temp_c": 30.0}
            msg.update({a: round(float(v), 4) for a, v in zip(C.AXES, x)})
            await ws.send(json.dumps(msg))
            seq += 1
            await asyncio.sleep(max(0.0, t0 + seq / C.SAMPLE_RATE_HZ - time.time()))


# --------------------------------------------------------------------------- serial mode
class SimulatedDevice:
    """Behaves like the inference firmware's serial port: readline() / write()."""

    def __init__(self, artifacts=None, threshold=C.THRESHOLD, speed=1.0, seed=0, random_signs=False):
        art = Path(artifacts or C.ARTIFACTS_DIR)
        norm = json.loads((art / "norm_params.json").read_text())
        self.classes = norm["classes"]
        self.mean = np.array(norm["mean"], np.float32)
        self.std = np.array(norm["std"], np.float32)
        self.predict = self._load_model(art)
        self.stream = Stream(seed)
        self.rng = np.random.default_rng(seed)
        self.random_signs = random_signs
        self.threshold = threshold
        self.speed = speed
        self.buffer = self.stream.next(C.WINDOW)
        self.lines = queue.Queue()
        self.lines.put("#CLASSES," + ",".join(self.classes))
        self.last_accept = -1e9
        self.t_ms = 0
        self.next_due = time.time()
        self.invoke_us = []

    @staticmethod
    def _load_model(art):
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        path = art / "gesture_model_float32.tflite"
        if path.exists():
            interp = tf.lite.Interpreter(model_path=str(path))
            interp.allocate_tensors()
            i, o = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]

            def run(x):
                interp.set_tensor(i, x[None].astype(np.float32))
                interp.invoke()
                return interp.get_tensor(o)[0]
            return run
        model = tf.keras.models.load_model(art / "gesture_model.keras")
        return lambda x: model(x[None], training=False).numpy()[0]

    # serial-port interface ---------------------------------------------------------
    def write(self, data):
        text = data.decode() if isinstance(data, bytes) else data
        for cmd in text.strip().splitlines():
            if cmd.startswith("t "):
                self.threshold = float(cmd[2:])
                self.lines.put(f"# threshold = {self.threshold:.2f}")
            elif cmd.startswith("sim "):           # test hook: sign this word now
                self.stream.perform(cmd[4:].strip())

    def perform(self, label):
        self.stream.perform(label)

    def readline(self):
        if self.lines.empty():
            self._step()
        return (self.lines.get() + "\n").encode()

    def close(self):
        pass

    # one 0.5 s inference step --------------------------------------------------------
    def _step(self):
        if self.speed > 0:
            time.sleep(max(0.0, self.next_due - time.time()))
            self.next_due = max(self.next_due, time.time() - 1) + C.STRIDE / C.SAMPLE_RATE_HZ / self.speed
        if self.random_signs and self.stream.performing is None and self.rng.random() < 0.25:
            self.stream.perform(self.classes[self.rng.integers(len(self.classes))])
        self.buffer = np.concatenate([self.buffer[C.STRIDE:], self.stream.next(C.STRIDE)])
        self.t_ms += 1000 * C.STRIDE // C.SAMPLE_RATE_HZ
        t0 = time.perf_counter()
        p = self.predict((self.buffer - self.mean) / self.std)
        us = int((time.perf_counter() - t0) * 1e6)
        self.invoke_us.append(us)
        top = int(np.argmax(p))
        accepted = (self.classes[top] != C.REST_CLASS and p[top] >= self.threshold
                    and self.t_ms - self.last_accept >= C.COOLDOWN_S * 1000)
        if accepted:
            self.last_accept = self.t_ms
        self.lines.put(f"#PRED,{self.t_ms},{self.classes[top]},{p[top]:.3f},{int(accepted)},{us},"
                       + ",".join(f"{v:.3f}" for v in p))
        if self.t_ms % 10_000 == 0:
            inv = self.invoke_us[-20:]
            self.lines.put(f"#STAT,invoke_avg_us={int(np.mean(inv))},invoke_min_us={min(inv)},"
                           f"invoke_max_us={max(inv)},pre_us=0,arena=0,heap=0,rate_hz=100.0,lost=0,"
                           f"bad_reads=0,reinit=0,threshold={self.threshold:.2f},int8=0")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["dataset", "capture", "serial"])
    ap.add_argument("--classes", choices=list(C.CLASS_SETS), default=C.CLASS_SET)
    ap.add_argument("--out", default=str(C.DATA_DIR / "synthetic"))
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=C.WS_PORT)
    ap.add_argument("--seconds", type=float, default=0, help="0 = run until Ctrl+C")
    ap.add_argument("--artifacts", default=str(C.ARTIFACTS_DIR))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    C.use_class_set(args.classes)

    if args.mode == "dataset":
        make_dataset(args.out, args.per_class, C.CLASSES, args.seed)
    elif args.mode == "capture":
        asyncio.run(stream_capture(args.host, args.port, args.seconds, args.seed))
    else:
        dev = SimulatedDevice(args.artifacts, seed=args.seed, random_signs=True)
        t0 = time.time()
        while args.seconds <= 0 or time.time() - t0 < args.seconds:
            sys.stdout.write(dev.readline().decode())
            sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
