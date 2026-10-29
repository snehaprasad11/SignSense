"""Record labelled 1 s windows streamed by the capture firmware over a Wi-Fi WebSocket.

The ESP32 (firmware/esp32s3_capture) connects to ws://<this-laptop>:8765 and sends one JSON
object per sample:

    {"seq": 1234, "timestamp_us": 12345678, "ax": 0.01, ..., "gz": -0.4, "temp_c": 31.2}

Keys (generated from config.SIGN_KEY_ORDER, so any word set works):
    1-6    sign words (1 = HELLO, 2 = THANK YOU, 3 = YES, 4 = PLEASE, 5 = HELP, 6 = COME)
    space  rest
    r      toggle auto-rest (records a rest window every 1.5 s)
    z      undo the last key press
    q      quit

Press the key right AFTER finishing a sign. With the default alignments (0, 25, 50 samples)
each press saves 3 windows ending 0, 0.25 and 0.5 s after the press, so the sign sits at
different positions in the window, as it does when the device classifies every 0.5 s live.
Every window goes through quality checks first; rejected windows are not saved.

    python capture_server.py
    python capture_server.py --alignments 0 --classes directions
"""
import argparse
import asyncio
import json
import sys
import time
from collections import deque

import numpy as np

import config as C
import curate_dataset as io  # window CSV helpers

try:
    import websockets
except ImportError:  # pragma: no cover
    sys.exit("pip install websockets")


# --------------------------------------------------------------------------- quality checks
MAX_MISSING = 0                 # samples lost over Wi-Fi inside one window
DT_TOLERANCE = 0.10             # mean sample period must be 10 ms ± 10 %
MAX_DT_US = 25_000              # no single gap longer than 25 ms
SAT_FRACTION = 0.05             # > 5 % saturated samples = reject
SIGN_MIN_GYRO_STD = 8.0         # °/s; a real sign moves the hand
REST_MAX_GYRO_STD = 6.0         # °/s; rest must be (almost) still
REST_MAX_GYRO_MEAN = 5.0        # °/s; larger mean at rest = bad gyro calibration


def check_window(label, seqs, ts_us, x):
    """Return a list of problems with one window (empty list = OK)."""
    problems = []
    missing = int(seqs[-1] - seqs[0] + 1 - len(seqs))
    if missing > MAX_MISSING:
        problems.append(f"{missing} samples missing (Wi-Fi drop)")
    dt = np.diff(ts_us)
    expected = 1e6 / C.SAMPLE_RATE_HZ
    if abs(dt.mean() - expected) > DT_TOLERANCE * expected or dt.max() > MAX_DT_US:
        problems.append(f"bad timing (mean {dt.mean() / 1000:.1f} ms, max {dt.max() / 1000:.1f} ms)")
    sat = (np.abs(x[:, :3]) >= 0.99 * C.ACCEL_RANGE_G).any(axis=1) | \
          (np.abs(x[:, 3:]) >= 0.99 * C.GYRO_RANGE_DPS).any(axis=1)
    if sat.mean() > SAT_FRACTION:
        problems.append(f"sensor saturated in {sat.mean():.0%} of samples")
    gyro_std = float(x[:, 3:].std(axis=0).max())
    if label == C.REST_CLASS:
        if gyro_std > REST_MAX_GYRO_STD:
            problems.append(f"rest that moves (gyro std {gyro_std:.1f} °/s)")
        gyro_mean = float(np.abs(x[:, 3:].mean(axis=0)).max())
        if gyro_mean > REST_MAX_GYRO_MEAN:
            problems.append(f"gyro offset {gyro_mean:.1f} °/s at rest: recalibrate the device")
    elif gyro_std < SIGN_MIN_GYRO_STD:
        problems.append(f"sign with no motion (gyro std {gyro_std:.1f} °/s): key pressed too late?")
    return problems


# --------------------------------------------------------------------------- recorder
class Recorder:
    def __init__(self, args):
        self.args = args
        self.data_dir = args.data_dir
        self.buffer = deque(maxlen=4 * C.WINDOW)      # (seq, ts_us, [6 axes])
        self.pending = []                             # (label, end_seq_list, press_id)
        self.history = []                             # [(label, n_saved)] for undo
        self.session = int(time.time())
        self.auto_rest = False
        self.next_auto_rest = 0.0
        self.counts = {c: 0 for c in C.CLASSES}
        self.rx = 0
        self.last_rate_print = time.time()
        for c in C.CLASSES:
            p = io.csv_path(c, self.data_dir)
            if p.exists():
                n = io.next_window_id(p)
                self.counts[c] = n
        self.keys = {str(i + 1): lbl for i, lbl in enumerate(C.SIGN_KEY_ORDER[C.CLASS_SET])}
        self.keys[" "] = C.REST_CLASS

    # ---- incoming samples
    def on_sample(self, msg):
        if msg.get("type") == "stats":
            print(f"\n[ESP32] {msg}")
            return
        if "seq" not in msg:
            return
        self.buffer.append((int(msg["seq"]), int(msg["timestamp_us"]),
                            [float(msg[a]) for a in C.AXES]))
        self.rx += 1
        self.process_pending()
        now = time.time()
        if self.auto_rest and now >= self.next_auto_rest:
            self.next_auto_rest = now + 1.5
            self.request(C.REST_CLASS, alignments=[0])
        if now - self.last_rate_print >= 5:
            rate = self.rx / (now - self.last_rate_print)
            self.rx, self.last_rate_print = 0, now
            print(f"\r[rx {rate:5.1f} Hz]", end="", flush=True)

    # ---- labelling
    def request(self, label, alignments=None):
        if not self.buffer:
            print("\nno data from the ESP32 yet")
            return
        press_seq = self.buffer[-1][0]
        aligns = alignments if alignments is not None else self.args.alignments
        self.pending.append((label, [press_seq + a for a in aligns], []))

    def process_pending(self):
        if not self.pending:
            return
        newest = self.buffer[-1][0]
        done = []
        for item in self.pending:
            label, ends, results = item
            if newest < max(ends):
                continue
            for end in ends:
                results.append(self.extract(end))
            done.append(item)
            self.save(label, results)
        for item in done:
            self.pending.remove(item)

    def extract(self, end_seq):
        rows = [r for r in self.buffer if end_seq - C.WINDOW < r[0] <= end_seq]
        if len(rows) < C.WINDOW // 2:
            return None
        rows = rows[-C.WINDOW:]
        seqs = np.array([r[0] for r in rows])
        ts = np.array([r[1] for r in rows], dtype=np.int64)
        x = np.array([r[2] for r in rows], dtype=np.float32)
        return seqs, ts, x

    def save(self, label, results):
        path = io.csv_path(label, self.data_dir)
        saved = 0
        for res in results:
            if res is None:
                print(f"\n  ✗ {label}: buffer too short")
                continue
            seqs, ts, x = res
            problems = check_window(label, seqs, ts, x) if len(x) == C.WINDOW else ["window incomplete"]
            if problems and not self.args.force:
                print(f"\n  ✗ {label}: " + "; ".join(problems))
                continue
            io.append_window(path, io.next_window_id(path), label, self.session, ts, x)
            saved += 1
        self.counts[label] += saved
        if saved:
            self.history.append((label, saved))
        self.print_counts(f"✓ {label} +{saved}")

    def undo(self):
        if not self.history:
            print("\nnothing to undo")
            return
        label, n = self.history.pop()
        for _ in range(n):
            io.remove_last_window(io.csv_path(label, self.data_dir))
        self.counts[label] -= n
        self.print_counts(f"↶ removed {n} × {label}")

    def print_counts(self, head):
        counts = "  ".join(f"{c}:{self.counts[c]}" for c in C.CLASSES)
        print(f"\n{head}   [{counts}]")

    def on_key(self, ch):
        if ch in self.keys:
            label = self.keys[ch]
            self.request(label)
        elif ch == "r":
            self.auto_rest = not self.auto_rest
            print(f"\nauto-rest {'ON' if self.auto_rest else 'OFF'}")
        elif ch == "z":
            self.undo()
        elif ch == "q":
            return False
        return True


# --------------------------------------------------------------------------- keyboard
async def keyboard_loop(rec, stop):
    if rec.args.pynput:
        from pynput import keyboard  # optional dependency
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        def on_press(key):
            ch = " " if key == keyboard.Key.space else getattr(key, "char", None)
            if ch:
                loop.call_soon_threadsafe(queue.put_nowait, ch)

        keyboard.Listener(on_press=on_press).start()
        while not stop.is_set():
            if not rec.on_key(await queue.get()):
                stop.set()
        return

    if sys.platform == "win32":
        import msvcrt
        while not stop.is_set():
            while msvcrt.kbhit():
                if not rec.on_key(msvcrt.getwch().lower()):
                    stop.set()
            await asyncio.sleep(0.02)
    else:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        loop.add_reader(fd, lambda: queue.put_nowait(sys.stdin.read(1)))
        try:
            while not stop.is_set():
                if not rec.on_key((await queue.get()).lower()):
                    stop.set()
        finally:
            loop.remove_reader(fd)
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# --------------------------------------------------------------------------- main
async def main(args):
    rec = Recorder(args)
    stop = asyncio.Event()

    async def handler(ws, *_):
        print(f"\nESP32 connected from {ws.remote_address}")
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                for m in (msg if isinstance(msg, list) else [msg]):
                    rec.on_sample(m)
        except websockets.ConnectionClosed:
            pass
        print("\nESP32 disconnected")

    async with websockets.serve(handler, "0.0.0.0", args.port, ping_interval=None):
        print(f"Listening on ws://0.0.0.0:{args.port}  (put this laptop's IP in secrets.h)")
        for k, lbl in rec.keys.items():
            name = "space" if k == " " else k
            print(f"  {name:>5} = {C.word(lbl) or 'rest':10s} ({C.SIGN_HINTS.get(lbl, '')})")
        print("      r = auto-rest   z = undo   q = quit")
        rec.print_counts("existing windows")
        await keyboard_loop(rec, stop)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=C.WS_PORT)
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    ap.add_argument("--classes", choices=list(C.CLASS_SETS), default=C.CLASS_SET)
    ap.add_argument("--alignments", type=lambda s: [int(v) for v in s.split(",")], default=[0, 25, 50],
                    help="window ends, in samples after the key press (default 0,25,50)")
    ap.add_argument("--force", action="store_true", help="save windows even if quality checks fail")
    ap.add_argument("--pynput", action="store_true", help="read keys with pynput (global hotkeys)")
    a = ap.parse_args()
    C.use_class_set(a.classes)
    try:
        asyncio.run(main(a))
    except KeyboardInterrupt:
        pass
