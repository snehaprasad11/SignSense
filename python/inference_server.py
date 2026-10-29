"""Show what the ESP32 recognises: big word, sentence, speech, and a guided live test.

    ESP32 serial ─► parse #PRED/#STAT ─┬─► big word + sentence + speech (pyttsx3)
                                       ├─► CSV log of every window (logs/inference_live_*.csv)
                                       └─► guided test (--test): "get ready: YES (fist nodding)… GO!"
                                             -> logs/inference_test_*.csv -> analyze_live.py

Live use
    python inference_server.py --speak --quiet --threshold 0.84 --port COM5
Guided live test (random prompts, every prediction labelled with the expected word)
    python inference_server.py --test --reps 8 --ready 2 --port COM5
Without hardware (needs artifacts from train.py + export_model.py)
    python inference_server.py --sim --test --reps 2
"""
import argparse
import csv
import queue
import random
import sys
import threading
import time
from datetime import datetime

import config as C

BAR_WIDTH = 30


# --------------------------------------------------------------------------- device link
def open_device(args):
    if args.sim:
        from simulate_esp32 import SimulatedDevice
        return SimulatedDevice(args.artifacts, threshold=args.threshold, speed=args.sim_speed,
                               random_signs=not args.test)
    try:
        import serial
    except ImportError:
        sys.exit("pip install pyserial")
    if not args.port:
        from serial.tools import list_ports
        ports = [p.device for p in list_ports.comports()]
        sys.exit(f"--port is required; available ports: {', '.join(ports) or 'none'}")
    dev = serial.Serial(args.port, C.SERIAL_BAUD, timeout=1)
    time.sleep(0.5)
    return dev


def parse_pred(line, classes):
    # #PRED,<t_ms>,<label>,<conf>,<accepted>,<invoke_us>,<p0>,...
    parts = line.split(",")
    probs = [float(v) for v in parts[6:]]
    return {"t_ms": int(parts[1]), "label": parts[2], "conf": float(parts[3]),
            "accepted": parts[4] == "1", "invoke_us": int(parts[5]),
            "probs": dict(zip(classes, probs))}


def parse_stat(line):
    return dict(kv.split("=", 1) for kv in line.split(",")[1:] if "=" in kv)


# --------------------------------------------------------------------------- speech
class Speaker:
    def __init__(self, enabled):
        self.q = queue.Queue()
        self.enabled = enabled
        if enabled:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
        except Exception as e:  # pragma: no cover
            print(f"[speech disabled: {e}]")
            return
        while True:
            engine.say(self.q.get())
            engine.runAndWait()

    def say(self, text):
        if self.enabled and text:
            self.q.put(text)


# --------------------------------------------------------------------------- server
class Server:
    def __init__(self, args):
        self.args = args
        self.dev = open_device(args)
        self.classes = list(C.CLASSES)
        self.speaker = Speaker(args.speak)
        self.sentence, self.last_word_t = [], 0.0
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        kind = "test" if args.test else "live"
        self.log_path = C.LOGS_DIR / f"inference_{kind}_{stamp}.csv"
        self.stats_path = C.LOGS_DIR / f"device_stats_{stamp}.csv"
        self.log_file = open(self.log_path, "w", newline="")
        self.log = None
        self.stats_log = None
        self.stats_file = None
        self.expected, self.trial, self.phase = "", -1, ""
        self.trial_hits = []
        self.dev.write(f"t {args.threshold:.2f}\n".encode())

    def _writer(self):
        if self.log is None:
            self.log = csv.writer(self.log_file)
            self.log.writerow(["host_time", "t_ms", "label", "conf", "accepted", "invoke_us", "trial",
                               "expected", "phase"] + [f"p_{c}" for c in self.classes])
        return self.log

    def read_line(self):
        raw = self.dev.readline()
        return raw.decode(errors="replace").strip() if raw else ""

    def handle(self, line):
        if line.startswith("#CLASSES"):
            self.classes = line.split(",")[1:]
        elif line.startswith("#PRED"):
            try:
                p = parse_pred(line, self.classes)
            except (ValueError, IndexError):
                return None
            self._writer().writerow([f"{time.time():.3f}", p["t_ms"], p["label"], f"{p['conf']:.3f}",
                                     int(p["accepted"]), p["invoke_us"], self.trial, self.expected, self.phase]
                                    + [f"{p['probs'].get(c, 0):.3f}" for c in self.classes])
            self.log_file.flush()
            if not self.args.quiet and not self.args.test:
                self.draw_bars(p)
            if p["accepted"]:
                self.on_word(p)
            return p
        elif line.startswith("#STAT"):
            s = parse_stat(line)
            if self.stats_log is None:
                self.stats_file = open(self.stats_path, "w", newline="")
                self.stats_log = csv.DictWriter(self.stats_file, fieldnames=["host_time"] + list(s))
                self.stats_log.writeheader()
            self.stats_log.writerow({"host_time": f"{time.time():.3f}", **s})
            self.stats_file.flush()
            if not self.args.quiet:
                print(f"[device] Invoke {int(s.get('invoke_avg_us', 0)) / 1000:.1f} ms · "
                      f"rate {s.get('rate_hz')} Hz · lost {s.get('lost')} · heap {s.get('heap')}")
        elif line and not self.args.quiet:
            print(f"[esp32] {line}")
        return None

    def draw_bars(self, p):
        print("\x1b[2J\x1b[H", end="")
        print(f"t = {p['t_ms'] / 1000:7.1f} s   Invoke {p['invoke_us'] / 1000:.0f} ms   "
              f"threshold {self.args.threshold:.2f}\n")
        for c in self.classes:
            v = p["probs"].get(c, 0)
            mark = "◄" if c == p["label"] else " "
            print(f"  {c:>10s} {'█' * int(v * BAR_WIDTH):<{BAR_WIDTH}} {v:5.2f} {mark}")
        print(f"\n  sentence: {' '.join(self.sentence)}")

    def on_word(self, p):
        word = C.word(p["label"])
        if not word:
            return
        now = time.time()
        if now - self.last_word_t > C.SENTENCE_GAP_S:
            self.sentence = []
        self.sentence.append(word)
        self.last_word_t = now
        self.trial_hits.append(p["label"])
        if not self.args.test:
            print(f"\n   >>>  {word}  <<<   ({p['conf']:.2f})      sentence: {' '.join(self.sentence)}\n")
        self.speaker.say(word)

    # ---- modes
    def run_live(self):
        print(f"logging to {self.log_path}  (Ctrl+C to stop)")
        while True:
            self.handle(self.read_line())

    def pump(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self.handle(self.read_line())

    def run_test(self):
        words = [c for c in self.classes if c != C.REST_CLASS]
        trials = [c for c in words for _ in range(self.args.reps)]
        if not self.args.no_rest:
            trials += [C.REST_CLASS] * self.args.reps
        random.Random(self.args.seed).shuffle(trials)
        print(f"guided live test: {len(trials)} trials, logging to {self.log_path}\n")
        correct = 0
        for i, label in enumerate(trials):
            self.trial, self.expected, self.phase = i, label, "ready"
            name = C.word(label) or "REST (keep still)"
            print(f"[{i + 1}/{len(trials)}] get ready: {name}  ({C.SIGN_HINTS.get(label, '')})")
            self.pump(self.args.ready)
            print("    GO!")
            self.phase, self.trial_hits = "go", []
            if self.args.sim:
                self.dev.perform(label)
            self.pump(self.args.trial)
            got = self.trial_hits[0] if self.trial_hits else C.REST_CLASS
            ok = got == label
            correct += ok
            print(f"    -> {C.word(got) or 'nothing'}  {'✓' if ok else '✗'}\n")
            self.phase = "pause"
            self.pump(self.args.pause)
        print(f"trial accuracy {correct}/{len(trials)} = {correct / len(trials):.1%}")
        print(f"analyse: python analyze_live.py {self.log_path} --threshold {self.args.threshold}")

    def close(self):
        self.log_file.close()
        if self.stats_file:
            self.stats_file.close()
        self.dev.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port, e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--threshold", type=float, default=C.THRESHOLD)
    ap.add_argument("--speak", action="store_true", help="speak each word (pyttsx3)")
    ap.add_argument("--quiet", action="store_true", help="no confidence bars / device chatter")
    ap.add_argument("--test", action="store_true", help="guided live test")
    ap.add_argument("--reps", type=int, default=8, help="trials per word in --test")
    ap.add_argument("--ready", type=float, default=2.0, help="seconds between prompt and GO")
    ap.add_argument("--trial", type=float, default=3.0, help="seconds recorded after GO")
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--no-rest", action="store_true", help="no rest trials in --test")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sim", action="store_true", help="use simulate_esp32.SimulatedDevice")
    ap.add_argument("--sim-speed", type=float, default=1.0, help="simulator time factor (2 = twice as fast)")
    ap.add_argument("--artifacts", default=str(C.ARTIFACTS_DIR))
    args = ap.parse_args()
    if args.sim and args.sim_speed != 1.0:
        for k in ("ready", "trial", "pause"):
            setattr(args, k, getattr(args, k) / args.sim_speed)

    server = Server(args)
    try:
        server.run_test() if args.test else server.run_live()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        print(f"\nlog: {server.log_path}")


if __name__ == "__main__":
    main()
