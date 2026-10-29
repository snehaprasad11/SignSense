"""Shared settings for every SignSense script.

Anything that must agree between capture, training, export and the live server lives here:
class list, window size, sampling rate, thresholds and folder paths. The firmware gets the
same values through the generated `model_config.h` (see export_model.py), so nothing is
copied by hand.
"""
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
ARTIFACTS_DIR = ROOT / "artifacts"
LOGS_DIR = ROOT / "logs"
FIRMWARE_INFERENCE_DIR = ROOT / "firmware" / "esp32s3_inference"

# --------------------------------------------------------------------------- classes
# Class order is alphabetical; it fixes the output index of the network, so it must not change
# between training and deployment (export_model.py writes it into model_config.h).
CLASS_SETS = {
    "signs": ["come", "hello", "help", "please", "rest", "thank_you", "yes"],
    "directions": ["down", "left", "rest", "right", "up"],
}
CLASS_SET = "signs"
CLASSES = CLASS_SETS[CLASS_SET]
REST_CLASS = "rest"

# Word shown / spoken for each class ("" = say nothing).
WORDS = {
    "hello": "HELLO",
    "thank_you": "THANK YOU",
    "yes": "YES",
    "please": "PLEASE",
    "help": "HELP",
    "come": "COME",
    "rest": "",
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
}

# How to perform each sign; shown by the capture tool and the guided live test.
SIGN_HINTS = {
    "hello": "open hand at the temple, repeated wave outwards",
    "thank_you": "flat hand from the chin, forward and down",
    "yes": "fist nodding up and down",
    "please": "flat hand circling on the chest",
    "help": "fist on the open palm, lift both up",
    "come": "index finger beckoning towards you",
    "rest": "hand still, relaxed",
    "up": "move hand up",
    "down": "move hand down",
    "left": "move hand left",
    "right": "move hand right",
}

# Keyboard labels for capture_server.py: 1..6 = signs in this order, space = rest.
SIGN_KEY_ORDER = {
    "signs": ["hello", "thank_you", "yes", "please", "help", "come"],
    "directions": ["up", "down", "left", "right"],
}

# --------------------------------------------------------------------------- signal
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]   # g (±2 g) and °/s (±250 °/s)
N_AXES = len(AXES)
SAMPLE_RATE_HZ = 100
WINDOW = 100                  # 1 s window = 100 timesteps x 6 axes
STRIDE = 50                   # device classifies every 0.5 s (50 % overlap)
ACCEL_RANGE_G = 2.0
GYRO_RANGE_DPS = 250.0

# --------------------------------------------------------------------------- live decision
THRESHOLD = 0.84              # tuned from live test #4 (see analyze_live.py)
COOLDOWN_S = 0.9              # one sign is reported once
SENTENCE_GAP_S = 6.0          # words closer than this join one sentence

# --------------------------------------------------------------------------- network
WS_PORT = 8765                # capture firmware connects to ws://<laptop-ip>:8765
SERIAL_BAUD = 115200

SEED = 42


def use_class_set(name):
    """Switch every module that reads config.CLASSES to another class set."""
    global CLASS_SET, CLASSES
    CLASS_SET = name
    CLASSES = CLASS_SETS[name]
    return CLASSES


def word(label):
    return WORDS.get(label, label.upper())
