"""
GuardNet Configuration Settings
"""

from dotenv import load_dotenv
import os
load_dotenv() 

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODEL_DIR, "guardnet_model.h5")
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolov8n.pt")
LOG_FILE = os.path.join(LOG_DIR, "violence_events.csv")

# ─── Frame / Sequence Settings ────────────────────────────────────────────────
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
SEQUENCE_LENGTH = 16          # frames per LSTM sequence
FRAME_SKIP = 2               # process every Nth frame

# ─── Model Architecture ───────────────────────────────────────────────────────
CNN_BACKBONE = "MobileNetV2"  # or "ResNet50"
LSTM_UNITS = 256
DROPOUT_RATE = 0.5
NUM_CLASSES = 2              # violent, non-violent

# ─── Detection Thresholds ─────────────────────────────────────────────────────
VIOLENCE_THRESHOLD = 0.70    # probability above this → alert
CONFIDENCE_DISPLAY = True

# ─── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 7

# ─── Alert Settings ───────────────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 5   # min seconds between repeated alerts
ENABLE_SOUND_ALERT = True
ENABLE_EMAIL_ALERT = True    # set True + fill SMTP below to enable
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = os.getenv("SMTP_USER", "your@email.com")
SMTP_PASS = os.getenv("SMTP_PASS", "yourpassword")
ALERT_RECIPIENT = os.getenv("ALERT_RECIPIENT", "security@example.com")

# ─── YOLO Person Detection ────────────────────────────────────────────────────
ENABLE_YOLO = True
YOLO_CONFIDENCE = 0.4
YOLO_IOU = 0.45

# ─── Heatmap ──────────────────────────────────────────────────────────────────
ENABLE_HEATMAP = True
HEATMAP_ALPHA = 0.4          # overlay transparency

# ─── Display ──────────────────────────────────────────────────────────────────
WINDOW_NAME = "GuardNet – Real-Time Violence Detection"
DISPLAY_FPS = True
DISPLAY_HEATMAP = True
RECORD_OUTPUT = False        # save annotated video
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "output_annotated.avi")

# ─── Colour Palette (BGR) ─────────────────────────────────────────────────────
COLOR_NORMAL   = (0,   200,  50)   # green
COLOR_VIOLENCE = (0,    30, 220)   # red
COLOR_BOX      = (255, 165,   0)   # orange person bbox
COLOR_TEXT_BG  = (20,   20,  20)
COLOR_HEATMAP_LOW  = (0, 255, 0)
COLOR_HEATMAP_HIGH = (0, 0, 255)
