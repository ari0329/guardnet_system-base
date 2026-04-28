"""
GuardNet Production Configuration
"""
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
CLIPS_DIR  = os.path.join(BASE_DIR, "clips")
DATA_DIR   = os.path.join(BASE_DIR, "data")

for _d in [MODEL_DIR, LOG_DIR, CLIPS_DIR]:
    os.makedirs(_d, exist_ok=True)

MODEL_PATH   = os.path.join(MODEL_DIR, "guardnet_model.h5")
LOG_FILE     = os.path.join(LOG_DIR,   "violence_events.csv")
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolov8n.pt")

# Frame / Sequence
FRAME_WIDTH     = 224
FRAME_HEIGHT    = 224
SEQUENCE_LENGTH = 16
FRAME_SKIP      = 2

# Model
CNN_BACKBONE  = "MobileNetV2"
LSTM_UNITS    = 256
DROPOUT_RATE  = 0.5
NUM_CLASSES   = 2

# Detection
VIOLENCE_THRESHOLD     = 0.70
ALERT_COOLDOWN_SECONDS = 5

# Clip extraction
PRE_BUFFER_SECONDS  = 3
POST_BUFFER_SECONDS = 3

# YOLO
ENABLE_YOLO     = True
YOLO_CONFIDENCE = 0.4
YOLO_IOU        = 0.45

# Heatmap
HEATMAP_ALPHA = 0.45

# Training
BATCH_SIZE              = 8
EPOCHS                  = 30
LEARNING_RATE           = 1e-4
VALIDATION_SPLIT        = 0.2
EARLY_STOPPING_PATIENCE = 7

# Colours (BGR)
COLOR_NORMAL   = (0,   200,  50)
COLOR_VIOLENCE = (0,    30, 220)
COLOR_BOX      = (255, 165,   0)