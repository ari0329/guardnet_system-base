# 🛡️ GuardNet — AI-Based Real-Time Violence Detection System

GuardNet is a production-ready, modular AI system that detects violent activity (fighting, hitting, aggressive motion) from live webcam feeds or recorded video using a **CNN + LSTM** deep learning pipeline.

---

## 📁 Project Structure

```
guardnet/
├── config/
│   └── config.py              # All tuneable settings
├── models/
│   └── guardnet_model.py      # CNN-LSTM architecture & inference wrapper
├── utils/
│   ├── preprocessing.py       # Frame extraction, normalisation, heatmap
│   ├── detection_engine.py    # Real-time orchestration loop
│   ├── alerts.py              # Console / sound / email alerts + CSV logger
│   └── person_detector.py     # YOLOv8 multi-person bounding boxes
├── logs/
│   └── violence_events.csv    # Auto-created event log
├── train.py                   # Training pipeline entry point
├── demo.py                    # Real-time detection entry point
├── dashboard.py               # Streamlit GUI dashboard
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Demo Mode (no training needed)
```bash
# Webcam (index 0)
python demo.py --demo_mode

# Pre-recorded video
python demo.py --source /path/to/video.mp4 --demo_mode

# Custom threshold
python demo.py --demo_mode --threshold 0.65
```

### 3. Train on Your Dataset
Organise your data as:
```
data/
├── violence/        ← .mp4 / .avi clips of violent scenes
└── non-violence/    ← .mp4 / .avi clips of normal scenes
```

Then run:
```bash
python train.py --data_dir ./data --epochs 30
```

Public datasets that work out of the box:
- **Hockey Fight Dataset** – <https://www.kaggle.com/datasets/yassershrief/hockey-fight-videos>
- **RWF-2000** – <https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection>
- **UCF Crime Dataset** – <https://www.crcv.ucf.edu/projects/real-world/>

### 4. Real-Time Detection (after training)
```bash
python demo.py --source 0          # webcam
python demo.py --source video.mp4  # file
```

### 5. Streamlit Dashboard (GUI)
```bash
streamlit run dashboard.py
```
Open <http://localhost:8501> in your browser.

---

## 🧠 Model Architecture

```
Input (batch, 16, 224, 224, 3)
        │
        ▼
TimeDistributed(MobileNetV2 – ImageNet pretrained)   ← spatial features
        │
TimeDistributed(BatchNorm + Dropout)
        │
        ▼
Bidirectional LSTM (256 units, return_sequences=True) ← temporal reasoning
        │  Dropout
Bidirectional LSTM (128 units)
        │  Dropout
        ▼
Dense(128, ReLU) → Dropout → Dense(2, Softmax)
        │
        ▼
[P(non-violent), P(violent)]
```

- **Backbone**: MobileNetV2 (default) or ResNet50 (set in `config/config.py`)
- **Sequence length**: 16 frames  
- **Input resolution**: 224 × 224  
- **Classes**: 0 = non-violent, 1 = violent  

---

## ⚙️ Configuration (`config/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `VIOLENCE_THRESHOLD` | `0.70` | Probability above which an alert fires |
| `SEQUENCE_LENGTH` | `16` | Frames fed to LSTM per inference |
| `FRAME_SKIP` | `2` | Process every Nth frame (speed/accuracy trade-off) |
| `CNN_BACKBONE` | `MobileNetV2` | Feature extractor (`MobileNetV2` or `ResNet50`) |
| `LSTM_UNITS` | `256` | BiLSTM hidden units |
| `ENABLE_YOLO` | `True` | Person detection overlay |
| `ENABLE_HEATMAP` | `True` | Motion intensity heatmap |
| `ENABLE_SOUND_ALERT` | `True` | Terminal bell on detection |
| `ENABLE_EMAIL_ALERT` | `False` | SMTP email alert (fill SMTP_* fields) |
| `RECORD_OUTPUT` | `False` | Save annotated video to disk |

---

## 🖥️ Keyboard Controls (demo window)

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `H` | Toggle heatmap overlay |

---

## 🔔 Alert System

- **Console**: ASCII banner with confidence bar, timestamp, source
- **Sound**: Cross-platform terminal bell (configurable with `playsound`)
- **Email**: SMTP email with HTML template (set `ENABLE_EMAIL_ALERT=True` + SMTP config)
- **Log**: `logs/violence_events.csv` – timestamp, confidence, source

### Email Alert Setup
```python
# config/config.py
ENABLE_EMAIL_ALERT = True
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "your@gmail.com"
SMTP_PASS = "your-app-password"   # Gmail App Password
ALERT_RECIPIENT = "security@example.com"
```

---

## 📈 Performance Tips

| Optimisation | Effect |
|---|---|
| `FRAME_SKIP = 3` | +50% FPS, slight accuracy loss |
| `CNN_BACKBONE = MobileNetV2` | Fastest; use ResNet50 for accuracy |
| `SEQUENCE_LENGTH = 8` | Halves LSTM latency |
| `ENABLE_YOLO = False` | Eliminates YOLO overhead |
| Run on GPU | 10–20× faster than CPU |

---

## 📦 Deliverables Checklist

- ✅ `config/config.py` — all settings in one place  
- ✅ `models/guardnet_model.py` — CNN-LSTM architecture + training  
- ✅ `utils/preprocessing.py` — frame pipeline + heatmap  
- ✅ `utils/detection_engine.py` — real-time loop  
- ✅ `utils/alerts.py` — multi-channel alert + CSV logger  
- ✅ `utils/person_detector.py` — YOLOv8 person boxes  
- ✅ `train.py` — full training pipeline  
- ✅ `demo.py` — real-time detection script  
- ✅ `dashboard.py` — Streamlit GUI dashboard  
- ✅ `requirements.txt`  
- ✅ `README.md`  

---

## 📝 License
MIT – free to use, modify, and distribute.

















py -3.11 -m venv myenv
myenv\Scripts\activate
pip install ultralytics
pip install tensorflow
pip install tensorboard
pip install opencv-python 
pip install opencv-python numpy ultralytics streamlit pandas plotly Pillow 