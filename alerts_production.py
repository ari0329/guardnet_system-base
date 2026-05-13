"""
GuardNet – Alert & Logging System (Production v5)
==================================================
Changes in v5
  ✓ Violence threshold lowered to 20% (0.20) — catches minimal violence early
  ✓ Local alert sound played on detection (via playsound / pygame fallback)
  ✓ Twilio SMS: sends violence alert to nearest police station number
  ✓ Twilio Voice: automated phone call with TwiML reading out incident details
  ✓ All contact numbers configured at runtime from the dashboard sidebar —
    no hardcoded phone numbers anywhere in this file
"""

import os
import sys
import csv
import time
import smtplib
import threading
from datetime import datetime
from email.mime.text      import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base      import MIMEBase
from email                import encoders

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import LOG_FILE, LOG_DIR, ALERT_COOLDOWN_SECONDS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Violence detection threshold ──────────────────────────────────────────────
# Lowered to 0.20 (20%) to catch minimal / early-stage violence.
# Override via environment variable VIOLENCE_THRESHOLD if needed.
VIOLENCE_THRESHOLD: float = float(os.getenv("VIOLENCE_THRESHOLD", "0.20"))

# ── SMTP config ───────────────────────────────────────────────────────────────
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

# ── Twilio config (set in .env or environment) ────────────────────────────────
TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID",  "")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN",   "")
TWILIO_FROM_NUMBER  = os.getenv("TWILIO_FROM_NUMBER",  "")   # E.164: +1XXXXXXXXXX
# Public URL Twilio can reach to fetch TwiML for voice calls.
# Use Twilio TwiML Bins (free, no hosting) or your own server / ngrok tunnel.
TWILIO_TWIML_URL    = os.getenv("TWILIO_TWIML_URL",    "")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Local alert sound
# ══════════════════════════════════════════════════════════════════════════════

def _play_alert_sound() -> None:
    """
    Play a local siren / beep when violence is detected.

    Priority:
        1. playsound  (pip install playsound) — plays assets/alert.wav if present
        2. pygame     (pip install pygame)    — synthesises 440 Hz beep otherwise
        3. Terminal bell \a                   — always available, last resort

    Put your siren file at  assets/alert.wav  (or .mp3) next to this script.
    """
    ALERT_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "alert.wav"
    )

    def _run():
        # ── playsound ─────────────────────────────────────────────────────
        try:
            from playsound import playsound
            if os.path.exists(ALERT_FILE):
                playsound(ALERT_FILE, block=False)
                return
        except Exception:
            pass

        # ── pygame ────────────────────────────────────────────────────────
        try:
            import pygame
            import numpy as np
            pygame.mixer.init(frequency=44100, size=-16, channels=2)
            if os.path.exists(ALERT_FILE):
                pygame.mixer.music.load(ALERT_FILE)
                pygame.mixer.music.play()
            else:
                # Synthesise a 440 Hz alert tone for 1 second
                sr   = 44100
                t    = np.linspace(0, 1.0, sr, False)
                wave = (np.sin(2 * 3.14159 * 440 * t) * 32767).astype("int16")
                stereo = np.column_stack([wave, wave])
                pygame.sndarray.make_sound(stereo).play()
            time.sleep(1.2)
            return
        except Exception:
            pass

        # ── terminal bell ─────────────────────────────────────────────────
        print("\a\a\a", end="", flush=True)

    threading.Thread(target=_run, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Twilio SMS to police station
# ══════════════════════════════════════════════════════════════════════════════

def _send_twilio_sms(
    to_number:      str,
    confidence:     float,
    camera_id:      str,
    timestamp:      str,
    location_label: str = "the monitored premises",
) -> None:
    """
    Send an SMS alert to the nearest police station via Twilio.

    Requires in .env / environment:
        TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER

    to_number must be in E.164 format, e.g. +911234567890
    """
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER):
        print("[WARN] Twilio credentials not configured — SMS skipped.")
        return
    if not to_number:
        print("[WARN] No police phone number provided — SMS skipped.")
        return

    def _run():
        try:
            from twilio.rest import Client
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            body = (
                f"[GUARDNET ALERT] VIOLENCE DETECTED\n"
                f"Time      : {timestamp}\n"
                f"Camera    : {camera_id}\n"
                f"Confidence: {confidence * 100:.1f}%\n"
                f"Location  : {location_label}\n"
                f"Action    : Immediate response required."
            )
            msg = client.messages.create(
                body=body,
                from_=TWILIO_FROM_NUMBER,
                to=to_number,
            )
            print(f"[INFO] Twilio SMS sent → {to_number}  (SID: {msg.sid})")
        except ImportError:
            print("[WARN] 'twilio' package missing — run: pip install twilio")
        except Exception as exc:
            print(f"[WARN] Twilio SMS error: {exc}")

    threading.Thread(target=_run, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# 3. Twilio automated voice call to police station
# ══════════════════════════════════════════════════════════════════════════════

def _make_twilio_voice_call(
    to_number:      str,
    confidence:     float,
    camera_id:      str,
    timestamp:      str,
    location_label: str = "the monitored premises",
) -> None:
    """
    Place an automated voice call that reads out the incident details.

    Two modes (auto-selected):
        A) TWILIO_TWIML_URL is set → Twilio fetches TwiML from that URL.
           Use a Twilio TwiML Bin (console.twilio.com → TwiML Bins) for
           zero-hosting-cost deployment.
        B) TWILIO_TWIML_URL is empty → inline TwiML is passed directly
           to the Twilio Calls API (twiml= parameter). No hosting needed.
    """
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER):
        print("[WARN] Twilio credentials not configured — voice call skipped.")
        return
    if not to_number:
        print("[WARN] No police phone number provided — voice call skipped.")
        return

    def _run():
        try:
            from twilio.rest import Client
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

            spoken = (
                f"Attention. This is an automated alert from GuardNet, "
                f"a CCTV intelligence system. "
                f"Violence has been detected at {location_label}. "
                f"Camera {camera_id} recorded an incident at {timestamp} "
                f"with a confidence level of {confidence * 100:.0f} percent. "
                f"Immediate response is required. "
                f"Repeating — Violence detected at {location_label}. "
                f"Camera {camera_id}. Time {timestamp}. "
                f"Confidence {confidence * 100:.0f} percent. "
                f"Please respond immediately. Thank you."
            )

            if TWILIO_TWIML_URL:
                call = client.calls.create(
                    to=to_number,
                    from_=TWILIO_FROM_NUMBER,
                    url=TWILIO_TWIML_URL,
                )
            else:
                twiml = (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    "<Response>"
                    f'  <Say voice="alice" language="en-IN">{spoken}</Say>'
                    "  <Pause length=\"1\"/>"
                    "</Response>"
                )
                call = client.calls.create(
                    to=to_number,
                    from_=TWILIO_FROM_NUMBER,
                    twiml=twiml,
                )
            print(f"[INFO] Twilio voice call initiated → {to_number}  (SID: {call.sid})")
        except ImportError:
            print("[WARN] 'twilio' package missing — run: pip install twilio")
        except Exception as exc:
            print(f"[WARN] Twilio voice call error: {exc}")

    threading.Thread(target=_run, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# Event logger
# ══════════════════════════════════════════════════════════════════════════════

class EventLogger:
    FIELDNAMES = ["timestamp", "camera_id", "confidence", "clip_path"]

    def __init__(self, path=LOG_FILE):
        self.path  = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def log(self, confidence: float, camera_id: str = "cam0",
            clip_path: str = "") -> dict:
        row = {
            "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id"  : camera_id,
            "confidence" : f"{confidence:.4f}",
            "clip_path"  : clip_path,
        }
        with self._lock:
            with open(self.path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)
        return row

    def read_all(self):
        try:
            with open(self.path, "r") as f:
                return list(csv.DictReader(f))
        except Exception:
            return []


# ══════════════════════════════════════════════════════════════════════════════
# Alert manager — single entry-point for all alert channels
# ══════════════════════════════════════════════════════════════════════════════

class AlertManager:
    """
    Central alert dispatcher.  Called from the inference thread on every
    violence detection event.  Respects ALERT_COOLDOWN_SECONDS per camera.
    """

    def __init__(self):
        self._last  = {}          # camera_id → epoch time of last alert
        self._lock  = threading.Lock()
        self.logger = EventLogger()

    def trigger(
        self,
        confidence:     float,
        camera_id:      str = "cam0",
        alert_email:    str = "",
        clip_path:      str = "",
        police_number:  str = "",    # E.164, e.g. +911234567890
        location_label: str = "the monitored premises",
    ) -> None:
        """
        Dispatch all alert channels if the per-camera cooldown has elapsed.

        Parameters
        ----------
        confidence      : Model probability (0–1) — threshold is 0.20
        camera_id       : Camera label (e.g. "CAM-1")
        alert_email     : Operator/security email
        clip_path       : Path to the saved incident video clip (may be empty)
        police_number   : E.164 phone of the nearest police station
        location_label  : Human-readable description of the monitored location
        """
        now = time.time()
        with self._lock:
            if now - self._last.get(camera_id, 0) < ALERT_COOLDOWN_SECONDS:
                return
            self._last[camera_id] = now

        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.log(confidence, camera_id, clip_path)

        # ── 1. Console banner ─────────────────────────────────────────────
        self._console(confidence, camera_id)

        # ── 2. Local alert sound ──────────────────────────────────────────
        _play_alert_sound()

        # ── 3. Email alert ────────────────────────────────────────────────
        if alert_email and SMTP_USER and SMTP_PASS:
            threading.Thread(
                target=self._email,
                args=(confidence, camera_id, ts, alert_email, clip_path),
                daemon=True,
            ).start()

        # ── 4. Twilio SMS to police station ───────────────────────────────
        if police_number:
            _send_twilio_sms(
                to_number      = police_number,
                confidence     = confidence,
                camera_id      = camera_id,
                timestamp      = ts,
                location_label = location_label,
            )

        # ── 5. Twilio automated voice call to police station ──────────────
        if police_number:
            _make_twilio_voice_call(
                to_number      = police_number,
                confidence     = confidence,
                camera_id      = camera_id,
                timestamp      = ts,
                location_label = location_label,
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _console(self, conf: float, cam: str) -> None:
        ts  = datetime.now().strftime("%H:%M:%S")
        bar = "█" * int(conf * 20)
        print(
            f"\n{'='*55}\n"
            f"  ⚠️  VIOLENCE DETECTED [{ts}]  Camera: {cam}\n"
            f"  Confidence: {bar} {conf*100:.1f}%\n"
            f"{'='*55}\n"
        )

    def _email(self, conf: float, cam: str, ts: str,
               recipient: str, clip_path: str) -> None:
        try:
            msg            = MIMEMultipart()
            msg["Subject"] = f"[GuardNet] ⚠️ Violence Detected – {ts}"
            msg["From"]    = SMTP_USER
            msg["To"]      = recipient
            body = f"""
            <html><body style="font-family:Arial;background:#0d1117;color:#e6edf3;padding:20px">
            <h2 style="color:#f85149">⚠️ GuardNet – Violence Alert</h2>
            <table style="border-collapse:collapse;width:100%">
              <tr><td style="padding:8px;color:#8892a4"><b>Timestamp</b></td><td>{ts}</td></tr>
              <tr><td style="padding:8px;color:#8892a4"><b>Camera</b></td><td>{cam}</td></tr>
              <tr><td style="padding:8px;color:#8892a4"><b>Confidence</b></td><td>{conf*100:.1f}%</td></tr>
            </table>
            <p style="color:#f85149">Please review the footage immediately.</p>
            </body></html>"""
            msg.attach(MIMEText(body, "html"))
            if clip_path and os.path.exists(clip_path):
                with open(clip_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f'attachment; filename="{os.path.basename(clip_path)}"',
                    )
                    msg.attach(part)
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
                s.starttls()
                s.login(SMTP_USER, SMTP_PASS)
                s.sendmail(SMTP_USER, recipient, msg.as_string())
            print(f"[INFO] Email alert sent → {recipient}")
        except Exception as exc:
            print(f"[WARN] Email failed: {exc}")


# ── Singleton accessor ────────────────────────────────────────────────────────

_alert_mgr = None

def get_alert_manager() -> AlertManager:
    global _alert_mgr
    if _alert_mgr is None:
        _alert_mgr = AlertManager()
    return _alert_mgr
